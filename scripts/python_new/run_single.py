#!/usr/bin/env python3
"""Unified entrypoint for single CLP phases and modality pretraining."""

import hashlib
import logging
import os
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from src.data.probes import build_fim_probe
from src.data.normalization import normalization_from_transform
from src.modules.aux_modules import TraceFIM
from src.modules.architectures.wrappers import BiModalModelwithPretrainedBranches
from src.modules.metrics import RunStatsBiModal
from src.trainer.trainer_classification_mm_clp import (
    TrainerClassification,
    fim_measurement_due,
)
from src.trainer.trainer_classification_mm_clp_umt import (
    TrainerClassification as UMTTrainerClassification,
    validation_controlled_umt_trainer_class,
)
from src.trainer.trainer_validation_clp import (
    ValidationControlledTrainer,
    validation_protocol_enabled,
)
from src.trainer.unimodal_references import (
    INITIALIZATION_POLICY,
    load_and_validate_unimodal_reference_pair,
)
from src.utils.prepare import (
    prepare_criterion,
    prepare_loaders_clp,
    prepare_model,
    prepare_optim_and_scheduler,
)
from src.utils.prepare_clp_data import (
    prepare_test_loaders_clp,
    prepare_training_loaders_clp,
)
from src.utils.utils_criterion import get_samples_weights, load_criterion_specific_params
from src.utils.utils_data import count_classes
from src.utils.utils_model import load_model_specific_params
from src.utils.utils_trainer import load_branch, load_training_checkpoint, manual_seed


@dataclass(frozen=True)
class ModeSpec:
    phase: int | None = None
    pretraining: str | None = None
    all_at_once: bool = False
    run_stats: bool = True
    trace_fim: bool = False
    label: str = ""


MODE_SPECS = {
    "all_at_once": ModeSpec(
        all_at_once=True, trace_fim=True, label="four-phase CLP training"
    ),
    "normal": ModeSpec(phase=2, run_stats=False, label="normal training"),
    "phase1": ModeSpec(phase=1, label="phase 1"),
    "phase2": ModeSpec(phase=2, label="phase 2"),
    "phase3": ModeSpec(phase=3, label="phase 3 intervention"),
    # Standalone Phase 4 is also the supported entrypoint for post-intervention
    # TFIM trajectories.  A zero measurement count still disables the probe.
    "phase4": ModeSpec(phase=4, trace_fim=True, label="phase 4"),
    "left_proper": ModeSpec(
        pretraining="left_proper", trace_fim=True, label="left modality pretraining"
    ),
    "right_proper": ModeSpec(
        pretraining="right_proper", trace_fim=True, label="right modality pretraining"
    ),
    "right_blurred": ModeSpec(
        pretraining="right_blurred",
        trace_fim=True,
        label="right blurred modality pretraining",
    ),
}


class Phase3StepWarmupScheduler:
    """Linearly warm up a phase LR per optimizer step, then keep it constant."""

    def __init__(
        self, optimizer, total_steps, start_factor, *, metric_prefix="phase3"
    ):
        total_steps = int(total_steps)
        start_factor = float(start_factor)
        if total_steps < 0:
            raise ValueError("phase3_lr_warmup total_steps must be non-negative")
        if not 0.0 < start_factor <= 1.0:
            raise ValueError(
                "phase3_lr_warmup_start_factor must be in (0, 1]"
            )
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.start_factor = start_factor
        self.completed_steps = 0
        self.metric_prefix = str(metric_prefix)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if self.total_steps:
            self._set_warmup_lr()

    @property
    def in_warmup(self):
        return self.completed_steps < self.total_steps

    @property
    def warmup_progress(self):
        if self.total_steps == 0:
            return 1.0
        return min(self.completed_steps / self.total_steps, 1.0)

    @property
    def warmup_factor(self):
        return self.start_factor + (
            (1.0 - self.start_factor) * self.warmup_progress
        )

    def _set_warmup_lr(self):
        factor = self.warmup_factor
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def step_batch(self):
        """Advance warm-up after an optimizer update."""
        if not self.in_warmup:
            return
        self.completed_steps += 1
        self._set_warmup_lr()

    def get_last_lr(self):
        """Expose current LR through the standard scheduler logging API."""
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "version": 1,
            "total_steps": self.total_steps,
            "start_factor": self.start_factor,
            "completed_steps": self.completed_steps,
            "base_lrs": list(self.base_lrs),
        }

    def load_state_dict(self, state_dict):
        if int(state_dict.get("version", 1)) != 1:
            raise ValueError("Unsupported Phase3StepWarmupScheduler state version")
        if int(state_dict["total_steps"]) != self.total_steps:
            raise ValueError("Phase-3 warm-up step count changed across resume")
        if float(state_dict["start_factor"]) != self.start_factor:
            raise ValueError("Phase-3 warm-up start factor changed across resume")
        self.completed_steps = int(state_dict["completed_steps"])
        self.base_lrs = [float(value) for value in state_dict["base_lrs"]]
        if self.in_warmup:
            self._set_warmup_lr()


def _add_phase3_lr_warmup(
    optimizer, scheduler, epochs, start_factor, steps_per_epoch=1
):
    """Wrap the phase scheduler with a checkpointable per-step warm-up."""
    epochs = int(epochs)
    steps_per_epoch = int(steps_per_epoch)
    start_factor = float(start_factor)
    if epochs < 0:
        raise ValueError("phase3_lr_warmup_epochs must be non-negative")
    if steps_per_epoch < 1:
        raise ValueError("phase3_lr_warmup steps_per_epoch must be positive")
    if not 0.0 < start_factor <= 1.0:
        raise ValueError(
            "phase3_lr_warmup_start_factor must be in (0, 1]"
        )
    return Phase3StepWarmupScheduler(
        optimizer=optimizer,
        total_steps=epochs * steps_per_epoch,
        start_factor=start_factor,
        metric_prefix="phase3",
    )


def _add_phase4_lr_warmup(
    optimizer, scheduler, epochs, start_factor, steps_per_epoch=1
):
    """Add checkpointable Phase-4 warm-up updated per optimizer step."""
    epochs = int(epochs)
    steps_per_epoch = int(steps_per_epoch)
    start_factor = float(start_factor)
    if epochs < 0:
        raise ValueError("phase4_lr_warmup_epochs must be non-negative")
    if steps_per_epoch < 1:
        raise ValueError("phase4_lr_warmup steps_per_epoch must be positive")
    if not 0.0 < start_factor <= 1.0:
        raise ValueError(
            "phase4_lr_warmup_start_factor must be in (0, 1]"
        )
    return Phase3StepWarmupScheduler(
        optimizer=optimizer,
        total_steps=epochs * steps_per_epoch,
        start_factor=start_factor,
        metric_prefix="phase4",
    )


def _required(config: DictConfig, name: str):
    if name not in config or config[name] is None:
        raise ValueError(f"Missing required CLI override: {name}=...")
    return config[name]


def _state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, metadata and values in a model state dict."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"state_dict entry {name!r} is not a tensor and cannot be hashed"
            )
        tensor = value.detach().cpu().contiguous()
        fields = (
            name.encode("utf-8"),
            str(tensor.dtype).encode("ascii"),
            repr(tuple(tensor.shape)).encode("ascii"),
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"),
        )
        for field in fields:
            digest.update(len(field).to_bytes(8, byteorder="big"))
            digest.update(field)
    return digest.hexdigest()


def configure_phase_trainability(
    model, phase, *, phase3_rule="", phase3_intervention="deactivation"
):
    """Set trainable parameter groups before constructing a phase optimizer."""
    model.requires_grad_(True)
    if phase in {"left_proper", "right_proper"}:
        model.requires_grad_(False)
        branch_name = (
            "left_branch" if phase == "left_proper" else "right_branch"
        )
        getattr(model, branch_name).requires_grad_(True)
        model.main_branch.requires_grad_(True)
    elif int(phase) == 3 and phase3_rule == "relative_unimodal_parity":
        model.requires_grad_(False)
        model.right_branch.requires_grad_(True)
    elif int(phase) == 3 and phase3_intervention == "frozen_left_active":
        model.left_branch.requires_grad_(False)



def _uses_validation_protocol(spec: ModeSpec, config: DictConfig) -> bool:
    controlled_phase = bool(
        (spec.all_at_once or spec.phase in {3, 4})
        and validation_protocol_enabled(config)
    )
    controlled_reference = bool(
        spec.pretraining in {"left_proper", "right_proper"}
        and config.get("unimodal_reference_training", False)
    )
    return controlled_phase or controlled_reference

def _run_bounds(mode: str, spec: ModeSpec, config: DictConfig):
    if spec.pretraining is not None:
        return 0, int(_required(config, "epochs")), None
    if spec.all_at_once:
        durations = [int(_required(config, f"phase{index}")) for index in range(1, 5)]
        return 0, sum(durations), config.get("resume_checkpoint")
    if mode == "normal":
        epochs = config.get("epochs", config.get("phase1"))
        if epochs is None:
            raise ValueError("Normal mode requires epochs=... (or legacy phase1=...)")
        return 0, int(epochs), None

    phase = spec.phase
    durations = [int(_required(config, f"phase{index}")) for index in range(1, phase + 1)]
    start_epoch = sum(durations[:-1])
    end_epoch = start_epoch + durations[-1]
    checkpoint = None if phase == 1 else _required(config, "model_checkpoint")
    return start_epoch, end_epoch, checkpoint


def _checkpoint_restore_policy(config: DictConfig, *, explicit_resume: bool) -> bool:
    """Choose checkpoint semantics without conflating phase transfer and resume.

    ``resume_checkpoint`` continues an interrupted run and therefore restores
    optimizer, scheduler, epoch, global step, and RNG by default.
    ``model_checkpoint`` starts a new phase from model weights only by default,
    because the phase can use a different data/intervention regime.
    """
    option = "restore_training_state" if explicit_resume else "transfer_training_state"
    return bool(config.get(option, explicit_resume))


def _validate_resume_state(
    resume_state,
    *,
    explicit_resume,
    restore_training_state,
    resume_start_epoch,
):
    if not explicit_resume:
        return
    if not resume_state["is_training_checkpoint"]:
        raise ValueError(
            "resume_checkpoint requires a versioned training checkpoint; "
            "use model_checkpoint for a legacy model-only state_dict"
        )
    if not restore_training_state and resume_start_epoch is None:
        raise ValueError(
            "restore_training_state=false requires an explicit "
            "resume_start_epoch to avoid repeating completed phases"
        )


def _manifest_differences(saved, current, path=""):
    """Return dotted paths whose protocol values differ."""
    if isinstance(saved, Mapping) and isinstance(current, Mapping):
        differences = []
        for key in sorted(set(saved) | set(current)):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in saved or key not in current:
                differences.append(child_path)
            else:
                differences.extend(
                    _manifest_differences(saved[key], current[key], child_path)
                )
        return differences
    if isinstance(saved, (list, tuple)) and isinstance(current, (list, tuple)):
        differences = []
        if len(saved) != len(current):
            return [path]
        for index, (saved_item, current_item) in enumerate(zip(saved, current)):
            differences.extend(
                _manifest_differences(
                    saved_item, current_item, f"{path}[{index}]"
                )
            )
        return differences
    return [] if type(saved) is type(current) and saved == current else [path]


def _validate_resume_protocol(
    resume_state,
    current_manifest,
    *,
    explicit_resume,
    allow_missing_manifest=False,
):
    if not explicit_resume:
        return
    saved_manifest = resume_state.get("metadata", {}).get("protocol_manifest")
    if saved_manifest is None:
        if allow_missing_manifest:
            logging.warning(
                "Resume checkpoint has no protocol manifest; compatibility "
                "cannot be verified."
            )
            return
        raise ValueError(
            "resume checkpoint has no protocol manifest; use "
            "allow_resume_without_protocol_manifest=true only after manually "
            "verifying the protocol"
        )
    differences = _manifest_differences(saved_manifest, current_manifest)
    if differences:
        displayed = ", ".join(differences[:8])
        suffix = " ..." if len(differences) > 8 else ""
        raise ValueError(
            f"resume checkpoint protocol mismatch: {displayed}{suffix}"
        )


def _subset_manifest(indices):
    if indices is None:
        return None
    canonical = np.ascontiguousarray(indices, dtype=np.int64).reshape(-1)
    return {
        "samples": int(canonical.size),
        "indices_sha256": hashlib.sha256(canonical.tobytes()).hexdigest(),
    }


def _dataset_root(dataset):
    current = dataset
    while hasattr(current, "dataset"):
        current = current.dataset
    root = getattr(current, "root", None)
    if root is None:
        raise AttributeError(
            f"Dataset {type(current).__name__} does not expose a root path"
        )
    return str(root)


def _load_held_out(dataset_name: str, split: str, device: torch.device):
    prefix = f"data/{split}_{dataset_name}_held_out"
    return {
        "proper_x_left": torch.load(f"{prefix}_proper_x_left.pt", map_location=device, weights_only=True),
        "proper_x_right": torch.load(f"{prefix}_proper_x_right.pt", map_location=device, weights_only=True),
        "blurred_x_right": torch.load(f"{prefix}_blurred_x_right.pt", map_location=device, weights_only=True),
        "y": torch.load(f"{prefix}_y.pt", map_location=device, weights_only=True),
    }


def run(mode: str, config: DictConfig, *, umt: bool = False):
    if mode not in MODE_SPECS:
        choices = ", ".join(sorted(MODE_SPECS))
        raise ValueError(f"Unsupported mode {mode!r}; choose one of: {choices}")

    spec = MODE_SPECS[mode]
    model_name = str(_required(config, "model_name"))
    dataset_name = str(_required(config, "dataset_name"))
    lr = float(_required(config, "lr"))
    wd = float(_required(config, "wd"))
    start_epoch, end_epoch, checkpoint = _run_bounds(mode, spec, config)
    explicit_resume = config.get("resume_checkpoint") is not None
    if explicit_resume:
        checkpoint = config.resume_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(config.get("batch_size", 125))
    num_workers = int(config.get("num_workers", 12))
    overlap = float(config.get("overlap", 0.0))
    random_seed = int(config.get("seed", 83))
    logs_per_epoch = int(config.get("logs_per_epoch", 0))
    clip_value = float(config.get("clip_value", 0.0))
    lr_lambda = float(config.get("lr_lambda", 1.0))
    resize_factor = float(config.get("resize_factor", 0.25))
    fim_measurements_per_epoch = int(
        config.get("fim_measurements_per_epoch", 2)
    )
    fim_eval_interval_epochs = int(
        config.get("fim_eval_interval_epochs", 1)
    )
    if fim_eval_interval_epochs < 1:
        raise ValueError("fim_eval_interval_epochs must be positive")
    raw_fim_eval_epochs = config.get("fim_eval_epochs")
    fim_eval_epochs = (
        None
        if raw_fim_eval_epochs is None
        else [int(epoch) for epoch in raw_fim_eval_epochs]
    )
    if fim_eval_epochs is not None:
        # Validate once before any dataset or model work begins.
        fim_measurement_due(
            fim_eval_epochs[0] if fim_eval_epochs else 1,
            fim_eval_interval_epochs,
            fim_eval_epochs,
        )
    if fim_measurements_per_epoch < 0:
        raise ValueError("fim_measurements_per_epoch must be non-negative")
    phase1_validation_interval_epochs = int(
        config.get("phase1_validation_interval_epochs", 5)
    )
    diagnostic_validation_interval_epochs = int(
        config.get("diagnostic_validation_interval_epochs", 1)
    )
    train_probe_interval_epochs = int(
        config.get("train_probe_interval_epochs", 10)
    )
    resume_checkpoint_interval_epochs = int(
        config.get("resume_checkpoint_interval_epochs", 1)
    )
    cadence = {
        "phase1_validation_interval_epochs": (
            phase1_validation_interval_epochs
        ),
        "diagnostic_validation_interval_epochs": (
            diagnostic_validation_interval_epochs
        ),
        "train_probe_interval_epochs": train_probe_interval_epochs,
        "resume_checkpoint_interval_epochs": (
            resume_checkpoint_interval_epochs
        ),
    }
    if any(value < 1 for value in cadence.values()):
        raise ValueError("validation and checkpoint intervals must be positive")

    type_names = {
        "model": model_name,
        "criterion": "cls",
        "dataset": dataset_name,
        "optim": "sgd",
        "scheduler": "multiplicative",
    }
    manual_seed(random_seed=random_seed, device=device)

    use_validation_protocol = _uses_validation_protocol(spec, config)
    trace_fim = bool(spec.trace_fim)
    subset_path = config.get("proper_right_subset_path")
    proper_right_subset = np.load(str(subset_path)) if subset_path is not None else None
    dataset_params = {
        "dataset_path": None,
        "overlap": overlap,
        "resize_factor": resize_factor,
        "subset": proper_right_subset,
    }
    loader_params = {
        "batch_size": batch_size,
        "pin_memory": device.type == "cuda",
        "num_workers": num_workers,
    }
    normalization_profile = config.get("normalization_profile")
    split_profile = config.get("split_profile")
    final_test_loader_factory = None
    if use_validation_protocol:
        loaders = prepare_training_loaders_clp(
            dataset_name,
            dataset_params=dataset_params,
            loader_params=loader_params,
            split_profile=str(split_profile),
            normalization_profile=str(normalization_profile),
            generator_seed=random_seed,
            verify_dataset_files=bool(
                config.get("verify_dataset_files", True)
            ),
        )
        fim_probe = loaders.fim if trace_fim else None
        fim_probe_metadata = (
            {
                "source": "protocol_split",
                "profile": str(split_profile),
                "excluded_from_training": True,
                "samples": int(loaders.fim.probe_indices.size),
                "indices_sha256": hashlib.sha256(
                    loaders.fim.probe_indices.tobytes()
                ).hexdigest(),
            }
            if trace_fim
            else None
        )

        def final_test_loader_factory():
            return prepare_test_loaders_clp(
                dataset_name,
                dataset_params=dataset_params,
                loader_params=loader_params,
                normalization_profile=str(normalization_profile),
            )

        proper_eval_dataset = loaders["validation_proper"].dataset
        blurred_eval_dataset = loaders["validation_blurred"].dataset
        while not hasattr(proper_eval_dataset, "transform1"):
            proper_eval_dataset = proper_eval_dataset.dataset
        while not hasattr(blurred_eval_dataset, "transform1"):
            blurred_eval_dataset = blurred_eval_dataset.dataset
    else:
        loaders = prepare_loaders_clp(
            dataset_name,
            dataset_params=dataset_params,
            loader_params=loader_params,
        )
        fim_probe = None
        fim_probe_metadata = None
        fim_probe_source = str(config.get("fim_probe_source", "generated"))
        if trace_fim and fim_probe_source not in {"generated", "files"}:
            raise ValueError("fim_probe_source must be generated or files")
        if trace_fim and fim_probe_source == "files":
            fim_probe_metadata = {
                "source": "files",
                "excluded_from_training": False,
            }
        if trace_fim and fim_probe_source == "generated":
            fim_fraction = float(config.get("fim_probe_fraction", 0.02))
            fim_seed = int(config.get("fim_probe_seed", random_seed))
            fim_probe = build_fim_probe(
                loaders["train"].dataset,
                dataset_name,
                overlap=overlap,
                resize_factor=resize_factor,
                fraction=fim_fraction,
                seed=fim_seed,
            )
            exclude_probe = bool(config.get("fim_exclude_from_training", True))
            if exclude_probe:
                loaders["train"] = DataLoader(
                    Subset(
                        loaders["train"].dataset,
                        fim_probe.train_indices.tolist(),
                    ),
                    shuffle=True,
                    **loader_params,
                )
            fim_probe_metadata = {
                "source": "generated",
                "fraction": fim_fraction,
                "seed": fim_seed,
                "excluded_from_training": exclude_probe,
                "samples": int(fim_probe.probe_indices.size),
                "indices_sha256": hashlib.sha256(
                    fim_probe.probe_indices.tobytes()
                ).hexdigest(),
            }
        proper_eval_dataset = loaders["test_proper"].dataset
        blurred_eval_dataset = loaders["test_blurred"].dataset

    fim_estimator_metadata = (
        {
            "measurements_per_epoch": fim_measurements_per_epoch,
            "eval_interval_epochs": fim_eval_interval_epochs,
            "eval_epochs": fim_eval_epochs,
            "samples_per_input": int(config.get("fim_samples_per_input", 5)),
            "sampling_seed": int(
                config.get("fim_sampling_seed", random_seed + 1)
            ),
            "chunk_size": int(config.get("fim_chunk_size", 16)),
        }
        if trace_fim
        else None
    )
    logged_dataset_params = {
        key: value for key, value in dataset_params.items() if key != "subset"
    }
    logged_dataset_params["split_profile"] = split_profile
    logged_dataset_params["normalization_profile"] = normalization_profile
    if use_validation_protocol:
        logged_dataset_params["split"] = loaders.split_manifest
    logged_dataset_params["normalization"] = {
        "proper_left": normalization_from_transform(proper_eval_dataset.transform1),
        "proper_right": normalization_from_transform(proper_eval_dataset.transform2),
        "blurred_right": normalization_from_transform(blurred_eval_dataset.transform2),
    }
    num_classes = count_classes(loaders["train"].dataset)

    input_channels, img_height, img_width = loaders["train"].dataset[0][0][0].shape
    model_params = {
        "num_classes": num_classes,
        "input_channels": input_channels,
        "img_height": img_height,
        "img_width": img_width,
        "overlap": overlap,
        **load_model_specific_params(model_name),
    }
    unimodal_reference_training = bool(
        config.get("unimodal_reference_training", False)
    )
    unimodal_initialization_policy = None
    if unimodal_reference_training:
        unimodal_initialization_policy = str(
            config.get(
                "unimodal_initialization_policy", INITIALIZATION_POLICY
            )
        )
        if unimodal_initialization_policy != INITIALIZATION_POLICY:
            raise ValueError("unsupported unimodal initialization policy")
        manual_seed(random_seed=random_seed, device=device)
    student_model = prepare_model(model_name, model_params=model_params)
    source_bimodal_initial_state_sha256 = None
    if unimodal_reference_training:
        source_bimodal_initial_state_sha256 = _state_dict_sha256(
            student_model.state_dict()
        )
    if (
        spec.pretraining in {"left_proper", "right_proper"}
        and unimodal_reference_training
    ):
        if not (
            hasattr(student_model, "left_branch")
            and hasattr(student_model, "right_branch")
            and hasattr(student_model, "main_branch")
        ):
            raise ValueError(
                "unimodal reference training requires left_branch, "
                "right_branch and main_branch"
            )
    if umt:
        teacher_model = prepare_model(model_name, model_params=model_params)
        left_teacher = load_branch(
            teacher_model.left_branch,
            str(_required(config, "left_branch_pretrained_path")),
            "left_branch",
            device,
        )
        right_teacher = load_branch(
            teacher_model.right_branch,
            str(_required(config, "right_branch_pretrained_path")),
            "right_branch",
            device,
        )
        model = BiModalModelwithPretrainedBranches(
            student_model, left_teacher, right_teacher
        ).to(device)
        metric_model = model.main_model
        trainer_class = UMTTrainerClassification
    else:
        model = student_model.to(device)
        metric_model = model
        trainer_class = TrainerClassification
    if use_validation_protocol:
        trainer_class = (
            validation_controlled_umt_trainer_class()
            if umt
            else ValidationControlledTrainer
        )

    samples_weights = get_samples_weights(loaders, num_classes).to(device)
    if spec.pretraining is None:
        criterion_params = {"criterion_name": "ce", "weight": samples_weights}
    else:
        criterion_params = load_criterion_specific_params(type_names["criterion"])
        criterion_params["weight"] = samples_weights
    criterion = prepare_criterion(type_names["criterion"], criterion_params=criterion_params)
    criterion_params["weight"] = samples_weights.tolist()

    def configure_trainable_parameters(phase):
        phase3_rule = str(
            (config.get("phase3_stopping") or {}).get("decision_rule", "")
        )
        phase3_intervention = str(
            (config.get("phase3_intervention") or {}).get(
                "mode", "deactivation"
            )
        )
        configure_phase_trainability(
            metric_model,
            phase,
            phase3_rule=phase3_rule,
            phase3_intervention=phase3_intervention,
        )

    if spec.pretraining in {"left_proper", "right_proper"} and bool(
        config.get("unimodal_reference_training", False)
    ):
        configure_trainable_parameters(spec.pretraining)

    optim_params = {"lr": lr, "weight_decay": wd}
    scheduler_params = {"lr_lambda": lambda _: lr_lambda}
    optim, lr_scheduler = prepare_optim_and_scheduler(
        metric_model,
        optim_name=type_names["optim"],
        optim_params=optim_params,
        scheduler_name=type_names["scheduler"],
        scheduler_params=scheduler_params,
    )

    train_steps_per_epoch = len(loaders["train"])
    max_train_batches = int(config.get("max_train_batches", 0) or 0)
    if max_train_batches:
        train_steps_per_epoch = min(train_steps_per_epoch, max_train_batches)

    def optimizer_factory(phase):
        configure_trainable_parameters(phase)
        phase_optim, phase_scheduler = prepare_optim_and_scheduler(
            metric_model,
            optim_name=type_names["optim"],
            optim_params=dict(optim_params),
            scheduler_name=type_names["scheduler"],
            scheduler_params={
                "lr_lambda": (
                    lambda _epoch, factor=lr_lambda: factor
                )
            },
        )
        if isinstance(phase, int) and int(phase) == 3:
            phase_scheduler = _add_phase3_lr_warmup(
                phase_optim,
                phase_scheduler,
                config.get("phase3_lr_warmup_epochs", 0),
                config.get("phase3_lr_warmup_start_factor", 0.1),
                steps_per_epoch=train_steps_per_epoch,
            )
        elif isinstance(phase, int) and int(phase) == 4:
            phase_scheduler = _add_phase4_lr_warmup(
                phase_optim,
                phase_scheduler,
                config.get("phase4_lr_warmup_epochs", 0),
                config.get("phase4_lr_warmup_start_factor", 0.1),
                steps_per_epoch=train_steps_per_epoch,
            )
        return phase_optim, phase_scheduler

    scheduler_params["lr_lambda"] = lr_lambda
    protocol_manifest = OmegaConf.to_container(
        OmegaConf.create(
            {
                "version": 3 if use_validation_protocol else 2,
                "mode": mode,
                "model": {
                    "name": model_name,
                    "parameters": model_params,
                    "umt": bool(umt),
                },
                "dataset": {
                    "name": dataset_name,
                    "overlap": overlap,
                    "resize_factor": resize_factor,
                    "split_profile": split_profile,
                    "normalization_profile": normalization_profile,
                    "split": (
                        loaders.split_manifest
                        if use_validation_protocol
                        else None
                    ),
                    "normalization": logged_dataset_params["normalization"],
                    "proper_right_subset": _subset_manifest(proper_right_subset),
                },
                "loader": {
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "max_train_batches": int(
                        config.get("max_train_batches", 0) or 0
                    ),
                    "max_eval_batches": int(
                        config.get("max_eval_batches", 0) or 0
                    ),
                    "max_test_batches": int(
                        config.get("max_test_batches", 0) or 0
                    ),
                },
                "phase2_test_policy": str(
                    config.get("phase2_test_policy", "disabled")
                ),
                "test_policy": (
                    str(config.get("phase4_test_policy", "final_only"))
                    if use_validation_protocol
                    else "historical"
                ),
                "fim_probe": fim_probe_metadata,
                "fim_estimator": fim_estimator_metadata,
                "training": {
                    "seed": random_seed,
                    "epochs": (
                        int(config.epochs) if config.get("epochs") is not None else None
                    ),
                    "phase_durations": {
                        f"phase{index}": int(config.get(f"phase{index}", 0))
                        for index in range(1, 5)
                    },
                    "phase_controllers": {
                        name: (
                            OmegaConf.to_container(
                                config.get(name), resolve=True
                            )
                            if OmegaConf.is_config(config.get(name))
                            else dict(config.get(name, {}) or {})
                        )
                        for name in (
                            "phase2_stopping",
                            "phase3_stopping",
                            "phase3_intervention",
                            "phase4_selection",
                            "phase4_diagnostics",
                            "phase4_staged_unfreezing",
                            "phase4_auxiliary_loss",
                        )
                    },
                    "cadence": cadence,
                    "optimizer": {
                        "name": type_names["optim"],
                        "lr": lr,
                        "weight_decay": wd,
                    },
                    "scheduler": {
                        "name": type_names["scheduler"],
                        "lr_lambda": lr_lambda,
                    },
                    "phase3_lr_warmup_epochs": int(
                        config.get("phase3_lr_warmup_epochs", 0)
                    ),
                    "phase3_lr_warmup_start_factor": float(
                        config.get("phase3_lr_warmup_start_factor", 0.1)
                    ),
                    "phase3_lr_warmup_unit": "optimizer_step",
                    "phase3_lr_warmup_steps_per_epoch": train_steps_per_epoch,
                    "phase4_lr_warmup_epochs": int(
                        config.get("phase4_lr_warmup_epochs", 0)
                    ),
                    "phase4_lr_warmup_start_factor": float(
                        config.get("phase4_lr_warmup_start_factor", 0.1)
                    ),
                    "phase4_lr_warmup_unit": "optimizer_step",
                    "phase4_lr_warmup_steps_per_epoch": train_steps_per_epoch,
                    "unimodal_reference_training": bool(
                        config.get("unimodal_reference_training", False)
                    ),
                    "unimodal_initialization_policy": (
                        unimodal_initialization_policy
                    ),
                    **(
                        {
                            "source_bimodal_initial_state_sha256": (
                                source_bimodal_initial_state_sha256
                            )
                        }
                        if unimodal_reference_training
                        else {}
                    ),
                    "clip_value": clip_value,
                    "distill": float(config.get("distill", 1.0)),
                    "phase4_target_train_acc": (
                        float(config.phase4_target_train_acc)
                        if config.get("phase4_target_train_acc") is not None
                        else None
                    ),
                    "phase4_weight_decay": (
                        float(config.phase4_weight_decay)
                        if config.get("phase4_weight_decay") is not None
                        else None
                    ),
                    "phase4_lr_lambda": (
                        float(config.phase4_lr_lambda)
                        if config.get("phase4_lr_lambda") is not None
                        else None
                    ),
                    "phase4_bn_recalibration_batches": int(
                        config.get("phase4_bn_recalibration_batches", 0)
                    ),
                    "phase4_bn_recalibration_scope": str(
                        config.get(
                            "phase4_bn_recalibration_scope", "main_branch"
                        )
                    ),
                },
            }
        ),
        resolve=True,
    )
    unimodal_reference_pair = None
    phase3_section = config.get("phase3_stopping") or {}
    if str(phase3_section.get("decision_rule", "")) == (
        "relative_unimodal_parity"
    ):
        reference_section = config.get("unimodal_references") or {}
        unimodal_reference_pair = load_and_validate_unimodal_reference_pair(
            reference_section,
            seed=random_seed,
            model_name=model_name,
            dataset_name=dataset_name,
            split_profile=str(split_profile),
            normalization_profile=str(normalization_profile),
            split_manifest=protocol_manifest["dataset"]["split"],
            normalization_manifest=protocol_manifest["dataset"][
                "normalization"
            ],
        )
        protocol_manifest["training"]["unimodal_references"] = {
            "left": unimodal_reference_pair[0].state_dict(),
            "right": unimodal_reference_pair[1].state_dict(),
        }
    resume_state = None
    restore_training_state = _checkpoint_restore_policy(
        config, explicit_resume=explicit_resume
    )
    if checkpoint is not None:
        preliminary_state = load_training_checkpoint(
            str(checkpoint),
            model,
            device=device,
        )
        if (
            restore_training_state
            and use_validation_protocol
            and preliminary_state["is_training_checkpoint"]
        ):
            resume_phase = (
                preliminary_state.get("metadata") or {}
            ).get("phase")
            if resume_phase is not None:
                optim, lr_scheduler = optimizer_factory(int(resume_phase))
        resume_state = load_training_checkpoint(
            str(checkpoint),
            model,
            optim if restore_training_state else None,
            lr_scheduler if restore_training_state else None,
            device=device,
        )
        _validate_resume_state(
            resume_state,
            explicit_resume=explicit_resume,
            restore_training_state=restore_training_state,
            resume_start_epoch=config.get("resume_start_epoch"),
        )
        _validate_resume_protocol(
            resume_state,
            protocol_manifest,
            explicit_resume=explicit_resume,
            allow_missing_manifest=bool(
                config.get("allow_resume_without_protocol_manifest", False)
            ),
        )
        if restore_training_state and resume_state["is_training_checkpoint"]:
            checkpoint_epoch = resume_state["next_epoch"]
            expected_epoch = start_epoch
            if (
                not spec.all_at_once
                and not explicit_resume
                and checkpoint_epoch != expected_epoch
                and not bool(config.get("allow_checkpoint_epoch_mismatch", False))
            ):
                raise ValueError(
                    f"Checkpoint resumes at epoch {checkpoint_epoch}, but mode "
                    f"{mode!r} expects epoch {expected_epoch}."
                )
            start_epoch = checkpoint_epoch
        elif config.get("resume_start_epoch") is not None:
            start_epoch = int(config.resume_start_epoch)
    if start_epoch >= end_epoch:
        raise ValueError(
            f"No epochs left to run: start_epoch={start_epoch}, end_epoch={end_epoch}."
        )

    extra_modules = defaultdict(lambda: None)
    run_stats_enabled = bool(
        spec.run_stats
        and not config.get("unimodal_reference_training", False)
    )
    if run_stats_enabled:
        extra_modules["run_stats"] = RunStatsBiModal(metric_model, optim)
        if resume_state is not None and restore_training_state:
            diagnostics_state = resume_state.get("diagnostics_state") or {}
            run_stats_state = diagnostics_state.get("run_stats")
            if run_stats_state is not None:
                extra_modules["run_stats"].load_diagnostic_state_dict(
                    run_stats_state
                )
            elif explicit_resume:
                logging.warning(
                    "Checkpoint has no RunStats state; model training resumes "
                    "exactly, but trajectory diagnostics restart at the checkpoint."
                )
    if trace_fim:
        if fim_probe is None:
            held_out_train = _load_held_out(dataset_name, "train", device)
        else:
            held_out_train = fim_probe.tensors
        extra_modules["trace_fim_train"] = TraceFIM(
            held_out_train,
            metric_model,
            num_classes=num_classes,
            postfix="train",
            m_sampling=int(config.get("fim_samples_per_input", 5)),
            sampling_seed=int(config.get("fim_sampling_seed", random_seed + 1)),
            chunk_size=int(config.get("fim_chunk_size", 16)),
        )

    trainer_kwargs = {
        "model": model,
        "criterion": criterion,
        "loaders": loaders,
        "optim": optim,
        "lr_scheduler": lr_scheduler,
        "device": device,
        "extra_modules": extra_modules,
    }
    if use_validation_protocol:
        trainer_kwargs.update(
            {
                "metric_model": metric_model,
                "optimizer_factory": optimizer_factory,
                "final_test_loader_factory": final_test_loader_factory,
            }
        )
    trainer = trainer_class(**trainer_kwargs)
    if resume_state is not None and restore_training_state:
        trainer.resume_rng_state = resume_state["rng_state"]
        trainer.resume_global_step = resume_state["global_step"]
        trainer.resume_training_state = resume_state
        if hasattr(loaders, "load_state_dict"):
            loaders.load_state_dict(resume_state.get("loader_state"))
    if (
        resume_state is not None
        and not restore_training_state
        and hasattr(trainer, "selected_source_step")
    ):
        trainer.selected_source_step = resume_state.get("global_step")

    completed = ", ".join(
        f"phase{index}={config[f'phase{index}']}"
        for index in range(1, 5 if spec.all_at_once else (spec.phase or 1))
        if f"phase{index}" in config
    )
    context = f", trained with {completed}" if completed else ""
    group_name = (
        f"{dataset_name}, {model_name}, sgd, epochs={end_epoch - start_epoch}"
        f"_overlap={overlap}_lr={lr}_wd={wd}_lambda={lr_lambda}"
    )
    run_name = f"umt_{mode}" if umt else mode
    exp_name = f"{run_name}, {spec.label}{context}, {group_name}"
    h_params_overall = {
        "model": model_params,
        "criterion": criterion_params,
        "dataset": logged_dataset_params,
        "fim_probe": fim_probe_metadata,
        "loaders": loader_params,
        "optim": optim_params,
        "scheduler": scheduler_params,
        "type_names": type_names,
        "mode": mode,
        "checkpoint": {
            "path": str(checkpoint) if checkpoint is not None else None,
            "kind": (
                "resume"
                if explicit_resume
                else ("phase_transfer" if checkpoint is not None else None)
            ),
            "restored_training_state": bool(
                restore_training_state
                and resume_state is not None
                and resume_state["is_training_checkpoint"]
            ),
        },
        "umt": {"enabled": umt, "distill": float(config.get("distill", 1.0))},
        "phase4_optimizer": {
            "weight_decay": config.get("phase4_weight_decay"),
            "lr_lambda": config.get("phase4_lr_lambda"),
        },
        "phase4_batchnorm": {
            "recalibration_batches": int(
                config.get("phase4_bn_recalibration_batches", 0)
            ),
            "scope": str(config.get("phase4_bn_recalibration_scope", "main_branch")),
        },
        "phase4_auxiliary_loss": (
            OmegaConf.to_container(
                config.get("phase4_auxiliary_loss"), resolve=True
            )
            if OmegaConf.is_config(config.get("phase4_auxiliary_loss"))
            else dict(config.get("phase4_auxiliary_loss", {}) or {})
        ),
        "unimodal_references": (
            {
                "left": unimodal_reference_pair[0].state_dict(),
                "right": unimodal_reference_pair[1].state_dict(),
            }
            if unimodal_reference_pair is not None
            else None
        ),
    }
    logger_config = {
        "logger_name": str(config.get("logger", "wandb")),
        "entity": str(
            config.get("wandb_entity", os.environ.get("WANDB_ENTITY") or "")
        ) or None,
        "project_name": str(
            config.get("wandb_project", os.environ.get("WANDB_PROJECT") or "")
        ) or None,
        "hyperparameters": h_params_overall,
        "mode": str(config.get("logger_mode", "online")),
    }

    batches_per_epoch = len(loaders["train"])
    phase1_end = int(config.get("phase1", 0))
    phase2_end = phase1_end + int(config.get("phase2", 0))
    phase3_end = phase2_end + int(config.get("phase3", 0))
    phase4_end = phase3_end + int(config.get("phase4", 0))
    run_config = OmegaConf.create(
        {
            "exp_starts_at_epoch": start_epoch,
            "exp_ends_at_epoch": end_epoch,
            "phase1_starts_at_epoch": 0,
            "phase1_ends_at_epoch": phase1_end,
            "phase2_ends_at_epoch": phase2_end,
            "phase3_ends_at_epoch": phase3_end,
            "phase4_ends_at_epoch": phase4_end,
            "phase1": int(config.get("phase1", 0)),
            "phase2": int(config.get("phase2", 0)),
            "phase3": int(config.get("phase3", 0)),
            "phase4": int(config.get("phase4", 0)),
            "protocol_manifest": protocol_manifest,
            "normalization_profile": normalization_profile,
            "phase2_stopping": config.get("phase2_stopping", {}),
            "phase3_stopping": config.get("phase3_stopping", {}),
            "phase3_intervention": config.get(
                "phase3_intervention", {}
            ),
            "unimodal_references": (
                {
                    "left": unimodal_reference_pair[0].state_dict(),
                    "right": unimodal_reference_pair[1].state_dict(),
                }
                if unimodal_reference_pair is not None
                else None
            ),
            "phase4_selection": config.get("phase4_selection", {}),
            "phase4_diagnostics": config.get("phase4_diagnostics", {}),
            "phase4_staged_unfreezing": config.get(
                "phase4_staged_unfreezing", {}
            ),
            "phase4_auxiliary_loss": config.get(
                "phase4_auxiliary_loss", {}
            ),
            "phase2_test_policy": str(
                config.get("phase2_test_policy", "disabled")
            ),
            "phase4_test_policy": str(
                config.get("phase4_test_policy", "final_only")
            ),
            "protocol_smoke": bool(config.get("protocol_smoke", False)),
            "max_train_batches": int(config.get("max_train_batches", 0) or 0),
            "max_eval_batches": int(config.get("max_eval_batches", 0) or 0),
            "max_test_batches": int(config.get("max_test_batches", 0) or 0),
            "phase4_target_train_acc": config.get("phase4_target_train_acc"),
            "phase3_lr_warmup_epochs": int(
                config.get("phase3_lr_warmup_epochs", 0)
            ),
            "phase3_lr_warmup_start_factor": float(
                config.get("phase3_lr_warmup_start_factor", 0.1)
            ),
            "phase3_lr_warmup_unit": "optimizer_step",
            "phase3_lr_warmup_steps_per_epoch": train_steps_per_epoch,
            "phase4_lr_warmup_epochs": int(
                config.get("phase4_lr_warmup_epochs", 0)
            ),
            "phase4_lr_warmup_start_factor": float(
                config.get("phase4_lr_warmup_start_factor", 0.1)
            ),
            "phase4_lr_warmup_unit": "optimizer_step",
            "phase4_lr_warmup_steps_per_epoch": train_steps_per_epoch,
            "unimodal_reference_training": bool(
                config.get("unimodal_reference_training", False)
            ),
            "unimodal_reference_eval_interval_epochs": int(
                config.get("unimodal_reference_eval_interval_epochs", 5)
            ),
            "phase4_weight_decay": config.get("phase4_weight_decay"),
            "phase4_lr_lambda": config.get("phase4_lr_lambda"),
            "phase4_bn_recalibration_batches": int(config.get("phase4_bn_recalibration_batches", 0)),
            "phase4_bn_recalibration_scope": str(config.get("phase4_bn_recalibration_scope", "main_branch")),
            "log_multi": max(
                1,
                batches_per_epoch
                // (logs_per_epoch if logs_per_epoch else batches_per_epoch),
            ),
            "run_stats_multi": max(1, batches_per_epoch // 2),
            "fim_measurements_per_epoch": fim_measurements_per_epoch,
            "fim_eval_interval_epochs": fim_eval_interval_epochs,
            "fim_eval_epochs": fim_eval_epochs,
            "phase1_validation_interval_epochs": (
                phase1_validation_interval_epochs
            ),
            "diagnostic_validation_interval_epochs": (
                diagnostic_validation_interval_epochs
            ),
            "train_probe_interval_epochs": train_probe_interval_epochs,
            "resume_checkpoint_interval_epochs": (
                resume_checkpoint_interval_epochs
            ),
            "stiffness_multi": batches_per_epoch * 20,
            "rank_multi": batches_per_epoch * 20,
            "clip_value": clip_value,
            "distill": float(config.get("distill", 1.0)),
            "overlap": overlap,
            "resize_factor": resize_factor,
            "random_seed": random_seed,
            "whether_disable_tqdm": bool(config.get("disable_tqdm", True)),
            "base_path": os.environ["REPORTS_DIR"],
            "exp_name": exp_name,
            "logger_config": logger_config,
            "run_manifest_context": {
                "repo_root": os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..")
                ),
                "dataset_path": _dataset_root(loaders["train"].dataset),
                "input_checkpoint": (
                    str(checkpoint) if checkpoint is not None else None
                ),
                "config": {
                    "requested": OmegaConf.to_container(config, resolve=True),
                    "protocol": protocol_manifest,
                },
            },
        }
    )

    logging.info(
        "Starting %s: model=%s dataset=%s epochs=%d..%d lr=%s wd=%s",
        mode,
        model_name,
        dataset_name,
        start_epoch,
        end_epoch,
        lr,
        wd,
    )
    try:
        if spec.all_at_once:
            trainer.run_all_at_once(run_config)
        elif spec.pretraining is None:
            trainer.run_phase(spec.phase, run_config)
        elif bool(config.get("unimodal_reference_training", False)):
            trainer.run_unimodal_reference(spec.pretraining, run_config)
        else:
            trainer.run_pretraining(spec.pretraining, run_config)
    except BaseException as error:
        trainer.finalize_run_manifest("failed", error=error)
        raise
    else:
        trainer.finalize_run_manifest("completed")


def main(default_mode: str | None = None, *, umt: bool = False):
    logging.basicConfig(
        format="[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    cli_config = OmegaConf.from_cli()
    frozen_path = cli_config.get("frozen_config")
    config_path = cli_config.get("config")
    if frozen_path is not None and config_path is not None:
        raise ValueError("use either config=... or frozen_config=..., not both")
    if config_path is not None:
        config = OmegaConf.load(str(config_path))
        overrides = OmegaConf.create(
            {
                key: value
                for key, value in cli_config.items()
                if key != "config"
            }
        )
        config = OmegaConf.merge(config, overrides)
        config.config_path = os.path.abspath(str(config_path))
    elif frozen_path is None:
        config = cli_config
    else:
        allowed_overrides = {"frozen_config", "resume_checkpoint"}
        unexpected = sorted(set(cli_config.keys()) - allowed_overrides)
        if unexpected:
            raise ValueError(
                "Frozen publication configs reject scientific CLI overrides: "
                + ", ".join(unexpected)
            )
        config = OmegaConf.load(str(frozen_path))
        if config.get("frozen") is not True or int(
            config.get("frozen_config_version", 0)
        ) != 1:
            raise ValueError("Invalid frozen publication configuration")
        config.frozen_config_path = os.path.abspath(str(frozen_path))
        if cli_config.get("resume_checkpoint") is not None:
            config.resume_checkpoint = cli_config.resume_checkpoint
    umt = bool(umt or config.get("umt", False))
    mode = default_mode or str(_required(config, "mode"))
    run(mode, config, umt=umt)


if __name__ == "__main__":
    main()
