"""Validation-controlled four-phase CLP trainer."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.trainer.modality_evaluation import (
    DOMINANT_ONLY_MODE,
    FULL_MODE,
    WEAK_ONLY_MODE,
    ModalityMode,
    evaluate_modalities,
    evaluate_single_mode,
)
from src.trainer.phase4_diagnostics import (
    Phase4HybridAnchor,
    configure_phase4_trainability,
    evaluate_phase4_hybrids,
)
from src.trainer.gradient_diagnostics import (
    evaluate_phase3_gradient_diagnostics,
)
from src.trainer.trainer_classification_mm_clp import (
    PHASE_STAGES,
    PRETRAIN_STAGES,
    TrainingStage,
    TrainerClassification,
)
from src.trainer.unimodal_references import (
    INITIALIZATION_POLICY,
    REFERENCE_VERSION,
)
from src.trainer.validation_control import (
    InterventionCheckpointRecord,
    ModalityEvaluationResult,
    ModeMetrics,
    Phase2CheckpointRecord,
    Phase2CheckpointSelector,
    Phase2PlateauConfig,
    Phase2PlateauDetector,
    Phase3InterventionStopper,
    Phase3LocalAccuracyStopper,
    Phase3RelativeUnimodalStopper,
    Phase3RecoveryStopper,
    Phase3StopConfig,
    Phase4CheckpointRecord,
    Phase4CheckpointSelector,
    UnimodalCheckpointSelector,
    phase3_trajectory_record,
    should_evaluate_phase_epoch,
)
from src.utils.utils_trainer import (
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)


VALIDATION_PROTOCOL_MODES = {"disabled", "observe_only", "enforce"}


def _section(config, name):
    value = config.get(name, {}) or {}
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return dict(value)


def validation_protocol_enabled(config):
    if bool(config.get("validation_protocol", False)):
        return True
    modes = {
        str(_section(config, name).get("mode", "disabled"))
        for name in ("phase2_stopping", "phase3_stopping")
    }
    invalid = modes - VALIDATION_PROTOCOL_MODES
    if invalid:
        raise ValueError(f"invalid validation protocol modes: {sorted(invalid)}")
    return modes != {"disabled"}


class ValidationControlledTrainer(TrainerClassification):
    def __init__(
        self,
        *args,
        metric_model,
        optimizer_factory,
        final_test_loader_factory,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metric_model = metric_model
        self.optimizer_factory = optimizer_factory
        self.final_test_loader_factory = final_test_loader_factory
        self.executed_global_epoch = 0
        self.selected_source_step = None
        self._candidate_paths = set()
        self._phase_boundary_paths = set()
        self._active_config = None
        self._phase2_posthoc_records = {}
        self._final_selected_checkpoint_path = None

    @staticmethod
    def _phase_stage(phase, config):
        """Resolve an explicitly configured Phase-3 intervention."""
        stage = PHASE_STAGES[int(phase)]
        if int(phase) != 3:
            return stage
        mode = str(
            _section(config, "phase3_intervention").get(
                "mode", "deactivation"
            )
        )
        if mode == "deactivation":
            return stage
        if mode in {"frozen_left_active", "full_active"}:
            return TrainingStage(
                "proper",
                True,
                True,
                right_transform="proper",
            )
        raise ValueError(f"unknown phase3 intervention mode: {mode}")

    def _resume_phase(self):
        state = getattr(self, "resume_training_state", None)
        if not state or not state.get("is_training_checkpoint"):
            return None
        phase = (state.get("metadata") or {}).get("phase")
        return None if phase is None else int(phase)

    def _resume_for_phase(self, phase):
        return (
            self.resume_training_state
            if self._resume_phase() == int(phase)
            else None
        )

    @staticmethod
    def _checkpoint_metrics(state):
        if state is None:
            return None
        metrics = (state.get("metadata") or {}).get("metrics")
        if metrics is None:
            return None
        return ModalityEvaluationResult.from_state_dict(metrics)

    def _prepare_phase(self, phase, config):
        state = self._resume_for_phase(phase)
        if state is None:
            self._start_phase(phase, config)
            return 0, None
        config.phase = phase
        self._apply_stage(config, self._phase_stage(phase, config))
        self.executed_global_epoch = int(state["next_epoch"])
        self.global_step = int(state["global_step"])
        return int(state.get("phase_epoch") or 0), state

    def _complete_resumed_phase(self, phase):
        if self._resume_phase() == int(phase):
            self.resume_training_state = None

    def _reset_optimizer(self, phase, config):
        self.optim, self.lr_scheduler = self.optimizer_factory(phase)
        run_stats = self.extra_modules.get("run_stats")
        if run_stats is not None:
            run_stats.optim = self.optim
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    snapshot = run_stats.optimizer_step_parameters.get(name)
                    if snapshot is not None:
                        snapshot.copy_(parameter)
        if phase == 4:
            self._apply_phase4_optimizer_overrides(config)

    def _start_phase(self, phase, config):
        config.phase = phase
        self._reset_optimizer(phase, config)
        self._apply_stage(config, self._phase_stage(phase, config))
        run_stats = self.extra_modules.get("run_stats")
        if run_stats is not None:
            run_stats.start_phase(phase)

    def _train_one_epoch(self, config, phase_epoch):
        config.active_phase_epoch = int(phase_epoch)
        self.epoch = self.executed_global_epoch
        phase4_trainability = None
        if int(getattr(config, "phase", 0) or 0) == 4:
            section = _section(config, "phase4_staged_unfreezing")
            if bool(section.get("enabled", False)):
                phase4_trainability = configure_phase4_trainability(
                    self.metric_model,
                    phase_epoch,
                    int(section.get("shared_only_epochs", 0)),
                )
        if phase4_trainability is None:
            self.model.train()
        if self._relative_unimodal_parity_active(config):
            self.metric_model.left_branch.eval()
            self.metric_model.main_branch.eval()
            self.metric_model.right_branch.train()
        elif (
            int(getattr(config, "phase", 0) or 0) == 3
            and str(
                _section(config, "phase3_intervention").get(
                    "mode", "deactivation"
                )
            )
            == "frozen_left_active"
        ):
            self.metric_model.left_branch.eval()
            self.metric_model.right_branch.train()
            self.metric_model.main_branch.train()
        if phase4_trainability is not None:
            self.logger.log_scalars(
                {
                    "phase4/shared_only_active": float(
                        phase4_trainability.shared_only_active
                    ),
                    "phase4/trainable_left": float(
                        phase4_trainability.trainable_left
                    ),
                    "phase4/trainable_right": float(
                        phase4_trainability.trainable_right
                    ),
                    "phase4/trainable_shared": 1.0,
                    "phase4/trainability_phase_epoch": int(phase_epoch),
                },
                int(self.global_step or 0),
            )
        self.criterion.train()
        self.resume_global_step = int(self.global_step or 0)
        metrics = self.run_epoch("train", config)
        self.executed_global_epoch += 1
        return metrics

    @staticmethod
    def _relative_unimodal_parity_active(config):
        return (
            int(getattr(config, "phase", 0) or 0) == 3
            and str(
                _section(config, "phase3_stopping").get(
                    "decision_rule", ""
                )
            )
            == "relative_unimodal_parity"
        )

    @staticmethod
    def _intervention_mode(stage):
        return ModalityMode(
            enable_left_branch=stage.enable_left_branch,
            enable_right_branch=stage.enable_right_branch,
            left_branch_intervention=stage.left_branch_intervention,
            right_branch_intervention=stage.right_branch_intervention,
        )

    def _evaluate(self, phase, phase_epoch):
        return evaluate_modalities(
            self.metric_model,
            self.criterion,
            self.loaders["validation_proper"],
            self.device,
            intervention_mode=self._intervention_mode(
                self._phase_stage(phase, self._active_config)
            ),
            phase_epoch=phase_epoch,
            global_epoch=self.executed_global_epoch,
            global_step=int(self.global_step or 0),
            max_batches=self._batch_limit("max_eval_batches"),
        )

    def _batch_limit(self, name):
        if self._active_config is None:
            return None
        value = int(self._active_config.get(name, 0) or 0)
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value or None

    def _evaluate_full_loader(self, loader, *, max_batches=None):
        model_modes = [
            (module, module.training) for module in self.metric_model.modules()
        ]
        criterion_modes = [
            (module, module.training) for module in self.criterion.modules()
        ]
        self.metric_model.eval()
        self.criterion.eval()
        total_loss = 0.0
        correct = 0
        samples = 0
        try:
            with torch.no_grad():
                for batch_index, ((x_left, x_right), targets) in enumerate(loader):
                    if max_batches is not None and batch_index >= max_batches:
                        break
                    x_left = x_left.to(self.device)
                    x_right = x_right.to(self.device)
                    targets = targets.to(self.device)
                    logits = self.metric_model(
                        x_left, x_right, **FULL_MODE.kwargs()
                    )
                    criterion_output = self.criterion(logits, targets)
                    loss = (
                        criterion_output[0]
                        if isinstance(criterion_output, tuple)
                        else criterion_output
                    )
                    batch_size = targets.size(0)
                    total_loss += float(loss.detach()) * batch_size
                    correct += int((logits.argmax(dim=1) == targets).sum())
                    samples += batch_size
        finally:
            for module, training in model_modes:
                module.training = training
            for module, training in criterion_modes:
                module.training = training
        if samples == 0:
            raise ValueError("evaluation loader must not be empty")
        return ModeMetrics(total_loss / samples, correct / samples)

    def _log_modality_metrics(self, prefix, metrics, extra=None):
        values = {
            f"{prefix}/full_val_loss": metrics.full.loss,
            f"{prefix}/full_val_accuracy": metrics.full.accuracy,
            f"{prefix}/dominant_only_val_loss": metrics.dominant_only.loss,
            f"{prefix}/dominant_only_val_accuracy": (
                metrics.dominant_only.accuracy
            ),
            f"{prefix}/weak_only_val_loss": metrics.weak_only.loss,
            f"{prefix}/weak_only_val_accuracy": metrics.weak_only.accuracy,
            f"{prefix}/weak_only_loss": metrics.weak_only.loss,
            f"{prefix}/weak_only_accuracy": metrics.weak_only.accuracy,
            f"{prefix}/phase_epoch": metrics.phase_epoch,
            f"{prefix}/intervention_val_loss": metrics.intervention.loss,
            f"{prefix}/intervention_val_accuracy": (
                metrics.intervention.accuracy
            ),
            f"{prefix}/weak_utility_loss": metrics.weak_utility_loss,
            f"{prefix}/weak_utility_accuracy": metrics.weak_utility_accuracy,
            "epochs/global": metrics.global_epoch,
            "epochs/phase": metrics.phase_epoch,
            "steps/global_optimizer": metrics.global_step,
            "global_epoch": metrics.global_epoch,
            "phase_epoch": metrics.phase_epoch,
            "global_step": metrics.global_step,
            "selected_source_step": (
                -1
                if self.selected_source_step is None
                else int(self.selected_source_step)
            ),
        }
        calibration_fields = (
            "nll",
            "brier",
            "ece",
            "mean_confidence",
            "mean_incorrect_confidence",
        )
        for mode_name, mode_metrics in (
            ("full", metrics.full),
            ("dominant_only", metrics.dominant_only),
            ("weak_only", metrics.weak_only),
            ("intervention", metrics.intervention),
        ):
            for field in calibration_fields:
                value = getattr(mode_metrics, field)
                if value is not None:
                    values[f"{prefix}/{mode_name}_val_{field}"] = value
        values.update(extra or {})
        self.logger.log_scalars(values, metrics.global_step)

    def _log_blurred_diagnostic(
        self, prefix, step, phase_epoch, *, force=False
    ):
        interval = int(
            self._active_config.get(
                "diagnostic_validation_interval_epochs", 1
            )
        )
        if interval < 1:
            raise ValueError(
                "diagnostic_validation_interval_epochs must be positive"
            )
        if not force and int(phase_epoch) % interval:
            return
        metrics = self._evaluate_full_loader(
            self.loaders["validation_blurred"],
            max_batches=self._batch_limit("max_eval_batches"),
        )
        self.logger.log_scalars(
            {
                f"{prefix}/blurred_full_val_loss": metrics.loss,
                f"{prefix}/blurred_full_val_accuracy": metrics.accuracy,
            },
            step,
        )

    def _log_train_probe(self, validation_metrics):
        interval = int(
            self._active_config.get("train_probe_interval_epochs", 10)
        )
        if interval < 1:
            raise ValueError(
                "train_probe_interval_epochs must be positive"
            )
        if int(validation_metrics.phase_epoch) % interval:
            return
        train_metrics = evaluate_single_mode(
            self.metric_model,
            self.criterion,
            self.loaders.train_probe,
            self.device,
            mode=WEAK_ONLY_MODE,
            max_batches=self._batch_limit("max_eval_batches"),
        )
        self.logger.log_scalars(
            {
                "phase3/train_probe_weak_only_loss": train_metrics.loss,
                "phase3/train_probe_weak_only_accuracy": (
                    train_metrics.accuracy
                ),
                "phase3/weak_only_generalization_gap": (
                    validation_metrics.weak_only.loss - train_metrics.loss
                ),
            },
            validation_metrics.global_step,
        )

    def _phase3_gradient_diagnostics(self, validation_metrics):
        section = _section(self._active_config, "phase3_stopping")
        if section.get("decision_rule") != "local_accuracy":
            return None
        interval = int(section.get("gradient_diagnostic_interval_epochs", 10))
        if interval < 1:
            raise ValueError(
                "gradient_diagnostic_interval_epochs must be positive"
            )
        if int(validation_metrics.phase_epoch) % interval:
            return None
        diagnostics = evaluate_phase3_gradient_diagnostics(
            self.metric_model,
            self.criterion,
            self.loaders["validation_proper"],
            self.device,
            max_batches=int(section.get("gradient_probe_max_batches", 1)),
        )
        self.logger.log_scalars(
            {
                f"phase3/gradient_probe_{name}": value
                for name, value in diagnostics.items()
            },
            validation_metrics.global_step,
        )
        return diagnostics

    def _save_resume_if_due(
        self,
        config,
        *,
        phase,
        phase_epoch,
        metrics=None,
        phase_state=None,
        force=False,
    ):
        interval = int(config.get("resume_checkpoint_interval_epochs", 1))
        if interval < 1:
            raise ValueError(
                "resume_checkpoint_interval_epochs must be positive"
            )
        if not force and int(phase_epoch) % interval:
            return None
        return self._save_candidate(
            config,
            "resume_current",
            phase=phase,
            phase_epoch=phase_epoch,
            metrics=metrics,
            phase_state=phase_state,
        )

    def _phase_state(self, **controllers):
        state = {
            name: (
                controller.state_dict()
                if hasattr(controller, "state_dict")
                else controller
            )
            for name, controller in controllers.items()
        }
        if self._phase2_posthoc_records:
            state["phase2_posthoc_records"] = {
                name: record.state_dict()
                for name, record in self._phase2_posthoc_records.items()
            }
        return state

    def _save_candidate(
        self,
        config,
        label,
        *,
        phase,
        phase_epoch,
        metrics=None,
        phase_state=None,
        extra_metadata=None,
    ):
        path = self.save_path(label)
        protocol_manifest = getattr(config, "protocol_manifest", None)
        if OmegaConf.is_config(protocol_manifest):
            protocol_manifest = OmegaConf.to_container(
                protocol_manifest, resolve=True
            )
        loader_state = (
            self.loaders.state_dict()
            if hasattr(self.loaders, "state_dict")
            else None
        )
        metadata = {
            "kind": getattr(config, "kind", None),
            "phase": phase,
            "protocol_manifest": protocol_manifest,
            "metrics": (
                metrics.state_dict() if metrics is not None else None
            ),
        }
        metadata.update(dict(extra_metadata or {}))
        save_training_checkpoint(
            self.model,
            self.optim,
            self.lr_scheduler,
            path,
            next_epoch=self.executed_global_epoch,
            global_step=int(self.global_step or 0),
            metadata=metadata,
            diagnostics_state=self._diagnostics_state(),
            phase_state=phase_state,
            loader_state=loader_state,
            phase_epoch=phase_epoch,
        )
        self._candidate_paths.add(path)
        return path

    def _set_unimodal_training_modes(self, modality):
        self.model.train()
        self.metric_model.main_branch.train()
        if modality == "left_proper":
            self.metric_model.left_branch.train()
            self.metric_model.right_branch.eval()
        elif modality == "right_proper":
            self.metric_model.left_branch.eval()
            self.metric_model.right_branch.train()
        else:
            raise ValueError("unknown unimodal reference modality")

    @staticmethod
    def _unimodal_reference_metadata(config, modality, record):
        protocol = config.protocol_manifest
        if OmegaConf.is_config(protocol):
            protocol = OmegaConf.to_container(protocol, resolve=True)
        dataset = protocol["dataset"]
        training = protocol["training"]
        return {
            "version": REFERENCE_VERSION,
            "modality": modality,
            "validation_accuracy": record.metrics.accuracy,
            "validation_loss": record.metrics.loss,
            "selected_epoch": record.epoch,
            "seed": int(training["seed"]),
            "model_name": str(protocol["model"]["name"]),
            "dataset_name": str(dataset["name"]),
            "split_profile": str(dataset["split_profile"]),
            "normalization_profile": str(dataset["normalization_profile"]),
            "split_manifest": dataset["split"],
            "normalization_manifest": dataset["normalization"],
            "initialization_policy": INITIALIZATION_POLICY,
            "source_bimodal_initial_state_sha256": training[
                "source_bimodal_initial_state_sha256"
            ],
        }

    def run_unimodal_reference(self, modality, config):
        """Train and validation-select one clean unimodal reference."""
        if modality not in {"left_proper", "right_proper"}:
            raise ValueError("only proper unimodal references are supported")
        self._active_config = config
        self._initialize_run(config)
        config.phase = 0
        self._apply_stage(config, PRETRAIN_STAGES[modality])
        self.optim, self.lr_scheduler = self.optimizer_factory(modality)
        selector = UnimodalCheckpointSelector(modality)
        interval = int(config.unimodal_reference_eval_interval_epochs)
        if interval < 1:
            raise ValueError(
                "unimodal_reference_eval_interval_epochs must be positive"
            )
        duration = int(config.exp_ends_at_epoch - config.exp_starts_at_epoch)
        mode = (
            DOMINANT_ONLY_MODE
            if modality == "left_proper"
            else WEAK_ONLY_MODE
        )
        try:
            for local_epoch in range(1, duration + 1):
                config.active_phase_epoch = local_epoch
                self.epoch = local_epoch - 1
                self._set_unimodal_training_modes(modality)
                self.criterion.train()
                self.resume_global_step = int(self.global_step or 0)
                self.run_epoch("train", config)
                self.executed_global_epoch = local_epoch
                if local_epoch % interval and local_epoch != duration:
                    continue
                metrics = evaluate_single_mode(
                    self.metric_model,
                    self.criterion,
                    self.loaders["validation_proper"],
                    self.device,
                    mode=mode,
                    max_batches=self._batch_limit("max_eval_batches"),
                )
                candidate_path = self.save_path(
                    f"unimodal_{modality}_epoch_{local_epoch}"
                )
                improved, record = selector.update(
                    metrics, local_epoch, candidate_path
                )
                self.logger.log_scalars(
                    {
                        "unimodal/validation_accuracy": metrics.accuracy,
                        "unimodal/validation_loss": metrics.loss,
                        "unimodal/best_epoch": selector.best.epoch,
                        "unimodal/epoch": local_epoch,
                    },
                    int(self.global_step or 0),
                )
                if improved:
                    self._save_candidate(
                        config,
                        f"unimodal_{modality}_epoch_{local_epoch}",
                        phase=0,
                        phase_epoch=local_epoch,
                        phase_state={"selector": selector.state_dict()},
                        extra_metadata={
                            "unimodal_reference": (
                                self._unimodal_reference_metadata(
                                    config, modality, record
                                )
                            )
                        },
                    )
                self._prune_candidates(selector.retained_checkpoint_paths)
            best = selector.best
            if best is None:
                raise RuntimeError("unimodal reference produced no validation")
            self._load_selected(best.checkpoint_path)
            self._phase_boundary_paths.add(best.checkpoint_path)
            self._write_phase_summary(
                {
                    "phase": "unimodal_reference",
                    "modality": modality,
                    "executed_epochs": duration,
                    "selected_checkpoint": best.checkpoint_path,
                    "selected_epoch": best.epoch,
                    "validation_accuracy": best.metrics.accuracy,
                    "validation_loss": best.metrics.loss,
                    "test_used": False,
                    "initialization_policy": INITIALIZATION_POLICY,
                }
            )
        finally:
            if self.logger is not None:
                self.logger.close()
        return selector.best

    def _load_selected(self, path):
        executed_step = int(self.global_step or 0)
        state = load_training_checkpoint(
            path, self.model, optimizer=None, scheduler=None, device=self.device
        )
        diagnostics = state.get("diagnostics_state") or {}
        for name, diagnostic_state in diagnostics.items():
            module = self.extra_modules.get(name)
            if module is not None and hasattr(
                module, "load_diagnostic_state_dict"
            ):
                module.load_diagnostic_state_dict(diagnostic_state)
        if hasattr(self.loaders, "load_state_dict"):
            self.loaders.load_state_dict(state.get("loader_state"))
        restore_rng_state(state.get("rng_state"))
        self.selected_source_step = state.get("global_step")
        self.global_step = executed_step
        return state

    def _prune_candidates(self, keep):
        keep = {str(path) for path in keep if path}
        keep.update(self._phase_boundary_paths)
        resume_path = str(self.save_path("resume_current"))
        if Path(resume_path).is_file():
            keep.add(resume_path)
        for path in tuple(self._candidate_paths):
            if path not in keep:
                Path(path).unlink(missing_ok=True)
                self._candidate_paths.remove(path)


    @staticmethod
    def _summary_metrics(metrics):
        if metrics is None:
            return None
        return {
            "full_loss": metrics.full.loss,
            "full_accuracy": metrics.full.accuracy,
            "dominant_only_loss": metrics.dominant_only.loss,
            "dominant_only_accuracy": metrics.dominant_only.accuracy,
            "weak_only_loss": metrics.weak_only.loss,
            "weak_only_accuracy": metrics.weak_only.accuracy,
            "weak_utility_loss": metrics.weak_utility_loss,
            "phase_epoch": metrics.phase_epoch,
            "global_epoch": metrics.global_epoch,
            "global_step": metrics.global_step,
            "full_nll": metrics.full.nll,
            "full_brier": metrics.full.brier,
            "full_ece": metrics.full.ece,
            "full_mean_confidence": metrics.full.mean_confidence,
            "full_mean_incorrect_confidence": (
                metrics.full.mean_incorrect_confidence
            ),
            "dominant_only_nll": metrics.dominant_only.nll,
            "dominant_only_brier": metrics.dominant_only.brier,
            "dominant_only_ece": metrics.dominant_only.ece,
            "dominant_only_mean_confidence": (
                metrics.dominant_only.mean_confidence
            ),
            "dominant_only_mean_incorrect_confidence": (
                metrics.dominant_only.mean_incorrect_confidence
            ),
            "weak_only_nll": metrics.weak_only.nll,
            "weak_only_brier": metrics.weak_only.brier,
            "weak_only_ece": metrics.weak_only.ece,
            "weak_only_mean_confidence": (
                metrics.weak_only.mean_confidence
            ),
            "weak_only_mean_incorrect_confidence": (
                metrics.weak_only.mean_incorrect_confidence
            ),
        }

    def _write_phase_summary(self, summary):
        path = Path(self.base_path) / "phase_summaries.jsonl"
        with path.open("a", encoding="utf-8") as summary_file:
            summary_file.write(json.dumps(summary, sort_keys=True) + "\n")
        logging.info("Phase summary: %s", json.dumps(summary, sort_keys=True))

    def _append_phase3_trajectory(self, record):
        path = Path(self.base_path) / "phase3_trajectory.jsonl"
        with path.open("a", encoding="utf-8") as trajectory_file:
            trajectory_file.write(json.dumps(record, sort_keys=True) + "\n")
        return str(path)

    def _append_phase4_diagnostic(self, record):
        path = Path(self.base_path) / "phase4_hybrid_trajectory.jsonl"
        with path.open("a", encoding="utf-8") as trajectory_file:
            trajectory_file.write(json.dumps(record, sort_keys=True) + "\n")
        return str(path)

    def _evaluate_phase4_hybrids(self, anchor, metrics):
        hybrids = evaluate_phase4_hybrids(
            self.metric_model,
            anchor,
            self.criterion,
            self.loaders["validation_proper"],
            self.device,
            max_batches=self._batch_limit("max_eval_batches"),
        )
        values = {
            "phase4_diagnostic/current_right_anchor_shared_weak_loss": (
                hybrids.current_right_anchor_shared.loss
            ),
            "phase4_diagnostic/current_right_anchor_shared_weak_accuracy": (
                hybrids.current_right_anchor_shared.accuracy
            ),
            "phase4_diagnostic/anchor_right_current_shared_weak_loss": (
                hybrids.anchor_right_current_shared.loss
            ),
            "phase4_diagnostic/anchor_right_current_shared_weak_accuracy": (
                hybrids.anchor_right_current_shared.accuracy
            ),
            "phase4_diagnostic/current_model_weak_loss": (
                metrics.weak_only.loss
            ),
            "phase4_diagnostic/current_model_weak_accuracy": (
                metrics.weak_only.accuracy
            ),
            "phase4_diagnostic/phase_epoch": metrics.phase_epoch,
        }
        self.logger.log_scalars(values, metrics.global_step)
        artifact = self._append_phase4_diagnostic(
            {
                "version": 1,
                "phase_epoch": metrics.phase_epoch,
                "global_epoch": metrics.global_epoch,
                "global_step": metrics.global_step,
                "current_model": metrics.weak_only.state_dict(),
                **hybrids.state_dict(),
            }
        )
        return hybrids, artifact

    def _run_phase1(self, config):
        duration = int(config.get("phase1", 0))
        if duration <= 0:
            return
        start_epoch, resume_state = self._prepare_phase(1, config)
        interval = int(config.get("phase1_validation_interval_epochs", 5))
        if interval < 1:
            raise ValueError(
                "phase1_validation_interval_epochs must be positive"
            )
        metrics = self._checkpoint_metrics(resume_state)
        for local_epoch in range(start_epoch + 1, duration + 1):
            self._train_one_epoch(config, local_epoch)
            if local_epoch % interval == 0 or local_epoch == duration:
                metrics = self._evaluate(1, local_epoch)
                self._log_modality_metrics("phase1", metrics)
                self._log_blurred_diagnostic(
                    "phase1", metrics.global_step, metrics.phase_epoch
                )
            self._save_resume_if_due(
                config,
                phase=1,
                phase_epoch=local_epoch,
                metrics=metrics,
            )
        if metrics is None:
            raise RuntimeError("Phase 1 finished without validation metrics")
        path = self._save_candidate(
            config,
            "phase1_boundary",
            phase=1,
            phase_epoch=duration,
            metrics=metrics,
        )
        self._phase_boundary_paths.add(path)
        self._write_phase_summary(
            {
                "phase": 1,
                "executed_epochs": duration,
                "checkpoint": path,
            }
        )
        self._prune_candidates({path})
        self._complete_resumed_phase(1)

    def _phase2_config(self, config):
        section = _section(config, "phase2_stopping")
        return section, Phase2PlateauConfig(
            **{
                key: value
                for key, value in section.items()
                if key in Phase2PlateauConfig.__dataclass_fields__
            }
        )

    def _run_phase2(self, config):
        duration = int(config.get("phase2", 0))
        section, detector_config = self._phase2_config(config)
        mode = str(section.get("mode", "disabled"))
        duration_policy = str(
            section.get("duration_policy", "publication_binary")
        )

        if duration_policy not in {
            "publication_binary",
            "diagnostic_fixed",
        }:
            raise ValueError("unsupported phase2 duration policy")

        if (
            duration_policy == "publication_binary"
            and duration not in {0, 200}
            and not bool(config.get("protocol_smoke", False))
        ):
            raise ValueError(
                "publication_binary phase2 must be configured as 0 or 200"
            )
        if duration == 0:
            self._write_phase_summary(
                {"phase": 2, "executed_epochs": 0, "stop_reason": "skipped"}
            )
            return None
        start_epoch, resume_state = self._prepare_phase(2, config)
        detector = Phase2PlateauDetector(detector_config)
        selector = Phase2CheckpointSelector(
            detector_config.selection_window,
            detector_config.selection_scope,
        )
        if resume_state is not None:
            controller_state = resume_state.get("phase_state") or {}
            if controller_state.get("detector") is not None:
                detector.load_state_dict(controller_state["detector"])
            if controller_state.get("selector") is not None:
                selector.load_state_dict(controller_state["selector"])
        interval = int(section.get("eval_interval_epochs", 5))
        if interval < 1:
            raise ValueError("phase2 eval_interval_epochs must be positive")
        shadow_decision = (
            detector._decision(True, "plateau_detected")
            if detector.stop_epoch is not None
            else None
        )
        executed = start_epoch
        metrics = self._checkpoint_metrics(resume_state)
        phase2_end = (
            start_epoch
            if shadow_decision is not None and mode == "enforce"
            else duration
        )
        for local_epoch in range(start_epoch + 1, phase2_end + 1):
            self._train_one_epoch(config, local_epoch)
            executed = local_epoch
            if local_epoch % interval and local_epoch != duration:
                self._save_resume_if_due(
                    config,
                    phase=2,
                    phase_epoch=local_epoch,
                    phase_state=self._phase_state(
                        detector=detector, selector=selector
                    ),
                )
                continue
            metrics = self._evaluate(2, local_epoch)
            if shadow_decision is None:
                decision = detector.update(metrics)
                path = self.save_path(
                    f"phase2_candidate_epoch_{local_epoch}"
                )
                record = Phase2CheckpointRecord(metrics, path)
                selector.add(record)
                self._save_candidate(
                    config,
                    f"phase2_candidate_epoch_{local_epoch}",
                    phase=2,
                    phase_epoch=local_epoch,
                    metrics=metrics,
                    phase_state=self._phase_state(
                        detector=detector, selector=selector
                    ),
                )
                self._prune_candidates(
                    {record.checkpoint_path for record in selector.records}
                )
                if decision.should_stop:
                    shadow_decision = decision
            else:
                decision = shadow_decision
            self._log_modality_metrics(
                "phase2",
                metrics,
                {
                    "phase2/full_loss_bad_checks": (
                        decision.full_loss_bad_checks
                    ),
                    "phase2/weak_loss_slope": (
                        decision.weak_loss_slope
                        if decision.weak_loss_slope is not None
                        else float("nan")
                    ),
                    "phase2/weak_utility_slope": (
                        decision.weak_utility_slope
                        if decision.weak_utility_slope is not None
                        else float("nan")
                    ),
                    "phase2/plateau_confirmations": (
                        decision.plateau_confirmations
                    ),
                    "phase2/best_checkpoint_epoch": (
                        selector.best.metrics.phase_epoch
                        if selector.best is not None
                        else -1
                    ),
                    "phase2/best_loss_checkpoint_epoch": (
                        selector.best_loss.metrics.phase_epoch
                        if selector.best_loss is not None
                        else -1
                    ),
                    "phase2/best_accuracy_checkpoint_epoch": (
                        selector.best_accuracy.metrics.phase_epoch
                        if selector.best_accuracy is not None
                        else -1
                    ),
                    "phase2/selection_scope_global": float(
                        detector_config.selection_scope == "global"
                    ),
                },
            )
            self._log_blurred_diagnostic(
                "phase2", metrics.global_step, metrics.phase_epoch
            )
            self._save_resume_if_due(
                config,
                phase=2,
                phase_epoch=local_epoch,
                metrics=metrics,
                phase_state=self._phase_state(
                    detector=detector, selector=selector
                ),
            )
            if decision.should_stop and mode == "enforce":
                break

        best_loss = selector.best_loss
        best_accuracy = selector.best_accuracy
        best = selector.best_for(detector_config.primary_metric)
        self._phase2_posthoc_records = {
            name: record
            for name, record in (
                ("best_loss", best_loss),
                ("best_accuracy", best_accuracy),
            )
            if record is not None
        }
        hypothetical_stop_reason = (
            "plateau_detected"
            if shadow_decision is not None
            and shadow_decision.should_stop
            else "max_epochs_reached"
        )
        stop_reason = hypothetical_stop_reason if mode == "enforce" else mode
        if mode == "enforce" and best is not None:
            self._load_selected(best.checkpoint_path)
        boundary_metrics = (
            best.metrics
            if mode == "enforce" and best is not None
            else metrics
        )
        boundary_path = self._save_candidate(
            config,
            "phase2_boundary",
            phase=2,
            phase_epoch=(
                best.metrics.phase_epoch
                if mode == "enforce" and best is not None
                else executed
            ),
            metrics=boundary_metrics,
            phase_state=self._phase_state(
                detector=detector, selector=selector
            ),
        )
        self._phase_boundary_paths.add(boundary_path)
        if best is not None:
            self._phase_boundary_paths.add(best.checkpoint_path)
        for retained in (best_loss, best_accuracy):
            if retained is not None:
                self._phase_boundary_paths.add(retained.checkpoint_path)
        self._write_phase_summary(
            {
                "phase": 2,
                "mode": mode,
                "executed_epochs": executed,
                "stop_reason": stop_reason,
                "hypothetical_stop_reason": hypothetical_stop_reason,
                "selected_checkpoint": (
                    best.checkpoint_path if best is not None else None
                ),
                "selected_source_step": self.selected_source_step,
                "selection_scope": detector_config.selection_scope,
                "selected_metrics": self._summary_metrics(
                    best.metrics if best is not None else None
                ),
                "primary_metric": detector_config.primary_metric,
                "best_loss_checkpoint": (
                    best_loss.checkpoint_path if best_loss is not None else None
                ),
                "best_loss_metrics": self._summary_metrics(
                    best_loss.metrics if best_loss is not None else None
                ),
                "best_accuracy_checkpoint": (
                    best_accuracy.checkpoint_path
                    if best_accuracy is not None
                    else None
                ),
                "best_accuracy_metrics": self._summary_metrics(
                    best_accuracy.metrics
                    if best_accuracy is not None
                    else None
                ),
                "boundary_checkpoint": boundary_path,
            }
        )
        self._prune_candidates(
            {
                retained.checkpoint_path
                for retained in (best_loss, best_accuracy)
                if retained is not None
            }
        )
        self._complete_resumed_phase(2)
        return best

    def _phase3_config(self, config, duration):
        section = _section(config, "phase3_stopping")
        values = {
            key: value
            for key, value in section.items()
            if key in Phase3StopConfig.__dataclass_fields__
        }
        values["max_epochs"] = duration
        interval = int(section.get("eval_interval_epochs", 5))
        if "max_looks" not in values:
            values["max_looks"] = max(1, (duration + interval - 1) // interval)
        return section, Phase3StopConfig(**values)

    def _log_phase3(self, metrics, record, decision, stopper):
        extra = {
            "phase3/weak_quality_gain": record.weak_quality_gain,
            "phase3/weak_accuracy_gain": record.weak_accuracy_gain,
            "phase3/weak_only_loss": metrics.weak_only.loss,
            "phase3/weak_only_accuracy": metrics.weak_only.accuracy,
            "phase3/phase_epoch": metrics.phase_epoch,
            "phase3/weak_utility_gain": record.weak_utility_gain,
            "phase3/full_loss_increase": record.full_loss_increase,
            "phase3/dominant_loss_increase": (
                record.dominant_loss_increase
            ),
            "phase3/is_feasible": float(record.is_feasible),
            "phase3/is_safe": float(record.is_safe),
            "phase3/bad_checks": decision.bad_checks,
            "phase3/best_checkpoint_epoch": (
                stopper.best_feasible.metrics.phase_epoch
                if stopper.best_feasible is not None
                else (
                    stopper.best_safe.metrics.phase_epoch
                    if stopper.best_safe is not None
                    else -1
                )
            ),
            "phase3/safety_bad_checks": decision.safety_bad_checks,
            "phase3/reversal_bad_checks": stopper.reversal_bad_checks,
            "phase3/futility_bad_checks": stopper.futility_bad_checks,
            "phase3/recovery_plateau_bad_checks": stopper.futility_bad_checks,
            "phase3/evaluation_count": stopper.evaluation_count,
            "phase3/compatibility_drift_loss": (
                record.dominant_loss_increase
            ),
            "phase3/compatibility_drift_accuracy": (
                record.compatibility_drift_accuracy
            ),
            "phase3/full_accuracy_change": (
                metrics.full.accuracy - stopper.baseline.full.accuracy
            ),
            "phase3/dominant_accuracy_change": (
                metrics.dominant_only.accuracy
                - stopper.baseline.dominant_only.accuracy
            ),
            "phase3/reactivation_full_loss_gap": (
                record.reactivation_full_loss_gap
            ),
        }
        if record.weak_ratio is not None:
            extra.update(
                {
                    "phase3/unimodal_left_accuracy": (
                        stopper.unimodal_left_accuracy
                    ),
                    "phase3/unimodal_right_accuracy": (
                        stopper.unimodal_right_accuracy
                    ),
                    "phase3/dominant_ratio": record.dominant_ratio,
                    "phase3/weak_ratio": record.weak_ratio,
                    "phase3/parity_gap": record.parity_gap,
                    "phase3/recovery_fraction": (
                        record.recovery_fraction
                    ),
                    "phase3/recovery_fraction_threshold": (
                        record.recovery_fraction_threshold
                    ),
                    "phase3/parity_confirmations": (
                        stopper.parity_confirmations
                    ),
                    "phase3/first_parity_epoch": (
                        stopper.first_parity_candidate.metrics.phase_epoch
                        if stopper.first_parity_candidate is not None
                        else -1
                    ),
                }
            )
        for name, epoch in getattr(stopper, "first_trigger_epochs", {}).items():
            extra[f"phase3/first_{name}_epoch"] = epoch
        for name, estimate in (record.paired_estimates or {}).items():
            extra[f"phase3/{name}_ci_lower"] = estimate["lower"]
            extra[f"phase3/{name}_ci_upper"] = estimate["upper"]
            extra[f"phase3/{name}_standard_error"] = estimate[
                "standard_error"
            ]
        for name, estimate in (stopper.last_trend_estimates or {}).items():
            extra[f"phase3/{name}"] = estimate.mean
            extra[f"phase3/{name}_ci_lower"] = estimate.lower
            extra[f"phase3/{name}_ci_upper"] = estimate.upper
        for name, value in (stopper.last_optimistic_bounds or {}).items():
            extra[f"phase3/optimistic_{name}_upper"] = value
        for name, value in getattr(stopper, "last_local_flags", {}).items():
            extra[f"phase3/local_{name}"] = float(value)
        for name in (
            "target_bad_checks",
            "pareto_bad_checks",
            "futility_harm_bad_checks",
        ):
            if hasattr(stopper, name):
                extra[f"phase3/{name}"] = getattr(stopper, name)
        self._log_modality_metrics("phase3", metrics, extra)

    def _run_phase3(self, config):
        duration = int(config.get("phase3", 0))
        if duration <= 0:
            self._write_phase_summary(
                {"phase": 3, "executed_epochs": 0, "stop_reason": "skipped"}
            )
            return 0
        section, stopper_config = self._phase3_config(config, duration)
        mode = str(section.get("mode", "disabled"))
        observe_phase4_transition = str(
            section.get("observe_phase4_transition", "endpoint")
        )
        if observe_phase4_transition not in {
            "endpoint",
            "hypothetical_selected",
        }:
            raise ValueError("unknown observe_phase4_transition")
        if (
            observe_phase4_transition != "endpoint"
            and mode != "observe_only"
        ):
            raise ValueError(
                "observe_phase4_transition requires observe_only mode"
            )
        stopper_class = {
            "weak_recovery": Phase3RecoveryStopper,
            "local_accuracy": Phase3LocalAccuracyStopper,
            "relative_unimodal_parity": Phase3RelativeUnimodalStopper,
        }.get(stopper_config.decision_rule, Phase3InterventionStopper)

        def build_stopper(baseline_metrics):
            if stopper_class is not Phase3RelativeUnimodalStopper:
                return stopper_class(stopper_config, baseline_metrics)
            references = config.get("unimodal_references")
            if not references:
                raise ValueError(
                    "relative_unimodal_parity requires validated references"
                )
            return stopper_class(
                stopper_config,
                baseline_metrics,
                unimodal_left_accuracy=float(
                    references["left"]["validation_accuracy"]
                ),
                unimodal_right_accuracy=float(
                    references["right"]["validation_accuracy"]
                ),
            )
        start_epoch, resume_state = self._prepare_phase(3, config)
        if resume_state is None:
            baseline = self._evaluate(3, 0)
            metrics = baseline
            self._log_modality_metrics("phase3_baseline", baseline)
            self._log_blurred_diagnostic(
                "phase3_baseline",
                baseline.global_step,
                baseline.phase_epoch,
                force=True,
            )
            pre_path = self.save_path("phase3_pre_intervention")
            stopper = build_stopper(baseline)
            baseline_record = stopper._record(baseline, pre_path)
            baseline_decision = None
            if isinstance(stopper, Phase3RelativeUnimodalStopper):
                baseline_decision = stopper.initialize_baseline(pre_path)
                baseline_record = baseline_decision.current
                self.logger.log_scalars(
                    {
                        "phase3/unimodal_left_accuracy": (
                            stopper.unimodal_left_accuracy
                        ),
                        "phase3/unimodal_right_accuracy": (
                            stopper.unimodal_right_accuracy
                        ),
                        "phase3/dominant_ratio": stopper.dominant_ratio,
                        "phase3/weak_ratio": baseline_record.weak_ratio,
                        "phase3/parity_gap": baseline_record.parity_gap,
                        "phase3/recovery_fraction": (
                            baseline_record.recovery_fraction
                        ),
                        "phase3/recovery_fraction_threshold": (
                            baseline_record.recovery_fraction_threshold
                        ),
                        "phase3/parity_confirmations": 0,
                        "phase3/first_parity_epoch": (
                            0 if baseline_decision.should_stop else -1
                        ),
                    },
                    int(self.global_step or 0),
                )
            trajectory_path = self._append_phase3_trajectory(
                phase3_trajectory_record(
                    baseline,
                    decision_rule=stopper_config.decision_rule,
                    checkpoint_path=pre_path,
                    checkpoint_retained=True,
                    current_record=baseline_record,
                    decision=baseline_decision,
                    unimodal_references=(
                        OmegaConf.to_container(
                            config.get("unimodal_references"), resolve=True
                        )
                        if isinstance(stopper, Phase3RelativeUnimodalStopper)
                        else None
                    ),
                )
            )
            self._save_candidate(
                config,
                "phase3_pre_intervention",
                phase=3,
                phase_epoch=0,
                metrics=baseline,
                phase_state={
                    **self._phase_state(stopper=stopper),
                    "pre_checkpoint_path": pre_path,
                    "milestone_checkpoint_paths": [],
                },
            )
        else:
            controller_state = resume_state.get("phase_state") or {}
            stopper_state = controller_state.get("stopper")
            if stopper_state is None:
                raise ValueError("phase 3 resume checkpoint lacks stopper state")
            baseline = ModalityEvaluationResult.from_state_dict(
                stopper_state["baseline"]
            )
            metrics = self._checkpoint_metrics(resume_state)
            stopper = build_stopper(baseline)
            stopper.load_state_dict(stopper_state)
            trajectory_path = str(
                Path(self.base_path) / "phase3_trajectory.jsonl"
            )
            pre_path = controller_state.get("pre_checkpoint_path")
            if pre_path is None:
                raise ValueError(
                    "phase 3 resume checkpoint lacks pre-intervention path"
                )
        milestone_epochs = {
            int(epoch)
            for epoch in section.get("calibration_milestone_epochs", [])
        }
        if any(epoch <= 0 or epoch > duration for epoch in milestone_epochs):
            raise ValueError("phase 3 calibration milestone is out of range")
        materialization_epochs = {
            int(epoch)
            for epoch in section.get(
                "materialization_checkpoint_epochs", []
            )
        }
        if any(
            epoch <= 0 or epoch > duration
            for epoch in materialization_epochs
        ):
            raise ValueError(
                "phase 3 materialization checkpoint is out of range"
            )
        milestone_paths = set(
            (controller_state if resume_state is not None else {}).get(
                "milestone_checkpoint_paths", []
            )
        )

        def phase3_state():
            return {
                **self._phase_state(stopper=stopper),
                "pre_checkpoint_path": pre_path,
                "milestone_checkpoint_paths": sorted(milestone_paths),
            }
        interval = int(section.get("eval_interval_epochs", 5))
        if interval < 1:
            raise ValueError("phase3 eval_interval_epochs must be positive")
        initial_dense_epochs = int(
            section.get("initial_dense_eval_epochs", 0)
        )
        if initial_dense_epochs < 0:
            raise ValueError(
                "phase3 initial_dense_eval_epochs must be non-negative"
            )
        final_decision = stopper.final_decision
        executed = start_epoch
        current_path = pre_path
        phase3_end = (
            start_epoch
            if final_decision is not None and mode == "enforce"
            else duration
        )
        for local_epoch in range(start_epoch + 1, phase3_end + 1):
            self._train_one_epoch(config, local_epoch)
            executed = local_epoch
            if local_epoch in materialization_epochs:
                materialized_path = self._save_candidate(
                    config,
                    f"phase3_materialized_epoch_{local_epoch}",
                    phase=3,
                    phase_epoch=local_epoch,
                    phase_state=phase3_state(),
                    extra_metadata={
                        "checkpoint_role": "materialization_without_evaluation"
                    },
                )
                milestone_paths.add(materialized_path)
            if not should_evaluate_phase_epoch(
                local_epoch,
                duration,
                interval,
                initial_dense_epochs,
            ):
                self._save_resume_if_due(
                    config,
                    phase=3,
                    phase_epoch=local_epoch,
                    phase_state=phase3_state(),
                )
                continue
            metrics = self._evaluate(3, local_epoch)
            gradient_diagnostics = self._phase3_gradient_diagnostics(metrics)
            should_update = (
                final_decision is None
                or stopper_config.shadow_continue_after_stop
            )
            current_path = self.save_path(
                f"phase3_candidate_epoch_{local_epoch}"
            )
            if should_update:
                if isinstance(stopper, Phase3LocalAccuracyStopper):
                    decision = stopper.update(
                        metrics,
                        current_path,
                        diagnostics=gradient_diagnostics,
                    )
                else:
                    decision = stopper.update(metrics, current_path)
                if decision.should_stop and final_decision is None:
                    final_decision = stopper.final_decision or decision
            else:
                decision = final_decision
            current_record = (
                decision.current
                if should_update
                else stopper._record(metrics, current_path)
            )
            if local_epoch in milestone_epochs:
                milestone_paths.add(current_path)
            keep = {pre_path, *milestone_paths}
            if final_decision is not None and final_decision.selected is not None:
                keep.add(final_decision.selected.checkpoint_path)
            if stopper.best_feasible is not None:
                keep.add(stopper.best_feasible.checkpoint_path)
            if stopper.best_safe is not None:
                keep.add(stopper.best_safe.checkpoint_path)
            keep.update(
                getattr(stopper, "retained_checkpoint_paths", set())
            )
            if current_path in keep:
                self._save_candidate(
                    config,
                    f"phase3_candidate_epoch_{local_epoch}",
                    phase=3,
                    phase_epoch=local_epoch,
                    metrics=metrics,
                    phase_state=phase3_state(),
                )
            self._append_phase3_trajectory(
                phase3_trajectory_record(
                    metrics,
                    decision_rule=stopper_config.decision_rule,
                    checkpoint_path=current_path,
                    checkpoint_retained=current_path in keep,
                    current_record=current_record,
                    decision=decision,
                    unimodal_references=(
                        OmegaConf.to_container(
                            config.get("unimodal_references"), resolve=True
                        )
                        if isinstance(stopper, Phase3RelativeUnimodalStopper)
                        else None
                    ),
                )
            )
            self._prune_candidates(keep)
            self._log_phase3(metrics, current_record, decision, stopper)
            self._log_train_probe(metrics)
            self._log_blurred_diagnostic(
                "phase3", metrics.global_step, metrics.phase_epoch
            )
            self._save_resume_if_due(
                config,
                phase=3,
                phase_epoch=local_epoch,
                metrics=metrics,
                phase_state=phase3_state(),
            )
            if decision.should_stop and mode == "enforce":
                break

        if final_decision is None:
            selected, status = stopper.selection()
            hypothetical_stop_reason = "max_epochs"
        else:
            selected = final_decision.selected
            status = final_decision.selection_status
            hypothetical_stop_reason = final_decision.stop_reason
        stop_reason = (
            hypothetical_stop_reason
            if mode == "enforce"
            else mode
        )
        replay_hypothetical = (
            mode == "observe_only"
            and observe_phase4_transition == "hypothetical_selected"
        )
        if mode == "enforce" or replay_hypothetical:
            transition_path = (
                selected.checkpoint_path
                if selected is not None
                else pre_path
            )
            self._load_selected(transition_path)
            intervention_epochs = (
                selected.metrics.phase_epoch if selected is not None else 0
            )
            boundary_metrics = (
                selected.metrics if selected is not None else baseline
            )
            phase4_transition = (
                "hypothetical_selected"
                if replay_hypothetical
                else "enforced_selected"
            )
        else:
            intervention_epochs = executed
            boundary_metrics = metrics
            phase4_transition = "phase3_endpoint"
        selected_path = (
            selected.checkpoint_path if selected is not None else pre_path
        )
        boundary_path = self._save_candidate(
            config,
            "phase3_boundary",
            phase=3,
            phase_epoch=intervention_epochs,
            metrics=boundary_metrics,
            phase_state=phase3_state(),
        )
        active_best_paths = {
            record.checkpoint_path
            for record in (stopper.best_feasible, stopper.best_safe)
            if record is not None
        }
        self._phase_boundary_paths.update(
            {
                boundary_path,
                pre_path,
                selected_path,
                *milestone_paths,
                *active_best_paths,
            }
        )
        self._write_phase_summary(
            {
                "phase": 3,
                "mode": mode,
                "phase4_transition": phase4_transition,
                "executed_epochs": executed,
                "intervention_epochs": intervention_epochs,
                "stop_reason": stop_reason,
                "hypothetical_stop_reason": hypothetical_stop_reason,
                "selection_status": status,
                "selected_checkpoint": selected_path,
                "selected_source_step": self.selected_source_step,
                "boundary_checkpoint": boundary_path,
                "baseline_metrics": self._summary_metrics(baseline),
                "final_observed_metrics": self._summary_metrics(metrics),
                "selected_metrics": self._summary_metrics(
                    selected.metrics if selected is not None else baseline
                ),
                "best_feasible_epoch": (
                    stopper.best_feasible.metrics.phase_epoch
                    if stopper.best_feasible is not None
                    else None
                ),
                "best_feasible_checkpoint": (
                    stopper.best_feasible.checkpoint_path
                    if stopper.best_feasible is not None
                    else None
                ),
                "best_safe_epoch": (
                    stopper.best_safe.metrics.phase_epoch
                    if stopper.best_safe is not None
                    else None
                ),
                "best_safe_checkpoint": (
                    stopper.best_safe.checkpoint_path
                    if stopper.best_safe is not None
                    else None
                ),
                "last_optimistic_bounds": stopper.last_optimistic_bounds,
                "first_trigger_epochs": dict(
                    getattr(stopper, "first_trigger_epochs", {})
                ),
                "milestone_checkpoints": sorted(milestone_paths),
                "trajectory_artifact": trajectory_path,
                "unimodal_references": (
                    OmegaConf.to_container(
                        config.get("unimodal_references"), resolve=True
                    )
                    if isinstance(stopper, Phase3RelativeUnimodalStopper)
                    else None
                ),
                "dominant_ratio": getattr(
                    stopper, "dominant_ratio", None
                ),
                "parity_confirmations": getattr(
                    stopper, "parity_confirmations", None
                ),
                "recovery_fraction_threshold": getattr(
                    stopper, "recovery_fraction_threshold", None
                ),
            }
        )
        self._prune_candidates(
            {selected_path, *milestone_paths, *active_best_paths}
        )
        self._complete_resumed_phase(3)
        return intervention_epochs

    def _run_phase4(self, config, intervention_epochs):
        duration = int(config.get("phase4", 0))
        if duration < 0:
            raise ValueError("phase4 must be non-negative")
        section = _section(config, "phase4_selection")
        interval = int(section.get("eval_interval_epochs", 5))
        total_budget = int(section.get("budget_total_epochs", 200))
        primary_metric = str(section.get("primary_metric", "loss"))
        if interval < 1:
            raise ValueError("phase4 eval_interval_epochs must be positive")
        if primary_metric not in {"loss", "accuracy"}:
            raise ValueError("unknown phase 4 primary_metric")
        if duration > total_budget:
            raise ValueError(
                "phase4 exceeds phase4_selection budget_total_epochs"
            )
        diagnostic_section = _section(config, "phase4_diagnostics")
        trainability_section = _section(
            config, "phase4_staged_unfreezing"
        )
        staged_unfreezing_enabled = bool(
            trainability_section.get("enabled", False)
        )
        shared_only_epochs = int(
            trainability_section.get("shared_only_epochs", 0)
        )
        if shared_only_epochs < 0:
            raise ValueError(
                "phase4_staged_unfreezing shared_only_epochs must be "
                "non-negative"
            )
        if staged_unfreezing_enabled and shared_only_epochs > duration:
            raise ValueError(
                "phase4_staged_unfreezing shared_only_epochs exceeds Phase 4"
            )
        diagnostic_enabled = bool(diagnostic_section.get("enabled", False))
        dense_eval_epochs = {
            int(epoch)
            for epoch in diagnostic_section.get("dense_eval_epochs", [])
        }
        hybrid_eval_epochs = {
            int(epoch)
            for epoch in diagnostic_section.get(
                "hybrid_eval_epochs", dense_eval_epochs
            )
        }
        invalid_diagnostic_epochs = sorted(
            epoch
            for epoch in dense_eval_epochs | hybrid_eval_epochs
            if epoch < 0 or epoch > duration
        )
        if invalid_diagnostic_epochs:
            raise ValueError(
                "phase4 diagnostic epochs must be within the Phase-4 budget: "
                f"{invalid_diagnostic_epochs}"
            )
        if hybrid_eval_epochs - dense_eval_epochs:
            raise ValueError(
                "phase4 hybrid_eval_epochs must be a subset of "
                "dense_eval_epochs"
            )
        start_epoch, resume_state = self._prepare_phase(4, config)
        if (
            resume_state is None
            and int(config.get("phase4_bn_recalibration_batches", 0) or 0)
        ):
            self._maybe_recalibrate_phase4_batchnorm(config)
        selector = Phase4CheckpointSelector(
            total_budget, intervention_epochs
        )
        diagnostic_anchor = None
        diagnostic_artifact = None
        diagnostic_state = None
        if resume_state is not None:
            controller_state = resume_state.get("phase_state") or {}
            selector_state = controller_state.get("selector")
            if selector_state is None:
                raise ValueError("phase 4 resume checkpoint lacks selector state")
            selector.load_state_dict(selector_state)
            if diagnostic_enabled:
                diagnostic_state = controller_state.get("phase4_diagnostic")
                if not diagnostic_state:
                    raise ValueError(
                        "phase 4 diagnostic resume lacks anchor state"
                    )
                anchor_path = Path(diagnostic_state["anchor_path"])
                if not anchor_path.is_file():
                    raise FileNotFoundError(
                        f"Phase-4 diagnostic anchor not found: {anchor_path}"
                    )
                diagnostic_anchor = Phase4HybridAnchor.from_state_dict(
                    torch.load(
                        anchor_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                )
                artifact_path = Path(self.base_path) / (
                    "phase4_hybrid_trajectory.jsonl"
                )
                diagnostic_artifact = (
                    str(artifact_path) if artifact_path.is_file() else None
                )
        elif diagnostic_enabled:
            diagnostic_anchor = Phase4HybridAnchor.capture(self.metric_model)
            anchor_path = Path(self.base_path) / "phase4_hybrid_anchor.pt"
            torch.save(diagnostic_anchor.state_dict(), anchor_path)
            diagnostic_state = {
                "version": 1,
                "anchor_path": str(anchor_path),
                "dense_eval_epochs": sorted(dense_eval_epochs),
                "hybrid_eval_epochs": sorted(hybrid_eval_epochs),
            }
        if resume_state is None:
            initial = self._evaluate(4, 0)
            self._log_modality_metrics("phase4", initial)
            self._log_blurred_diagnostic(
                "phase4", initial.global_step, initial.phase_epoch
            )
            initial_path = self.save_path("phase4_candidate_epoch_0")
            initial_record = Phase4CheckpointRecord(initial, initial_path)
            selector.add(initial_record)
            if diagnostic_enabled and 0 in hybrid_eval_epochs:
                _hybrids, diagnostic_artifact = (
                    self._evaluate_phase4_hybrids(
                        diagnostic_anchor, initial
                    )
                )
            self._save_candidate(
                config,
                "phase4_candidate_epoch_0",
                phase=4,
                phase_epoch=0,
                metrics=initial,
                phase_state=self._phase_state(
                    selector=selector,
                    phase4_diagnostic=diagnostic_state,
                ),
            )
        for local_epoch in range(start_epoch + 1, duration + 1):
            self._train_one_epoch(config, local_epoch)
            should_evaluate = (
                local_epoch in dense_eval_epochs
                or local_epoch % interval == 0
                or local_epoch == duration
            )
            if not should_evaluate:
                self._save_resume_if_due(
                    config,
                    phase=4,
                    phase_epoch=local_epoch,
                    phase_state=self._phase_state(
                        selector=selector,
                        phase4_diagnostic=diagnostic_state,
                    ),
                )
                continue
            metrics = self._evaluate(4, local_epoch)
            if diagnostic_enabled and local_epoch in hybrid_eval_epochs:
                _hybrids, diagnostic_artifact = (
                    self._evaluate_phase4_hybrids(
                        diagnostic_anchor, metrics
                    )
                )
            path = self.save_path(f"phase4_candidate_epoch_{local_epoch}")
            record = Phase4CheckpointRecord(metrics, path)
            selector.add(record)
            keep = {
                retained.checkpoint_path
                for retained in (
                    selector.best_full,
                    selector.best_budget_matched,
                    selector.best_full_accuracy,
                    selector.best_budget_matched_accuracy,
                )
                if retained is not None
            }
            if path in keep:
                self._save_candidate(
                    config,
                    f"phase4_candidate_epoch_{local_epoch}",
                    phase=4,
                    phase_epoch=local_epoch,
                    metrics=metrics,
                    phase_state=self._phase_state(
                        selector=selector,
                        phase4_diagnostic=diagnostic_state,
                    ),
                )
            self._prune_candidates(keep)
            self._log_modality_metrics("phase4", metrics)
            self._log_blurred_diagnostic(
                "phase4", metrics.global_step, metrics.phase_epoch
            )
            self._save_resume_if_due(
                config,
                phase=4,
                phase_epoch=local_epoch,
                metrics=metrics,
                phase_state=self._phase_state(
                    selector=selector,
                    phase4_diagnostic=diagnostic_state,
                ),
            )

        selected = {
            "best_full": selector.best_full,
            "best_budget_matched": selector.best_budget_matched,
            "best_full_accuracy": selector.best_full_accuracy,
            "best_budget_matched_accuracy": (
                selector.best_budget_matched_accuracy
            ),
        }
        test_policy = str(config.get("phase4_test_policy", "final_only"))
        if test_policy not in {"final_only", "disabled"}:
            raise ValueError("unknown phase4_test_policy")
        test_summary = {}
        if test_policy == "final_only":
            test_loaders = self.final_test_loader_factory()
            evaluated = {}
            for label, record in selected.items():
                if record is None:
                    continue
                if record.checkpoint_path not in evaluated:
                    self._load_selected(record.checkpoint_path)
                    proper = self._evaluate_full_loader(
                        test_loaders["test_proper"],
                        max_batches=self._batch_limit("max_test_batches"),
                    )
                    blurred = self._evaluate_full_loader(
                        test_loaders["test_blurred"],
                        max_batches=self._batch_limit("max_test_batches"),
                    )
                    evaluated[record.checkpoint_path] = (proper, blurred)
                proper, blurred = evaluated[record.checkpoint_path]
                self.logger.log_scalars(
                    {
                        f"final_test/{label}/proper_loss": proper.loss,
                        f"final_test/{label}/proper_accuracy": proper.accuracy,
                        f"final_test/{label}/blurred_loss": blurred.loss,
                        f"final_test/{label}/blurred_accuracy": blurred.accuracy,
                    },
                    int(self.global_step or 0),
                )
                test_summary[label] = {
                    "proper_loss": proper.loss,
                    "proper_accuracy": proper.accuracy,
                    "blurred_loss": blurred.loss,
                    "blurred_accuracy": blurred.accuracy,
                }
        primary = selector.best_full_for(primary_metric)
        if primary is not None:
            self._load_selected(primary.checkpoint_path)
            self._final_selected_checkpoint_path = primary.checkpoint_path
        kept = {
            record.checkpoint_path
            for record in selected.values()
            if record is not None
        }
        self._prune_candidates(kept)
        self._write_phase_summary(
            {
                "phase": 4,
                "executed_epochs": duration,
                "intervention_epochs": intervention_epochs,
                "primary_metric": primary_metric,
                "primary_checkpoint": (
                    primary.checkpoint_path if primary is not None else None
                ),
                "primary_metrics": self._summary_metrics(
                    primary.metrics if primary is not None else None
                ),
                "best_full_checkpoint": (
                    selector.best_full.checkpoint_path
                    if selector.best_full is not None
                    else None
                ),
                "best_budget_matched_checkpoint": (
                    selector.best_budget_matched.checkpoint_path
                    if selector.best_budget_matched is not None
                    else None
                ),
                "best_full_accuracy_checkpoint": (
                    selector.best_full_accuracy.checkpoint_path
                    if selector.best_full_accuracy is not None
                    else None
                ),
                "best_budget_matched_accuracy_checkpoint": (
                    selector.best_budget_matched_accuracy.checkpoint_path
                    if selector.best_budget_matched_accuracy is not None
                    else None
                ),
                "best_full_metrics": self._summary_metrics(
                    selector.best_full.metrics
                    if selector.best_full is not None
                    else None
                ),
                "best_budget_matched_metrics": self._summary_metrics(
                    selector.best_budget_matched.metrics
                    if selector.best_budget_matched is not None
                    else None
                ),
                "best_full_accuracy_metrics": self._summary_metrics(
                    selector.best_full_accuracy.metrics
                    if selector.best_full_accuracy is not None
                    else None
                ),
                "best_budget_matched_accuracy_metrics": (
                    self._summary_metrics(
                        selector.best_budget_matched_accuracy.metrics
                        if selector.best_budget_matched_accuracy is not None
                        else None
                    )
                ),
                "test_policy": test_policy,
                "test_metrics": test_summary,
                "selected_source_step": self.selected_source_step,
                "phase4_diagnostics": {
                    "enabled": diagnostic_enabled,
                    "dense_eval_epochs": sorted(dense_eval_epochs),
                    "hybrid_eval_epochs": sorted(hybrid_eval_epochs),
                    "anchor_path": (
                        diagnostic_state["anchor_path"]
                        if diagnostic_state is not None
                        else None
                    ),
                    "trajectory_artifact": diagnostic_artifact,
                },
                "phase4_staged_unfreezing": {
                    "enabled": staged_unfreezing_enabled,
                    "shared_only_epochs": (
                        shared_only_epochs if staged_unfreezing_enabled else 0
                    ),
                    "optimizer_contains_all_phase4_parameters": True,
                },
            }
        )
        self._complete_resumed_phase(4)

    def _restore_phase2_posthoc_records(self, state):
        raw_records = (state.get("phase_state") or {}).get(
            "phase2_posthoc_records", {}
        )
        if raw_records:
            self._phase2_posthoc_records = {
                name: Phase2CheckpointRecord.from_state_dict(record)
                for name, record in raw_records.items()
            }

    def _run_phase2_posthoc_test(self, config):
        config_get = getattr(config, "get", None)
        policy = str(
            config_get("phase2_test_policy", "disabled")
            if config_get is not None
            else getattr(config, "phase2_test_policy", "disabled")
        )
        if policy == "disabled":
            return
        if policy != "posthoc_final":
            raise ValueError("unknown phase2_test_policy")
        if not self._phase2_posthoc_records:
            self._write_phase_summary(
                {
                    "phase": "posthoc_phase2_test",
                    "policy": policy,
                    "status": "no_phase2_checkpoints",
                }
            )
            return
        final_path = self._final_selected_checkpoint_path
        if final_path is None:
            raise RuntimeError(
                "post-hoc Phase 2 test requires a final selected checkpoint"
            )
        test_loaders = self.final_test_loader_factory()
        evaluated = {}
        summary = {
            "phase": "posthoc_phase2_test",
            "policy": policy,
            "status": "completed",
            "checkpoints": {},
        }
        try:
            for label, record in self._phase2_posthoc_records.items():
                if record.checkpoint_path not in evaluated:
                    self._load_selected(record.checkpoint_path)
                    proper = self._evaluate_full_loader(
                        test_loaders["test_proper"],
                        max_batches=self._batch_limit("max_test_batches"),
                    )
                    blurred = self._evaluate_full_loader(
                        test_loaders["test_blurred"],
                        max_batches=self._batch_limit("max_test_batches"),
                    )
                    evaluated[record.checkpoint_path] = (proper, blurred)
                proper, blurred = evaluated[record.checkpoint_path]
                values = {
                    f"posthoc_test/phase2/{label}/proper_loss": proper.loss,
                    f"posthoc_test/phase2/{label}/proper_accuracy": (
                        proper.accuracy
                    ),
                    f"posthoc_test/phase2/{label}/blurred_loss": blurred.loss,
                    f"posthoc_test/phase2/{label}/blurred_accuracy": (
                        blurred.accuracy
                    ),
                }
                self.logger.log_scalars(values, int(self.global_step or 0))
                summary["checkpoints"][label] = {
                    "checkpoint": record.checkpoint_path,
                    "validation_metrics": self._summary_metrics(record.metrics),
                    "test_proper": {
                        "loss": proper.loss,
                        "accuracy": proper.accuracy,
                    },
                    "test_blurred": {
                        "loss": blurred.loss,
                        "accuracy": blurred.accuracy,
                    },
                }
        finally:
            self._load_selected(final_path)
        self._write_phase_summary(summary)

    def run_phase(self, phase, config):
        if int(phase) not in {3, 4}:
            return super().run_phase(phase, config)
        self._active_config = config
        self._initialize_run(config)
        self.executed_global_epoch = int(config.exp_starts_at_epoch)
        self.global_step = 0
        try:
            if int(phase) == 3:
                self._run_phase3(config)
            else:
                self._run_phase4(config, int(config.get("phase3", 0)))
        finally:
            if self.logger is not None:
                self.logger.close()

    def run_all_at_once(self, config):
        self._active_config = config
        self._initialize_run(config)
        resume_phase = self._resume_phase()
        if resume_phase is not None:
            state = self.resume_training_state
            self.executed_global_epoch = int(state["next_epoch"])
            self.global_step = int(state["global_step"])
            self._restore_phase2_posthoc_records(state)
        try:
            if resume_phase is None or resume_phase <= 1:
                self._run_phase1(config)
            if resume_phase is None or resume_phase <= 2:
                self._run_phase2(config)
            if resume_phase is None or resume_phase <= 3:
                intervention_epochs = self._run_phase3(config)
            else:
                selector_state = (
                    (self.resume_training_state.get("phase_state") or {})
                    .get("selector")
                )
                if selector_state is None:
                    raise ValueError(
                        "phase 4 resume checkpoint lacks intervention budget"
                    )
                intervention_epochs = int(
                    selector_state["intervention_epochs"]
                )
            self._run_phase4(config, intervention_epochs)
            self._run_phase2_posthoc_test(config)
        finally:
            if self.logger is not None:
                self.logger.close()
