import logging
import os
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict

import torch
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from src.modules.batchnorm import recalibrate_batchnorm
from src.utils.run_manifest import RunManifest
from src.data.transforms import TRANSFORMS_BLURRED_RIGHT_NAME_MAP, TRANSFORMS_PROPER_RIGHT_NAME_MAP
from src.utils.common import create_logger

from src.utils.utils_trainer import (
    adjust_evaluators,
    adjust_evaluators_pre_log,
    create_paths,
    restore_rng_state,
    manual_seed as seed_everything,
    save_training_checkpoint,
)
from src.utils.utils_optim import clip_grad_norm


@dataclass(frozen=True)
class TrainingStage:
    kind: str
    enable_left_branch: bool
    enable_right_branch: bool
    left_branch_intervention: str | None = None
    right_branch_intervention: str | None = None
    right_transform: str | None = None


PHASE_STAGES = {
    1: TrainingStage("blurred", True, True, right_transform="blurred"),
    2: TrainingStage("proper", True, True, right_transform="proper"),
    3: TrainingStage(
        "proper",
        False,
        True,
        left_branch_intervention="deactivation",
        right_transform="proper",
    ),
    4: TrainingStage("proper", True, True, right_transform="proper"),
}

PRETRAIN_STAGES = {
    "left_proper": TrainingStage("proper", True, False, right_branch_intervention="deactivation"),
    "right_proper": TrainingStage("proper", False, True, left_branch_intervention="deactivation"),
    "right_blurred": TrainingStage("blurred", False, True, left_branch_intervention="deactivation", right_transform="blurred"),
}


def phase4_auxiliary_loss_weights(config):
    """Return validated (weak, dominant) auxiliary Phase-4 loss weights."""
    section = config.get("phase4_auxiliary_loss", {}) or {}
    if not bool(section.get("enabled", False)):
        return 0.0, 0.0
    weak_weight = float(section.get("weak_weight", 0.0))
    dominant_weight = float(section.get("dominant_weight", 0.0))
    if weak_weight < 0.0 or dominant_weight < 0.0:
        raise ValueError("Phase-4 auxiliary loss weights must be non-negative")
    return weak_weight, dominant_weight


def _measurement_batch_indices(loader_size, measurements_per_epoch):
    """Choose exactly K deterministic local batch indices for one epoch."""
    loader_size = int(loader_size)
    measurements_per_epoch = int(measurements_per_epoch)
    if loader_size < 0:
        raise ValueError("loader_size must be non-negative")
    if measurements_per_epoch < 0:
        raise ValueError("fim_measurements_per_epoch must be non-negative")
    count = min(loader_size, measurements_per_epoch)
    if count == 0:
        return frozenset()
    return frozenset(
        index * loader_size // count for index in range(count)
    )


def fim_measurement_due(phase_epoch, interval, explicit_epochs=None):
    """Return whether TFIM is due at a local phase epoch.

    An explicit epoch list takes precedence over the legacy periodic cadence.
    Requiring a strictly increasing list makes the measurement contract
    unambiguous in manifests and prevents duplicate measurements.
    """
    phase_epoch = int(phase_epoch)
    interval = int(interval)
    if interval < 1:
        raise ValueError("fim_eval_interval_epochs must be positive")
    if explicit_epochs is None:
        return phase_epoch % interval == 0
    epochs = tuple(int(epoch) for epoch in explicit_epochs)
    if not epochs:
        raise ValueError("fim_eval_epochs must not be empty")
    if any(epoch < 1 for epoch in epochs):
        raise ValueError("fim_eval_epochs must contain only positive epochs")
    if tuple(sorted(set(epochs))) != epochs:
        raise ValueError("fim_eval_epochs must be strictly increasing and unique")
    return phase_epoch in epochs


class TrainerClassification:
    def __init__(self, model, criterion, loaders, optim, lr_scheduler, extra_modules, device):
        self.model = model
        self.criterion = criterion
        self.loaders = loaders
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.device = device

        self.logger = None
        self.base_path = None
        self.save_path = None
        self.epoch = -1
        self.global_step = None
        self.resume_rng_state = None
        self.resume_global_step = None
        self.run_manifest = None

        self.extra_modules = extra_modules
        
        
    def _apply_stage(self, config, stage):
        config.extra = {
            "left_branch_intervention": stage.left_branch_intervention,
            "right_branch_intervention": stage.right_branch_intervention,
            "enable_left_branch": stage.enable_left_branch,
            "enable_right_branch": stage.enable_right_branch,
        }
        config.kind = stage.kind

        dataset_name = config.logger_config["hyperparameters"]["type_names"]["dataset"]
        normalization_profile = getattr(
            config, "normalization_profile", None
        )
        if stage.right_transform == "blurred":
            transform_factory = TRANSFORMS_BLURRED_RIGHT_NAME_MAP[dataset_name]
            transform_args = (
                config.overlap,
                getattr(config, "resize_factor", 0.25),
                normalization_profile,
            )
        elif stage.right_transform == "proper":
            transform_factory = TRANSFORMS_PROPER_RIGHT_NAME_MAP[dataset_name]
            transform_args = (config.overlap, normalization_profile)
        else:
            return
        train_dataset = self.loaders["train"].dataset
        while not hasattr(train_dataset, "transform2") and hasattr(train_dataset, "dataset"):
            train_dataset = train_dataset.dataset
        if not hasattr(train_dataset, "transform2"):
            raise AttributeError("training dataset does not expose the right-modality transform")
        train_dataset.transform2 = transform_factory(*transform_args)

    def _run_stage(
        self, config, stage, start_epoch, end_epoch, close_logger, target_train_acc=None
    ):
        self._apply_stage(config, stage)
        run_stats = self.extra_modules.get("run_stats")
        if run_stats is not None:
            run_stats.start_phase(getattr(config, "phase", stage.kind))
        self.run_loop(
            start_epoch, end_epoch, config, target_train_acc=target_train_acc
        )
        step = f"epoch_{self.epoch + 1}"
        self._save_checkpoint(config, step, next_epoch=self.epoch + 1)
        if close_logger:
            self.logger.close()

    def _apply_phase4_optimizer_overrides(self, config):
        weight_decay = getattr(config, "phase4_weight_decay", None)
        if weight_decay is not None:
            weight_decay = float(weight_decay)
            if weight_decay < 0:
                raise ValueError("phase4_weight_decay must be non-negative")
            for group in self.optim.param_groups:
                if group.get("weight_decay_enabled", True):
                    group["weight_decay"] = weight_decay

        lr_lambda = getattr(config, "phase4_lr_lambda", None)
        if lr_lambda is not None:
            if self.lr_scheduler is None or not hasattr(self.lr_scheduler, "lr_lambdas"):
                raise ValueError(
                    "phase4_lr_lambda requires a MultiplicativeLR-compatible scheduler"
                )
            factor = float(lr_lambda)
            if factor <= 0:
                raise ValueError("phase4_lr_lambda must be positive")
            self.lr_scheduler.lr_lambdas = [
                lambda _epoch, factor=factor: factor
                for _ in self.optim.param_groups
            ]

    def _maybe_recalibrate_phase4_batchnorm(self, config):
        batches = int(getattr(config, "phase4_bn_recalibration_batches", 0) or 0)
        if batches == 0:
            return None
        if batches < 0:
            raise ValueError("phase4_bn_recalibration_batches must be non-negative")
        scope = str(
            getattr(config, "phase4_bn_recalibration_scope", "main_branch")
        )
        self._apply_stage(config, PHASE_STAGES[4])
        report = recalibrate_batchnorm(
            self.model,
            self.loaders["train"],
            self.device,
            num_batches=batches,
            scope=scope,
        )
        step = int(getattr(config, "phase3_ends_at_epoch", 0)) * len(
            self.loaders["train"]
        )
        self.logger.log_scalars(report, step)
        logging.info(
            "Recalibrated %d BatchNorm modules on %d batches before phase 4.",
            report["bn_recalibration/modules"],
            report["bn_recalibration/batches"],
        )
        return report

    def run_phase(self, phase, config):
        if phase not in PHASE_STAGES:
            raise ValueError(f"Unsupported phase: {phase}")
        self._initialize_run(config)
        config.phase = phase
        if phase == 4:
            self._apply_phase4_optimizer_overrides(config)
            if config.exp_starts_at_epoch == config.phase3_ends_at_epoch:
                self._maybe_recalibrate_phase4_batchnorm(config)
        self._run_stage(
            config, PHASE_STAGES[phase], config.exp_starts_at_epoch,
            config.exp_ends_at_epoch, close_logger=True,
            target_train_acc=(
                getattr(config, "phase4_target_train_acc", None)
                if phase == 4
                else None
            ),
        )

    def run_phase1(self, config):
        self.run_phase(1, config)

    def run_phase2(self, config):
        self.run_phase(2, config)

    def run_phase3(self, config):
        self.run_phase(3, config)

    def run_phase4(self, config):
        self.run_phase(4, config)

    def run_all_at_once(self, config):
        self._initialize_run(config)
        boundaries = (
            (config.phase1_starts_at_epoch, config.phase1_ends_at_epoch),
            (config.phase1_ends_at_epoch, config.phase2_ends_at_epoch),
            (config.phase2_ends_at_epoch, config.phase3_ends_at_epoch),
            (config.phase3_ends_at_epoch, config.phase4_ends_at_epoch),
        )
        resume_epoch = int(
            getattr(config, "exp_starts_at_epoch", config.phase1_starts_at_epoch)
        )
        for phase, (start_epoch, end_epoch) in enumerate(boundaries, start=1):
            start_epoch = max(start_epoch, resume_epoch)
            if start_epoch >= end_epoch:
                continue
            config.phase = phase
            if phase == 4:
                self._apply_phase4_optimizer_overrides(config)
                if start_epoch == config.phase3_ends_at_epoch:
                    self._maybe_recalibrate_phase4_batchnorm(config)
            self._run_stage(
                config, PHASE_STAGES[phase], start_epoch, end_epoch, close_logger=False,
                target_train_acc=(
                    getattr(config, "phase4_target_train_acc", None)
                    if phase == 4
                    else None
                ),
            )
        self.logger.close()

    def run_loop(
        self, exp_starts_at_epoch, exp_ends_at_epoch, config, target_train_acc=None
    ):
        """
        Main method of trainer.
        Set seed, run train-val in the loop.
        Args:
            config (dict): Consists of:
                exp_starts_at_epoch (int): A number representing the beginning of run
                exp_ends_at_epoch (int): A number representing the end of run
                grad_accum_steps (int):
                step_multi (int):
                base_path (str): Base path
                exp_name (str): Base name of experiment
                logger_name (str): Logger type
                random_seed (int): Seed generator
        """
        if target_train_acc is not None and not 0.0 <= target_train_acc <= 1.0:
            raise ValueError("target_train_acc must be in [0, 1]")
        logging.info('Training started.')
        for epoch in trange(exp_starts_at_epoch, exp_ends_at_epoch, desc='run_exp', leave=True, position=0,
                            colour='green', disable=config.whether_disable_tqdm):
            self.epoch = epoch
            if epoch % 20 == 0:# or (epoch % 1 == 0 and epoch < 80 and epoch > 60):  # there is a problem when till this epoch > 80, to powinno być zapisywane według relatywnego numerowania
                step = f'epoch_{epoch}'
                self._save_checkpoint(config, step, next_epoch=epoch)
                
            self.model.train()
            self.criterion.train()
            train_metrics = self.run_epoch(phase='train', config=config)
            self.model.eval()
            self.criterion.eval()
            with torch.no_grad():
                self.run_epoch(phase='test_proper', config=config)
                self.run_epoch(phase='test_blurred', config=config)
            if (
                target_train_acc is not None
                and train_metrics["epoch_acc/train"] >= target_train_acc
            ):
                logging.info(
                    "Target train accuracy %.6f reached at epoch %d (%.6f).",
                    target_train_acc, epoch, train_metrics["epoch_acc/train"],
                )
                break
                
        logging.info('Training completed.')
        
        
    def run_pretraining(self, mode, config):
        if mode not in PRETRAIN_STAGES:
            raise ValueError(f"Unsupported pretraining mode: {mode}")
        self._initialize_run(config)
        self._run_stage(
            config, PRETRAIN_STAGES[mode], config.exp_starts_at_epoch,
            config.exp_ends_at_epoch, close_logger=True,
        )

    def run_left_modality_pretraining_proper(self, config):
        self.run_pretraining("left_proper", config)

    def run_right_modality_pretraining_proper(self, config):
        self.run_pretraining("right_proper", config)

    def run_right_modality_pretraining_blurred(self, config):
        self.run_pretraining("right_blurred", config)

    def at_exp_start(self, config):
        """
        Initialization of experiment.
        Creates fullname, dirs and logger.
        """
        self.base_path, self.save_path = create_paths(config.base_path, config.exp_name)
        manifest_context = getattr(config, "run_manifest_context", None)
        if manifest_context is not None:
            if OmegaConf.is_config(manifest_context):
                manifest_context = OmegaConf.to_container(manifest_context, resolve=True)
            self.run_manifest = RunManifest.create(
                self.base_path,
                config=manifest_context["config"],
                repo_root=manifest_context["repo_root"],
                dataset_path=manifest_context["dataset_path"],
                input_checkpoint=manifest_context.get("input_checkpoint"),
            )
        config.logger_config['log_dir'] = f'{self.base_path}/{config.logger_config["logger_name"]}'
        self.logger = create_logger(config.logger_config['logger_name'], config)
        
        self.logger.log_model(self.model, self.criterion, log=None)
        
        if 'run_stats' in self.extra_modules:
            self.extra_modules['run_stats'].logger = self.logger
        if 'stiffness_train' in self.extra_modules:
            self.extra_modules['stiffness_train'].logger = self.logger
        if 'stiffness_test' in self.extra_modules:
            self.extra_modules['stiffness_test'].logger = self.logger
            
        if 'dead_relu_left' in self.extra_modules:
            self.extra_modules['dead_relu_left'].logger = self.logger
        if 'dead_relu_right' in self.extra_modules:
            self.extra_modules['dead_relu_right'].logger = self.logger
            
        if 'trace_fim_train' in self.extra_modules:
            self.extra_modules['trace_fim_train'].logger = self.logger
            self.extra_modules['trace_fim_train'].artifact_path = os.path.join(
                self.base_path, "trace_fim_train.jsonl"
            )
        if 'trace_fim_test' in self.extra_modules:
            self.extra_modules['trace_fim_test'].logger = self.logger
            
        if 'rank_left_train' in self.extra_modules:
            self.extra_modules['rank_left_train'].logger = self.logger
        if 'rank_right_train' in self.extra_modules:
            self.extra_modules['rank_right_train'].logger = self.logger
        if 'rank_left_test' in self.extra_modules:
            self.extra_modules['rank_left_test'].logger = self.logger
        if 'rank_right_test' in self.extra_modules:
            self.extra_modules['rank_right_test'].logger = self.logger
            
            
            
            
    def _initialize_run(self, config):
        self.manual_seed(config)
        resume_rng_state = getattr(self, "resume_rng_state", None)
        self.at_exp_start(config)
        if resume_rng_state is not None:
            restore_rng_state(resume_rng_state)
            self.resume_rng_state = None

    def _diagnostics_state(self):
        state = {}
        for name, module in self.extra_modules.items():
            if module is not None and hasattr(module, "diagnostic_state_dict"):
                state[name] = module.diagnostic_state_dict()
        return state or None

    def _save_checkpoint(self, config, step, *, next_epoch):
        checkpoint_path = self.save_path(step)
        train_loader_size = len(self.loaders.get("train", ()))
        protocol_manifest = getattr(config, "protocol_manifest", None)
        if OmegaConf.is_config(protocol_manifest):
            protocol_manifest = OmegaConf.to_container(
                protocol_manifest, resolve=True
            )
        save_training_checkpoint(
            self.model,
            self.optim,
            self.lr_scheduler,
            checkpoint_path,
            next_epoch=next_epoch,
            global_step=next_epoch * train_loader_size,
            metadata={
                "kind": getattr(config, "kind", None),
                "phase": getattr(config, "phase", None),
                "protocol_manifest": protocol_manifest,
            },
            diagnostics_state=self._diagnostics_state(),
        )
        if self.run_manifest is not None:
            self.run_manifest.add_artifact(checkpoint_path, "checkpoint")

    def finalize_run_manifest(self, status, error=None):
        if self.run_manifest is not None:
            self.run_manifest.finalize(status, error=error)

    def compute_loss(self, x_left, x_right, targets, config):
        weak_weight, dominant_weight = phase4_auxiliary_loss_weights(config)
        auxiliary_active = bool(
            int(getattr(config, "phase", 0) or 0) == 4
            and self.model.training
            and torch.is_grad_enabled()
            and (weak_weight > 0.0 or dominant_weight > 0.0)
        )
        model_output = self.model(
            x_left,
            x_right,
            left_branch_intervention=config.extra["left_branch_intervention"],
            right_branch_intervention=config.extra["right_branch_intervention"],
            enable_left_branch=config.extra["enable_left_branch"],
            enable_right_branch=config.extra["enable_right_branch"],
            return_features=auxiliary_active,
        )
        if auxiliary_active:
            predictions, features_left, features_right = model_output
            if not hasattr(self.model, "classify_encoded_modalities"):
                raise TypeError(
                    "phase4_auxiliary_loss requires a model with "
                    "classify_encoded_modalities()"
                )
        else:
            predictions = model_output
        full_loss, evaluators = self.criterion(predictions, targets)
        if not auxiliary_active:
            return full_loss, evaluators

        total_loss = full_loss
        evaluators["phase4_loss/full"] = full_loss.item()
        evaluators["phase4_acc/full"] = evaluators["acc"]

        if weak_weight > 0.0:
            weak_predictions = self.model.classify_encoded_modalities(
                torch.zeros_like(features_left), features_right
            )
            weak_loss, weak_evaluators = self.criterion(
                weak_predictions, targets
            )
            total_loss = total_loss + weak_weight * weak_loss
            evaluators["phase4_loss/weak_only"] = weak_loss.item()
            evaluators["phase4_loss/weak_weighted"] = (
                weak_weight * weak_loss.item()
            )
            evaluators["phase4_acc/weak_only"] = weak_evaluators["acc"]

        # A zero dominant weight intentionally avoids this third shared-trunk
        # pass. It is kept configurable for a later paired ablation only.
        if dominant_weight > 0.0:
            dominant_predictions = self.model.classify_encoded_modalities(
                features_left, torch.zeros_like(features_right)
            )
            dominant_loss, dominant_evaluators = self.criterion(
                dominant_predictions, targets
            )
            total_loss = total_loss + dominant_weight * dominant_loss
            evaluators["phase4_loss/dominant_only"] = dominant_loss.item()
            evaluators["phase4_loss/dominant_weighted"] = (
                dominant_weight * dominant_loss.item()
            )
            evaluators["phase4_acc/dominant_only"] = dominant_evaluators["acc"]

        evaluators["phase4_loss/weak_weight"] = weak_weight
        evaluators["phase4_loss/dominant_weight"] = dominant_weight
        evaluators["phase4_loss/total"] = total_loss.item()
        evaluators["loss"] = total_loss.item()
        return total_loss, evaluators

    def run_epoch(self, phase, config):
        """
        Run single epoch
        Args:
            phase (str): phase of the trening
            config (dict):
        """
        logging.info(f'Epoch: {self.epoch}, Phase: {phase}.')
        
        running_assets = {
            'evaluators': defaultdict(float),
            'denom': 0.0,
        }
        epoch_assets = {
            'evaluators': defaultdict(float),
            'denom': 0.0,
        }
        loader_size = len(self.loaders[phase])
        configured_limit = int(
            getattr(
                config,
                "max_train_batches" if phase == "train" else "max_eval_batches",
                0,
            )
            or 0
        )
        if configured_limit < 0:
            raise ValueError("batch limits must be non-negative")
        if configured_limit:
            loader_size = min(loader_size, configured_limit)
        fim_interval = int(getattr(config, "fim_eval_interval_epochs", 1))
        if fim_interval < 1:
            raise ValueError("fim_eval_interval_epochs must be positive")
        fim_epoch = int(
            getattr(config, "active_phase_epoch", self.epoch + 1)
        )
        fim_batch_indices = (
            _measurement_batch_indices(
                loader_size,
                getattr(config, "fim_measurements_per_epoch", 2),
            )
            if fim_measurement_due(
                fim_epoch,
                fim_interval,
                getattr(config, "fim_eval_epochs", None),
            )
            else frozenset()
        )
        progress_bar = tqdm(self.loaders[phase], desc=f'run_epoch: {phase}',
                            leave=False, position=1, total=loader_size, colour='red', disable=config.whether_disable_tqdm)
        if phase == "train" and self.resume_global_step is not None:
            self.global_step = self.resume_global_step
            self.resume_global_step = None
        else:
            self.global_step = self.epoch * loader_size
        
        if self.epoch < 20:
            config.stiffness_multi = loader_size * 5
            config.rank_multi = loader_size * 5
        elif self.epoch < 40:
            config.stiffness_multi = loader_size * 10
            config.rank_multi = loader_size * 10
        else:
            config.stiffness_multi = loader_size * 20
            config.rank_multi = loader_size * 20
        
        
        # ════════════════════════ training / inference ════════════════════════ #
        
        
        for i, data in enumerate(progress_bar):
            if i >= loader_size:
                break
            (x_true1, x_true2), y_true = data
            x_true1, x_true2, y_true = x_true1.to(self.device), x_true2.to(self.device), y_true.to(self.device)
            if self.extra_modules['dead_relu_left']:
                self.extra_modules['dead_relu_left'].enable()
                self.extra_modules['dead_relu_right'].enable()
            loss, evaluators = self.compute_loss(x_true1, x_true2, y_true, config)
            if self.extra_modules["dead_relu_left"]:
                self.extra_modules["dead_relu_left"].disable()
                self.extra_modules["dead_relu_right"].disable()
            step_assets = {
                'evaluators': evaluators,
                'denom': y_true.size(0),
            }
            if 'train' == phase:
                loss.backward()
                if config.clip_value > 0:
                    norm = clip_grad_norm(torch.nn.utils.clip_grad_norm_, self.model, config.clip_value)
                    step_assets['evaluators']['run_stats/model_gradient_norm_squared_from_pytorch'] = norm.item() ** 2
                    
                run_stats = self.extra_modules['run_stats']
                step_warmup = (
                    self.lr_scheduler
                    if self.lr_scheduler is not None
                    and hasattr(self.lr_scheduler, 'step_batch')
                    else None
                )
                warmup_active = bool(
                    step_warmup is not None and step_warmup.in_warmup
                )
                warmup_lr_used = self.optim.param_groups[0]['lr']
                warmup_progress = (
                    step_warmup.warmup_progress if warmup_active else None
                )
                warmup_factor = (
                    step_warmup.warmup_factor if warmup_active else None
                )
                self.optim.step()
                if run_stats is not None:
                    run_stats.record_optimizer_step()
                    if (
                        config.run_stats_multi
                        and self.global_step % config.run_stats_multi == 0
                    ):
                        run_stats('l2', self.global_step)
                
                if step_warmup is not None:
                    step_warmup.step_batch()
                    if warmup_active:
                        warmup_prefix = getattr(
                            step_warmup, 'metric_prefix', 'phase3'
                        )
                        self.logger.log_scalars(
                            {
                                f'{warmup_prefix}/lr_used': warmup_lr_used,
                                f'{warmup_prefix}/lr_warmup_factor': warmup_factor,
                                f'{warmup_prefix}/lr_warmup_progress': warmup_progress,
                                f'{warmup_prefix}/lr_warmup_step': (
                                    step_warmup.completed_steps
                                ),
                            },
                            self.global_step,
                        )

                epoch_scheduler_step = (
                    (self.global_step + 1) % loader_size == 0
                )
                non_multiplicative_step = (
                    config.logger_config['hyperparameters']['type_names'][
                        'scheduler'
                    ] != 'multiplicative'
                )
                if (
                    self.lr_scheduler is not None
                    and step_warmup is None
                    and (epoch_scheduler_step or non_multiplicative_step)
                ):
                    self.lr_scheduler.step()
                    self.logger.log_scalars(
                        {
                            'lr/training': self.optim.param_groups[0]['lr'],
                            'steps/lr': self.global_step,
                        },
                        self.global_step,
                    )
                    
                self.optim.zero_grad(set_to_none=True)
                
                
                    
                if self.extra_modules['trace_fim_train'] is not None and i in fim_batch_indices:
                    self.extra_modules['trace_fim_train'](self.global_step, config, kind=config.kind)
                    
                if self.extra_modules['trace_fim_test'] is not None and i in fim_batch_indices:
                    self.extra_modules['trace_fim_test'](self.global_step, config, kind=config.kind)
                    
                if self.extra_modules['stiffness_train'] is not None and config.stiffness_multi and self.global_step % config.stiffness_multi == 0:
                    self.extra_modules['stiffness_train'](self.global_step, config, scope='periodic', phase='train', kind=config.kind)
                    
                if self.extra_modules['stiffness_test'] is not None and config.stiffness_multi and self.global_step % config.stiffness_multi == 0:
                    self.extra_modules['stiffness_test'](self.global_step, config, scope='periodic', phase='test', kind=config.kind)
                    
                if self.extra_modules['rank_left_train'] is not None and config.rank_multi and self.global_step % config.rank_multi == 0:
                    self.extra_modules['rank_left_train'].enable()
                    self.extra_modules['rank_left_train'].analysis(self.global_step, scope='periodic', phase='train', kind=config.kind)
                    self.extra_modules['rank_left_train'].disable()
                    
                if self.extra_modules['rank_right_train'] is not None and config.rank_multi and self.global_step % config.rank_multi == 0:
                    self.extra_modules['rank_right_train'].enable()
                    self.extra_modules['rank_right_train'].analysis(self.global_step, scope='periodic', phase='train', kind=config.kind)
                    self.extra_modules['rank_right_train'].disable()
                    
                if self.extra_modules['rank_left_test'] is not None and config.rank_multi and self.global_step % config.rank_multi == 0:
                    self.extra_modules['rank_left_test'].enable()
                    self.extra_modules['rank_left_test'].analysis(self.global_step, scope='periodic', phase='test', kind=config.kind)
                    self.extra_modules['rank_left_test'].disable()
                    
                if self.extra_modules['rank_right_test'] is not None and config.rank_multi and self.global_step % config.rank_multi == 0:
                    self.extra_modules['rank_right_test'].enable()
                    self.extra_modules['rank_right_test'].analysis(self.global_step, scope='periodic', phase='test', kind=config.kind)
                    self.extra_modules['rank_right_test'].disable()
            
            
            # ════════════════════════ logging ════════════════════════ #
            
            
            running_assets = self.update_assets(running_assets, step_assets, step_assets['denom'], 'running', phase)

            whether_log = (i + 1) % config.log_multi == 0
            whether_epoch_end = (i + 1) == loader_size

            if whether_log or whether_epoch_end:
                epoch_assets = self.update_assets(epoch_assets, running_assets, 1.0, 'epoch', phase)

            if whether_log:
                self.log(running_assets, phase, 'running', progress_bar, self.global_step)
                running_assets['evaluators'] = defaultdict(float)
                running_assets['denom'] = 0.0

            if whether_epoch_end:
                self.log(
                    epoch_assets, phase, 'epoch', progress_bar,
                    self.global_step, series_step=self.epoch,
                )

            self.global_step += 1
            
        if self.extra_modules['dead_relu_left']:
            self.extra_modules['dead_relu_left'].at_the_epoch_end(phase, epoch_assets['denom'], self.global_step)
            self.extra_modules['dead_relu_right'].at_the_epoch_end(phase, epoch_assets['denom'], self.global_step)
        return {
            key: value / epoch_assets["denom"]
            for key, value in epoch_assets["evaluators"].items()
        }
            


    def log(
        self, assets: Dict, phase: str, scope: str, progress_bar: tqdm,
        step: int, series_step: int | None = None,
    ):
        '''
        Send chosen assets to logger and progress bar
        Args:
            assets (Dict):
            phase:
            scope:
            progress_bar:
        '''
        # Persist full-precision values. Rounding is only a terminal-display
        # concern; W&B can smooth the stored raw series in its UI.
        evaluators_log = adjust_evaluators_pre_log(
            assets['evaluators'], assets['denom']
        )
        evaluators_log[f'steps/{phase}_{scope}'] = (
            step if series_step is None else series_step
        )
        self.logger.log_scalars(evaluators_log, step)
        progress_bar.set_postfix(
            {
                key: round(value, 4) if isinstance(value, float) else value
                for key, value in evaluators_log.items()
            }
        )

        if self.lr_scheduler is not None and phase == 'train' and scope == 'running':
            self.logger.log_scalars({f'lr_scheduler': self.lr_scheduler.get_last_lr()[0]}, step)


    def update_assets(self, assets_target: Dict, assets_source: Dict, multiplier, scope, phase: str):
        '''
        Update epoch assets
        Args:
            assets_target (Dict): Assets to which assets should be transferred
            assets_source (Dict): Assets from which assets should be transferred
            multiplier (int): Number to get rid of the average
            scope (str): Either running or epoch
            phase (str): Phase of the traning
        '''
        assets_target['evaluators'] = adjust_evaluators(assets_target['evaluators'], assets_source['evaluators'],
                                                        multiplier, scope, phase)
        assets_target['denom'] += assets_source['denom']
        return assets_target


    def manual_seed(self, config: defaultdict):
        """Seed a new run through the shared RNG implementation."""
        seed_everything(config.random_seed, self.device)
