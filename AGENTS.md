# AGENTS Instructions for CLPIntervention

## Scope

- Active research code lives in `src/`, `scripts/python_new/`, and `scripts/bash/`.
- Treat `scripts/python_old/` and `scripts/python_backup/` as archival. Do not repair or refactor them unless explicitly requested.
- Notebooks are analysis artifacts. Avoid bulk notebook rewrites and generated-output churn.
- Prefer focused changes that preserve the scientific protocol and checkpoint compatibility.

## Scientific protocol

The bimodal input is a pair of left and right visual fields cropped from one image. The core critical-period protocol is:

1. Phase 1: both branches are enabled; the left field is proper and the right field is blurred, except for optional indices in `subset` that receive the proper right transform.
2. Phase 2: both branches are enabled and both fields are proper.
3. Phase 3: the left branch is disabled with the `deactivation` intervention; the right branch and shared trunk train on proper input.
4. Phase 4: both branches are enabled again on proper input.

Do not change this phase meaning silently. Any alternative intervention, phase order, or transform schedule must be an explicit experiment parameter and must be documented.

The explicit `relative_unimodal_parity` variant is the exception to the
historical Phase-3 trainability rule: only `right_branch` trains, while
`left_branch` and `main_branch` are frozen in `eval()` including BatchNorm
buffers. Phase 4 must unfreeze the complete model before constructing its
fresh optimizer.

Paper-specific reference behavior:

- The historical S-ResNet-18 study uses `mm_resnet` with `backbone_type=resnet18`, `modify_resnet=true`, additive fusion, and the split after `layer2`.
- CIFAR-10 and Fashion-MNIST training apply one shared horizontal flip, then independent translations of each field by at most `1/8`; they do not rotate during training.
- `resize_factor=0.25` must be honored both when the dataset is constructed and when phase 1 reapplies the blurred-right transform.
- Phase 4 may stop at `phase4_target_train_acc`, measured on the training set. Altered recovery uses optional phase-4-only weight decay and LR multiplier overrides.

## Core flow

```text
scripts/python_new/run_single.py or scripts/python_new/run_all_at_once.py
  -> historical profile:
     -> src/utils/prepare.py
     -> src/data/datasets.py
     -> src/trainer/trainer_classification_mm_clp.py
  -> validation-controlled CIFAR-10 profile:
     -> src/utils/prepare_clp_data.py
     -> src/data/cifar10_protocol.py
     -> src/data/cifar10_datasets.py
     -> src/trainer/trainer_validation_clp.py
        -> src/trainer/modality_evaluation.py
        -> src/trainer/validation_control.py
     -> src/modules/architectures/*
     -> src/modules/losses.py
```

UMT flow:

```text
scripts/python_new/run_all_at_once_umt.py
  -> scripts/python_new/run_single.py (umt=True)
  -> src/modules/architectures/wrappers.py
  -> src/trainer/trainer_classification_mm_clp_umt.py
```

The top-level `run_all_at_once.py` is a separate balance-penalty experiment, not an alias for the standard runner.

## Critical invariants

- Model forward methods accept two inputs and the intervention arguments used by the trainers.
- A disabled branch must receive an explicit `deactivation` or `occlusion` intervention.
- Fusion by concatenation is along the channel or feature dimension, never a spatial dimension.
- Synthetic intervention tensors preserve input device and dtype.
- Shape inference during model construction must run without gradients, must not update BatchNorm statistics, and must restore the original module mode.
- Evaluation runs under `torch.no_grad()` with both model and criterion in evaluation mode.
- Temporary diagnostic evaluation must restore the previous model mode.
- Validation-controlled stopping uses only `validation_proper`; `validation_blurred`, weak-only train probe, FIM and test are never stopper inputs.
- Relative-unimodal-parity references must match seed, model, dataset split,
  normalization and canonical paired initialization. The dominant ratio is
  frozen at e3=0; two consecutive parity hits confirm the first checkpoint in
  the streak. Reference accuracy and Phase-3 stopping must never use test.
- Optional Phase-2 test diagnostics must run only post hoc after all phases from retained checkpoints, under `posthoc_test/phase2/*`. Test loaders and metrics must never be passed to phase controllers or selectors, and the final selected checkpoint must be restored afterward.
- PAIS must keep per-example losses aligned across modes, apply the configured repeated-look/metric-family correction, and preserve its trend window across resume.
- Weak-recovery PAIS must also keep per-example correctness aligned, use validation proper only, treat compatibility drift as diagnostic/tie-breaker, and reserve numerical emergency stop for non-finite metrics. Calibration shadow mode records independent trigger epochs after freezing the first hypothetical decision.
- The weak-only train probe is deterministic, unaugmented, evaluated in `eval()` and diagnostic only; never optimize or select checkpoints on it.
- FIM diagnostics use a deterministic class-balanced probe disjoint from training by default, the same model-label samples for both branches, isolated RNG, and no BatchNorm or shared-trunk parameters.
- RSV uses only Kleinman et al.'s convention: `(SV_left - SV_right) / (SV_left + SV_right)`, with `+1=left` and `-1=right`. Artifacts must store both source variances and this convention. Historical opposite-sign values must be negated before comparison; do not add a runtime sign switch.
- RSV measurements must keep the unmodified field as variant zero, use deterministic sampling, run in eval mode without gradients, restore the original model mode, and remove hooks.
- Publication RSV is post hoc on phase checkpoints. Record `stage3_avgpool` (`main_branch.0` plus analysis-only adaptive pooling) and `stage4_avgpool` (native pooling after `main_branch.1`) separately during shared forward passes.
- Paired RSV comparisons must match seed/model names, probe indices, labels, sign convention, layer and pooling. Aggregate units within images before hierarchical resampling of images and models.
- Phase-4 BatchNorm recalibration is an opt-in control and defaults off. It must not update weights, optimizer, scheduler, affine BN parameters, or training RNG, and must not rerun when resuming inside phase 4.
- The primary BatchNorm control scope is `main_branch`; compare it with native BN from the same phase-3 checkpoint.
- UMT teacher branches stay frozen and in evaluation mode. Distillation applies only to enabled student branches.
- Hooks must not leak. A diagnostic must tolerate a branch that produces no activations during a phase.
- Class utilities support torchvision datasets with either `targets` or `labels` and nested dataset wrappers.
- Do not add guessed or copied normalization statistics. Add a new overlap only after computing dataset-specific left, right, and blurred-right statistics.

## Configuration and data

- Model registry entries in `src/utils/common.py::MODEL_NAME_MAP` require a matching `src/configs/<model_name>.json` file.
- Dataset paths come from parameters or environment variables. Never commit machine-specific dataset, report, or checkpoint paths.
- Expected variables include `REPORTS_DIR`, `CIFAR10_PATH`, `FMNIST_PATH`, `SVHN_PATH`, `MNIST_PATH`, `KMNIST_PATH`, and `TINYIMAGENET_PATH`.
- Never commit API keys or tokens. `src/configs/env_variables.sh` must remain secret-free and may only provide defaults from the current environment.
- New checkpoints use the versioned `clpintervention.training` format and preserve model, optimizer, scheduler, next epoch, global step, and RNG state. Model-only legacy `state_dict` files remain readable.
- `model_checkpoint` means a phase transfer and loads weights only by default. `resume_checkpoint` means continuation of an interrupted run and restores the full training state by default. Opt-in phase-state transfer uses `transfer_training_state=true`; do not conflate it with resume.
- Loading a UMT teacher may use a branch-only checkpoint or extract `left_branch.*`, `right_branch.*`, or `main_model.<branch>.*` from a full checkpoint.
- A resumed run must not silently repeat completed phases. Keep model, optimizer family, dataset/subset, batch size, transforms, and phase boundaries compatible with the checkpoint.
- Generated FIM probes are the supported default and log their index hash. Legacy held-out tensors and subset arrays under `data/` are local artifacts and stay untracked.

## Environment

This repository uses two isolated PLGrid environments:

- CPU/login: `/net/storage/pr3/plgrid/plggdnnp/conda_envs/clpintervention-local-cpu`;
- GH200/aarch64: `/net/storage/pr3/plgrid/plggdnnp/conda_envs/clpintervention-gh200`.

Create or verify the local environment with `bash scripts/bash/create_local_env.sh`. Run Python through `scripts/bash/clp_python_local.sh`. Create the GPU environment only on a GH200 compute node with `bash scripts/bash/create_gpu_env.sh`; `scripts/bash/run_gpu_tests.sh` performs this step automatically. `environment.yml` remains a portable dependency specification, not the active cluster prefix.

Do not use the legacy `lapsum-*` environments directly for repository operations; they are clone sources only. Submit experiments through `scripts/bash/submit_experiment.sh`; it delegates to `scripts/bash/run_experiment.sh` while forcing the Slurm stdout/stderr path under `$REPORTS_DIR/slurm_logs` on persistent storage. Do not invoke `sbatch scripts/bash/run_experiment.sh` directly. Do not add dataset/model-specific Slurm copies when CLI overrides suffice. Do not install or upgrade dependencies during routine repository work unless requested. `environment.yml` is the core runtime/test set; optional visualization backends and Captum live in `environment-optional.yml` and must remain lazy imports.

The login node is for lightweight repository inspection, orchestration, static checks, and pure controller/configuration unit tests only. Never run any tensor-batch workload on the login node, including tiny synthetic batch forwards used by unit tests. In particular, do not iterate data loaders, compute dataset normalization or features, preprocess datasets, run model forward/backward passes over batches, or execute training/evaluation loops there. Submit every such operation to a compute node, even when it is CPU-only; GPU or whole-model workloads must use the appropriate GPU compute node.

Before starting a dataset or tensor workload, verify that the process is running inside an allocated compute job. If a login-node command would touch a large dataset or instantiate/iterate its loader, stop and submit it through Slurm instead.

## Testing

For every behavior-changing fix, add or update a targeted regression test. On the login node run only static checks and explicitly selected pure-logic tests that never create tensor batches:

```bash
scripts/bash/clp_python_local.sh -m compileall -q src scripts/python_new tests
bash -n scripts/bash/*.sh
```

Submit every test that creates tensor batches, iterates a loader, or performs a model forward/backward to a compute node through Slurm. Do not use a broad local pytest command whose collected tests have not been audited for tensor work.

Do not launch GPU training, Slurm jobs, or dataset downloads unless the user explicitly requests them. Report GPU and end-to-end dataset runs as skipped when they were not executed.

## Safety and generated artifacts

- `REPORTS_DIR` for every training, evaluation, TFIM/RSV computation and
  compute test must resolve under `/net/storage`. Checkpoints, raw traces,
  per-example predictions, W&B local data and Slurm stdout/stderr belong
  there, never in the repository worktree. The repository may contain only
  compact, path-sanitized analysis tables, manifests and selected figures
  needed to reproduce a scientific conclusion.
- Submit training and compute analysis with
  `scripts/bash/submit_experiment.sh`; the helper rejects a non-storage
  `REPORTS_DIR` and writes Slurm logs to `$REPORTS_DIR/slurm_logs`.
- The tracked storage layout and artifact-location contract are documented in
  `docs/ARTIFACT_STORAGE.md`. Keep it path-portable: tracked files name
  `$REPORTS_DIR`, not a user- or machine-specific resolved directory.
- Do not delete datasets, checkpoints, reports, Slurm logs, notebooks, or local analysis artifacts without explicit approval.
- Do not commit `data/`, `reports/`, `reports2/`, `slurm_logs/`, cache files, generated feature images, or credentials.
- Do not overwrite user experiment outputs.
- Keep launchers explicit about their Python module and checkpoint arguments.

## Documentation and reporting

Update `README.md` when setup, protocol, entrypoints, environment variables, or artifact contracts change. Record audit findings and unresolved scientific limitations in `docs/AUDIT.md`. Append concise dated work notes under `docs/work_logs/` and keep stable terminology in `docs/CONCEPTS.md`.

For non-trivial changes, report:

1. behavior fixed and files changed;
2. commands and key validation results;
3. tests and skipped GPU or end-to-end gates;
4. unresolved limitations that require data, GPU execution, or a scientific decision.
