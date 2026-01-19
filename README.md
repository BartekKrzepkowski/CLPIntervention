# CLPIntervention

Research codebase for studying **`plasticity interventions`** in neural networks, with an emphasis on **multi-stage training** and **(multi-)modality setups**.

This repository contains:
- **`end-to-end experiment runners`** (single command pipelines),
- **`phase-by-phase scripts`** (more granular control),
- **`notebooks`** for analysis/visualization (trajectories, weight distributions, held-out samples),
- utilities for **`weight/trajectory visualization`**.

> Note: the code is organized around **phases** (`phase1`–`phase4`) and also provides **dataset-specific** runners (e.g., `fmnist`, `tinyin`).

---

## Project layout (high level)

- **`src/`** — core training / model / data logic
- **`scripts/`** — helper scripts, experiment launchers, job utilities
- **`run_all_at_once.py`** — Python entry point to run the full pipeline
- **`run_all_at_once*.sh`** — one-shot bash pipelines:
  - `run_all_at_once.sh`
  - `run_all_at_once_fmnist.sh`
  - `run_all_at_once_tinyin.sh`
- **`run_phase*.sh`** — phase-by-phase runners:
  - `run_phase1.sh`
  - `run_phase2_part1.sh`, `run_phase2_part2.sh`
  - `run_phase3_part1.sh`, `run_phase3_part2.sh`
  - `run_phase4_part1.sh`, `run_phase4_part2.sh`
- **`run_pretrain_*.sh`** — modality-specific pretraining:
  - `run_pretrain_modality1_proper.sh`
  - `run_pretrain_modality2_proper.sh`
  - `run_pretrain_modality2_blurred.sh`
- **`weights_visualisation.py`** + `run_weight_visualization.sh` — weight visualization utilities
- **`*.ipynb`** — analysis notebooks (trajectories, stable rank, held-out samples, distributions)

---

## Quickstart

### 1) Create environment
Use the provided helper:

```bash
bash create_env.sh
