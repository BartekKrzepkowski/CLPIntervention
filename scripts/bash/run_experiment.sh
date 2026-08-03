#!/bin/bash
#SBATCH --job-name=clpi-experiment
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null

set -euo pipefail

if [[ "${CLP_STORAGE_LOG_CONFIGURED:-0}" != "1" ]]; then
  echo "Submit through scripts/bash/submit_experiment.sh so the Slurm log is stored under REPORTS_DIR." >&2
  exit 2
fi

if [[ $# -lt 1 ]]; then
  echo "Use scripts/bash/submit_experiment.sh PYTHON_MODULE [ARG ...]" >&2
  exit 2
fi

CLP_GPU_PREFIX="${CLP_GPU_PREFIX:-/net/storage/pr3/plgrid/plggdnnp/conda_envs/clpintervention-gh200}"
if [[ "$(uname -m)" != "aarch64" || ! -x "${CLP_GPU_PREFIX}/bin/python" ]]; then
  echo "This launcher requires the prepared CLPIntervention GH200 environment." >&2
  echo "Create it first with: sbatch scripts/bash/run_gpu_tests.sh" >&2
  exit 1
fi

source src/configs/env_variables.sh
export LD_LIBRARY_PATH="${CLP_GPU_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

module="$1"
shift
exec "${CLP_GPU_PREFIX}/bin/python" -m "${module}" "$@"
