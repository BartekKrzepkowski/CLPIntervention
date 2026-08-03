#!/bin/bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/bash/submit_experiment.sh PYTHON_MODULE [ARG ...]" >&2
  exit 2
fi

source src/configs/env_variables.sh
: "${REPORTS_DIR:?Set REPORTS_DIR to a persistent storage directory}"

case "${REPORTS_DIR}" in
  /net/storage/*) ;;
  *)
    echo "REPORTS_DIR must be located under /net/storage for Slurm runs." >&2
    exit 2
    ;;
esac

slurm_log_dir="${REPORTS_DIR%/}/slurm_logs"
mkdir -p "${slurm_log_dir}"

sbatch_args=(
  --output="${slurm_log_dir}/%x-%j.out"
  --export=ALL,CLP_STORAGE_LOG_CONFIGURED=1
)
if [[ -n "${CLP_JOB_NAME:-}" ]]; then
  sbatch_args+=(--job-name="${CLP_JOB_NAME}")
fi
if [[ -n "${CLP_DEPENDENCY:-}" ]]; then
  sbatch_args+=(--dependency="${CLP_DEPENDENCY}")
fi

exec sbatch "${sbatch_args[@]}" scripts/bash/run_experiment.sh "$@"
