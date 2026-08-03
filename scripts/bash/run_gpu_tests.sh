#!/bin/bash
#SBATCH --job-name=clpi-gpu-tests
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --qos=now
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/gpu-tests-%j.out

set -euo pipefail


CLP_GPU_PREFIX="${CLP_GPU_PREFIX:-/net/storage/pr3/plgrid/plggdnnp/conda_envs/clpintervention-gh200}"
bash scripts/bash/create_gpu_env.sh
source src/configs/env_variables.sh
export LD_LIBRARY_PATH="${CLP_GPU_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
"${CLP_GPU_PREFIX}/bin/python" -c "import platform, torch; print(platform.machine()); print(torch.__version__); print(torch.cuda.get_device_name()); assert torch.cuda.is_available()"
if [[ $# -gt 0 ]]; then
  "${CLP_GPU_PREFIX}/bin/python" -m pytest "$@"
else
  "${CLP_GPU_PREFIX}/bin/python" -m pytest -q -m gpu tests/test_gpu_models.py
fi
