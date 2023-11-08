#!/bin/bash
##GMUM
#SBATCH --job-name=eff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=3G
#SBATCH --partition=rtx3080
#SBATCH --qos=normal
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate clpi_env
nvidia-smi
# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python_new.$1 7e-3 0.0 1 &
# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python_new.$1 5e-3 0.0 1 &
# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python_new.$1 3e-3 0.0 1 &
# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python_new.$1 1e-3 0.0 1 &
# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python_new.$1 7e-3 0.0 2 &
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python_new.$1 5e-1 0.0 2 &
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python_new.$1 3e-1 0.0 2 &
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python_new.$1 1e-1 0.0 2 &
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python_new.$1 5e-2 0.0 2 &
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python_new.$1 3e-2 0.0 2 &
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python_new.$1 1e-2 0.0 2 &
# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python_new.$1 3e-4 0.0 3 &
# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python_new.$1 1e-4 0.0 3 &
wait