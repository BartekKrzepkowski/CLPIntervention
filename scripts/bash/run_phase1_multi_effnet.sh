#!/bin/bash
##GMUM
#SBATCH --job-name=eff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=rtx3080
#SBATCH --qos=normal
#SBATCH --time=13:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate clpi_env
source src/configs/env_variables.sh

# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python_new.$1 800 3e-0 1e-4 0.2 0.2 &
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python_new.$1 800 1e-0 1e-4 0.2 0.2 &
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python_new.$1 800 7e-1 1e-4 0.2 0.2 &
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python_new.$1 800 1.5e-0 1e-4 0.2 0.2 &
wait