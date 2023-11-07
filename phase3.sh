#!/bin/bash
##ENTROPY
#SBATCH --job-name=clpi
#SBATCH --partition=common
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=3G
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate clpi_env
nvidia-smi
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python.$1 6e-1 0.0 120 &
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python.$1 6e-1 0.0 160 &
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python.$1 6e-1 0.0 200 &
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python.$1 6e-1 0.0 240 &
wait

