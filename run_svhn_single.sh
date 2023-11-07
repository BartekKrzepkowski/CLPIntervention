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
WANDB__SERVICE_WAIT=300 python -m scripts.python_new.$1 5e-2 0.0 3 &
wait