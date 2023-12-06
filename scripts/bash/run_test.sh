#!/bin/bash
##GMUM
#SBATCH --job-name=test
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate clpi_env
# WANDB__SERVICE_WAIT=300 python -m scripts.python_new.$1 10 1e-1 0.0 80 200 50
WANDB__SERVICE_WAIT=300 python -m scripts.python_new.$1