#!/bin/bash
##GMUM
#SBATCH --job-name=eff
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3G
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate clpi_env
WANDB__SERVICE_WAIT=300 python -m scripts.python_new.$1 6e-1 0.0 1 1 1 1