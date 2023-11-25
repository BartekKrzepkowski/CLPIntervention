#!/bin/bash
##GMUM
#SBATCH --job-name=eff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3G
#SBATCH --partition=rtx2080
#SBATCH --qos=quick
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate clpi_env
WANDB__SERVICE_WAIT=300 python -m scripts.python_new.$1 200 5e-2 0.0