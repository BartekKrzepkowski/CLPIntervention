#!/bin/bash
##DGX
#SBATCH --job-name=critical_period_step
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=batch
#SBATCH --nodelist=gpu01
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate clpi_env
WANDB__SERVICE_WAIT=300 python3 -m scripts.python.$1 $2 $3