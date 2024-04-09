#!/bin/bash
##ATHENA
#SBATCH --job-name=ncollapse
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate clpi_env

WANDB__SERVICE_WAIT=300 python -m $1