#!/bin/bash
##ATHENA
#SBATCH --job-name=ncollapse
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate clpi_env
source src/configs/env_variables.sh
WANDB__SERVICE_WAIT=300 python -m scripts.python_new.$1 model_name=mm_resnet dataset_name=mm_tinyimagenet lr=5e-1 wd=0.0 phase1=40 phase2=150 phase3=30 phase4=150