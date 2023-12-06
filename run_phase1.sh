#!/bin/bash
##GMUM
#SBATCH --job-name=eff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=rtx3080
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate clpi_env
source src/configs/env_variables.sh
WANDB__SERVICE_WAIT=300 python -m scripts.python_new.$1 model_name=mm_effnetv2s dataset_name=mm_tinyimagenet phase1=800 lr=1e-0 wd=0.0