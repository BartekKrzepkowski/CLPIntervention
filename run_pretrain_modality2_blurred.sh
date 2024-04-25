#!/bin/bash
##ATHENA
#SBATCH --job-name=eff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate clpi_env
source src/configs/env_variables.sh
WANDB__SERVICE_WAIT=300 python -m scripts.python_new.run_pretrain_modality2_proper model_name=mm_resnet dataset_name=mm_cifar10 lr=5e-1 wd=0.0 epochs=300