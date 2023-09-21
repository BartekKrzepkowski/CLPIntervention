#!/bin/bash
##DGX
#SBATCH --job-name=critical_period_step
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate $HOME/anaconda3/envs/fp2

WANDB__SERVICE_WAIT=300 python3 -u run_exp.py