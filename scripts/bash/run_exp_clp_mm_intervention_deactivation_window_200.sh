#!/bin/bash
##DGX
#SBATCH --job-name=critical_period_step
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=batch
#SBATCH --time=2-0
#SBATCH --output=slurm-%j.out

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/clp_env

WANDB__SERVICE_WAIT=300 python3 -u run_exp_clp_mm_intervention_deactivation_window_200.py