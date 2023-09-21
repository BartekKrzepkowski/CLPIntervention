#!/bin/bash
##DGX
#SBATCH --job-name=critical_period_step
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=5G
#SBATCH --partition=batch
#SBATCH --nodelist=login01
#SBATCH --time=2-0
#SBATCH --output=slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate $HOME/miniconda3/envs/clpi_env
for name in 0 40 80 120 160 200; do
    CUDA_VISIBLE_DEVICES=0 python -u run_exp_clp_mm_intervention_deactivation_window_${name}.py &
done
wait