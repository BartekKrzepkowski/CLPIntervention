#!/bin/bash
##DGX
#SBATCH --job-name=critical_period_step
#SBATCH --gpus=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=batch
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate $HOME/miniconda3/envs/clpi_env
for name in 0 40 80 120 160 200; do
    WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python3 -u run_exp_clp_mm_intervention_deactivation_window_${name}.py &
done

for name in 40 80 120 160 200; do
    WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python3 -u run_exp_clp_mm_intervention_full_occlusion_window_${name}.py &
done
wait