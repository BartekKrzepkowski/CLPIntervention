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

for PHASE1 in 80 120
do
    CHECKPOINT="/net/pr2/projects/plgrid/plgg_ccbench/bartek/reports2/phase1, mm_tinyimagenet, mm_resnet, sgd, epochs=400_overlap=0.0_lr=0.5_wd=0.0_lambda=1.0/2024-02-03_16-19-13/checkpoints/model_step_epoch_${PHASE1}.pth"
    echo $CHECKPOINT
    WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python_new.run_phase2 model_name=mm_resnet dataset_name=mm_tinyimagenet lr=5e-1 wd=0.0 phase1=$PHASE1 phase2=400 model_checkpoint="${CHECKPOINT}" &
done

wait