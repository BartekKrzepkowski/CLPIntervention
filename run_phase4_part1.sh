#!/bin/bash
##ATHENA
#SBATCH --job-name=ncollapse
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate clpi_env
source src/configs/env_variables.sh

PHASE1=40
PHASE2=0
PHASE3=120
# for PHASE3 in 0 20 40 60
# do
CHECKPOINT="/net/pr2/projects/plgrid/plgg_ccbench/bartek/reports2/phase3, intervention deactivation, trained with phase1=40 and phase2=0, mm_tinyimagenet, mm_resnet, sgd, epochs=200_overlap=0.0_lr=0.5_wd=0.0_lambda=1.0/2024-02-04_07-38-41/checkpoints/model_step_epoch_${PHASE3}.pth"
echo $CHECKPOINT
WANDB__SERVICE_WAIT=300 python -m scripts.python_new.run_phase4 model_name=mm_resnet dataset_name=mm_tinyimagenet lr=5e-1 wd=0.0 phase1=$PHASE1 phase2=$PHASE2 phase3=$PHASE3 phase4=500 model_checkpoint="${CHECKPOINT}"
# CUDA_VISIBLE_DEVICES=0
# done

# wait