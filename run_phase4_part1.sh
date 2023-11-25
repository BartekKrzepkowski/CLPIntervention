#!/bin/bash
##GMUM
#SBATCH --job-name=eff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3G
#SBATCH --partition=rtx3080
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate clpi_env


PHASE1=0
PHASE2=0
for PHASE3 in 0 20 40 60
do
    CHECKPOINT="/model_step_epoch_${PHASE3}.pth"
    echo $CHECKPOINT
    WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python_new.$1 1e-1 0.0 $PHASE1 $PHASE2 $PHASE3 "${CHECKPOINT}" &
done

wait