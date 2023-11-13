#!/bin/bash
##ENTROPY
#SBATCH --job-name=clpi
#SBATCH --partition=common
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate clpi_env


PHASE1=60
PHASE2=0
for PHASE3 in 80 120 160 200
do
    # Uruchomienie procesu treningu w tle z użyciem zmiennej PHASE3
    CHECKPOINT="/home/barkrz/reports/just_run, sgd, dual_fmnist, mm_simple_cnn_fp_0.0_lr_0.1_wd_0.0_N_3 overlap=0.0, phase3, intervention deactivation, trained with phase1=60 and phase2=0/2023-11-10_15-27-20/checkpoints/model_step_epoch_${PHASE3}.pth"
    echo $CHECKPOINT
    WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python_new.$1 1e-1 0.0 $PHASE1 $PHASE2 $PHASE3 "${CHECKPOINT}" &
done

# Czekanie na zakończenie wszystkich procesów uruchomionych w tle
wait