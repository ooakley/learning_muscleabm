#!/bin/bash
#SBATCH --job-name=nn_train
#SBATCH --partition=ncpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-3
#SBATCH --output=logs/nn_train_%A_%a.out

# Script inputs:
usage="Usage: sbatch nn_regression.sh <experiment_dirpath>"
experiment_dirpath=${1:?$usage}

ml load uv

metrics=(speeds meander_ratios ann_indices coherency)
# metrics=(speeds)
metric=${metrics[$SLURM_ARRAY_TASK_ID]}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

uv run --no-sync python python_scripts/emulation/nn_cv_training.py \
    --experiment_dirpath "$experiment_dirpath" \
    --metric_name "$metric"