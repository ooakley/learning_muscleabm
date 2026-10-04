#!/bin/bash
#SBATCH --job-name=gp_noise
#SBATCH --partition=ncpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-11
#SBATCH --output=logs/%x_%A_%a.out

# Trains a GP of the log standard error of each model metric (one array task per metric)
# on the sample matrix and summary data of an experiment folder. Usage, from the
# repository root:
#
#     sbatch bash_scripts/gp_noise_regression.sh <experiment_dirpath>
set -eo pipefail

# --- Arguments ---
usage="Usage: sbatch bash_scripts/gp_noise_regression.sh <experiment_dirpath>"
experiment_dirpath=${1:?$usage}

ml load uv
# Lmod is not guaranteed to work with unset variables treated as errors, so only from here:
set -u

metrics=(
    # Movement metrics:
    speeds
    meander_ratios
    ann_indices
    coherency
    interaction
    order_parameters
    # Centre of mass metrics:
    com_speeds
    com_meander_ratios
    com_ann_indices
    com_coherency
    com_interaction
    com_order_parameters
)
metric=${metrics[$SLURM_ARRAY_TASK_ID]}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

uv run --no-sync python python_scripts/emulation/gp_noise_training.py \
    --experiment_dirpath "$experiment_dirpath" \
    --metric_name "$metric"
