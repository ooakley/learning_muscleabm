#!/bin/bash
#SBATCH --job-name=gee_estimation
#SBATCH --partition=ncpu
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=18G
#SBATCH --array=1-128
#SBATCH --output=logs/sensitivity/%x_%A_%a.out

# Global eigenparameter estimation from the op65 Hessians of an experiment, split over the
# array tasks. Usage, from the repository root:
#
#     sbatch bash_scripts/gee_estimation.sh <experiment_dirpath>
set -eo pipefail

# --- Arguments ---
usage="Usage: sbatch bash_scripts/gee_estimation.sh <experiment_dirpath>"
experiment_dirpath=${1:?$usage}

# --- Settings ---
world_size=128

ml load uv
# Lmod is not guaranteed to work with unset variables treated as errors, so only from here:
set -u

estimate_gee () {
    local task_id=$1
    echo "Running task $task_id..."
    srun --ntasks 1 \
        uv run --no-sync python python_scripts/sensitivity/global_eigenparameter_estimation.py \
        --experiment_dirpath "$experiment_dirpath" \
        --world_size "$world_size" \
        --task_id "$task_id"
}

estimate_gee $(( SLURM_ARRAY_TASK_ID - 1 ))
