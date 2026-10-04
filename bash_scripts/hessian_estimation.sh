#!/bin/bash
#SBATCH --job-name=hessian
#SBATCH --partition=ncpu
#SBATCH --time=6:00:00
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --hint=nomultithread
#SBATCH --output=logs/%x_%j.out

# Estimates op65 Hessians across parameter space for an experiment, split over 512
# processes. Usage, from the repository root:
#
#     sbatch bash_scripts/hessian_estimation.sh <experiment_dirpath>
set -eo pipefail

# --- Arguments ---
usage="Usage: sbatch bash_scripts/hessian_estimation.sh <experiment_dirpath>"
experiment_dirpath=${1:?$usage}

# --- Settings ---
world_size=512

ml load uv parallel
# Lmod is not guaranteed to work with unset variables treated as errors, so only from here:
set -u

estimate_hessian () {
    local task_id=$1
    echo "Running task $task_id..."

    # Single-value Hessian estimation:
    # srun --exact --ntasks 1 --cpus-per-task 1 --nodes=1-1 --mem-per-cpu=8G --hint=nomultithread \
    #     uv run --no-sync python python_scripts/sensitivity/generate_hessians.py \
    #     --experiment_dirpath "$experiment_dirpath" \
    #     --metric op65 \
    #     --world_size "$world_size" \
    #     --task_id "$task_id"

    # Design point Hessian estimation:
    srun --exact --ntasks 1 --cpus-per-task 1 --nodes=1-1 --mem-per-cpu=8G --hint=nomultithread \
        uv run --no-sync python python_scripts/sensitivity/generate_full_rank_hessians.py \
        --experiment_dirpath "$experiment_dirpath" \
        --metric op65 \
        --world_size "$world_size" \
        --task_id "$task_id"

    # Signal finished:
    echo "Finished task $task_id..."
}

# Export the function and its settings so we can use them with GNU parallel:
export -f estimate_hessian
export world_size experiment_dirpath

# Run with GNU parallel:
echo "Running hessian estimation in parallel..."
parallel -u -j "$world_size" estimate_hessian ::: $(seq 0 $(( world_size - 1 )))
