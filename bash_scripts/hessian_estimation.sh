#!/bin/bash
#SBATCH --job-name=hessian
#SBATCH --partition=ncpu
#SBATCH --time=6:00:00
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --hint=nomultithread

# Script inputs:
usage="Usage: sbatch hessian_estimation.sh <experiment_dirpath>"
experiment_dirpath=${1:?$usage}

ml load uv parallel
uv sync

world_size=512

estimate_hessian () {
    local task_id=$1
    echo "Running task $task_id..."

    # Single-value Hessian estimation:
    # srun --exact --ntasks 1 --cpus-per-task 1 --nodes=1-1 --mem-per-cpu=8G --hint=nomultithread \
    #     uv run --frozen python_scripts/sensitivity/generate_hessians.py \
    #     --experiment_dirpath "$experiment_dirpath" \
    #     --metric op65 \
    #     --world_size $world_size \
    #     --task_id $task_id

    # Design point Hessian estimation:
    srun --exact --ntasks 1 --cpus-per-task 1 --nodes=1-1 --mem-per-cpu=8G --hint=nomultithread \
        uv run --frozen python_scripts/sensitivity/generate_full_rank_hessians.py \
        --experiment_dirpath "$experiment_dirpath" \
        --metric op65 \
        --world_size $world_size \
        --task_id $task_id

    # Signal finished:
    echo "Finished task $task_id..."

}

export -f estimate_hessian
export world_size=$world_size
export experiment_dirpath

task_list=$(seq 0 $(( $world_size-1 )))

# Run with GNU parallel:
echo "Running hessian estimation in parallel..."
parallel -u -j $world_size estimate_hessian ::: $task_list
