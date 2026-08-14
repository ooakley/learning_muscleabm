#!/bin/bash
#SBATCH --job-name=hessian
#SBATCH --partition=ncpu
#SBATCH --time=6:00:00
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G

ml load uv parallel
world_size=512

estimate_hessian () {
    local task_id=$1
    echo "Running task $task_id..."
    srun --exact --ntasks 1 --nnodes 1 --cpus-per-task 1 \
        uv run python_scripts/generate_hessians.py \
        --metric op65 \
        --world_size $world_size \
        --task_id $task_id
}

export -f estimate_hessian
export world_size=$world_size

task_list=$(seq 0 $(( $world_size-1 )))

# Run with GNU parallel:
echo "Running hessian estimation in parallel..."
parallel -u -j $world_size estimate_hessian ::: $task_list
