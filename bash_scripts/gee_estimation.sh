#!/bin/bash
#SBATCH --job-name=gee_estimation
#SBATCH --partition=ncpu
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=18G
#SBATCH --array=1-128

ml load uv
world_size=128

estimate_gee () {
    local task_id=$1
    echo "Running task $task_id..."
    srun --ntasks 1 \
        uv run python_scripts/global_eigenparameter_estimation.py \
        --world_size $world_size \
        --task_id $task_id
}

estimate_gee $(($SLURM_ARRAY_TASK_ID-1))