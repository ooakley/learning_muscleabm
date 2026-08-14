#!/bin/bash
#SBATCH --job-name=gp_noise
#SBATCH --partition=ncpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=14
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G

ml load uv
experiment_dirpath="model_experiments/2026-05-31-collisions_shape"

# Movement metrics:
srun --ntasks 1 uv run python python_scripts/gp_noise_training.py \
    --experiment_dirpath $experiment_dirpath \
    --metric_name speeds &
sleep 60

srun --ntasks 1 uv run python python_scripts/gp_noise_training.py \
    --experiment_dirpath $experiment_dirpath \
    --metric_name meander_ratios &
srun --ntasks 1 uv run python python_scripts/gp_noise_training.py \
    --experiment_dirpath $experiment_dirpath \
    --metric_name ann_indices &
srun --ntasks 1 uv run python python_scripts/gp_noise_training.py \
    --experiment_dirpath $experiment_dirpath \
    --metric_name coherency &
srun --ntasks 1 uv run python python_scripts/gp_noise_training.py \
    --experiment_dirpath $experiment_dirpath \
    --metric_name interaction &
srun --ntasks 1 uv run python python_scripts/gp_noise_training.py \
    --experiment_dirpath $experiment_dirpath \
    --metric_name order_parameters &

srun --ntasks 1 uv run python python_scripts/gp_noise_training.py \
    --experiment_dirpath $experiment_dirpath \
    --metric_name com_speeds &
srun --ntasks 1 uv run python python_scripts/gp_noise_training.py \
    --experiment_dirpath $experiment_dirpath \
    --metric_name com_meander_ratios &
srun --ntasks 1 uv run python python_scripts/gp_noise_training.py \
    --experiment_dirpath $experiment_dirpath \
    --metric_name com_ann_indices &
srun --ntasks 1 uv run python python_scripts/gp_noise_training.py \
    --experiment_dirpath $experiment_dirpath \
    --metric_name com_coherency &
srun --ntasks 1 uv run python python_scripts/gp_noise_training.py \
    --experiment_dirpath $experiment_dirpath \
    --metric_name com_interaction &
srun --ntasks 1 uv run python python_scripts/gp_noise_training.py \
    --experiment_dirpath $experiment_dirpath \
    --metric_name com_order_parameters &

wait