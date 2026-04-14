#!/bin/bash
#SBATCH --job-name=gp_train
#SBATCH --partition=ncpu
#SBATCH --time=4:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G

ml load uv

# Movement metrics:
# uv run python python_scripts/gp_training.py --experiment_dirpath model_experiments/2026-03-05-collisions_only --metric_name ann_indices
# uv run python python_scripts/gp_training.py --experiment_dirpath model_experiments/2026-03-05-collisions_only --metric_name coherency_fractions
# uv run python python_scripts/gp_training.py --experiment_dirpath model_experiments/2026-03-05-collisions_only --metric_name magnitude_cellmeans
# uv run python python_scripts/gp_training.py --experiment_dirpath model_experiments/2026-03-05-collisions_only --metric_name meander_ratios

# Matrix metrics:

# COM Movement Metrics:
uv run python python_scripts/gp_training.py --experiment_dirpath model_experiments/2026-03-20-collisions_shape --metric_name com_ann_indices
uv run python python_scripts/gp_training.py --experiment_dirpath model_experiments/2026-03-20-collisions_shape --metric_name com_coherency_fractions
uv run python python_scripts/gp_training.py --experiment_dirpath model_experiments/2026-03-20-collisions_shape --metric_name com_meander_ratios
uv run python python_scripts/gp_training.py --experiment_dirpath model_experiments/2026-03-20-collisions_shape --metric_name com_speeds