#!/bin/bash
#SBATCH --job-name=collate
#SBATCH --partition=ncpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G

ml load uv
uv run python python_scripts/collate_site_analyses.py --experiment_folderpath model_experiments/2026-03-20-collisions_shape --sample_count 131072
# uv run python python_scripts/collate_matrix_analyses.py --experiment_folderpath model_experiments/2026-03-07-matrix_collisions --sample_count 131072
# uv run python python_scripts/collate_speed_persistence.py --experiment_folderpath model_experiments/2026-03-07-matrix_collisions --sample_count 131072