#!/bin/bash
#SBATCH --job-name=archive
#SBATCH --ntasks=1
#SBATCH --partition=ncpu
#SBATCH --time=12:00:00
#SBATCH --mem=1G

rm -rf model_experiments/2026-08-13-collisions_shape

# ml load uv
# experiment_dirpath="model_experiments/2026-06-03-matrix_shape"

# uv run python python_scripts/archive_image_data.py \
#     --experiment_dirpath $experiment_dirpath \
#     --image_filename stadia
# uv run python python_scripts/archive_image_data.py \
#     --experiment_dirpath $experiment_dirpath \
#     --image_filename trajectory
# uv run python python_scripts/archive_image_data.py \
#     --experiment_dirpath $experiment_dirpath \
#     --image_filename matrix_heading
# uv run python python_scripts/archive_image_data.py \
#     --experiment_dirpath $experiment_dirpath \
#     --image_filename matrix_density

# uv run python python_scripts/archive_image_data.py \
#     --experiment_dirpath $experiment_dirpath \
#     --image_filename com_trajectory