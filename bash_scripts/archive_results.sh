#!/bin/bash
#SBATCH --job-name=archive
#SBATCH --ntasks=1
#SBATCH --partition=ncpu
#SBATCH --time=24:00:00
#SBATCH --mem=1G

rm -rf model_experiments/2026-10-02-collisions_shape/delete

# ml load uv
# uv run python_scripts/sensitivity/get_bd_matrix.py --experiment_dirpath "$experiment_dirpath"

# ml load uv
# experiment_dirpath="model_experiments/2026-09-19-matrix_shape"

# uv run python python_scripts/simulation/archive_image_data.py \
#     --experiment_dirpath $experiment_dirpath \
#     --image_filename trajectory

# uv run python python_scripts/simulation/archive_image_data.py \
#     --experiment_dirpath $experiment_dirpath \
#     --image_filename stadia

# uv run python python_scripts/simulation/archive_image_data.py \
#     --experiment_dirpath $experiment_dirpath \
#     --image_filename matrix_heading

# uv run python python_scripts/simulation/archive_image_data.py \
#     --experiment_dirpath $experiment_dirpath \
#     --image_filename matrix_density

# uv run python python_scripts/simulation/archive_image_data.py \
#     --experiment_dirpath $experiment_dirpath \
#     --image_filename angular_variance

# uv run python python_scripts/simulation/archive_image_data.py \
#     --experiment_dirpath $experiment_dirpath \
#     --image_filename com_trajectory