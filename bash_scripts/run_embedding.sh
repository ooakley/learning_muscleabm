#!/bin/bash
#SBATCH --job-name=run_embd
#SBATCH --partition=ncpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH --hint=nomultithread

# Script inputs:
usage="Usage: sbatch run_embedding.sh <experiment_dirpath>"
experiment_dirpath=${1:?$usage}

ml load uv
uv run python python_scripts/sensitivity/run_isomap_embedding.py --experiment_dirpath "$experiment_dirpath"
# uv run python python_scripts/sensitivity/rotate_hessians.py --experiment_dirpath "$experiment_dirpath"
