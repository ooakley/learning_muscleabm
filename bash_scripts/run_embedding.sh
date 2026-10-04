#!/bin/bash
#SBATCH --job-name=run_embd
#SBATCH --partition=ncpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH --hint=nomultithread

ml load uv
uv run python python_scripts/run_isomap_embedding.py
# uv run python python_scripts/rotate_hessians.py
