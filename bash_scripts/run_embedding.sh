#!/bin/bash
#SBATCH --job-name=kpca
#SBATCH --partition=ncpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2G

ml load uv
uv run python python_scripts/run_embedding.py