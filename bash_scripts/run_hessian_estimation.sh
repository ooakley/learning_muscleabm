#!/bin/bash
#SBATCH --job-name=hessian
#SBATCH --partition=ncpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G

ml load uv
uv run python_scripts/generate_hessians.py