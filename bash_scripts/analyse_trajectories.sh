#!/bin/bash
#SBATCH --job-name=an_traj
#SBATCH --ntasks=1
#SBATCH --partition=ncpu
#SBATCH --time=24:00:00
#SBATCH --mem=16G

ml load uv
uv run python python_scripts/analyse_wetlab_data.py