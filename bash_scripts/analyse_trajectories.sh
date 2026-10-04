#!/bin/bash
#SBATCH --job-name=an_traj
#SBATCH --ntasks=1
#SBATCH --partition=ncpu
#SBATCH --time=24:00:00
#SBATCH --mem=16G

# Script inputs, defaulting to the shared analysed data folder on CAMP:
usage="Usage: sbatch analyse_trajectories.sh [analysed_data_dirpath]"
analysed_data_dirpath=${1:-/camp/home/eloaklo/home/shared/eloaklo/analysed_data}

ml load uv
uv run python python_scripts/wetlab/analyse_wetlab_data.py \
    --analysed_data_dirpath "$analysed_data_dirpath"