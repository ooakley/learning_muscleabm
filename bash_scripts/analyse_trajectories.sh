#!/bin/bash
#SBATCH --job-name=an_traj
#SBATCH --ntasks=1
#SBATCH --partition=ncpu
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out

# Analyses the tracked wet lab trajectories into wetlab_data/site_dataframe.csv and
# wetlab_data/particle_dataframe.csv. Usage, from the repository root:
#
#     sbatch bash_scripts/analyse_trajectories.sh [analysed_data_dirpath]
#
# analysed_data_dirpath defaults to the shared analysed data folder on CAMP.
set -eo pipefail

# --- Arguments ---
analysed_data_dirpath=${1:-/camp/home/eloaklo/home/shared/eloaklo/analysed_data}

ml load uv
# Lmod is not guaranteed to work with unset variables treated as errors, so only from here:
set -u

uv run --no-sync python python_scripts/wetlab/analyse_wetlab_data.py \
    --analysed_data_dirpath "$analysed_data_dirpath"
