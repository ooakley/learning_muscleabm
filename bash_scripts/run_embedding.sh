#!/bin/bash
#SBATCH --job-name=run_embd
#SBATCH --partition=ncpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH --hint=nomultithread
#SBATCH --output=logs/sensitivity/%x_%j.out

# Embeds the op65 Fisher information matrices of an experiment. Usage, from the repository
# root:
#
#     sbatch bash_scripts/run_embedding.sh <experiment_dirpath>
set -eo pipefail

# --- Arguments ---
usage="Usage: sbatch bash_scripts/run_embedding.sh <experiment_dirpath>"
experiment_dirpath=${1:?$usage}

ml load uv
# Lmod is not guaranteed to work with unset variables treated as errors, so only from here:
set -u

uv run --no-sync python python_scripts/sensitivity/run_isomap_embedding.py --experiment_dirpath "$experiment_dirpath"
# uv run --no-sync python python_scripts/sensitivity/rotate_hessians.py --experiment_dirpath "$experiment_dirpath"
