#!/bin/bash
#SBATCH --job-name=mcmc_fit
#SBATCH --partition=ncpu
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --array=0-1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/%x_%A_%a.out

# Samples the posterior of one history matching wave against the wet lab data, one array
# task per phenotype. 16 threads per worker group, 4 worker groups, 64 threads in total.
# Usage, from the repository root:
#
#     sbatch bash_scripts/mcmc_fit.sh <experiment_dirpath> <hm_wave_id> <gp_models_dirpath>
set -eo pipefail

# --- Arguments ---
usage="Usage: sbatch bash_scripts/mcmc_fit.sh <experiment_dirpath> <hm_wave_id> <gp_models_dirpath>"
experiment_dirpath=${1:?$usage}
hm_wave_id=${2:?$usage}
gp_models_dirpath=${3:?$usage}

ml load uv
# Lmod is not guaranteed to work with unset variables treated as errors, so only from here:
set -u

phenotypes=(WT RD)
uv run --no-sync python python_scripts/inference/mcmc_gp_cov_fit.py \
    --experiment_dirpath "$experiment_dirpath" \
    --hm_wave_id "$hm_wave_id" \
    --gp_models_dirpath "$gp_models_dirpath" \
    --phenotype "${phenotypes[$SLURM_ARRAY_TASK_ID]}" \
    --experimental_lengthscales \
    --adaptive_proposals --burn_in_fraction 0.25 \
    --no_calibration
