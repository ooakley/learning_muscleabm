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

# Script inputs:
usage="Usage: sbatch bash_scripts/mcmc_fit.sh <experiment_dirpath> <hm_wave_id> <gp_models_dirpath>"
experiment_dirpath=${1:?$usage}
hm_wave_id=${2:?$usage}
gp_models_dirpath=${3:?$usage}

# 16 threads per worker group, 4 worker groups, 64 threads in total:
ml load uv

PHENOTYPES=(WT RD)
uv run --no-sync python python_scripts/inference/mcmc_gp_cov_fit.py \
    --experiment_dirpath $experiment_dirpath \
    --hm_wave_id $hm_wave_id \
    --gp_models_dirpath $gp_models_dirpath \
    --phenotype ${PHENOTYPES[$SLURM_ARRAY_TASK_ID]} \
    --experimental_lengthscales \
    --adaptive_proposals --burn_in_fraction 0.25 \
    --no_calibration