#!/bin/bash
#SBATCH --job-name=mcmc_fit
#SBATCH --partition=ncpu
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=1G

# Required lmod modules:
ml load uv

uv run python python_scripts/mcmc_gp_fit.py