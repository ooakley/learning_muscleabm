#!/bin/bash
#SBATCH --job-name=archive
#SBATCH --ntasks=1
#SBATCH --partition=ncpu
#SBATCH --time=24:00:00
#SBATCH --mem=1G

rm -rf model_experiments/2026-03-20-collisions_shape
# tar cfv model_experiments/2025-12-02-collisions_shape/run_data.tar -C model_experiments/2025-12-02-collisions_shape/run_data .