#!/bin/bash
#SBATCH --job-name=sym_reg
#SBATCH --ntasks=1
#SBATCH --partition=ncpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1

# Required lmod modules:
# ml purge (for some reason)

python3 ./python_scripts/run_symbolic_regression.py