#!/bin/bash
#SBATCH --job-name=sim_analysis
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=1
#SBATCH --array=1-1024

# Required lmod modules:
# Boost/1.81.0-GCC-12.2.0
# CMake/3.24.3-GCCcore-12.2.0

python3 simulationAnalysis.py --folder_id $SLURM_ARRAY_TASK_ID