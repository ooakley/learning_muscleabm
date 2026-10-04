#!/bin/bash
#SBATCH --job-name=hm_stage
#SBATCH --partition=ncpu
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH -o ./slurm_out/%x_%j.out

# Runs one light Python step of the history matching cycle (generating a wave, collating
# across waves, validating a wave). Usage:
#
#     sbatch run_python_stage.sh <python script> [arguments...]

# Required lmod modules:
ml load uv

echo "Running: $*"
uv run python3 "$@"