#!/bin/bash
#SBATCH --job-name=hm_stage
#SBATCH --partition=ncpu
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/%x_%j.out

# Runs one light Python step of the history matching cycle (generating a wave, collating
# across waves, validating a wave). Usage, from the repository root:
#
#     sbatch bash_scripts/run_python_stage.sh <python script> [arguments...]
set -eo pipefail

# --- Arguments ---
usage="Usage: sbatch bash_scripts/run_python_stage.sh <python script> [arguments...]"
: "${1:?$usage}"

ml load uv
# Lmod is not guaranteed to work with unset variables treated as errors, so only from here:
set -u

echo "Running: $*"
uv run --no-sync python "$@"
