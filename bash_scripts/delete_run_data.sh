#!/bin/bash
#SBATCH --job-name=delete_run_data
#SBATCH --partition=ncpu
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=1G
#SBATCH --output=logs/collation/%x_%j.out

# Deletes the run_data folder of one history matching wave, to save space on the cluster,
# once its simulations are collated into summary_data. Usage, from the repository root:
#
#     sbatch bash_scripts/delete_run_data.sh <experiment_dirpath> <hm_wave_id>
#
# submit_hm_waves.sh runs this once the wave is in the global dataset. Everything later
# stages read is kept: the parameters are in sample_matrix.npy and the metrics in
# summary_data. The simulations cannot be rerun or recollated afterwards, so only run it
# by hand on a wave that has passed collate_hm.py's check of failed simulations.
set -euo pipefail

# --- Arguments ---
usage="Usage: sbatch bash_scripts/delete_run_data.sh <experiment_dirpath> <hm_wave_id>"
experiment_dirpath=${1:?$usage}
hm_wave_id=${2:?$usage}
experiment_dirpath=${experiment_dirpath%/}
wave_dirpath="${experiment_dirpath}/hm${hm_wave_id}"
run_data_dirpath="${wave_dirpath}/run_data"

# --- Checks ---
if [ ! -d "$run_data_dirpath" ]; then
    echo "No run data folder found at ${run_data_dirpath}, so nothing to delete."
    exit 0
fi
if [ ! -f "${wave_dirpath}/sample_matrix.npy" ] || [ -z "$(ls -A "${wave_dirpath}/summary_data" 2>/dev/null)" ]; then
    echo "${wave_dirpath} has no sample matrix or summary data, so its run data is kept, exiting..."
    exit 1
fi

# Delete the folders of each thousand simulations in parallel, as removing many small files
# one at a time is slow on a parallel filesystem:
echo "Deleting ${run_data_dirpath}..."
find "$run_data_dirpath" -mindepth 1 -maxdepth 1 -print0 \
    | xargs -0 -r -n 1 -P "${SLURM_CPUS_PER_TASK:-1}" rm -rf
rmdir "$run_data_dirpath"
echo "Deleted ${run_data_dirpath}."
