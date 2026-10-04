#!/bin/bash
#SBATCH --job-name=collate
#SBATCH --partition=ncpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --output=logs/collation/%x_%j.out

# Collates the per-simulation outputs of one history matching wave into its summary_data
# folder. Usage, from the repository root:
#
#     sbatch bash_scripts/collate_gridsearch_data.sh <experiment_dirpath> <hm_wave_id>
set -eo pipefail

# --- Arguments ---
usage="Usage: sbatch bash_scripts/collate_gridsearch_data.sh <experiment_dirpath> <hm_wave_id>"
experiment_dirpath=${1:?$usage}
hm_wave_id=${2:?$usage}

ml load uv
# Lmod is not guaranteed to work with unset variables treated as errors, so only from here:
set -u

collate () {
    local collation_target=$1
    uv run --no-sync python python_scripts/collation/collate_site_analyses.py \
        --experiment_folderpath "$experiment_dirpath" \
        --hm_wave_id "$hm_wave_id" \
        --collation_target "$collation_target"
}

collation_targets=(
    speeds
    meander_ratios
    ann_indices
    coherency
    interaction
    order_parameters
    mean_directions
    # com_speeds
    # com_meander_ratios
    # com_ann_indices
    # com_coherency
    # com_interaction
    # com_order_parameters
    # com_mean_directions
    # cell_lengths
)

# Run the collations in parallel, keeping track of each so that failures can be detected:
process_ids=()
for collation_target in "${collation_targets[@]}"; do
    collate "$collation_target" &
    process_ids+=($!)
done

# uv run --no-sync python python_scripts/collation/collate_matrix_analyses.py \
#     --experiment_folderpath "$experiment_dirpath" &
# process_ids+=($!)

# A bare `wait` always succeeds, so wait for each collation in turn. The job fails if any
# of them did, which stops the jobs that are waiting on this one:
exit_status=0
for process_index in "${!process_ids[@]}"; do
    if ! wait "${process_ids[$process_index]}"; then
        echo "Collation of ${collation_targets[$process_index]} failed."
        exit_status=1
    fi
done
exit $exit_status
