#!/bin/bash
#SBATCH --job-name=collate
#SBATCH --partition=ncpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500M

ml load uv
experiment_dirpath="model_experiments/2026-07-01-csm_posterior"

collate () {
    local collation_target=$1
    uv run python python_scripts/collate_site_analyses.py \
        --experiment_folderpath $experiment_dirpath \
        --collation_target $collation_target
}

collate "speeds" &
sleep 30
collate "meander_ratios" &
collate "ann_indices" &
collate "coherency" &
collate "interaction" &
collate "order_parameters" &
collate "mean_directions" &

collate "com_speeds" &
collate "com_meander_ratios" &
collate "com_ann_indices" &
collate "com_coherency" &
collate "com_interaction" &
collate "com_order_parameters" &
collate "com_mean_directions" &

collate "cell_lengths" &

uv run python python_scripts/collate_matrix_analyses.py \
    --experiment_folderpath $experiment_dirpath &

wait