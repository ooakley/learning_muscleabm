#!/bin/bash
#SBATCH --job-name=gp_train
#SBATCH --partition=ncpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-3
#SBATCH --output=logs/emulation/%x_%A_%a.out

# Trains the GP emulator of each model metric (one array task per metric) on the global
# dataset of an experiment. Usage, from the repository root:
#
#     sbatch bash_scripts/gp_regression.sh <experiment_dirpath> <hm_wave_id> [gp_models_dirname] [arguments...]
#
# The models train on the rows of the global dataset from waves up to hm_wave_id, and are saved
# to that wave, inside the GP models folder if given, or trained_models/ if not (or given as "").
# Any further arguments are passed to gp_training.py:
#     the GP architecture and training settings (e.g. --training_steps 3840 --hidden_layer_count 2),
#         which otherwise take the defaults in muscleabm/history_matching.py;
#     --stage full, cross_validation or all (the default): submit_hm_waves.sh trains the full
#         models first (full), then cross-validates them and estimates their Sobol' indices in a
#         separate job (cross_validation), which needs the full models and the same settings.
#
# For example, to rerun only the cross-validation of hm2, with the settings of the default config:
#
#     sbatch bash_scripts/gp_regression.sh model_experiments/2026-10-02-collisions_shape 2 trained_models \
#         $(python3 -m muscleabm.history_matching configs/history_matching_configs/default.json | tail -n 1) \
#         --stage cross_validation
set -eo pipefail

# --- Arguments ---
usage="Usage: sbatch bash_scripts/gp_regression.sh <experiment_dirpath> <hm_wave_id> [gp_models_dirname] [arguments...]"
experiment_dirpath=${1:?$usage}
hm_wave_id=${2:?$usage}
gp_models_dirname=${3:-}
gp_settings_arguments=("${@:4}")

ml load uv
# Lmod is not guaranteed to work with unset variables treated as errors, so only from here:
set -u

metrics=(speeds meander_ratios ann_indices coherency)
metric=${metrics[$SLURM_ARRAY_TASK_ID]}
echo "Training the ${metric} GP model of hm${hm_wave_id}..."

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

gp_models_dirname_argument=()
if [ -n "$gp_models_dirname" ]; then
    gp_models_dirname_argument=(--gp_models_dirname "$gp_models_dirname")
fi

uv run --no-sync python python_scripts/emulation/gp_training.py \
    --experiment_dirpath "$experiment_dirpath" \
    --hm_wave_id "$hm_wave_id" \
    --metric_name "$metric" \
    ${gp_models_dirname_argument[@]+"${gp_models_dirname_argument[@]}"} \
    ${gp_settings_arguments[@]+"${gp_settings_arguments[@]}"}
