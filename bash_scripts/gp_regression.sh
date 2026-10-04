#!/bin/bash
#SBATCH --job-name=gp_train
#SBATCH --partition=ncpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-3
#SBATCH --output=logs/gp_train_%A_%a.out

ml load uv
# Script inputs. The models are saved to the latest wave in the global dataset, inside the
# GP models folder if given, or else a folder named after the training settings:
usage="Usage: sbatch gp_regression.sh <experiment_dirpath> <hm_wave_id> [gp_models_dirname]"
experiment_dirpath=${1:?$usage}
hm_wave_id=${2:?$usage}
gp_models_dirname=${3:-}

metrics=(speeds meander_ratios ann_indices coherency)
metric=${metrics[$SLURM_ARRAY_TASK_ID]}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

gp_models_dirname_argument=()
if [ -n "$gp_models_dirname" ]; then
    gp_models_dirname_argument=(--gp_models_dirname "$gp_models_dirname")
fi

uv run --no-sync python python_scripts/emulation/gp_training.py \
    --experiment_dirpath "$experiment_dirpath" \
    --metric_name "$metric" \
    ${gp_models_dirname_argument[@]+"${gp_models_dirname_argument[@]}"}