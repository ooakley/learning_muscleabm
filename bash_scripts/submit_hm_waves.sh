#!/bin/bash
# Submits a chain of history matching waves as dependent SLURM jobs.
#
# Run from the repository root, on a login node (with bash, not sbatch):
#
#     bash bash_scripts/submit_hm_waves.sh <experiment_dirpath> <hm_config_path> [first_wave_id] [first_stage]
#
# The history matching config (see configs/history_matching_configs/default.json, and
# muscleabm/history_matching.py for what each setting means) sets the number of waves, the
# size of each wave after the first, the MCMC temperature its samples are drawn from, and
# the architecture and training settings of the GP emulators.
#
# This runs waves first_wave_id (default 0) up to wave_count - 1. For each wave it submits:
#
#     generate  ->  simulate  ->  collate  ->  collate_hm  ->  train  ->  mcmc
#                                    \-> validate (tests the previous wave's GP models)
#
# Each job waits for the one before it, and the posterior of one wave generates the next.
# Wave 0 is never generated here: run the Sobol' search first, as it creates the experiment
# folder itself. A later wave is generated unless its run_data folder already exists.
#
# To pick a chain back up after a failure, give the wave and the stage to restart from
# (generate, simulate, collate, train or mcmc). first_stage only applies to the first wave:
#
#     bash bash_scripts/submit_hm_waves.sh model_experiments/2026-10-02-collisions_shape \
#         configs/history_matching_configs/default.json 2 train
#
# To print the sbatch commands without submitting anything:
#
#     DRY_RUN=true bash bash_scripts/submit_hm_waves.sh <experiment_dirpath> <hm_config_path>
set -euo pipefail

# --- Arguments ---
usage="Usage: bash submit_hm_waves.sh <experiment_dirpath> <hm_config_path> [first_wave_id] [first_stage]"
experiment_dirpath=${1:?$usage}
hm_config_path=${2:?$usage}
first_wave_id=${3:-0}
first_stage=${4:-generate}
experiment_dirpath=${experiment_dirpath%/}
dry_run=${DRY_RUN:-false}

# --- Stage scripts ---
# Each is called as: sbatch <script> <experiment_dirpath> <hm_wave_id>
# (the training script also gets the GP models folder name and the GP settings, and the
# MCMC script the GP models folder).
simulate_script="bash_scripts/gridsearch.sh"
collate_script="bash_scripts/collate_gridsearch_data.sh"
train_script="bash_scripts/gp_regression.sh"
mcmc_script="bash_scripts/mcmc_fit.sh"
# Generic script for the light Python steps (generate, collate_hm, validate):
python_stage_script="bash_scripts/run_python_stage.sh"
# Python scripts run by the generic script:
generate_python_script="python_scripts/search/generate_hm_sweep.py"
collate_hm_python_script="python_scripts/collation/collate_hm.py"
validate_python_script="python_scripts/inference/validate_wave.py"

# --- History matching config ---
# Read and checked by muscleabm/history_matching.py, so that a mistake stops the submission.
# It prints, one per line: wave_count, wave_size, samples_per_phenotype (half of wave_size),
# temperature_index, gp_folder_name (the folder the GP models are saved to inside each wave
# folder, named after the GP settings) and gp_training_arguments (the GP settings, as
# arguments of gp_training.py):
if [ ! -f "$hm_config_path" ]; then
    echo "No history matching config found at ${hm_config_path}, exiting..."
    exit 1
fi
if [ ! -f muscleabm/history_matching.py ]; then
    echo "muscleabm/history_matching.py not found: run this from the repository root, exiting..."
    exit 1
fi
hm_settings=$(python3 -m muscleabm.history_matching "$hm_config_path")
{
    read -r wave_count
    read -r wave_size
    read -r samples_per_phenotype
    read -r temperature_index
    read -r gp_folder_name
    read -r -a gp_training_arguments
} <<< "$hm_settings"
echo "History matching config ${hm_config_path}: ${wave_count} waves of ${wave_size} simulations" \
    "(${samples_per_phenotype} per phenotype, after wave 0), drawn from temperature ${temperature_index}," \
    "GP models in ${gp_folder_name}"

# --- Settings ---
# Folder name written by the MCMC script, inside each wave folder:
mcmc_dirname="disc_cov_mcmc_results"
# Must match the burn-in fraction used by the MCMC stage script:
burn_in_fraction=0.25
# Simulations run by each array task of the simulation script, and tasks allowed at once:
simulations_per_task=64
array_task_limit=90
# Stop the chain if more than this fraction of a wave's simulations failed:
max_failed_fraction=0.1
# Test each wave's GP models against the simulations of the next wave:
run_validation=true

# --- Checks ---
stage_names=(generate simulate collate train mcmc)
first_stage_index=-1
for stage_index in "${!stage_names[@]}"; do
    if [ "${stage_names[$stage_index]}" = "$first_stage" ]; then
        first_stage_index=$stage_index
    fi
done
if [ "$first_stage_index" -lt 0 ]; then
    echo "first_stage must be one of: ${stage_names[*]} (got ${first_stage}), exiting..."
    exit 1
fi
if ! [[ "$first_wave_id" =~ ^[0-9]+$ ]]; then
    echo "first_wave_id must be a whole number, exiting..."
    exit 1
fi
if [ "$first_wave_id" -ge "$wave_count" ]; then
    echo "first_wave_id (${first_wave_id}) must be below the config's wave_count (${wave_count}), exiting..."
    exit 1
fi
if [ ! -f "${experiment_dirpath}/config.json" ]; then
    echo "No config.json found in ${experiment_dirpath}, exiting..."
    exit 1
fi
for stage_script in "$simulate_script" "$collate_script" "$train_script" "$mcmc_script" "$python_stage_script" \
        "$generate_python_script" "$collate_hm_python_script" "$validate_python_script"; do
    if [ ! -f "$stage_script" ]; then
        echo "Stage script ${stage_script} not found: run this from the repository root, exiting..."
        exit 1
    fi
done

# The first wave either exists already, or is generated from the chains of the wave before it:
first_wave_dirpath="${experiment_dirpath}/hm${first_wave_id}"
if [ ! -d "${first_wave_dirpath}/run_data" ]; then
    if [ "$first_wave_id" -eq 0 ]; then
        echo "No hm0 found in ${experiment_dirpath}: generate the initial Sobol' search first, exiting..."
        exit 1
    fi
    if [ "$first_stage_index" -gt 0 ]; then
        echo "Cannot start from ${first_stage}: ${first_wave_dirpath}/run_data does not exist, exiting..."
        exit 1
    fi
    previous_mcmc_dirpath="${experiment_dirpath}/hm$(( first_wave_id - 1 ))/${mcmc_dirname}"
    for chain_filename in wt_mcmc_chain.npy rd_mcmc_chain.npy; do
        if [ ! -f "${previous_mcmc_dirpath}/${chain_filename}" ]; then
            echo "Cannot generate hm${first_wave_id}: ${previous_mcmc_dirpath}/${chain_filename} not found, exiting..."
            exit 1
        fi
    done
fi

# GP training uses every wave in the global dataset, so an earlier wave cannot be retrained
# while a later wave has summary data:
if [ "$first_stage_index" -le 3 ]; then
    for later_summary_dirpath in "${experiment_dirpath}"/hm*/summary_data; do
        later_wave_dirname=$(basename "$(dirname "$later_summary_dirpath")")
        later_wave_id=${later_wave_dirname#hm}
        if [[ "$later_wave_id" =~ ^[0-9]+$ ]] && [ "$later_wave_id" -gt "$first_wave_id" ] \
                && [ -n "$(ls -A "$later_summary_dirpath" 2>/dev/null)" ]; then
            echo "hm${later_wave_id} already has summary data, so the GP models of hm${first_wave_id} would be trained on it too."
            echo "Start from hm${later_wave_id} or later, or move its summary_data folder away first, exiting..."
            exit 1
        fi
    done
fi

# SLURM does not create log folders, and a job whose log folder is missing fails at once.
# Each stage logs to the folder of its part of the codebase:
mkdir -p logs/search logs/simulation logs/collation logs/emulation logs/inference

# The job scripts run with uv's --no-sync, so install the environment (including the
# shared muscleabm package that the Python scripts import) before anything is queued:
if [ "$dry_run" = true ]; then
    echo "Would run: uv sync"
else
    if ! command -v uv > /dev/null && command -v ml > /dev/null; then
        # Lmod is not guaranteed to work with unset variables treated as errors:
        set +u
        ml load uv
        set -u
    fi
    if ! command -v uv > /dev/null; then
        echo "uv not found: load it first (ml load uv), exiting..."
        exit 1
    fi
    uv sync
fi

# --- Helpers ---
# Submit a job and print its ID. Usage: submit <label> <sbatch arguments...>
submitted_job_ids=()
submit () {
    local label=$1
    shift
    local job_id
    if [ "$dry_run" = true ]; then
        job_id="<hm${wave}_${label%% *}>"
        echo "    sbatch $*" >&2
    else
        # Fail here if sbatch refuses the job: without an ID, the jobs after this one
        # would be submitted with nothing to wait for.
        job_id=$(sbatch --parsable "$@") || return 1
        job_id=${job_id%%;*}
        if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
            echo "    ${label}: sbatch did not return a job ID (got '${job_id}')" >&2
            return 1
        fi
        echo "    ${label}: ${job_id}" >&2
    fi
    echo "$job_id"
}

# Set the sbatch arguments for waiting on a job, or none if that job was not submitted.
# Usage: set_dependency <afterok|afterany> <job_id>
dependency=()
set_dependency () {
    local dependency_type=$1
    local job_id=$2
    dependency=()
    if [ -n "$job_id" ]; then
        dependency=("--dependency=${dependency_type}:${job_id}")
        # Dependents are cancelled, not left queued forever, if the job they wait for fails:
        if [ "$dependency_type" = "afterok" ]; then
            dependency+=("--kill-on-invalid-dep=yes")
        fi
    fi
}

# sbatch arguments that send the log of a Python stage to the folder of its script's stage,
# e.g. logs/search for python_scripts/search/generate_hm_sweep.py. Usage: log_argument <script>
log_argument () {
    echo "--output=logs/$(basename "$(dirname "$1")")/%x_%j.out"
}

# If a submission fails part-way, say how to cancel what is already queued:
report_failure () {
    echo "Submission failed. Jobs already queued: ${submitted_job_ids[*]:-none}"
    if [ "${#submitted_job_ids[@]}" -gt 0 ]; then
        echo "Cancel them with: scancel ${submitted_job_ids[*]}"
    fi
}
trap report_failure ERR

# Number of simulations in a wave: the Sobol' search for wave 0, posterior samples afterwards.
get_simulation_count () {
    local wave=$1
    local hm_config_filepath="${experiment_dirpath}/hm${wave}/hm_config.json"
    if [ "$wave" -eq 0 ]; then
        python3 -c "import json, sys; print(2 ** json.load(open(sys.argv[1]))['sample_exponent'])" \
            "${experiment_dirpath}/config.json"
    elif [ -f "$hm_config_filepath" ]; then
        # The wave already exists, so use the size it was generated with:
        python3 -c "import json, sys; c = json.load(open(sys.argv[1])); print(c['sample_count'] * len(c['phenotypes']))" \
            "$hm_config_filepath"
    else
        echo "$wave_size"
    fi
}

# --- Submission ---
mcmc=""
for wave in $(seq "$first_wave_id" $(( wave_count - 1 ))); do
    echo "Wave ${wave}:"
    wave_dirpath="${experiment_dirpath}/hm${wave}"
    gp_models_dirpath="${wave_dirpath}/${gp_folder_name}"

    # Stages before first_stage are skipped in the first wave only:
    start_index=0
    if [ "$wave" -eq "$first_wave_id" ]; then
        start_index=$first_stage_index
    fi
    generate=""; simulate=""; collate=""; collate_hm=""; train=""

    # 1. Generate the wave from the posterior of the previous one, unless it already exists:
    if [ "$start_index" -le 0 ]; then
        if [ ! -d "${wave_dirpath}/run_data" ]; then
            set_dependency afterok "$mcmc"
            generate=$(submit "generate" ${dependency[@]+"${dependency[@]}"} --job-name="hm${wave}_generate" \
                "$(log_argument "$generate_python_script")" "$python_stage_script" "$generate_python_script" \
                --experiment_dirpath "$experiment_dirpath" --hm_wave_id $(( wave - 1 )) \
                --mcmc_dirname "$mcmc_dirname" --sample_count "$samples_per_phenotype" \
                --burn_in_fraction "$burn_in_fraction" --temperature_index "$temperature_index")
            submitted_job_ids+=("$generate")
        else
            echo "    generate: ${wave_dirpath} already exists, skipping"
        fi
    fi

    # 2. Run the simulations, as an array sized to the wave:
    if [ "$start_index" -le 1 ]; then
        simulation_count=$(get_simulation_count "$wave")
        task_count=$(( (simulation_count + simulations_per_task - 1) / simulations_per_task ))
        set_dependency afterok "$generate"
        simulate=$(submit "simulate (${simulation_count} simulations, ${task_count} tasks)" \
            ${dependency[@]+"${dependency[@]}"} --job-name="hm${wave}_simulate" \
            --array="1-${task_count}%${array_task_limit}" \
            "$simulate_script" "$experiment_dirpath" "$wave")
        submitted_job_ids+=("$simulate")
    fi

    if [ "$start_index" -le 2 ]; then
        # 3. Collate the wave's summary data. This runs even if some simulations failed, as
        #    failed simulations are collated as NaN:
        set_dependency afterany "$simulate"
        collate=$(submit "collate" ${dependency[@]+"${dependency[@]}"} --job-name="hm${wave}_collate" \
            "$collate_script" "$experiment_dirpath" "$wave")
        submitted_job_ids+=("$collate")

        # 4. Rebuild the global dataset, stopping the chain if too much of the wave failed:
        set_dependency afterok "$collate"
        collate_hm=$(submit "collate_hm" "${dependency[@]}" --job-name="hm${wave}_collate_hm" \
            "$(log_argument "$collate_hm_python_script")" "$python_stage_script" "$collate_hm_python_script" \
            --experiment_dirpath "$experiment_dirpath" --max_failed_fraction "$max_failed_fraction" \
            --required_hm_wave_id "$wave")
        submitted_job_ids+=("$collate_hm")

        # 5. Test the previous wave's GP models against this wave's simulations (nothing waits on this).
        #    In the first wave submitted, those models were trained by an earlier submission,
        #    perhaps with other GP settings, so check they are where these settings put them:
        previous_gp_models_dirpath="${experiment_dirpath}/hm$(( wave - 1 ))/${gp_folder_name}"
        run_wave_validation=$run_validation
        if [ "$run_validation" = true ] && [ "$wave" -eq "$first_wave_id" ] && [ "$wave" -gt 0 ] \
                && [ ! -d "$previous_gp_models_dirpath" ]; then
            echo "    validate: ${previous_gp_models_dirpath} not found, skipping"
            run_wave_validation=false
        fi
        if [ "$run_wave_validation" = true ] && [ "$wave" -gt 0 ]; then
            validate=$(submit "validate" "${dependency[@]}" --job-name="hm${wave}_validate" \
                "$(log_argument "$validate_python_script")" "$python_stage_script" "$validate_python_script" \
                --experiment_dirpath "$experiment_dirpath" --hm_wave_id $(( wave - 1 )) \
                --gp_models_dirpath "$previous_gp_models_dirpath")
            submitted_job_ids+=("$validate")
        fi
    fi

    # 6. Train the GP models on the global dataset:
    if [ "$start_index" -le 3 ]; then
        set_dependency afterok "$collate_hm"
        train=$(submit "train" ${dependency[@]+"${dependency[@]}"} --job-name="hm${wave}_train" \
            "$train_script" "$experiment_dirpath" "$wave" "$gp_folder_name" "${gp_training_arguments[@]}")
        submitted_job_ids+=("$train")
    fi

    # 7. Sample the posterior of both phenotypes:
    set_dependency afterok "$train"
    mcmc=$(submit "mcmc" ${dependency[@]+"${dependency[@]}"} --job-name="hm${wave}_mcmc" \
        "$mcmc_script" "$experiment_dirpath" "$wave" "$gp_models_dirpath")
    submitted_job_ids+=("$mcmc")
done

if [ "$dry_run" = true ]; then
    echo "Dry run: nothing was submitted."
else
    echo "Submitted ${#submitted_job_ids[@]} jobs. Follow them with: squeue -u \$USER --format='%.12i %.22j %.10T %.20E'"
    echo "Cancel the whole chain with: scancel ${submitted_job_ids[*]}"
fi