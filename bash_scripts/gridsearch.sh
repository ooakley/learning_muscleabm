#!/bin/bash
#SBATCH --job-name=gridsearch
#SBATCH --partition=ncpu
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --array=1-512%100
#SBATCH -o ./slurm_out/%x_%A_%a.out

# Runs the simulations of one history matching wave. Usage:
#
#     sbatch run_simulations.sh <experiment_dirpath> <hm_wave_id>
#
# Each array task runs simulations_per_task simulations, so the array needs
# ceil(simulation count / simulations_per_task) tasks. Override the range above
# for a wave of a different size, e.g. sbatch --array=1-16%100 run_simulations.sh <...> 1
# (submit_hm_waves.sh does this itself, and expects the same simulations_per_task).

# --- Settings ---
usage="Usage: sbatch run_simulations.sh <experiment_dirpath> <hm_wave_id>"
experiment_dirpath=${1:?$usage}
hm_wave_id=${2:?$usage}
experiment_dirpath=${experiment_dirpath%/}
simulations_per_task=64
parallel_job_count=16

# Simulations live in the run_data folder of the wave:
run_data_dirpath="${experiment_dirpath}/hm${hm_wave_id}/run_data"
if [ ! -d "$run_data_dirpath" ]; then
    echo "No run data folder found at ${run_data_dirpath}, exiting..."
    exit 1
fi

# Required lmod modules:
ml load uv parallel Boost/1.81.0-GCC-12.2.0 CMake/3.24.3-GCCcore-12.2.0 OpenMPI/4.1.4-GCC-12.2.0

# Delete intermediate simulation outputs to save space on the cluster:
remove_intermediate_outputs () {
    local run_folderpath=$1
    rm -f "${run_folderpath}"/matrix_seed* "${run_folderpath}"/positions_seed*
    # rm -f "${run_folderpath}"/com_*
}

# Run and analyse the simulation with a given ID:
simulate () {
    local simulation_id=$1
    local hierarchy_folder=$((simulation_id / 1000))
    local run_folderpath="${run_data_dirpath}/${hierarchy_folder}/${simulation_id}"
    local arguments_filepath="${run_folderpath}/${simulation_id}_arguments.json"

    # Skip IDs past the end of the wave:
    if [ ! -f "$arguments_filepath" ]; then
        echo "${arguments_filepath} not found, skipping..."
        return 0
    fi

    # Skip simulations that are already done:
    if [ -f "${run_folderpath}/speeds.npy" ]; then
        echo "${run_folderpath} already complete..."
        remove_intermediate_outputs "$run_folderpath"
        return 0
    fi

    # Run simulations with given parameter set:
    uv run python3 ./python_scripts/simulation/call_json_parameters.py \
        --path_to_config "$arguments_filepath"

    # Analyse simulation:
    uv run python3 ./python_scripts/simulation/site_analysis.py --run_folderpath "$run_folderpath" --folder_id "$simulation_id"
    # uv run python3 ./python_scripts/simulation/site_analysis.py --run_folderpath "$run_folderpath" --folder_id "$simulation_id" --com_analysis
    # uv run python3 ./python_scripts/simulation/matrix_analysis.py --run_folderpath "$run_folderpath" --folder_id "$simulation_id"

    remove_intermediate_outputs "$run_folderpath"
}

# Export functions and the run data folder so we can use them with GNU parallel:
export -f simulate remove_intermediate_outputs
export run_data_dirpath

# Define range of simulation IDs covered by this array task:
first_simulation_id=$(( (SLURM_ARRAY_TASK_ID - 1) * simulations_per_task ))
last_simulation_id=$(( first_simulation_id + simulations_per_task - 1 ))

# Run in parallel with GNU parallel:
echo "Running simulations ${first_simulation_id} to ${last_simulation_id} of hm${hm_wave_id} in parallel..."
parallel -j "$parallel_job_count" simulate ::: $(seq "$first_simulation_id" "$last_simulation_id")