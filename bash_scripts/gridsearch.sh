#!/bin/bash
#SBATCH --job-name=gridsearch
#SBATCH --partition=ncpu
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --array=1-1
#SBATCH -o ./slurm_out/%a.out

# Required lmod modules:
ml load uv parallel Boost/1.81.0-GCC-12.2.0 CMake/3.24.3-GCCcore-12.2.0 OpenMPI/4.1.4-GCC-12.2.0
output_folderpath="model_experiments/2026-08-14-collisions_shape"
# --dependency after:48231538
# --array=1-2048%120

# Define simulation function:
simulate () {
    local arrayid=$1
    local hierarchy_folder=$(($arrayid/1000))
    local run_folderpath="${output_folderpath}/run_data/${hierarchy_folder}/${arrayid}"
    # local test_path="${run_folderpath}/angular_variance.png"

    # # Exit if simulation is already done:
    # if [ -f "$test_path" ]; then
    #     echo "${run_folderpath} already complete..."
    #     return 0
    # fi

    # Run simulations with given parameter set:
    uv run python3 ./python_scripts/call_json_parameters.py \
        --path_to_config ${run_folderpath}/${arrayid}_arguments.json

    # Analyse simulation:
    uv run python3 ./python_scripts/site_analysis.py --run_folderpath $run_folderpath --folder_id $arrayid
    uv run python3 ./python_scripts/site_analysis.py --run_folderpath $run_folderpath --folder_id $arrayid --com_analysis
    # uv run python3 ./python_scripts/matrix_analysis.py --run_folderpath $run_folderpath --folder_id $arrayid

    # Delete intermediate simulation outputs to save space on the cluster:
    rm ${output_folderpath}/run_data/${hierarchy_folder}/${arrayid}/matrix_seed*
    rm ${output_folderpath}/run_data/${hierarchy_folder}/${arrayid}/positions_seed*
}

# Export function and base filepath so we can use it with GNU parallel:
export -f simulate
export output_folderpath=$output_folderpath

# Define range over which we run our simulation:
subgroup=$(($SLURM_ARRAY_TASK_ID-1))
subgroup_index=$(($subgroup*64))
array_list=$(seq $subgroup_index $(($subgroup_index+64)))

# Run in parallel with GNU parallel:
echo "Running simulations in parallel..."
parallel -j 16 simulate ::: $array_list
