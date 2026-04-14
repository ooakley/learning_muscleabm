#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --partition=ncpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --array=2-3277
#SBATCH  -o ./slurm_out/%a.out


# 6554
# 3277
# 410

# Required lmod modules:
ml load uv parallel Boost/1.81.0-GCC-12.2.0 CMake/3.24.3-GCCcore-12.2.0 OpenMPI/4.1.4-GCC-12.2.0

output_folderpath="model_experiments/2026-04-09-collisions_shape"

# Define simulation function:
simulate () {
    local arrayid=$1
    local hierarchy_folder=$(($arrayid/1000))
    local run_folderpath="${output_folderpath}/run_data/${hierarchy_folder}/${arrayid}"

    # Run simulations with given parameter set:
    uv run python3 ./python_scripts/call_json_parameters.py \
    --path_to_config ${run_folderpath}/${arrayid}_arguments.json

    # Analyse simulation:
    uv run python3 ./python_scripts/site_analysis.py --run_folderpath $run_folderpath --folder_id $arrayid --com_analysis True
    # uv run python3 ./python_scripts/matrix_analysis.py --run_folderpath $run_folderpath --folder_id $arrayid
    # uv run python3 ./python_scripts/speed_persistence_analysis.py --run_folderpath $run_folderpath --folder_id $arrayid
    # python3 ./python_scripts/wasserstein_distance_analysis.py --folder_id $arrayid

    # # Delete intermediate simulation outputs to save space on the cluster:
    rm ${output_folderpath}/run_data/${hierarchy_folder}/${arrayid}/matrix_seed*
    rm ${output_folderpath}/run_data/${hierarchy_folder}/${arrayid}/positions_seed*
}

# Export function and base filepath so we can use it with GNU parallel:
export -f simulate
export output_folderpath=$output_folderpath

# Define range over which we run our simulation:
subgroup=$(($SLURM_ARRAY_TASK_ID-1))
subgroup_index=$(($subgroup*40))
array_list=$(seq $subgroup_index $(($subgroup_index+39)))

# Run in parallel with GNU parallel:
echo "Running in simulations in parallel..."
parallel -j 10 simulate ::: $array_list

# for i in {0..19}
# do
#     if (( (i % 5) == 0 )); then
#         wait
#     fi
    
#     arrayid=$(($subgroup_index+$i))
#     echo "--- --- --- --- --- --- ---"
#     echo $arrayid
#     simulate $arrayid &
# done

# wait