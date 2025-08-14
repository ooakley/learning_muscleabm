#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --ntasks=1
#SBATCH --partition=ncpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --array=1-820

# Required lmod modules:
# Boost/1.81.0-GCC-12.2.0 CMake/3.24.3-GCCcore-12.2.0 OpenMPI/4.1.4-GCC-12.2.0

# Defining simulation function:
simulate () {
    local arrayid=$1

    # Running simulations with given parameter set:
    python3 ./python_scripts/call_json_parameters.py \
    --path_to_config ./fileOutputs/${arrayid}/${arrayid}_arguments.json

    # Analysing simulation:
    # python3 ./python_scripts/order_parameter_analysis.py --folder_id $arrayid
    python3 ./python_scripts/speed_persistence_analysis.py --folder_id $arrayid
    python3 ./python_scripts/wasserstein_distance_analysis.py --folder_id $arrayid

    # Deleting intermediate simulation outputs to save space on the cluster:
    rm fileOutputs/$arrayid/matrix_seed*
    rm fileOutputs/$arrayid/positions_seed*
}

subgroup=$(($SLURM_ARRAY_TASK_ID-1))
subgroup_index=$(($subgroup*10))

for i in {0..9}
do
    arrayid=$(($subgroup_index+$i))
    echo "--- --- --- --- --- --- ---"
    echo $arrayid
    simulate $arrayid
done