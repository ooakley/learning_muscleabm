#!/bin/bash

# Command line arguments:
arrayid=$1

# Derived variables:
hierarchy_folder=$(($arrayid/1000))
run_folderpath="${output_folderpath}/run_data/${hierarchy_folder}/${arrayid}"

# Run simulations with given parameter set:
echo "Running array" $arrayid
uv run python3 ./python_scripts/simulation/call_json_parameters.py \
    --path_to_config ${run_folderpath}/${arrayid}_arguments.json

# # Analyse simulation:
# uv run python3 ./python_scripts/simulation/site_analysis.py --run_folderpath $run_folderpath --folder_id $arrayid
# uv run python3 ./python_scripts/simulation/site_analysis.py --run_folderpath $run_folderpath --folder_id $arrayid --com_analysis
# uv run python3 ./python_scripts/simulation/matrix_analysis.py --run_folderpath $run_folderpath --folder_id $arrayid

# Delete intermediate simulation outputs to save space on the cluster:
rm ${output_folderpath}/run_data/${hierarchy_folder}/${arrayid}/matrix_seed*
rm ${output_folderpath}/run_data/${hierarchy_folder}/${arrayid}/positions_seed*