#!/bin/bash
#SBATCH --job-name=collate
#SBATCH --partition=ncpu
#SBATCH --time=4:00:00
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G

output_folder="sobolMatrixOutputs"
collation_folder="matrixCollated"
python python_scripts/collate_site_analyses.py --folder_name $output_folder --output_name $collation_folder & 
python python_scripts/collate_speed_persistence.py --folder_name $output_folder --output_name $collation_folder &
python python_scripts/collate_pcp_dataframe.py --folder_name $output_folder --output_name $collation_folder &

wait