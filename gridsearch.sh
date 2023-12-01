#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=1
#SBATCH --array=1-3125

# Required lmod modules:
# Boost/1.81.0-GCC-12.2.0
# CMake/3.24.3-GCCcore-12.2.0

# Specify the path to the config file
config=./fileOutputs/gridsearch.txt

# Extracting argument names based on $SLURM_ARRAY_TASK_ID
superIterationCount=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $2}' $config)
numberOfCells=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $3}' $config)
worldSize=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $4}' $config)
gridSize=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $5}' $config)
cellTypeProportions=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $6}' $config)
matrixPersistence=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $7}' $config)

wbK=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $8}' $config)
kappa=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $9}' $config)
homotypicInhibition=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $10}' $config)
heterotypicInhibition=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $11}' $config)
polarityPersistence=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $12}' $config)
polarityTurningCoupling=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $13}' $config)
flowScaling=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $14}' $config)
flowPolarityCoupling=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $15}' $config)
collisionRepolarisation=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $16}' $config)
repolarisationRate=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $17}' $config)

# Running simulation with specified parameters:
./build/src/main --jobArrayID $SLURM_ARRAY_TASK_ID --superIterationCount $superIterationCount --numberOfCells $numberOfCells \
    --worldSize $worldSize --gridSize $gridSize \
    --cellTypeProportions $cellTypeProportions --matrixPersistence $matrixPersistence \
    --wbK $wbK --kappa $kappa --homotypicInhibition $homotypicInhibition \
    --heterotypicInhibition $heterotypicInhibition --polarityPersistence $polarityPersistence --polarityTurningCoupling $polarityTurningCoupling \
    --flowScaling $flowScaling --flowPolarityCoupling $flowPolarityCoupling \
    --collisionRepolarisation $collisionRepolarisation --repolarisationRate $repolarisationRate