#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --ntasks=1
#SBATCH --partition=ncpu
#SBATCH --time=2:30:00
#SBATCH --cpus-per-task=1
#SBATCH --array=1-70

# Required lmod modules:
# Boost/1.81.0-GCC-12.2.0 CMake/3.24.3-GCCcore-12.2.0 OpenMPI/4.1.4-GCC-12.2.0

# Specify the path to the config file
config=./fileOutputs/gridsearch.txt

# Defining simulation function:
simulate () {
    local arrayid=$1

    # Extracting argument values based on the array ID:
    superIterationCount=$(awk -v array_id=$arrayid '$1==array_id {print $2}' $config)
    numberOfCells=$(awk -v array_id=$arrayid '$1==array_id {print $3}' $config)
    worldSize=$(awk -v array_id=$arrayid '$1==array_id {print $4}' $config)
    gridSize=$(awk -v array_id=$arrayid '$1==array_id {print $5}' $config)

    cellTypeProportions=$(awk -v array_id=$arrayid '$1==array_id {print $6}' $config)
    matrixAdditionRate=$(awk -v array_id=$arrayid '$1==array_id {print $7}' $config)
    matrixTurnoverRate=$(awk -v array_id=$arrayid '$1==array_id {print $8}' $config)

    cellDepositionSigma=$(awk -v array_id=$arrayid '$1==array_id {print $9}' $config)
    cellSensationSigma=$(awk -v array_id=$arrayid '$1==array_id {print $10}' $config)

    poissonLambda=$(awk -v array_id=$arrayid '$1==array_id {print $11}' $config)
    kappa=$(awk -v array_id=$arrayid '$1==array_id {print $12}' $config)
    matrixKappa=$(awk -v array_id=$arrayid '$1==array_id {print $13}' $config)

    homotypicInhibition=$(awk -v array_id=$arrayid '$1==array_id {print $14}' $config)
    heterotypicInhibition=$(awk -v array_id=$arrayid '$1==array_id {print $15}' $config)
    polarityPersistence=$(awk -v array_id=$arrayid '$1==array_id {print $16}' $config)
    polarityTurningCoupling=$(awk -v array_id=$arrayid '$1==array_id {print $17}' $config)

    flowScaling=$(awk -v array_id=$arrayid '$1==array_id {print $18}' $config)
    flowPolarityCoupling=$(awk -v array_id=$arrayid '$1==array_id {print $19}' $config)
    polarityNoiseSigma=$(awk -v array_id=$arrayid '$1==array_id {print $20}' $config)

    collisionRepolarisation=$(awk -v array_id=$arrayid '$1==array_id {print $21}' $config)
    repolarisationRate=$(awk -v array_id=$arrayid '$1==array_id {print $22}' $config)

    # Running simulation with specified parameters:
    ./build/src/main --jobArrayID $arrayid --superIterationCount $superIterationCount \
        --numberOfCells $numberOfCells --timestepsToRun 5760 \
        --worldSize $worldSize --gridSize $gridSize \
        --cellTypeProportions $cellTypeProportions \
        --matrixAdditionRate $matrixAdditionRate --matrixTurnoverRate $matrixTurnoverRate \
        --cellDepositionSigma $cellDepositionSigma --cellSensationSigma $cellSensationSigma \
        --poissonLambda $poissonLambda --kappa $kappa --matrixKappa $matrixKappa \
        --homotypicInhibition $homotypicInhibition --heterotypicInhibition $heterotypicInhibition \
        --polarityPersistence $polarityPersistence --polarityTurningCoupling $polarityTurningCoupling \
        --flowScaling $flowScaling --flowPolarityCoupling $flowPolarityCoupling \
        --collisionRepolarisation $collisionRepolarisation --repolarisationRate $repolarisationRate \
        --polarityNoiseSigma $polarityNoiseSigma

    # Analysing simulation:
    python3 simulationAnalysis.py --folder_id $arrayid

    # Deleting intermediate simulation outputs to save space:
    rm fileOutputs/$arrayid/matrix_seed*
    rm fileOutputs/$arrayid/positions_seed*
}

subgroup=$(($SLURM_ARRAY_TASK_ID-1))
subgroup=$(($subgroup*10))

for i in {1..10}
do
    arrayid=$(($subgroup+$i))
    echo "--- --- --- --- --- --- ---"
    echo $arrayid
    simulate $arrayid
done