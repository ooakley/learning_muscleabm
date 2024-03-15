import os
import json
import numpy as np

# Defines the parameter ranges, with [start, stop, num_steps]:
GRIDSEARCH = {
    "superIterationCount": [10, 10, 1],
    "numberOfCells": [50, 190, 8],
    "worldSize": [2048, 2048, 1],
    "gridSize": [32, 32, 1],
    "cellTypeProportions": [0, 0, 1],
    "matrixAdditionRate": [0.1, 0.1, 1],
    "matrixTurnoverRate": [0.001, 0.001, 1],
    "cellDepositionSigma": [80, 80, 1],
    "cellSensationSigma": [80, 80, 1],
    "poissonLambda": [2, 4, 10],
    "kappa": [4, 4, 1],
    "matrixKappa": [9, 9, 1],
    "homotypicInhibition": [0.9, 0.9, 1],
    "heterotypicInhibition": [0, 0, 1],
    "polarityPersistence": [0.5, 2, 10],
    "polarityTurningCoupling": [2, 2, 1],
    "flowScaling": [3, 3, 1],
    "flowPolarityCoupling": [0.5, 8, 10],
    "polarityNoiseSigma": [0.01, 0.01, 1],
    "collisionRepolarisation": [0, 0, 1],
    "repolarisationRate": [0.8, 0.8, 1],
}


def main():
    # Generate the grid search variables:
    argument_grid = [
        (
            int(superIterationCount), int(numberOfCells), int(worldSize), int(gridSize),
            cellTypeProportions, matrixAdditionRate, matrixTurnoverRate,
            cellDepositionSigma, cellSensationSigma,
            poissonLambda, kappa, matrixKappa,
            homotypicInhibition, heterotypicInhibition,
            1 - (1/polarityPersistence), polarityTurningCoupling,
            flowScaling, flowPolarityCoupling, polarityNoiseSigma,
            collisionRepolarisation, repolarisationRate
        )
        for superIterationCount in np.linspace(*GRIDSEARCH["superIterationCount"])
        for numberOfCells in np.linspace(*GRIDSEARCH["numberOfCells"])
        for worldSize in np.linspace(*GRIDSEARCH["worldSize"])
        for gridSize in np.linspace(*GRIDSEARCH["gridSize"])
        for cellTypeProportions in np.linspace(*GRIDSEARCH["cellTypeProportions"])
        for matrixAdditionRate in np.linspace(*GRIDSEARCH["matrixAdditionRate"])
        for matrixTurnoverRate in np.linspace(*GRIDSEARCH["matrixTurnoverRate"])
        for cellDepositionSigma in np.linspace(*GRIDSEARCH["cellDepositionSigma"])
        for cellSensationSigma in np.linspace(*GRIDSEARCH["cellSensationSigma"])
        for poissonLambda in np.linspace(*GRIDSEARCH["poissonLambda"])
        for kappa in np.linspace(*GRIDSEARCH["kappa"])
        for matrixKappa in np.linspace(*GRIDSEARCH["matrixKappa"])
        for homotypicInhibition in np.linspace(*GRIDSEARCH["homotypicInhibition"])
        for heterotypicInhibition in np.linspace(*GRIDSEARCH["heterotypicInhibition"])
        for polarityPersistence in np.logspace(*GRIDSEARCH["polarityPersistence"], base=10)
        for polarityTurningCoupling in np.linspace(*GRIDSEARCH["polarityTurningCoupling"])
        for flowScaling in np.linspace(*GRIDSEARCH["flowScaling"])
        for flowPolarityCoupling in np.linspace(*GRIDSEARCH["flowPolarityCoupling"])
        for polarityNoiseSigma in np.linspace(*GRIDSEARCH["polarityNoiseSigma"])
        for collisionRepolarisation in np.linspace(*GRIDSEARCH["collisionRepolarisation"])
        for repolarisationRate in np.linspace(*GRIDSEARCH["repolarisationRate"])
    ]

    print(len(argument_grid))

    # Generating directory and output gridsearch to that directory:
    if not os.path.exists("fileOutputs"):
        os.mkdir("fileOutputs")
    output_filepath = os.path.join("fileOutputs", "gridsearch.txt")

    # Write the grid search variables to a file to be read by a job array on the cluster:
    with open(output_filepath, 'w') as f:
        # Writing column names:
        header_string = ""
        header_string += "array_id"
        for argument_name in GRIDSEARCH.keys():
            header_string += "\t"
            header_string += argument_name
        header_string += "\n"
        f.write(header_string)

        # Writing gridsearch values:
        for i, argtuple in enumerate(argument_grid):
            f.write(f"{i+1}")
            for argument_value in argtuple:
                f.write("\t")
                f.write(f"{argument_value}")
            f.write("\n")

    # Write up summary of gridsearch parameters:
    summary_filepath = os.path.join("fileOutputs", "gridsearch_summary.json")
    with open(summary_filepath, 'w') as writefile:
        json.dump(GRIDSEARCH, writefile, indent=4)

    return None


if __name__ == "__main__":
    main()
