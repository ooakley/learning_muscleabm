"""
Generates range of parameters to sample using Latin Hypercube sampling.

Sensitivity analyses for function of simulation will be required to call this function
several times.
"""
import os
import json
import numpy as np

# Defines sampling size of LHS method:
NUM_SAMPLES = 2000

# Ensures necessary parameters are integers:
INTEGER_PARAMETERS = [
    "superIterationCount",
    "numberOfCells",
    "worldSize",
    "gridSize"
]

# Defines the parameter ranges, with [start, stop]:
GRIDSEARCH = {
    "superIterationCount": [10, 10],
    "numberOfCells": [70, 70],
    "worldSize": [2048, 2048],
    "gridSize": [32, 32],
    "cellTypeProportions": [0, 0],
    "matrixAdditionRate": [0.01, 0.2],
    "matrixTurnoverRate": [0.0001, 0.01],
    "cellDepositionSigma": [60, 100],
    "cellSensationSigma": [60, 100],
    "poissonLambda": [1, 10],
    "kappa": [0, 4],
    "matrixKappa": [0, 6],
    "homotypicInhibition": [0.9, 0.9],
    "heterotypicInhibition": [0, 0],
    "polarityPersistence": [0.5, 1],
    "polarityTurningCoupling": [0, 6],
    "flowScaling": [2, 4],
    "flowPolarityCoupling": [0.5, 5],
    "polarityNoiseSigma": [0.01, 0.01],
    "collisionRepolarisation": [0, 0],
    "repolarisationRate": [0.6, 0.95],
}

REPARAMETERISATIONS = {
    "reparameterisedSmallT": [0.2, 2.5],
    "reparameterisationBeta": [1, 7.5]
}


def rand_seq(parameter_range, rng, is_int=False):
    parameter_array = np.linspace(*parameter_range, NUM_SAMPLES)
    if is_int:
        parameter_array = parameter_array.astype(int)
    rng.shuffle(parameter_array)
    return parameter_array


def main():
    # Generating random number generator for shuffling of variables:
    rng = np.random.default_rng(0)

    shuffled_parameters = {}
    for parameter in GRIDSEARCH.keys():
        shuffled_parameter_set = rand_seq(
            GRIDSEARCH[parameter], rng, parameter in INTEGER_PARAMETERS
        )
        shuffled_parameters[parameter] = shuffled_parameter_set

    # Overwriting poissonLamda and flowPolarityCoupling for reparameterisation:
    rand_t = rand_seq(REPARAMETERISATIONS["reparameterisedSmallT"], rng)
    rand_beta = rand_seq(REPARAMETERISATIONS["reparameterisationBeta"], rng)
    shuffled_parameters["poissonLambda"] = rand_beta * rand_t
    shuffled_parameters["flowPolarityCoupling"] = 1 / rand_t

    print("-- --- -- --- --")
    print(shuffled_parameters["poissonLambda"])
    print("-- --- -- --- --")
    print(shuffled_parameters["flowPolarityCoupling"])

    argument_grid = []
    for simulation_index in range(NUM_SAMPLES):
        parameter_set = []
        for parameter_name in GRIDSEARCH.keys():
            shuffled_parameter = shuffled_parameters[parameter_name]
            parameter_set.append(shuffled_parameter[simulation_index])
        argument_grid.append(parameter_set)

    # # Generate the grid search variables:
    # argument_grid = [
    #     (
    #         int(superIterationCount), int(numberOfCells), int(worldSize), int(gridSize),
    #         cellTypeProportions, matrixAdditionRate, matrixTurnoverRate,
    #         cellDepositionSigma, cellSensationSigma,
    #         poissonLambda, kappa, matrixKappa,
    #         homotypicInhibition, heterotypicInhibition,
    #         polarityPersistence, polarityTurningCoupling,
    #         flowScaling, flowPolarityCoupling, polarityNoiseSigma,
    #         collisionRepolarisation, repolarisationRate
    #     )
    #     for superIterationCount in rand_seq("superIterationCount", rng)
    #     for numberOfCells in rand_seq("numberOfCells", rng)
    #     for worldSize in rand_seq("worldSize", rng)
    #     for gridSize in rand_seq("gridSize", rng)
    #     for cellTypeProportions in rand_seq("cellTypeProportions", rng)
    #     for matrixAdditionRate in rand_seq("matrixAdditionRate", rng)
    #     for matrixTurnoverRate in rand_seq("matrixTurnoverRate", rng)
    #     for cellDepositionSigma in rand_seq("cellDepositionSigma", rng)
    #     for cellSensationSigma in rand_seq("cellSensationSigma", rng)
    #     for poissonLambda in rand_seq("poissonLambda", rng)
    #     for kappa in rand_seq("kappa", rng)
    #     for matrixKappa in rand_seq("matrixKappa", rng)
    #     for homotypicInhibition in rand_seq("homotypicInhibition", rng)
    #     for heterotypicInhibition in rand_seq("heterotypicInhibition", rng)
    #     for polarityPersistence in rand_seq("polarityPersistence", rng)
    #     for polarityTurningCoupling in rand_seq("polarityTurningCoupling", rng)
    #     for flowScaling in rand_seq("flowScaling", rng)
    #     for flowPolarityCoupling in rand_seq("flowPolarityCoupling", rng)
    #     for polarityNoiseSigma in rand_seq("polarityNoiseSigma", rng)
    #     for collisionRepolarisation in rand_seq("collisionRepolarisation", rng)
    #     for repolarisationRate in rand_seq("repolarisationRate", rng)
    # ]

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
            f.write(f"{i + 1}")
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
