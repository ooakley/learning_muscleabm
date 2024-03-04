import os
import json
import numpy as np
import copy

IS_INTEGER_PARAMETER = [
    True,  # "superIterationCount"
    True,  # "numberOfCells"
    True,  # "worldSize"
    True,  # "gridSize"
    False, # "cellTypeProportions"
    False, # "matrixAdditionRate"
    False, # "matrixTurnoverRate"
    False, # "cellDepositionSigma"
    False, # "cellSensationSigma"
    False, # "poissonLambda"
    False, # "kappa"
    False, # "matrixKappa"
    False, # "homotypicInhibition"
    False, # "heterotypicInhibition"
    False, # "polarityPersistence"
    False, # "polarityTurningCoupling"
    False, # "flowScaling"
    False, # "flowPolarityCoupling"
    False, # "polarityNoiseSigma"
    False, # "collisionRepolarisation"
    False, # "repolarisationRate"
]

# Defines the parameter ranges, with [start, stop, num_steps]:
BASE_PARAMETERS = {
    "superIterationCount": 10,
    "numberOfCells": 70,
    "worldSize": 2048,
    "gridSize": 32,
    "cellTypeProportions": 0,
    "matrixAdditionRate": 0.5,
    "matrixTurnoverRate": 0.001,
    "cellDepositionSigma": 80,
    "cellSensationSigma": 120,
    "poissonLambda": 4,
    "kappa": 0,
    "matrixKappa": 4,
    "homotypicInhibition": 0.9,
    "heterotypicInhibition": 0,
    "polarityPersistence": 0.975,
    "polarityTurningCoupling": 4,
    "flowScaling": 3,
    "flowPolarityCoupling": 3.5,
    "polarityNoiseSigma": 0.01,
    "collisionRepolarisation": 0,
    "repolarisationRate": 0.8,
}

PARAMETER_MODIFIERS = {
    "numberOfCells": [-50, 130, 50],
    "matrixAdditionRate": [-0.45, 0.2, 50],
    "matrixTurnoverRate": [0.001, 0.049, 50],
    "cellDepositionSigma": [-20, 80, 50],
    "cellSensationSigma": [-40, 60, 50],
    "poissonLambda": [-3, 4, 50],
    "kappa": [0.1, 1, 50],
    "matrixKappa": [-3.5, 4, 50],
    "homotypicInhibition": [-0.5, 0.09, 50],
    "polarityPersistence": [-0.2, 0.015, 50],
    "polarityTurningCoupling": [-3.5, 4, 50],
    "flowScaling": [-2, 3, 50],
    "flowPolarityCoupling": [-2, 2.5, 50],
    "collisionRepolarisation": [0.01, 0.5, 50],
    "repolarisationRate": [-0.5, 0.19, 50],
}

def main():

    argument_grid = []

    # Adding initial tuple:
    argument_grid.append(tuple(BASE_PARAMETERS.values()))

    # Adding modifiers:
    for argument_name, modifier_list in PARAMETER_MODIFIERS.items():
        modifier_array = np.linspace(*modifier_list)
        for modifier in modifier_array:
            if modifier == 0:
                continue
            base_copy = copy.deepcopy(BASE_PARAMETERS)
            if argument_name == "numberOfCells":
                base_copy[argument_name] += int(modifier)
            else:
                base_copy[argument_name] += modifier
            argument_grid.append(tuple(base_copy.values()))

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
        for argument_name in BASE_PARAMETERS.keys():
            header_string += "\t"
            header_string += argument_name
        header_string += "\n"
        f.write(header_string)

        # Writing gridsearch values:
        for i, argtuple in enumerate(argument_grid):
            f.write(f"{i+1}")
            for j, argument_value in enumerate(argtuple):
                f.write("\t")
                if IS_INTEGER_PARAMETER[j]:
                    f.write(f"{argument_value}")
                else:
                    f.write(f"{argument_value:.8f}")
            f.write("\n")

    # Write up summary of gridsearch parameters:
    baseparams_filepath = os.path.join("fileOutputs", "base_parameters_summary.json")
    modifiers_filepath = os.path.join("fileOutputs", "modifiers_summary.json")

    with open(baseparams_filepath, 'w') as writefile:
        json.dump(BASE_PARAMETERS, writefile, indent=4)

    with open(modifiers_filepath, 'w') as writefile:
        json.dump(PARAMETER_MODIFIERS, writefile, indent=4)

    return None


if __name__ == "__main__":
    main()
