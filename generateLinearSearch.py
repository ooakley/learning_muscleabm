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
    "matrixAdditionRate": 0.235317,
    "matrixTurnoverRate": 0.000515,
    "cellDepositionSigma": 66.614323,
    "cellSensationSigma": 72.169434,
    "poissonLambda": 3.495199,
    "kappa": 0.274455,
    "matrixKappa": 3.44689,
    "homotypicInhibition": 0.921284,
    "heterotypicInhibition": 0,
    "polarityPersistence": 0.716525,
    "polarityTurningCoupling": 0.288058,
    "flowScaling": 2.168234,
    "flowPolarityCoupling": 3.340568,
    "polarityNoiseSigma": 0.01,
    "collisionRepolarisation": 0,
    "repolarisationRate": 0.684087,
}

PARAMETER_MODIFIERS = {
    # "numberOfCells": [-20, 20, 50],
    "matrixAdditionRate": [-0.2, 0.4, 50],
    "matrixTurnoverRate": [-5e-4, 0.02, 50],
    "cellDepositionSigma": [-20, 60, 50],
    "cellSensationSigma": [-20, 60, 50],
    "poissonLambda": [-2, 2, 50],
    "kappa": [-0.2, 2, 50],
    "matrixKappa": [-3, 4, 50],
    "homotypicInhibition": [-0.5, 0.05, 50],
    "polarityPersistence": [-0.2, 0.28, 50],
    "polarityTurningCoupling": [-0.25, 6, 50],
    "flowScaling": [-0.5, 0.5, 50],
    "flowPolarityCoupling": [-2, 2, 50],
    "collisionRepolarisation": [0.05, 0.5, 50],
    "repolarisationRate": [-0.6, 0.3, 50],
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
