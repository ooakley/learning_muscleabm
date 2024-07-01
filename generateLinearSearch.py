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
    "superIterationCount": 15,
    "numberOfCells": 70,
    "worldSize": 2048,
    "gridSize": 32,
    "cellTypeProportions": 0.0,
    "matrixAdditionRate": 0.1652060824329732,
    "matrixTurnoverRate": 0.0006956782713085,
    "cellDepositionSigma": 71.00240096038415,
    "cellSensationSigma": 71.95878351340536,
    "poissonLambda": 3.1328384254862405,
    "kappa": 0.1270508203281312,
    "matrixKappa": 3.1368547418967587,
    "homotypicInhibition": 0.8974189675870348,
    "heterotypicInhibition": 0.0,
    "polarityPersistence": 0.5715686274509804,
    "polarityTurningCoupling": 0.921968787515006,
    "flowScaling": 2.6746698679471788,
    "flowPolarityCoupling": 0.751759821912039,
    "polarityNoiseSigma": 0.01,
    "collisionRepolarisation": 0.0,
    "repolarisationRate": 0.4715686274509804
}

PARAMETER_MODIFIERS = {
    # "numberOfCells": [-20, 20, 50],
    "matrixAdditionRate": [-BASE_PARAMETERS["matrixAdditionRate"], 0.4, 50],
    "matrixTurnoverRate": [-BASE_PARAMETERS["matrixTurnoverRate"], 0.05, 50],
    "cellDepositionSigma": [-20, 20, 50],
    "cellSensationSigma": [-20, 20, 50],
    "poissonLambda": [-2, 3, 50],
    "kappa": [-BASE_PARAMETERS["kappa"], 2, 50],
    "matrixKappa": [-BASE_PARAMETERS["matrixKappa"], 4, 50],
    "homotypicInhibition": [-BASE_PARAMETERS["homotypicInhibition"], 0.05, 50],
    "polarityPersistence": [-0.3, 0.4, 50],
    "polarityTurningCoupling": [-BASE_PARAMETERS["polarityTurningCoupling"], 5, 50],
    "flowScaling": [-2, 2, 50],
    "flowPolarityCoupling": [-0.5, 4, 50],
    "collisionRepolarisation": [0.05, 0.25, 50],
    "repolarisationRate": [-0.3, 0.5, 50],
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
    # baseparams_filepath = os.path.join("fileOutputs", "base_parameters_summary.json")
    baseparams_filepath = os.path.join("./", "base_parameters_summary.json")
    # modifiers_filepath = os.path.join("fileOutputs", "modifiers_summary.json")
    modifiers_filepath = os.path.join("./", "modifiers_summary.json")

    with open(baseparams_filepath, 'w') as writefile:
        json.dump(BASE_PARAMETERS, writefile, indent=4)

    with open(modifiers_filepath, 'w') as writefile:
        json.dump(PARAMETER_MODIFIERS, writefile, indent=4)

    return None


if __name__ == "__main__":
    main()
