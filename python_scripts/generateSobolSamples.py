"""Generates range of parameters to sample using Latin Hypercube sampling."""
import argparse
import os
import json
import copy

import numpy as np

# Defines the parameter ranges, with [start, stop]:
NUM_SAMPLES = 1000
SIMULATION_INDEX = 0

# All parameter names:
PARAMETER_NAMES = [
    "superIterationCount",
    "timestepsToRun",
    "numberOfCells",
    "worldSize",
    "gridSize",
    "matrixAdditionRate",
    "matrixTurnoverRate",
    "thereIsMatrixInteraction",
    "halfSatCellAngularConcentration",
    "maxCellAngularConcentration",
    "halfSatMeanActinFlow",
    "maxMeanActinFlow",
    "flowScaling",
    "polarityDiffusionRate",
    "actinAdvectionRate",
    "contactAdvectionRate",
    "halfSatMatrixAngularConcentration",
    "maxMatrixAngularConcentration",
    "cellBodyRadius",
    "eccentricity",
    "sharpness",
    "inhibitionStrength"
]

# Ensures necessary parameters are integers:
INTEGER_PARAMETERS = [
    "superIterationCount",
    "timestepsToRun",
    "numberOfCells",
    "worldSize",
    "gridSize",
    "thereIsMatrixInteraction"
]

CONSTANT_PARAMETERS = {
    "superIterationCount": 10,
    "timestepsToRun": 576,
    "numberOfCells": 150,
    "worldSize": 2048,
    "gridSize": 64,
    "thereIsMatrixInteraction": 1,
    "halfSatCellAngularConcentration": 0.5,
    "halfSatMeanActinFlow": 0.5,
    "halfSatMatrixAngularConcentration": 0.5,
    "maxMeanActinFlow": 6,
    "sharpness": 10,
}

GRIDSEARCH_PARAMETERS = {
    "matrixAdditionRate": [0.0, 0.2],
    "matrixTurnoverRate": [0.0, 0.2],
    "flowScaling": [0, 0.2],
    "cellBodyRadius": [60, 200],
    "eccentricity": [0, 0.98],
    "polarityDiffusionRate": [0, 0.1],
    "actinAdvectionRate": [0., 2.],
    "contactAdvectionRate": [0., 2.],
    "inhibitionStrength": [0.0, 1.0],
    "maxCellAngularConcentration": [1.5, 5],
    "maxMatrixAngularConcentration": [1.5, 5],
}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run simulation using given .json config file.')
    parser.add_argument('--path_to_config', type=str, help='Path to config file.')
    args = parser.parse_args()
    return args


def rand_seq(parameter_range, rng):
    parameter_array = np.linspace(*parameter_range, NUM_SAMPLES)
    rng.shuffle(parameter_array)
    return parameter_array


def generate_LHS_matrix(rng):
    # Generate unit hypercube sample:
    parameter_matrix = []
    for parameter_index in range(len(GRIDSEARCH_PARAMETERS)):
        # Generate columns of parameter matrix:
        # parameter_array = rng.uniform(size=(NUM_SAMPLES))
        parameter_array = np.linspace(0, 1, NUM_SAMPLES)
        rng.shuffle(parameter_array)
        parameter_matrix.append(parameter_array)
    # Stack along columns:
    parameter_matrix = np.stack(parameter_matrix, axis=1)
    return parameter_matrix


class JSONOutputManager:
    """Generates JSON files, while keeping track of number of files generated."""

    def __init__(self):
        """Initiliase count of simulations."""
        self.simulation_counter = 0

    def generate_json_configs(self, parameter_matrix):
        """Generate set of config files from parameter matrix."""
        for row in range(parameter_matrix.shape[0]):
            # Populating simulation config file:
            argument_json = {}
            for parameter_name in CONSTANT_PARAMETERS.keys():
                argument_json[parameter_name] = CONSTANT_PARAMETERS[parameter_name]
            for parameter_index, parameter_name in enumerate(GRIDSEARCH_PARAMETERS.keys()):
                min_value = GRIDSEARCH_PARAMETERS[parameter_name][0]
                max_value = GRIDSEARCH_PARAMETERS[parameter_name][1]
                parameter_value = parameter_matrix[row, parameter_index]
                scaled_value = ((max_value - min_value) * parameter_value) + min_value
                argument_json[parameter_name] = scaled_value
            simulation_id = self.simulation_counter
            argument_json["jobArrayID"] = simulation_id
            # Saving config file:
            output_subdirectory = os.path.join("fileOutputs", f"{simulation_id}")
            if not os.path.exists(output_subdirectory):
                os.mkdir(output_subdirectory)
            output_filepath = os.path.join(output_subdirectory, f"{simulation_id}_arguments.json")
            with open(output_filepath, 'w') as output:
                json.dump(argument_json, output, indent=4)
            # Ticking up total simulation count:
            self.simulation_counter += 1


def main():
    # Generating random number generator for shuffling of variables:
    random_number_generator = np.random.default_rng(0)

    # Generating the two parameter matrices:
    parameters_A = generate_LHS_matrix(random_number_generator)
    parameters_B = generate_LHS_matrix(random_number_generator)

    # Generating the combined parameter matrix:
    parameter_matrices = []
    for parameter_index in range(len(GRIDSEARCH_PARAMETERS)):
        parameters_ABi = copy.deepcopy(parameters_A)
        parameters_ABi[:, parameter_index] = parameters_B[:, parameter_index]
        parameter_matrices.append(parameters_ABi)

    # Shuffling individual parameter ranges:
    shuffled_parameters = {}
    for parameter in GRIDSEARCH_PARAMETERS.keys():
        shuffled_parameter_set = rand_seq(
            GRIDSEARCH_PARAMETERS[parameter],
            random_number_generator
        )
        shuffled_parameters[parameter] = shuffled_parameter_set

    # Ensuring output folder exists:
    if not os.path.exists("fileOutputs"):
        os.mkdir("fileOutputs")

    output_manager = JSONOutputManager()
    output_manager.generate_json_configs(parameters_A)
    output_manager.generate_json_configs(parameters_B)
    for parameter_matrix in parameter_matrices:
        output_manager.generate_json_configs(parameter_matrix)

    return None


if __name__ == "__main__":
    main()
