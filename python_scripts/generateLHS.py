"""Generates range of parameters to sample using Latin Hypercube sampling."""
import argparse
import os
import json

import numpy as np

# Defines the parameter ranges, with [start, stop]:
NUM_SAMPLES = 10

# Ensures necessary parameters are integers:
INTEGER_PARAMETERS = [
    "superIterationCount",
    "timestepsToRun",
    "numberOfCells",
    "worldSize",
    "gridSize",
    "thereIsMatrixInteraction"
]

GRIDSEARCH_PARAMETERS = {
    "superIterationCount": [10, 10],
    "timestepsToRun": [576, 576],
    "numberOfCells": [200, 200],
    "worldSize": [2048, 2048],
    "gridSize": [64, 64],
    "matrixAdditionRate": [0.0, 0.2],
    "matrixTurnoverRate": [0.0, 0.2],
    "thereIsMatrixInteraction": [1, 1],
    "halfSatCellAngularConcentration": [0.5, 0.5],
    "maxCellAngularConcentration": [4, 4],
    "halfSatMeanActinFlow": [0.5, 0.5],
    "maxMeanActinFlow": [6, 6],
    "flowScaling": [0, 0.2],
    "polarityDiffusionRate": [0, 0.05],
    "actinAdvectionRate": [0.5, 1.5],
    "contactAdvectionRate": [0.5, 1.5],
    "halfSatMatrixAngularConcentration": [0.5, 0.5],
    "maxMatrixAngularConcentration": [4, 4],
    "cellBodyRadius": [60, 150],
    "eccentricity": [0, 0.9],
    "sharpness": [10, 10],
    "inhibitionStrength": [0.0, 1.0]
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


def main():
    # Generating random number generator for shuffling of variables:
    random_number_generator = np.random.default_rng(0)

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

    # Combining parameters across shuffled ranges:
    for simulation_index in range(NUM_SAMPLES):
        # Generating .json:
        argument_json = {}
        for parameter_name in GRIDSEARCH_PARAMETERS.keys():
            shuffled_parameter_range = shuffled_parameters[parameter_name]
            if parameter_name in INTEGER_PARAMETERS:
                argument_json[parameter_name] = int(shuffled_parameter_range[simulation_index])
            else:
                argument_json[parameter_name] = shuffled_parameter_range[simulation_index]
        argument_json["jobArrayID"] = simulation_index

        # Saving .json:
        output_subdirectory = os.path.join("fileOutputs", f"{simulation_index}")
        if not os.path.exists(output_subdirectory):
            os.mkdir(output_subdirectory)
        output_filepath = os.path.join(output_subdirectory, f"{simulation_index}_arguments.json")
        with open(output_filepath, 'w') as output:
            json.dump(argument_json, output, indent=4)

    # Generating summary file:
    argument_grid = []
    for simulation_index in range(NUM_SAMPLES):
        parameter_set = []
        for parameter_name in GRIDSEARCH_PARAMETERS.keys():
            shuffled_parameter = shuffled_parameters[parameter_name]
            parameter_set.append(shuffled_parameter[simulation_index])
        argument_grid.append(parameter_set)

    # Gridsearch output summary:
    gs_output_filepath = os.path.join("fileOutputs", "gridsearch.txt")

    # Write the grid search variables to file:
    with open(gs_output_filepath, 'w') as f:
        # Writing column names:
        header_string = ""
        header_string += "array_id"
        for argument_name in GRIDSEARCH_PARAMETERS.keys():
            header_string += "\t"
            header_string += argument_name
        header_string += "\n"
        f.write(header_string)

        # Writing gridsearch values:
        for i, argtuple in enumerate(argument_grid):
            f.write(f"{i}")
            for argument_value in argtuple:
                f.write("\t")
                f.write(f"{argument_value}")
            f.write("\n")

    return None


if __name__ == "__main__":
    main()
