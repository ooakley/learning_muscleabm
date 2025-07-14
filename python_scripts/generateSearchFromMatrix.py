"""Generates range of parameters to sample using low discrepancy Sobol sequences."""
import os
import json
import argparse
import numpy as np

from scipy.stats import qmc

# Defines the parameter ranges, with [start, stop]:
NUM_SAMPLES_EXPONENT = 14
SIMULATION_INDEX = 0

CONSTANT_PARAMETERS = {
    "superIterationCount": 50,
    "timestepsToRun": 1440,
    "worldSize": 2048,
    "gridSize": 64,
    "dt": 1,
    "thereIsMatrixInteraction": 1,
    "matrixTurnoverRate": 0.05,
    "matrixAdditionRate": 0.05,
    "matrixAdvectionRate": 0.0,
    "cellBodyRadius": 75,
    "aspectRatio": 1,
    # # Collision parameters:
    # "collisionFlowReductionRate": 0,
    # "collisionAdvectionRate": 0,
    # "numberOfCells": 150,
}

GRIDSEARCH_PARAMETERS = {
    "cueDiffusionRate": [0.005, 0.5],
    "cueKa": [0.25, 1.25],
    "fluctuationAmplitude": [0.01, 0.2],
    "fluctuationTimescale": [1.5, 20],
    "actinAdvectionRate": [0.01, 1],
    "maximumSteadyStateActinFlow": [0.25, 5],
    # Collisions:
    "collisionFlowReductionRate": [0.0, 1.0],
    "collisionAdvectionRate": [0.0, 1.0],
    "numberOfCells": [50, 300],
}


class JSONOutputManager:
    """Generates JSON files, while keeping track of number of files generated."""

    def __init__(self):
        """Initiliase count of simulations."""
        self.simulation_counter = 0

    def generate_json_configs(self, parameter_matrix):
        """Generate set of config files from parameter matrix."""
        for row_index in range(parameter_matrix.shape[0]):
            # Populate simulation config file:
            argument_json = {}
            for parameter_name in CONSTANT_PARAMETERS.keys():
                argument_json[parameter_name] = CONSTANT_PARAMETERS[parameter_name]
            for parameter_index, parameter_name in enumerate(GRIDSEARCH_PARAMETERS.keys()):
                parameter_value = parameter_matrix[row_index, parameter_index]
                if parameter_name == "numberOfCells":
                    parameter_value = int(parameter_value)
                argument_json[parameter_name] = parameter_value
            simulation_id = self.simulation_counter
            argument_json["jobArrayID"] = simulation_id

            # Save config file:
            output_subdirectory = os.path.join("fileOutputs", f"{simulation_id}")
            if not os.path.exists(output_subdirectory):
                os.mkdir(output_subdirectory)
            output_filepath = os.path.join(output_subdirectory, f"{simulation_id}_arguments.json")
            with open(output_filepath, 'w') as output:
                json.dump(argument_json, output, indent=4)

            # Tick up total simulation count:
            self.simulation_counter += 1


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a folder with a given integer name.')
    parser.add_argument('--input_matrix', type=str, help='Matrix corresponding to parameter dimensions')
    args = parser.parse_args()
    return args


def main():
    # Generate random number generator for shuffling of variables:
    print("Retrieving samples...")
    args = parse_arguments()
    input_matrix = np.load(args.input_matrix)

    # Ensure output folder exists:
    if not os.path.exists("fileOutputs"):
        os.mkdir("fileOutputs")

    # Save gridsearch configuration space to folder:
    gridsearch_config_path = os.path.join("fileOutputs", "config.json")
    with open(gridsearch_config_path, 'w') as output:
        config_dict = {
            "constant": CONSTANT_PARAMETERS,
            "gridsearch": GRIDSEARCH_PARAMETERS
        }
        json.dump(config_dict, output, indent=4)

    # Output matrix as nested set of folders with .json files:
    print("Generating folder structure and writing samples to .json files...")
    output_manager = JSONOutputManager()
    output_manager.generate_json_configs(input_matrix)

    return None


if __name__ == "__main__":
    main()
