"""Generates range of parameters to sample using low discrepancy Sobol sequences."""
import os
import json

from scipy.stats import qmc

# Defines the parameter ranges, with [start, stop]:
NUM_SAMPLES_EXPONENT = 3
SIMULATION_INDEX = 0

CONSTANT_PARAMETERS = {
    "superIterationCount": 12,
    "timestepsToRun": 2880,
    "worldSize": 2048,
    "gridSize": 64,
    "dt": 1,
    "thereIsMatrixInteraction": 1,
    "matrixTurnoverRate": 0.05,
    "matrixAdditionRate": 0.05,
    "matrixAdvectionRate": 0.0,
    "aspectRatio": 1,
    "stretchFactor": 7.015735718887299,
    "slipFactor": 0.01
    # # Collision parameters:
    # "collisionFlowReductionRate": 0,
    # "collisionAdvectionRate": 0,
}

GRIDSEARCH_PARAMETERS = {
    "cueDiffusionRate": [0.0001, 2.5],
    "cueKa": [0.0001, 5],
    "fluctuationAmplitude": [5e-5, 5e-3],
    "fluctuationTimescale": [1, 250],
    "maximumSteadyStateActinFlow": [0.0, 4.5],
    "numberOfCells": [75, 150],
    "actinAdvectionRate": [0, 5],
    "cellBodyRadius": [50, 125],
    # Collisions:
    "collisionFlowReductionRate": [0.0, 1],
    "collisionAdvectionRate": [0.0, 4.5],
}


class JSONOutputManager:
    """Generates JSON files, while keeping track of number of files generated."""

    def __init__(self):
        """Initiliase count of simulations."""
        self.simulation_counter = 0

    def generate_json_configs(self, parameter_matrix):
        """Generate set of config files from parameter matrix."""
        for row_index in range(parameter_matrix.shape[0]):
            # Print progress:
            if (row_index + 1)% 1000 == 0:
                print(row_index + 1)

            # Populate simulation config file:
            argument_json = {}
            for parameter_name in CONSTANT_PARAMETERS.keys():
                argument_json[parameter_name] = CONSTANT_PARAMETERS[parameter_name]
            for parameter_index, parameter_name in enumerate(GRIDSEARCH_PARAMETERS.keys()):
                min_value = GRIDSEARCH_PARAMETERS[parameter_name][0]
                max_value = GRIDSEARCH_PARAMETERS[parameter_name][1]
                parameter_value = parameter_matrix[row_index, parameter_index]
                scaled_value = ((max_value - min_value) * parameter_value) + min_value
                if parameter_name == "numberOfCells":
                    argument_json[parameter_name] = int(scaled_value)
                else:
                    argument_json[parameter_name] = scaled_value
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


def main():
    # Generate random number generator for shuffling of variables:
    print("Generating samples...")
    sobol_sampler = qmc.Sobol(d=len(GRIDSEARCH_PARAMETERS), scramble=True, rng=0)
    sample_matrix = sobol_sampler.random_base2(m=NUM_SAMPLES_EXPONENT)

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
    output_manager.generate_json_configs(sample_matrix)

    return None


if __name__ == "__main__":
    main()
