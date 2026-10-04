"""Generates range of parameters to sample using low discrepancy Sobol sequences."""
import os
import json
import math
import argparse

import numpy as np

from datetime import datetime

# Define outputs folder:
OUTPUTS_FOLDER = "model_experiments"
MCMC_SOURCE_EXPERIMENT = "model_experiments/2026-09-16-collisions_shape"
MCMC_PARAM_DIMS = 12

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_config_path", required=True)
    return parser.parse_args()


class JSONOutputManager:
    """Generates JSON files, while keeping track of number of files generated."""

    def __init__(self, config_dictionary, experiment_folderpath):
        """Initiliase count of simulations."""
        self.simulation_counter = 0
        self.experiment_folderpath = experiment_folderpath
        self.constant_parameters = config_dictionary["constant_parameters"]
        self.gridsearch_parameters = config_dictionary["gridsearch_parameters"]

    def generate_json_configs(self, parameter_matrix):
        """Generate set of config files from parameter matrix."""
        for row_index in range(parameter_matrix.shape[0]):
            # Print progress:
            if (row_index + 1) % 1000 == 0:
                print(row_index + 1)

            # Populate simulation config file:
            argument_json = {}
            for parameter_name in self.constant_parameters.keys():
                argument_json[parameter_name] = self.constant_parameters[parameter_name]

            parameter_names = [name for name, _ in self.gridsearch_parameters]
            parameter_ranges = [p_range for _, p_range in self.gridsearch_parameters]
            for parameter_index in range(len(self.gridsearch_parameters)):
                # Get parameter info:
                parameter_name = parameter_names[parameter_index]
                min_value = parameter_ranges[parameter_index][0]
                max_value = parameter_ranges[parameter_index][1]

                # Get relevant numerical value:
                parameter_value = parameter_matrix[row_index, parameter_index]
                scaled_value = ((max_value - min_value) * parameter_value) + min_value
                if parameter_name == "numberOfCells":
                    argument_json[parameter_name] = int(scaled_value)
                else:
                    argument_json[parameter_name] = scaled_value
            simulation_id = self.simulation_counter
            argument_json["jobArrayID"] = simulation_id

            # Save config file:
            # --- Generate run data folder:
            run_data_folderpath = os.path.join(self.experiment_folderpath, "run_data")
            if not os.path.exists(run_data_folderpath):
                os.mkdir(run_data_folderpath)

            # --- Generate hashed hierarchy folder:
            hierarchy_folder_id = int(math.floor(simulation_id / 1000))
            hierarchy_directory = os.path.join(
                run_data_folderpath, str(hierarchy_folder_id)
            )
            if not os.path.exists(hierarchy_directory):
                os.mkdir(hierarchy_directory)

            # --- Generate output directory:
            output_subdirectory = os.path.join(hierarchy_directory, f"{simulation_id}")
            if not os.path.exists(output_subdirectory):
                os.mkdir(output_subdirectory)

            # --- Add to arguments .json:
            argument_json["outputFolder"] = output_subdirectory

            # --- Save arguments .json:
            output_filepath = os.path.join(output_subdirectory, f"{simulation_id}_arguments.json")
            with open(output_filepath, 'w') as output:
                json.dump(argument_json, output, indent=4)

            # Tick up total simulation count:
            self.simulation_counter += 1


def main():
    # Get command line arguments:
    arguments = parse_arguments()

    # Read in configuration .json file:
    with open(arguments.experiment_config_path) as config_fstream:
        config_dictionary = json.load(config_fstream)

    # Generate output folder:
    time_string = datetime.now().strftime("%Y-%m-%d-")
    experiment_folderpath = os.path.join(
        OUTPUTS_FOLDER, time_string + config_dictionary["experiment_name"]
    )
    if not os.path.exists(experiment_folderpath):
        os.mkdir(experiment_folderpath)

    # Load posterior:
    print("Loading samples...")
    mcmc_folderpath = os.path.join(MCMC_SOURCE_EXPERIMENT, "disc_cov_mcmc_results")
    wt_chain = np.load(os.path.join(mcmc_folderpath, "wt_mcmc_chain.npy"))
    rd_chain = np.load(os.path.join(mcmc_folderpath, "rd_mcmc_chain.npy"))

    # Subsample posterior:
    chain_length = wt_chain.shape[0]
    half_index = int(chain_length // 2)
    wt_posterior = wt_chain[half_index::128, :, 0, :].reshape(-1, MCMC_PARAM_DIMS)
    rd_posterior = rd_chain[half_index::128, :, 0, :].reshape(-1, MCMC_PARAM_DIMS)
    print(wt_posterior.shape)

    # # Apply adjustments:
    # base_intervention = np.load("model_experiments/2026-07-01-intervention-experiment/base_intervention.npz")["intervention"]
    # gee_intervention = np.load("model_experiments/2026-07-01-intervention-experiment/gee_intervention.npz")["intervention"]
    # base_intervention_mle = rd_mle * np.exp(base_intervention)
    # gee_intervention_mle = rd_mle * np.exp(gee_intervention)

    # Format sample matrix:
    sample_matrix = np.concatenate([wt_posterior, rd_posterior], axis=0)
    unit_coupling = 1
    unit_sample_rate = 1
    unit_cell_count = 0.35
    additional_parameters = np.array([unit_coupling, unit_sample_rate, unit_cell_count])
    additional_parameters = np.expand_dims(additional_parameters, axis=0)
    additional_parameters = np.repeat(additional_parameters, 2048, axis=0)
    print(additional_parameters.shape)
    sample_matrix = np.concatenate([sample_matrix, additional_parameters], axis=1)

    # Save sample matrix:
    sample_matrix_filepath = os.path.join(experiment_folderpath, "sample_matrix.npy")
    np.save(sample_matrix_filepath, sample_matrix)

    # Save gridsearch configuration space to folder:
    output_config_path = os.path.join(experiment_folderpath, "config.json")
    with open(output_config_path, 'w') as output:
        json.dump(config_dictionary, output, indent=4)

    # Output matrix as nested set of folders with .json files:
    print("Generating folder structure and writing samples to .json files...")
    output_manager = JSONOutputManager(config_dictionary, experiment_folderpath)
    output_manager.generate_json_configs(sample_matrix)
    return None


if __name__ == "__main__":
    main()
