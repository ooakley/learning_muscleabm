"""Generates range of parameters to sample using low discrepancy Sobol sequences."""
import os
import json
import argparse

import numpy as np

from datetime import datetime

from muscleabm.sampling import JSONOutputManager

# Define outputs folder:
OUTPUTS_FOLDER = "model_experiments"
MCMC_PARAM_DIMS = 12


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_config_path", required=True)
    parser.add_argument(
        "--mcmc_dirpath", required=True,
        help="Folder containing the WT and RD MCMC chains to sample from, e.g. "
             "model_experiments/2026-09-16-collisions_shape/disc_cov_mcmc_results."
    )
    return parser.parse_args()


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
    mcmc_folderpath = arguments.mcmc_dirpath
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
