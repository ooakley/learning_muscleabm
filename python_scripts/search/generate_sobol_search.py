"""Generates range of parameters to sample using low discrepancy Sobol sequences."""
import os
import json
import math
import argparse

import numpy as np

from datetime import datetime
from scipy.stats import qmc

# Define outputs folder:
OUTPUTS_FOLDER = "model_experiments"

# Design points used by the MCMC likelihood, and the name of the count parameter:
DEFAULT_DESIGN_COUNTS = [int(count) for count in np.linspace(50, 300, 11)]
COUNT_PARAMETER_NAME = "numberOfCells"

# Seeds for the scrambled Sobol' sequences. The curve gridsearch uses its own
# seed so its parameter sets do not coincide with those of the normal gridsearch:
GRIDSEARCH_SOBOL_SEED = 0
CURVE_SOBOL_SEED = 1


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_config_path", required=True)
    parser.add_argument(
        "--curve_gridsearch", action="store_true",
        help="Generate the curve gridsearch (every Sobol' parameter set simulated at every "
             "design count) instead of the normal gridsearch."
    )
    parser.add_argument(
        "--sobol_exponent", type=int, default=None,
        help="Generate 2^sobol_exponent Sobol' samples. Normal gridsearch: overrides "
             "'sample_exponent' in the config. Curve gridsearch: number of curves (required)."
    )
    parser.add_argument(
        "--design_counts", type=int, nargs="+", default=DEFAULT_DESIGN_COUNTS,
        help="Curve gridsearch only: cell counts to simulate for every curve."
    )
    arguments = parser.parse_args()

    # The config's exponent sizes the normal gridsearch; multiplying it by the number
    # of design counts would be far too large, so curves need an explicit exponent:
    if arguments.curve_gridsearch and arguments.sobol_exponent is None:
        parser.error("--curve_gridsearch requires --sobol_exponent.")
    return arguments


class JSONOutputManager:
    """Generates JSON files, while keeping track of number of files generated."""

    def __init__(self, config_dictionary, experiment_folderpath):
        """Initiliase count of simulations."""
        self.simulation_counter = 0
        self.experiment_folderpath = experiment_folderpath
        self.constant_parameters = config_dictionary["constant_parameters"]
        self.gridsearch_parameters = config_dictionary["gridsearch_parameters"]

    def generate_json_configs(self, parameter_matrix, exact_counts=None):
        """Generate set of config files from parameter matrix.

        If exact_counts is given (one integer per row), it is written as the
        cell count directly, instead of truncating the rescaled unit-cube value.
        """
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
                if parameter_name == COUNT_PARAMETER_NAME:
                    if exact_counts is not None:
                        argument_json[parameter_name] = int(exact_counts[row_index])
                    else:
                        argument_json[parameter_name] = int(scaled_value)
                else:
                    argument_json[parameter_name] = scaled_value
            simulation_id = self.simulation_counter
            argument_json["jobArrayID"] = simulation_id

            # Save config file:
            # --- Generate run data folder:
            run_data_folderpath = os.path.join(self.experiment_folderpath, "run_data")
            os.makedirs(run_data_folderpath, exist_ok=True)

            # --- Generate hashed hierarchy folder:
            hierarchy_folder_id = int(math.floor(simulation_id / 1000))
            hierarchy_directory = os.path.join(
                run_data_folderpath, str(hierarchy_folder_id)
            )
            os.makedirs(hierarchy_directory, exist_ok=True)

            # --- Generate output directory:
            output_subdirectory = os.path.join(hierarchy_directory, f"{simulation_id}")
            os.makedirs(output_subdirectory, exist_ok=True)

            # --- Add to arguments .json:
            argument_json["outputFolder"] = output_subdirectory

            # --- Save arguments .json:
            output_filepath = os.path.join(output_subdirectory, f"{simulation_id}_arguments.json")
            with open(output_filepath, 'w') as output:
                json.dump(argument_json, output, indent=4)

            # Tick up total simulation count:
            self.simulation_counter += 1


def write_design_point_curves(config_dictionary, experiment_folderpath, curve_parameters, design_counts):
    """Write one simulation per (parameter set, design count) into a sweep-style folder.

    curve_parameters: (n_curves, n_parameters - 1) unit-cube values, count column removed.
    design_counts:    integer cell counts simulated for every curve.

    Rows are ordered curve by curve, so simulation_id = curve_index * n_counts + count_index
    and any per-simulation output reshapes to (n_curves, n_counts, ...).
    """
    # Locate the count parameter and its range:
    gridsearch_parameters = config_dictionary["gridsearch_parameters"]
    parameter_names = [name for name, _ in gridsearch_parameters]
    count_index = parameter_names.index(COUNT_PARAMETER_NAME)
    count_min, count_max = gridsearch_parameters[count_index][1]

    # Validate inputs:
    curve_parameters = np.asarray(curve_parameters, dtype=float)
    design_counts = np.asarray(design_counts, dtype=int)
    if curve_parameters.ndim != 2 or curve_parameters.shape[1] != len(parameter_names) - 1:
        raise ValueError(
            f"curve_parameters must have shape (n_curves, {len(parameter_names) - 1}), "
            f"got {curve_parameters.shape}."
        )
    if np.any(curve_parameters < 0) or np.any(curve_parameters > 1):
        raise ValueError("curve_parameters must be unit-cube values in [0, 1].")
    if np.any(design_counts < count_min) or np.any(design_counts > count_max):
        raise ValueError(f"design_counts must lie within [{count_min}, {count_max}].")

    # Refuse to write over an existing sweep:
    sample_matrix_filepath = os.path.join(experiment_folderpath, "sample_matrix.npy")
    if os.path.exists(sample_matrix_filepath):
        raise FileExistsError(f"{experiment_folderpath} already contains a sample matrix.")
    os.makedirs(experiment_folderpath, exist_ok=True)

    # Build the full unit-cube matrix, re-inserting the count column:
    n_curves, n_counts = curve_parameters.shape[0], design_counts.shape[0]
    scaled_counts = (design_counts - count_min) / (count_max - count_min)
    sample_matrix = np.insert(
        np.repeat(curve_parameters, n_counts, axis=0),
        count_index, np.tile(scaled_counts, n_curves), axis=1
    )
    exact_counts = np.tile(design_counts, n_curves)

    # Save the matrix (same name as a sweep) and what is needed to rebuild the curves:
    np.save(sample_matrix_filepath, sample_matrix)
    np.save(os.path.join(experiment_folderpath, "curve_parameters.npy"), curve_parameters)
    np.save(os.path.join(experiment_folderpath, "design_counts.npy"), design_counts)
    with open(os.path.join(experiment_folderpath, "config.json"), 'w') as output:
        json.dump(config_dictionary, output, indent=4)

    # Write the folder structure and argument files:
    print(f"Writing {n_curves} curves x {n_counts} counts = {sample_matrix.shape[0]} simulations...")
    output_manager = JSONOutputManager(config_dictionary, experiment_folderpath)
    output_manager.generate_json_configs(sample_matrix, exact_counts=exact_counts)
    return sample_matrix


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

    parameter_count = len(config_dictionary["gridsearch_parameters"])

    # Curve gridsearch: Sobol' sample every parameter except the count, then
    # simulate each parameter set at every design count:
    if arguments.curve_gridsearch:
        print("Generating curve samples...")
        sobol_sampler = qmc.Sobol(d=parameter_count - 1, scramble=True, rng=CURVE_SOBOL_SEED)
        curve_parameters = sobol_sampler.random_base2(m=arguments.sobol_exponent)

        # Record how the curves were generated alongside the config:
        config_dictionary["curve_sobol_exponent"] = arguments.sobol_exponent
        config_dictionary["curve_design_counts"] = [int(count) for count in arguments.design_counts]
        write_design_point_curves(
            config_dictionary, experiment_folderpath + "_design_curves",
            curve_parameters, arguments.design_counts
        )
        return None

    # Normal gridsearch - the command line exponent takes precedence over the config:
    if arguments.sobol_exponent is not None:
        config_dictionary["sample_exponent"] = arguments.sobol_exponent

    if not os.path.exists(experiment_folderpath):
        os.mkdir(experiment_folderpath)

    # Generate random number generator for shuffling of variables:
    print("Generating samples...")
    sobol_sampler = qmc.Sobol(d=parameter_count, scramble=True, rng=GRIDSEARCH_SOBOL_SEED)
    sample_matrix = sobol_sampler.random_base2(m=config_dictionary["sample_exponent"])

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
