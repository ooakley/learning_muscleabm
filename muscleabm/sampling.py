"""Writing simulation argument files for parameter sweeps."""
import os
import json
import math

# Name of the count parameter:
COUNT_PARAMETER_NAME = "numberOfCells"


class JSONOutputManager:
    """Generates JSON files, while keeping track of number of files generated."""

    def __init__(self, config_dictionary, output_folderpath):
        """Initiliase count of simulations.

        output_folderpath is the folder the run_data folder is written to, e.g. a wave folder.
        """
        self.simulation_counter = 0
        self.output_folderpath = output_folderpath
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
            run_data_folderpath = os.path.join(self.output_folderpath, "run_data")
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
