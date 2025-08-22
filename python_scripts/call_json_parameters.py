"""A script to run the simulation based on a set of parameters in a .json file."""
# flake8: noqa: E501
# ^ disabling line length linting warnings.
import time
import argparse
import os
import json


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run simulation using given .json config file.')
    parser.add_argument('--path_to_config', type=str, help='Path to config file.')
    args = parser.parse_args()
    return args


def main():
    start_time = time.time()
    args = parse_arguments()

    with open(args.path_to_config, 'r') as file:
        parameter_dictionary = json.load(file)

    # TODO: just build the bash comand using the keys from the parameter dict

    full_command = "./build/src/main "
    for key, value in parameter_dictionary.items():
        argument_string = f"--{key} {value} "
        full_command += argument_string

    print(full_command)
    os.system(full_command)

    text_filepath = os.path.join(os.path.dirname(args.path_to_config), "sim_time.txt")
    with open(text_filepath, 'w') as output:
        time_taken = str(time.time() - start_time)
        output.write(time_taken)

    return None


if __name__ == '__main__':
    main()
