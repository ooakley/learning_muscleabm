import argparse
import json
import os
import math

import numpy as np


GRIDSEARCH_PARAMETERS = {
    "cueDiffusionRate": [0.0001, 2.5],
    "cueKa": [0.0001, 5],
    "fluctuationAmplitude": [1e-5, 1e-2],
    "fluctuationTimescale": [1, 75],
    "maximumSteadyStateActinFlow": [0.1, 3],
    "numberOfCells": [75, 175],
    "actinAdvectionRate": [0.1, 3],
    "cellBodyRadius": [15, 100],
    # Collisions:
    "collisionFlowReductionRate": [0.0, 1],
    "collisionAdvectionRate": [0.0, 1],
    # Shape:
    "stretchFactor": [0.25, 3],
    "slipFactor": [0.0001, 0.1],
}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process an outputs folder with a given name.')
    parser.add_argument('--folder_name', type=str, help='Folder name.')
    parser.add_argument('--output_name', type=str, help='Where to put output files.')
    args = parser.parse_args()
    return args


def main():
    # Read arguments:
    args = parse_arguments()

    # Ensure output directory exists:
    output_folder = f"out_{args.output_name}"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Get gridsearch parameter ranges:
    with open(f"{args.folder_name}/config.json", 'r') as file:
        config_dictionary = json.load(file)
    gridsearch_parameters = config_dictionary["gridsearch"]

    input_matrix = []
    for i in range(int(65536)):
        hierarchy_id = int(math.floor(i / 1000))
        json_filepath = f'./{args.folder_name}/{hierarchy_id}/{i}/{i}_arguments.json'
        with open(json_filepath) as json_data:
            # Reading data:
            parameter_dict = json.load(json_data)
            input_row = []
            for parameter_name in gridsearch_parameters.keys():
                input_row.append(parameter_dict[parameter_name])
            input_matrix.append(np.array(input_row))
    input_matrix = np.stack(input_matrix, axis=0)
    print(input_matrix.shape)
    np.save(f"./out_{args.output_name}/collated_inputs.npy", input_matrix)


if __name__ == "__main__":
    main()
