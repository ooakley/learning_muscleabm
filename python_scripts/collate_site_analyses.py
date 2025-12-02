import argparse
import math
import os
import json

import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process an outputs folder with a given name.')
    parser.add_argument('--experiment_folderpath', type=str)
    args = parser.parse_args()
    return args


def collate_data(experiment_folderpath, summarise_filename):
    # Code for discontinuous writing to numpy file taken & modified from:
    # https://stackoverflow.com/questions/65882709/how-to-write-ndarray-to-npy-file-iteratively-with-batches
    print(f"Collating data from {summarise_filename}...")

    # Get sample count from config file:
    with open(os.path.join(experiment_folderpath, "config.json")) as filestream:
        config_dict = json.load(filestream)
    sample_count = 2**config_dict["sample_exponent"]

    # Loop through all parameter sets:
    out_data = []
    for folder_id in range(sample_count):
        if (folder_id + 1) % 1000 == 0:
            print(folder_id + 1)
        try:
            hierarchy_id = int(math.floor(folder_id / 1000))
            id_filepath = os.path.join(
                experiment_folderpath, "run_data", str(hierarchy_id), str(folder_id), summarise_filename
            )
            superiteration_values = np.load(id_filepath)
            if superiteration_values.shape != (12,):  # (CELL TYPE) X (SS DISTANCE)
                print(f"Wrong shape found at {folder_id}, appending NaN...")
                print(f"Shape: {superiteration_values.shape}")
                blank_data = np.array([np.nan] * 12)
                out_data.append(blank_data)
            else:
                out_data.append(superiteration_values)
        except FileNotFoundError:
            print(f"No data file found at {folder_id}, appending NaN...")
            blank_data = np.array([np.nan] * 12)
            out_data.append(blank_data)

    out_data = np.stack(out_data, axis=0)
    out_filepath = os.path.join(experiment_folderpath, "summary_data", summarise_filename)
    np.save(out_filepath, out_data)


def main():
    # Parse arguments:
    args = parse_arguments()

    # Generate relevant directory if not present:
    summary_directory = os.path.join(args.experiment_folderpath, "summary_data")
    if not os.path.exists(summary_directory):
        os.mkdir(summary_directory)

    # Collate individual simulation data into set of comprehensive numpy arrays:
    collate_data(args.experiment_folderpath, "ann_indices.npy")
    collate_data(args.experiment_folderpath, "coherency_fractions.npy")


if __name__ == "__main__":
    main()
