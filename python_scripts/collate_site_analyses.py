import argparse
import math
import os
import json

import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process an outputs folder with a given name.')
    parser.add_argument('--experiment_folderpath', type=str)
    parser.add_argument('--collation_target', type=str)
    parser.add_argument('--sample_count', type=int, default=None)
    args = parser.parse_args()
    return args


def collate_data(experiment_folderpath, summarise_filename, sample_count):
    # Code for discontinuous writing to numpy file taken & modified from:
    # https://stackoverflow.com/questions/65882709/how-to-write-ndarray-to-npy-file-iteratively-with-batches
    print(f"Collating data from {summarise_filename}...", flush=True)

    # Get sample count from config file:
    with open(os.path.join(experiment_folderpath, "config.json")) as filestream:
        config_dict = json.load(filestream)
    superiteration_count = config_dict["constant_parameters"]["superIterationCount"]

    # Loop through all parameter sets:
    out_data = []
    for folder_id in range(sample_count):
        if (folder_id + 1) % 1000 == 0:
            print(folder_id + 1, flush=True)
        try:
            hierarchy_id = int(math.floor(folder_id / 1000))
            id_filepath = os.path.join(
                experiment_folderpath, "run_data", str(hierarchy_id), str(folder_id), summarise_filename
            )
            superiteration_values = np.load(id_filepath)
            if superiteration_values.shape != (superiteration_count,):
                print(f"{summarise_filename}: Wrong shape found at {folder_id}, appending NaN...", flush=True)
                print(f"{summarise_filename}: Shape: {superiteration_values.shape}", flush=True)
                blank_data = np.array([np.nan] * superiteration_count)
                out_data.append(blank_data)
            else:
                out_data.append(superiteration_values)
        except (FileNotFoundError, EOFError):
            print(f"{summarise_filename}: No data file found at {folder_id}, appending NaN...", flush=True)
            blank_data = np.array([np.nan] * superiteration_count)
            out_data.append(blank_data)

    out_data = np.stack(out_data, axis=0)
    out_filepath = os.path.join(experiment_folderpath, "summary_data", summarise_filename)
    np.save(out_filepath, out_data)


def main():
    # Parse arguments:
    args = parse_arguments()
    with open(os.path.join(args.experiment_folderpath, "config.json")) as filestream:
        config_dict = json.load(filestream)
    if args.sample_count is None:
        sample_count = 2**config_dict["sample_exponent"]
    else:
        sample_count = args.sample_count

    # Generate relevant directory if not present:
    summary_directory = os.path.join(args.experiment_folderpath, "summary_data")
    if not os.path.exists(summary_directory):
        os.mkdir(summary_directory)

    # Collate and save data:
    collate_data(args.experiment_folderpath, f"{args.collation_target}.npy", sample_count)


if __name__ == "__main__":
    main()
