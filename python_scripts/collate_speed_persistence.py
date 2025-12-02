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

    # Get necessary run information from config file:
    with open(os.path.join(experiment_folderpath, "config.json")) as filestream:
        config_dict = json.load(filestream)
    sample_count = 2**config_dict["sample_exponent"]
    superiteration_count = config_dict["constant_parameters"]["superIterationCount"]
    simulated_timesteps = config_dict["constant_parameters"]["timestepsToRun"]

    # Loop through all parameter sets:
    out_data = []
    for folder_id in range(0, sample_count):
        if (folder_id + 1) % 1000 == 0:
            print(folder_id + 1)
        hierarchy_id = int(math.floor(folder_id / 1000))
        id_filepath = os.path.join(
            experiment_folderpath, "run_data", str(hierarchy_id), str(folder_id), summarise_filename
        )
        try:
            superiteration_averages = np.load(id_filepath)

            # Assert that matrices have expected dimesions:
            if summarise_filename == "dtheta_cellmeans.npy":
                if superiteration_averages.shape != (superiteration_count, simulated_timesteps - 1):
                    print(folder_id)
                    assert superiteration_averages.shape == (superiteration_count, simulated_timesteps - 1)
            else:
                if superiteration_averages.shape != (superiteration_count, simulated_timesteps):
                    print(folder_id)
                    assert superiteration_averages.shape == (superiteration_count, simulated_timesteps)

            stationary_mean = np.mean(superiteration_averages[:, 1440:], axis=1)
            # ^ of shape (SUPERITERATIONS)

            # Get mean and standard deviation over simulation runs:
            overall_mean = np.mean(stationary_mean, axis=0)
            overall_std = np.std(stationary_mean, axis=0)
            out_data.append([overall_mean, overall_std])
        except:
            print(f"Empty or incorrect data encountered at {folder_id}, skipping...")
            out_data.append([np.nan, np.nan])

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
    collate_data(args.experiment_folderpath, "magnitude_cellmeans.npy")
    collate_data(args.experiment_folderpath, "dtheta_cellmeans.npy")
    collate_data(args.experiment_folderpath, "collision_cellmeans.npy")
    collate_data(args.experiment_folderpath, "order_parameters.npy")


if __name__ == "__main__":
    main()
