import argparse
import math
import os
import json

import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process an outputs folder with a given name.')
    parser.add_argument(
        '--experiment_folderpath', type=str, required=True,
        help='The {date}-{config} folder, containing config.json and the hm wave folders.'
    )
    parser.add_argument(
        '--hm_wave_id', type=int, required=True,
        help='History matching wave to collate, e.g. 0 for the hm0 folder.'
    )
    parser.add_argument('--collation_target', type=str, required=True)
    parser.add_argument(
        '--sample_count', type=int, default=None,
        help="Number of simulations to collate. Defaults to the number of rows in the wave's sample matrix."
    )
    args = parser.parse_args()
    return args


def get_sample_count(config_dict, wave_folderpath):
    # Each wave has its own sample matrix, with one row per simulation:
    sample_matrix_filepath = os.path.join(wave_folderpath, "sample_matrix.npy")
    if os.path.exists(sample_matrix_filepath):
        return np.load(sample_matrix_filepath, mmap_mode="r").shape[0]

    # Otherwise fall back on the size of the initial Sobol' search:
    print(f"No sample matrix found in {wave_folderpath}, using the config's sample exponent...", flush=True)
    return 2**config_dict["sample_exponent"]


def collate_data(config_dict, wave_folderpath, summarise_filename, sample_count):
    # Code for discontinuous writing to numpy file taken & modified from:
    # https://stackoverflow.com/questions/65882709/how-to-write-ndarray-to-npy-file-iteratively-with-batches
    print(f"Collating data from {summarise_filename}...", flush=True)

    # Get superiteration count from config file:
    superiteration_count = config_dict["constant_parameters"]["superIterationCount"]

    # Loop through all parameter sets:
    out_data = []
    for folder_id in range(sample_count):
        if (folder_id + 1) % 1000 == 0:
            print(folder_id + 1, flush=True)
        try:
            hierarchy_id = int(math.floor(folder_id / 1000))
            id_filepath = os.path.join(
                wave_folderpath, "run_data", str(hierarchy_id), str(folder_id), summarise_filename
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
    out_filepath = os.path.join(wave_folderpath, "summary_data", summarise_filename)
    np.save(out_filepath, out_data)


def main():
    # Parse arguments:
    args = parse_arguments()

    # The config file sits in the experiment folder, shared by every wave:
    with open(os.path.join(args.experiment_folderpath, "config.json")) as filestream:
        config_dict = json.load(filestream)

    # The run data and summary data sit in the folder of the given wave:
    wave_folderpath = os.path.join(args.experiment_folderpath, f"hm{args.hm_wave_id}")
    if not os.path.exists(os.path.join(wave_folderpath, "run_data")):
        raise FileNotFoundError(f"No run_data folder found in {wave_folderpath}.")
    print(f"Collating wave {args.hm_wave_id} from {wave_folderpath}...", flush=True)

    if args.sample_count is None:
        sample_count = get_sample_count(config_dict, wave_folderpath)
    else:
        sample_count = args.sample_count
    print(f"Sample count: {sample_count}", flush=True)

    # Generate relevant directory if not present:
    summary_directory = os.path.join(wave_folderpath, "summary_data")
    os.makedirs(summary_directory, exist_ok=True)

    # Collate and save data:
    collate_data(config_dict, wave_folderpath, f"{args.collation_target}.npy", sample_count)


if __name__ == "__main__":
    main()