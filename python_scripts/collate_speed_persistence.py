import argparse
import os

import numpy as np

SAMPLE_COUNT = int(16384 / 2)
# SAMPLE_COUNT = 50
TIMESTEPS = 2880
SUPERITERATION_NUMBER = 12


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process an outputs folder with a given name.')
    parser.add_argument('--folder_name', type=str, help='Folder name.')
    parser.add_argument('--output_name', type=str, help='Where to put output files.')
    args = parser.parse_args()
    return args


def collate_data(args, out_filename, in_file):
    # Code for discontinuous writing to numpy file taken & modified from:
    # https://stackoverflow.com/questions/65882709/how-to-write-ndarray-to-npy-file-iteratively-with-batches
    print(f"Collating data from {in_file}...")
    out_data = []
    for folder_id in range(0, SAMPLE_COUNT):
        if (folder_id + 1) % 1000 == 0:
            print(folder_id + 1)
        id_filepath = f'./{args.folder_name}/{folder_id}/{in_file}.npy'
        try:
            superiteration_averages = np.load(id_filepath)

            # Assert that matrices have expected dimesions:
            if in_file == "dtheta_cellmeans":
                if superiteration_averages.shape != (SUPERITERATION_NUMBER, TIMESTEPS - 1):
                    print(folder_id)
                    assert superiteration_averages.shape == (SUPERITERATION_NUMBER, TIMESTEPS - 1)
            else:
                if superiteration_averages.shape != (SUPERITERATION_NUMBER, TIMESTEPS):
                    print(folder_id)
                    assert superiteration_averages.shape == (SUPERITERATION_NUMBER, TIMESTEPS)

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
    out_filepath = os.path.join(f"out_{args.output_name}", out_filename)
    np.save(out_filepath, out_data)


def main():
    # Parse arguments:
    args = parse_arguments()
    output_folder = f"out_{args.output_name}"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Collate individual simulation data into set of comprehensive numpy arrays:
    collate_data(args, "collated_magnitudes.npy", "magnitude_cellmeans")
    collate_data(args, "collated_dtheta.npy", "dtheta_cellmeans")
    collate_data(args, "collated_collisions.npy", "collision_cellmeans")
    collate_data(args, "collated_order_parameters.npy", "order_parameters")


if __name__ == "__main__":
    main()
