import argparse
import os

import numpy as np

# SAMPLE_COUNT = 1000
SAMPLE_COUNT = int(16384)
# SAMPLE_COUNT = 2000


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
        try:
            id_filepath = f'./{args.folder_name}/{folder_id}/{in_file}.npy'
            superiteration_averages = np.load(id_filepath)
            if superiteration_averages.shape != (6, 6):  # (CELL TYPE) X (SS DISTANCE)
                blank_data = np.array([np.nan] * 36)
                blank_data = np.reshape(blank_data, (6, 6))
                out_data.append(blank_data)
            else:
                out_data.append(superiteration_averages)
        except FileNotFoundError:
            print(f"No data file found at {folder_id}, appending NaN...")
            blank_data = np.array([np.nan] * 36)
            blank_data = np.reshape(blank_data, (6, 6))
            out_data.append(blank_data)

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
    collate_data(args, "collated_distances.npy", "distributional_distances")


if __name__ == "__main__":
    main()
