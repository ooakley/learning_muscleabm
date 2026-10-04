import argparse
import os
import math

import numpy as np

import imageio.v3 as iio


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dirpath", required=True)
    parser.add_argument("--image_filename", required=True)
    parser.add_argument("--sample_count", default=2**17, type=int)
    parser.add_argument("--hash_count", default=1000, type=int)
    return parser.parse_args()


def main():
    args = parse_arguments()
    run_dirpath = os.path.join(args.experiment_dirpath, "run_data")

    # Get directory hashing info:
    hash_directory_count = math.ceil(args.sample_count / args.hash_count)
    hash_directory_ids = list(range(hash_directory_count))

    # Iterate through trajectories, collecting images:
    images = {}
    for hash_id in hash_directory_ids:
        print(f"Processing hashed folder: {hash_id}...", flush=True)
        for subhash_id in range(args.hash_count):
            # Calculate run ID, skip if out of range:
            run_id = (hash_id * args.hash_count) + subhash_id
            if run_id >= args.sample_count:
                continue

            # Copy image files from run data to image data:
            source_dirpath = os.path.join(run_dirpath, str(hash_id), str(run_id))
            source_filepath = os.path.join(source_dirpath, f"{args.image_filename}.png")
            if os.path.exists(source_filepath):
                images[str(run_id)] = iio.imread(source_filepath)
            else:
                print(f"No image file found at {run_id}...")
                images[str(run_id)] = 0

    # Compress and save numpy files:
    print("Compressing and saving image arrays...", flush=True)
    np.savez_compressed(os.path.join(args.experiment_dirpath, f"{args.image_filename}.npz"), **images)


if __name__ == "__main__":
    main()
