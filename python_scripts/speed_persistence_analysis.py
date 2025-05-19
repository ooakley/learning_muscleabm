"""Perform only basic order parameter calculations."""
import argparse
import os
import numpy as np
import pandas as pd

TIMESTEPS = 1140
SIMULATION_OUTPUTS_FOLDER = "./fileOutputs/"
SUPERITERATION_NUMBER = 10

OUTPUT_COLUMN_NAMES = [
    "frame", "particle", "x", "y",
    "shapeDirection",
    "orientation", "polarity_extent",
    "percept_direction", "percept_intensity",
    "actin_flow", "actin_mag",
    "movement_direction",
    "turning_angle", "sampled_angle"
]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a folder with a given integer name.')
    parser.add_argument('--folder_id', type=int, help='Integer corresponding to the folder name')
    args = parser.parse_args()
    return args


def main():
    """Run basic script logic."""
    # Parse arguments:
    args = parse_arguments()

    # Find specified simulation output files:
    subdirectory_path = os.path.join(SIMULATION_OUTPUTS_FOLDER, str(args.folder_id))
    actin_magnitudes = []
    actin_directions = []
    for seed in range(SUPERITERATION_NUMBER):
        # Read dataframe into memory:
        print(f"Reading subiteration {seed}...")
        filename = f"positions_seed{seed:03d}.csv"
        filepath = os.path.join(subdirectory_path, filename)
        trajectory_dataframe = pd.read_csv(
            filepath, index_col=None, header=None, names=OUTPUT_COLUMN_NAMES
        )

        # Sort by cell and then by frame:
        sorted_dataframe = trajectory_dataframe.sort_values(by=["particle", "frame"])

        # Get speed and angle data:
        actin_magnitude = np.asarray(sorted_dataframe["actin_mag"])
        actin_direction = np.asarray(sorted_dataframe["actin_flow"])
        actin_magnitude = np.reshape(actin_magnitude, (150, 1440))
        actin_direction = np.reshape(actin_direction, (150, 1440))

        # Accumulate to list:
        actin_magnitudes.append(actin_magnitude)
        actin_directions.append(actin_direction)

    # Convert to numpy array and save:
    actin_magnitudes = np.concatenate(actin_magnitudes, axis=0)
    actin_directions = np.concatenate(actin_directions, axis=0)
    np.save(os.path.join(subdirectory_path, "actin_magnitudes.npy"), actin_magnitudes)
    np.save(os.path.join(subdirectory_path, "actin_directions.npy"), actin_directions)


if __name__ == "__main__":
    main()
