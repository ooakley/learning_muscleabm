"""Perform only basic order parameter calculations."""
import argparse
import os
import json
import skimage

import numpy as np
import pandas as pd

SIMULATION_OUTPUTS_FOLDER = "./fileOutputs/"

OUTPUT_COLUMN_NAMES = [
    "frame", "particle", "x", "y",
    "shapeDirection",
    "orientation", "polarity_extent",
    "percept_direction", "percept_intensity",
    "actin_flow", "actin_mag",
    "collision_number",
    "cil_x", "cil_y",
    "movement_direction",
    "turning_angle",
    "stadium_x", "stadium_y",
    "sampled_angle"
]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a folder with a given integer name.')
    parser.add_argument('--folder_id', type=int, help='Integer corresponding to the folder name')
    args = parser.parse_args()
    return args


def interpolate_to_wetlab_frames(position_array):
    differences = np.diff(position_array, axis=1)
    differences[differences > 1024] -= 2048
    differences[differences < -1024] += 2048

    interpolants = position_array[:, :-1, :] + (differences / 2)
    interpolated_array = []
    for idx in range(288):
        interpolated_array.append(position_array[:, idx * 5, :])
        interpolated_array.append(interpolants[:, (idx * 5) + 2, :])

    interpolated_array = np.stack(interpolated_array, axis=1)
    interpolated_array[interpolated_array > 2048] -= 2048
    interpolated_array[interpolated_array < 0] += 2048
    return interpolated_array


def find_coherency_fraction(positions_array):
    # Estimate from trajectory dataframe as test:
    line_array = np.zeros((1024, 1024))
    for cell_index in range(positions_array.shape[0]):
        xy_data = positions_array[cell_index, :, :] / 2
        for frame_index in range(len(xy_data) - 1):
            # Get indices of line:
            xy_t = np.floor(xy_data[frame_index, :]).astype(int)
            xy_t1 = np.floor(xy_data[frame_index + 1, :]).astype(int)
            if np.any(np.concatenate([xy_t, xy_t1]) == 1024):
                continue
            # Account for periodic boundaries:
            distance = np.sqrt(np.sum((xy_t - xy_t1)**2, axis=0))
            if distance > 512:
                continue

            # Plot line indices on matrix:
            _rr, _cc = skimage.draw.line(*xy_t, *xy_t1)
            line_array[_rr, _cc] += 1

    # Find orientations of lines:
    structure_tensor = skimage.feature.structure_tensor(
        line_array, sigma=64,
        mode='constant', cval=0,
        order='rc'
    )

    eigenvalues = skimage.feature.structure_tensor_eigenvalues(structure_tensor)
    coherency_numerator = eigenvalues[0, :, :] - eigenvalues[1, :, :]
    coherency_denominator = eigenvalues[0, :, :] + eigenvalues[1, :, :]
    coherency = coherency_numerator / coherency_denominator

    line_array_mask = line_array > 0
    coherency_fraction = np.sum(coherency[line_array_mask]) / np.sum(line_array)
    idling_factor = np.sum(line_array) / np.sum(line_array_mask)
    return coherency_fraction, idling_factor, line_array


def find_anni(frame_positions):
    # Get distance matrix, taken from:
    # https://stackoverflow.com/questions/22720864/efficiently-calculating-a-euclidean-distance-matrix-using-numpy
    distance_sq = np.sum(
        (frame_positions[:, np.newaxis, :] - frame_positions[np.newaxis, :, :]) ** 2,
        axis=-1
    )
    distance_matrix = np.sqrt(distance_sq)

    # Set all diagonal entries to a large number, so minimum func can be broadcast:
    diagonal_idx = np.diag_indices(distance_matrix.shape[0], 2)
    distance_matrix[diagonal_idx] = 2048
    minimum_distances = np.min(distance_matrix, axis=1)

    # Get ratio of mean NN distance to expected distance:
    expected_minimum = 0.5 / np.sqrt(len(minimum_distances) / (2048 * 2048))
    anni = np.mean(minimum_distances) / expected_minimum
    return anni


def main():
    """Run basic script logic."""
    # Parse arguments:
    args = parse_arguments()

    # Find specified simulation output files:
    subdirectory_path = os.path.join(SIMULATION_OUTPUTS_FOLDER, str(args.folder_id))

    # Get arguments to simulation:
    json_filepath = os.path.join(subdirectory_path, f"{args.folder_id}_arguments.json")
    with open(json_filepath) as json_file:
        simulation_arguments = json.load(json_file)

    # Define variables needed for matrix reshaping later on:
    timesteps = simulation_arguments["timestepsToRun"]
    cell_number = simulation_arguments["numberOfCells"]
    superiteration_number = simulation_arguments["superIterationCount"]

    # Loop through subiterations:
    coherency_fractions = []
    ann_indices = []
    for seed in range(superiteration_number):
        # Read dataframe into memory:
        print(f"Reading subiteration {seed} for site analysis...")
        filename = f"positions_seed{seed:03d}.csv"
        filepath = os.path.join(subdirectory_path, filename)
        trajectory_dataframe = pd.read_csv(
            filepath, index_col=None, header=None, names=OUTPUT_COLUMN_NAMES
        )

        # Sort by cell and then by frame:
        positions = trajectory_dataframe.sort_values(['particle', 'frame']).loc[:, ('x', 'y')]
        position_array = np.array(positions).reshape(cell_number, timesteps, 2)
        position_array = position_array[:, 1440:, :]

        # Interpolate to match 2.5 minute timestep of wetlab data:
        interpolated_array = interpolate_to_wetlab_frames(position_array)

        # Get coherency fraction for site:
        coherency_fraction, _, _ = find_coherency_fraction(interpolated_array)

        # Loop through frames to get average ANNI:
        anni_timeseries = []
        for timepoint in range(position_array.shape[1]):
            anni_timeseries.append(find_anni(position_array[:, timepoint, :]))
        site_anni = np.mean(anni_timeseries)

        coherency_fractions.append(coherency_fraction)
        ann_indices.append(site_anni)

    # Save to .npy files as (SUPERITERATIONS) arrays:
    coherency_fractions = np.array(coherency_fractions)
    np.save(os.path.join(subdirectory_path, "coherency_fractions.npy"), coherency_fractions)

    ann_indices = np.array(ann_indices)
    np.save(os.path.join(subdirectory_path, "ann_indices.npy"), ann_indices)


if __name__ == "__main__":
    main()
