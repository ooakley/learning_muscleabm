"""Perform only basic order parameter calculations."""
import argparse
import os
import json
import skimage

import numpy as np
import pandas as pd

ANALYSE_COM = True
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
WORLD_SIZE = 1024


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a folder with a given integer name.')
    parser.add_argument('--run_folderpath', type=str)
    parser.add_argument('--folder_id', type=int)
    parser.add_argument('--com_analysis', type=bool)
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

    # Halve to match the (1024, 1024) pixel field of the real data:
    interpolated_array /= 2
    return interpolated_array


def find_coherency_fraction(positions_array):
    # Estimate order from trajectories plotted onto image, downsampling by 2:
    line_array = []
    disk_array = []
    for cell_index in range(positions_array.shape[0]):
        # Create empty array for placing trajectories:
        half_size = int(WORLD_SIZE / 2)
        particle_line_array = np.zeros((half_size, half_size))
        xy_data = positions_array[cell_index, :, :]
        # !!! Downsample:
        xy_data /= 2
        for frame_index in range(len(xy_data) - 1):
            # Get indices of line:
            xy_t = np.floor(xy_data[frame_index, :]).astype(int)
            xy_t1 = np.floor(xy_data[frame_index + 1, :]).astype(int)
            if np.any(np.concatenate([xy_t, xy_t1]) == half_size):
                continue
            # Account for periodic boundaries:
            distance = np.sqrt(np.sum((xy_t - xy_t1)**2, axis=0))
            if distance > half_size / 2:
                continue
            # Plot line indices on matrix:
            line_rr, line_cc = skimage.draw.line(*xy_t, *xy_t1)
            particle_line_array[line_rr, line_cc] += 1

        # Get frame of interaction area of trajectory:
        particle_line_array = np.clip(particle_line_array, 0, 1)
        particle_disk_array = skimage.morphology.isotropic_dilation(np.bool(particle_line_array), 8)
        line_array.append(particle_line_array)
        disk_array.append(particle_disk_array)

    # Get full arrays:
    line_array = np.stack(line_array, axis=0)
    disk_array = np.stack(disk_array, axis=0)
    trajectory_array = np.clip(np.sum(line_array, axis=0), 0, 1)

    # Find orientations of lines:
    structure_tensor = skimage.feature.structure_tensor(
        trajectory_array, sigma=8,
        mode='constant', cval=0,
        order='rc'
    )

    # Get coherency of trajectory shapes:
    eigenvalues = skimage.feature.structure_tensor_eigenvalues(structure_tensor)
    coherency_numerator = eigenvalues[0, :, :] - eigenvalues[1, :, :]
    coherency_denominator = eigenvalues[0, :, :] + eigenvalues[1, :, :]
    coherency = coherency_numerator / coherency_denominator

    # Get per-particle coherencies and interaction terms:
    # -- First filter coherency array for nan terms:
    filtered_coherency = np.copy(coherency)
    filtered_coherency[np.isnan(coherency)] = 0
    # -- Iterate through particle frames:
    interaction_values = []
    coherency_values = []
    disk_sum = np.sum(disk_array, axis=0)
    for particle_index in range(disk_array.shape[0]):
        indexed_path = disk_array[particle_index, :, :]
        comparator_paths = np.clip(disk_sum - indexed_path, 0, 1)
        interaction_value = np.sum(comparator_paths * indexed_path) / np.sum(indexed_path)
        coherency_value = np.sum(filtered_coherency * indexed_path) / np.sum(indexed_path)
        interaction_values.append(interaction_value)
        coherency_values.append(coherency_value)
    # -- Concatenate to arrays:
    interaction_values = np.stack(interaction_values)
    coherency_values = np.stack(coherency_values)
    return np.mean(interaction_values), np.mean(coherency_values)


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
    distance_matrix[diagonal_idx] = WORLD_SIZE
    minimum_distances = np.min(distance_matrix, axis=1)

    # Get ratio of mean NN distance to expected distance:
    expected_minimum = 0.5 / np.sqrt(len(minimum_distances) / (WORLD_SIZE * WORLD_SIZE))
    anni = np.mean(minimum_distances) / expected_minimum
    return anni


def find_motion_metrics(position_array):
    # Array shape: (CELL_NUMBER, TIMESTEPS, (X, Y))
    meander_ratios = []
    average_speeds = []
    for cell_index in range(position_array.shape[0]):
        # Unfold trajectory from the torus:
        trajectory_array = position_array[cell_index, :, :]
        step_differences = np.diff(trajectory_array, axis=0)
        step_differences[step_differences > WORLD_SIZE / 2] -= WORLD_SIZE
        step_differences[step_differences < -(WORLD_SIZE / 2)] += WORLD_SIZE

        # Get full trajectory length:
        step_lengths = np.sqrt(np.sum(step_differences ** 2, axis=1))
        average_speed = np.sum(step_lengths) / 1440  # Divide by number of minutes to get pixels per minute.

        # Get meander ratio:
        path_length = np.sum(step_lengths)
        if path_length <= 0:  # If cell is entirely stationary, skip
            continue
        total_displacement = np.sqrt(np.sum(np.sum(step_differences, axis=0) ** 2))
        meander_ratio = total_displacement / path_length

        # Record cell metrics:
        average_speeds.append(average_speed)
        meander_ratios.append(meander_ratio)

    # We want a per-site geometric mean of the particle speed distribution:
    site_average_speed = np.exp(np.mean(np.log(average_speeds)))

    return np.mean(meander_ratios), site_average_speed


def main():
    """Run basic script logic."""
    # Parse arguments:
    args = parse_arguments()
    run_folderpath = args.run_folderpath
    folder_id = args.folder_id

    # Get arguments to simulation:
    json_filepath = os.path.join(run_folderpath, f"{folder_id}_arguments.json")
    with open(json_filepath) as json_file:
        simulation_arguments = json.load(json_file)

    # Define variables needed for matrix reshaping later on:
    timesteps = simulation_arguments["timestepsToRun"]
    cell_number = simulation_arguments["numberOfCells"]
    superiteration_number = simulation_arguments["superIterationCount"]

    # Loop through subiterations:
    interaction_array = []
    coherency_array = []
    ann_indices = []
    meander_ratios = []
    speeds = []
    for seed in range(superiteration_number):
        # Read dataframe into memory:
        print(f"Reading subiteration {seed} for site analysis...")
        filename = f"positions_seed{seed:03d}.csv"
        filepath = os.path.join(run_folderpath, filename)
        trajectory_dataframe = pd.read_csv(
            filepath, index_col=None, header=None, names=OUTPUT_COLUMN_NAMES
        )

        # Sort by cell and then by frame:
        if not args.com_analysis:
            positions = trajectory_dataframe.sort_values(['particle', 'frame']).loc[:, ('x', 'y')]
            position_array = np.array(positions).reshape(cell_number, timesteps, 2)
            position_array = position_array[:, 1440:, :]
        else:
            # Get cell front position data:
            front_positions = trajectory_dataframe.sort_values(['particle', 'frame']).loc[:, ('x', 'y')]
            front_array = np.array(front_positions).reshape(cell_number, timesteps, 2)

            # Get cell back position data:
            back_positions = trajectory_dataframe.sort_values(['particle', 'frame']).loc[:, ('stadium_x', 'stadium_y')]
            back_array = np.array(back_positions).reshape(cell_number, timesteps, 2)

            # Get vectors from cell front to cell back, correcting for periodic boundaries:
            span_array = back_array - front_array
            span_array[span_array < -1024] += 2048
            span_array[span_array > +1024] -= 2048

            # Get centers of mass:
            position_array = front_array + (span_array / 2)
            position_array[position_array > 2048] -= 2048
            position_array[position_array < 0] += 2048

            # Restrict to second day of simulated culture:
            position_array = position_array[:, 1440:, :]

        # Interpolate to match 2.5 minute timestep of wetlab data:
        interpolated_array = interpolate_to_wetlab_frames(position_array)

        # Get coherency fraction for site:
        site_interaction, site_coherency = find_coherency_fraction(interpolated_array)

        # Loop through frames to get average ANNI:
        anni_timeseries = []
        for timepoint in range(position_array.shape[1]):
            anni_timeseries.append(find_anni(position_array[:, timepoint, :]))
        site_anni = np.mean(anni_timeseries)

        # Get average meander ratio across cells:
        meander_ratio, mean_speed = find_motion_metrics(interpolated_array)

        # Accumulate to lists:
        interaction_array.append(site_interaction)
        coherency_array.append(site_coherency)
        ann_indices.append(site_anni)
        meander_ratios.append(meander_ratio)
        speeds.append(mean_speed)

    # if not args.com_analysis:
    #     coherency_fractions = np.array(coherency_fractions)
    #     np.save(os.path.join(run_folderpath, "coherency_fractions.npy"), coherency_fractions)
    #     ann_indices = np.array(ann_indices)
    #     np.save(os.path.join(run_folderpath, "ann_indices.npy"), ann_indices)
    #     meander_ratios = np.array(meander_ratios)
    #     np.save(os.path.join(run_folderpath, "meander_ratios.npy"), meander_ratios)
    #     speeds = np.array(speeds)
    #     np.save(os.path.join(run_folderpath, "speeds.npy"), speeds)
    # else:

    interaction_array = np.array(interaction_array)
    np.save(os.path.join(run_folderpath, "com_interaction.npy"), interaction_array)
    coherency_array = np.array(coherency_array)
    np.save(os.path.join(run_folderpath, "com_coherency.npy"), coherency_array)
    ann_indices = np.array(ann_indices)
    np.save(os.path.join(run_folderpath, "com_ann_indices.npy"), ann_indices)
    meander_ratios = np.array(meander_ratios)
    np.save(os.path.join(run_folderpath, "com_meander_ratios.npy"), meander_ratios)
    speeds = np.array(speeds)
    np.save(os.path.join(run_folderpath, "com_speeds.npy"), speeds)


if __name__ == "__main__":
    main()
