"""Perform only basic order parameter calculations."""
import argparse
import os
import json
import skimage

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

OUTPUT_COLUMN_NAMES = [
    "frame",
    "particle",
    "x",
    "y",
    "stadium_x",
    "stadium_y",
]
WORLD_SIZE = 1024
PIXEL_SIZE = 0.3469 * 2  # Pixel size in µm, accounting for binning
TRAVEL_LIMIT = 75
FRAME_DURATION = 2.5

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a folder with a given integer name.')
    parser.add_argument('--run_folderpath', type=str)
    parser.add_argument('--folder_id', type=int)
    parser.add_argument('--com_analysis', action=argparse.BooleanOptionalAction)
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


def get_window_limits(positions_array):
    # Find total movement:
    # --- Array shape: (CELL_NUMBER, TIMESTEPS, (X, Y))
    x_diff = np.diff(positions_array[:, :, 0], axis=1)
    y_diff = np.diff(positions_array[:, :, 1], axis=1)
    # --- Account for periodic boundaries:
    x_diff[x_diff > WORLD_SIZE / 2] -= WORLD_SIZE
    x_diff[x_diff < (-WORLD_SIZE / 2)] += WORLD_SIZE
    y_diff[y_diff > WORLD_SIZE / 2] -= WORLD_SIZE
    y_diff[y_diff < (-WORLD_SIZE / 2)] += WORLD_SIZE
    # --- Get total travel:
    travel_array = np.sqrt(x_diff**2 + y_diff**2)
    travel_array = np.cumsum(travel_array, axis=1)
    particle_travel = np.mean(travel_array, axis=0)
    # -- Get cut-offs for travel control (approx. thirds):
    window_limits = [0]
    for i in range(3):
        upper_limit = TRAVEL_LIMIT * (i + 1)
        window_limits.append(np.argmax(particle_travel > upper_limit))
    # -- If final travel window exceeds data length, remove it:
    print(f"Raw window limits: {window_limits}", flush=True)
    while window_limits[-1] == 0:
        window_limits = window_limits[:-1]
        # Break if list is empty:
        if not window_limits:
            break
    print(f"Processed window limits: {window_limits}", flush=True)
    return window_limits


def find_coherency_fraction(positions_array):
    # Estimate order from trajectories plotted onto image, downsampling by 2:
    line_array = []
    disk_array = []
    for cell_index in range(positions_array.shape[0]):
        # Create empty array for placing trajectories:
        half_size = int(WORLD_SIZE / 2)
        particle_line_array = np.zeros((half_size, half_size))
        xy_data = np.copy(positions_array[cell_index, :, :])
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
        particle_disk_array = skimage.morphology.isotropic_dilation(np.bool(particle_line_array), 16)
        line_array.append(particle_line_array)
        disk_array.append(particle_disk_array)

    # Get full arrays:
    line_array = np.stack(line_array, axis=0)
    disk_array = np.stack(disk_array, axis=0)
    trajectory_array = np.clip(np.sum(line_array, axis=0), 0, 1)

    # Find orientations of lines:
    structure_tensor = skimage.feature.structure_tensor(
        trajectory_array, sigma=16,
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
    coherency_values = []
    interaction_values = []
    disk_sum = np.sum(disk_array, axis=0)
    for particle_index in range(disk_array.shape[0]):
        indexed_path = disk_array[particle_index, :, :]
        comparator_paths = np.clip(disk_sum - indexed_path, 0, 1)
        interaction_value = np.sum(comparator_paths * indexed_path) / np.sum(indexed_path)
        coherency_value = np.sum(filtered_coherency * indexed_path) / np.sum(indexed_path)
        interaction_values.append(interaction_value)
        coherency_values.append(coherency_value)
    # -- Concatenate to arrays:
    coherency_values = np.stack(coherency_values)
    interaction_values = np.stack(interaction_values)
    return np.mean(coherency_values), np.mean(interaction_values)


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
    expected_minimum = 0.5 * np.sqrt((WORLD_SIZE * WORLD_SIZE) / len(minimum_distances))
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
        average_speed = np.mean(step_lengths)  # Get mean pixels per frame.
        average_speed *= PIXEL_SIZE  # Multiply by µm per pixel to get µm per frame.
        average_speed /= FRAME_DURATION  # Divide by 2.5 minutes to get µm per minute.
        average_speeds.append(average_speed)

        # Get meander ratio:
        path_length = np.sum(step_lengths)
        if path_length <= 0:  # If cell is entirely stationary, skip
            continue
        total_displacement = np.sqrt(np.sum(np.sum(step_differences, axis=0) ** 2))
        meander_ratio = total_displacement / path_length
        meander_ratios.append(meander_ratio)

    # We want a per-site geometric mean of the particle speed distribution:
    site_average_speed = np.exp(np.mean(np.log(average_speeds)))
    return np.mean(meander_ratios), site_average_speed


def get_mean_length(trajectory_dataframe):
    # Get final frame:
    final_frame = np.max(trajectory_dataframe["frame"])
    final_frame_mask = trajectory_dataframe["frame"] == final_frame
    stadia_array = np.array(trajectory_dataframe.loc[final_frame_mask, ("x", "y", "stadium_x", "stadium_y")])

    # Plot stadia:
    cell_lengths = []
    for cell_index in range(stadia_array.shape[0]):
        # Get positions:
        base_x = stadia_array[cell_index, 0]
        base_y = stadia_array[cell_index, 1]
        stad_x = stadia_array[cell_index, 2]
        stad_y = stadia_array[cell_index, 3]

        # Roll positions:
        x_diff = base_x - stad_x
        y_diff = base_y - stad_y
        if x_diff > 1024:
            stad_x += 2048
        if x_diff < -1024:
            stad_x -= 2048
        if y_diff > 1024:
            stad_y += 2048
        if y_diff < -1024:
            stad_y -= 2048

        cell_length = np.sqrt((base_x - stad_x)**2 + (base_y - stad_y)**2)
        cell_lengths.append(cell_length)

    return np.mean(cell_lengths)


def plot_trajectory(cell_trajectory, ax, c, alpha=0.75):
    # Separate trajectories under crossing of torus boundary:
    movement = np.sqrt(np.sum(np.diff(cell_trajectory, axis=0) ** 2, axis=1))
    rollover_mask = movement > 512

    # Plot trajectory as normal:
    if np.count_nonzero(rollover_mask) == 0:
        ax.plot(
            cell_trajectory[:, 0],
            cell_trajectory[:, 1],
            c=c, alpha=alpha
        )
    else:
        rollover_indices = np.argwhere(rollover_mask)
        prev_index = 0
        for rollover_index in rollover_indices:
            rollover_index = rollover_index[0]
            ax.plot(
                cell_trajectory[prev_index:rollover_index + 1, 0],
                cell_trajectory[prev_index:rollover_index + 1, 1],
                c=c, alpha=alpha
            )
            prev_index = rollover_index + 1


def plot_csv(position_array, filepath):
    fig, ax = plt.subplots(figsize=(3, 3))
    for cell_index in range(position_array.shape[0]):
        cell_trajectory = position_array[cell_index, :, :]
        plot_trajectory(cell_trajectory, ax, 'k')

    ax.set_xlim(0, 1024)
    ax.set_ylim(0, 1024)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)
    plt.savefig(filepath, pad_inches=0.0, dpi=200)


def plot_stadia(trajectory_dataframe, filepath):
    # Get final frame:
    final_frame = np.max(trajectory_dataframe["frame"])
    final_frame_mask = trajectory_dataframe["frame"] == final_frame
    stadia_array = np.array(trajectory_dataframe.loc[final_frame_mask, ("x", "y", "stadium_x", "stadium_y")])

    # Plot stadia:
    fig, ax = plt.subplots(figsize=(3, 3))
    for cell_index in range(stadia_array.shape[0]):
        # Get positions:
        base_x = stadia_array[cell_index, 0]
        base_y = stadia_array[cell_index, 1]
        stad_x = stadia_array[cell_index, 2]
        stad_y = stadia_array[cell_index, 3]

        # Roll positions:
        x_diff = base_x - stad_x
        y_diff = base_y - stad_y
        if x_diff > 1024:
            stad_x += 2048
        if x_diff < -1024:
            stad_x -= 2048
        if y_diff > 1024:
            stad_y += 2048
        if y_diff < -1024:
            stad_y -= 2048

        # Plot stadium line:
        ax.plot([base_x, stad_x], [base_y, stad_y], c='k')
        ax.scatter(base_x, base_y, c='k', s=10)

    # Format axes:
    ax.set_xlim(0, 2048)
    ax.set_ylim(0, 2048)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # Save plots:
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)
    plt.savefig(filepath, pad_inches=0.0, dpi=200)


def main():
    """Run basic script logic."""
    # Parse arguments:
    args = parse_arguments()
    run_folderpath = args.run_folderpath
    folder_id = args.folder_id
    if args.com_analysis:
        print("Carrying out CoM analysis...", flush=True)
    else:
        print("Carrying out cell front analysis...", flush=True)

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
    order_parameters = []
    mean_directions = []
    if args.com_analysis:
        cell_lengths = []
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
            # Restrict to final day of simulated culture:
            position_array = position_array[:, -1440:, :]
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

            # Restrict to final day of simulated culture:
            position_array = position_array[:, -1440:, :]

        # Interpolate to match 2.5 minute timestep of wetlab data:
        interpolated_array = interpolate_to_wetlab_frames(position_array)

        # Get coherency fraction for site:
        window_limits = get_window_limits(interpolated_array)
        if not window_limits:
            site_coherency = np.nan
            site_interaction = np.nan
        else:
            windowed_coherencies = []
            windowed_interactions = []
            for window_index in range(len(window_limits) - 1):
                start_frame = window_limits[window_index]
                end_frame = window_limits[window_index+1]
                coherency, interaction = find_coherency_fraction(interpolated_array[:, start_frame:end_frame, :])
                windowed_coherencies.append(np.copy(coherency))
                windowed_interactions.append(np.copy(interaction))
            site_coherency = np.mean(windowed_coherencies)
            site_interaction = np.mean(windowed_interactions)

        # Loop through frames to get average ANNI:
        anni_timeseries = []
        for timepoint in range(interpolated_array.shape[1]):
            anni_timeseries.append(find_anni(interpolated_array[:, timepoint, :]))
        site_anni = np.mean(anni_timeseries)

        # Get average meander ratio across cells:
        meander_ratio, mean_speed = find_motion_metrics(interpolated_array)

        # Get estimated order parameter from positional data:
        x_diff = np.diff(interpolated_array[:, :, 0], axis=1)
        y_diff = np.diff(interpolated_array[:, :, 1], axis=1)
        particle_velocities = np.sqrt(x_diff**2 + y_diff**2)
        # -- Calculate x and y components of order parameter, averaging across cells:
        x_op_component = np.nanmean(x_diff / particle_velocities, axis=0)
        y_op_component = np.nanmean(y_diff / particle_velocities, axis=0)
        op_timeseries = np.sqrt(x_op_component**2 + y_op_component**2)
        order_parameter = np.mean(op_timeseries)
        # -- Get mean direction:
        mean_direction = np.arctan2(np.mean(y_op_component), np.mean(x_op_component))

        # Plot if first seed:
        if seed == 0:
            if args.com_analysis:
                plot_filepath = os.path.join(run_folderpath, "com_trajectory.png")
                plot_csv(interpolated_array, plot_filepath)
                stadia_filepath = os.path.join(run_folderpath, "stadia.png")
                plot_stadia(trajectory_dataframe, stadia_filepath)
            else:
                plot_filepath = os.path.join(run_folderpath, "trajectory.png")
                plot_csv(interpolated_array, plot_filepath)

        # Accumulate to lists:
        coherency_array.append(site_coherency)
        interaction_array.append(site_interaction)
        ann_indices.append(site_anni)
        meander_ratios.append(meander_ratio)
        speeds.append(mean_speed)
        order_parameters.append(order_parameter)
        mean_directions.append(mean_direction)
        if args.com_analysis:
            cell_lengths.append(get_mean_length(trajectory_dataframe) / 2)


    if not args.com_analysis:
        interaction_array = np.array(interaction_array)
        np.save(os.path.join(run_folderpath, "interaction.npy"), interaction_array)
        coherency_array = np.array(coherency_array)
        np.save(os.path.join(run_folderpath, "coherency.npy"), coherency_array)
        ann_indices = np.array(ann_indices)
        np.save(os.path.join(run_folderpath, "ann_indices.npy"), ann_indices)
        meander_ratios = np.array(meander_ratios)
        np.save(os.path.join(run_folderpath, "meander_ratios.npy"), meander_ratios)
        speeds = np.array(speeds)
        np.save(os.path.join(run_folderpath, "speeds.npy"), speeds)
        # Classical measures of organisation:
        np.save(os.path.join(run_folderpath, "order_parameters.npy"), np.array(order_parameters))
        np.save(os.path.join(run_folderpath, "mean_directions.npy"), np.array(mean_directions))
    else:
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
        # Classical measures of organisation:
        np.save(os.path.join(run_folderpath, "com_order_parameters.npy"), np.array(order_parameters))
        np.save(os.path.join(run_folderpath, "com_mean_directions.npy"), np.array(mean_directions))
        np.save(os.path.join(run_folderpath, "cell_lengths.npy"), np.array(cell_lengths))


if __name__ == "__main__":
    main()
