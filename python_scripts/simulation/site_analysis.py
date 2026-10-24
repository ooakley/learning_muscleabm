"""Perform only basic order parameter calculations."""
import argparse
import os
import json
import skimage
from skimage.draw import line as draw_line

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
# Distance in pixels below which a cell counts as not having moved, well below any real motion
# but above rounding error in the positions:
MOVEMENT_TOLERANCE = 1e-9
# Tracks need more steps than this to count towards a site's speed and meander ratio, as in the
# wet lab analysis (python_scripts/wetlab/analyse_wetlab_data.py):
MIN_TRACK_STEPS = 250

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


DILATION_RADIUS = 16


def draw_trajectory(xy_data, half_size):
    """Draw the line segments of a trajectory onto a boolean image indexed [x, y]."""
    line_image = np.zeros((half_size, half_size), dtype=bool)
    xy_floor = np.floor(xy_data).astype(int)
    starts, ends = xy_floor[:-1], xy_floor[1:]
    # Skip segments touching the far edge, or crossing the periodic boundary:
    on_edge = np.any(starts == half_size, axis=1) | np.any(ends == half_size, axis=1)
    distance = np.sqrt(np.sum((starts - ends)**2, axis=1))
    keep = ~on_edge & ~(distance > half_size / 2)
    for (x0, y0), (x1, y1) in zip(starts[keep], ends[keep]):
        line_rr, line_cc = draw_line(x0, y0, x1, y1)
        line_image[line_rr, line_cc] = True
    return line_image


def dilate_trajectory(line_image):
    """Dilate a trajectory image by DILATION_RADIUS, returning (crop slices, cropped disk).

    The dilation is computed only on a crop around the trajectory, padded by more than the radius,
    which gives the same result as dilating the whole image, since every trajectory pixel is in
    the crop and every pixel outside it is further than the radius from all of them.
    """
    rows, columns = np.nonzero(line_image)
    if rows.size == 0:
        disk = skimage.morphology.isotropic_dilation(line_image, DILATION_RADIUS)
        return (slice(None), slice(None)), disk
    pad = DILATION_RADIUS + 1
    region = (
        slice(max(rows.min() - pad, 0), rows.max() + pad + 1),
        slice(max(columns.min() - pad, 0), columns.max() + pad + 1),
    )
    return region, skimage.morphology.isotropic_dilation(line_image[region], DILATION_RADIUS)


def find_coherency_fraction(positions_array):
    # Estimate order from trajectories plotted onto image, downsampling by 2:
    half_size = int(WORLD_SIZE / 2)
    trajectory_array = np.zeros((half_size, half_size), dtype=bool)
    disk_sum = np.zeros((half_size, half_size), dtype=np.int64)
    disks = []
    for cell_index in range(positions_array.shape[0]):
        # Draw trajectory, and get the area it interacts with:
        line_image = draw_trajectory(positions_array[cell_index, :, :] / 2, half_size)
        trajectory_array |= line_image
        region, disk = dilate_trajectory(line_image)
        disk_sum[region] += disk
        disks.append((region, disk))

    # Find orientations of lines:
    structure_tensor = skimage.feature.structure_tensor(
        trajectory_array.astype(float), sigma=16,
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
    indexed_path = np.zeros((half_size, half_size), dtype=bool)
    for region, disk in disks:
        disk_area = np.sum(disk)
        # Fraction of the particle's area that other particles' areas overlap:
        overlapped = (disk_sum[region] - disk) > 0
        interaction_values.append(np.sum(overlapped & disk) / disk_area)
        # Mean coherency over the particle's area (summed over the whole image, as before):
        indexed_path[region] = disk
        coherency_values.append(np.sum(filtered_coherency * indexed_path) / disk_area)
        indexed_path[region] = False
    # -- Concatenate to arrays:
    coherency_values = np.stack(coherency_values)
    interaction_values = np.stack(interaction_values)

    # Mean coherency over the pixels of every trajectory, as the wet lab coherency_fraction:
    coherency_fraction = np.sum(coherency[trajectory_array]) / np.sum(trajectory_array)
    return np.mean(coherency_values), np.mean(interaction_values), coherency_fraction


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


def split_into_tracks(trajectory):
    """Split a cell's trajectory, (TIMESTEPS, (X, Y)), wherever it crosses the periodic boundary.

    Wet lab tracks end when a cell leaves the field of view, and a new track starts when it (or
    another cell) enters it, so each crossing ends one track and starts another.
    """
    crossings = np.any(np.abs(np.diff(trajectory, axis=0)) > WORLD_SIZE / 2, axis=1)
    return np.split(trajectory, np.flatnonzero(crossings) + 1)


def find_motion_metrics(position_array):
    """Site meander ratio and geometric mean speed, from tracks as the wet lab analysis builds them.

    As in python_scripts/wetlab/analyse_wetlab_data.py, each track's speed is its mean step length,
    and its meander ratio is its displacement over its path length. Only tracks with more than
    MIN_TRACK_STEPS steps, that moved, count towards the site's values. Tracks end at periodic
    boundary crossings (see split_into_tracks), which shortens the tracks of fast cells, as leaving
    the field of view does.
    """
    # Array shape: (CELL_NUMBER, TIMESTEPS, (X, Y))
    meander_ratios = []
    average_speeds = []
    for cell_index in range(position_array.shape[0]):
        for track in split_into_tracks(position_array[cell_index, :, :]):
            step_differences = np.diff(track, axis=0)
            if step_differences.shape[0] <= MIN_TRACK_STEPS:
                continue

            # Get full track length:
            step_lengths = np.sqrt(np.sum(step_differences ** 2, axis=1))
            path_length = np.sum(step_lengths)

            # Skip entirely stationary tracks, which have no meander ratio, and whose zero speed
            # would make the site's geometric mean speed zero:
            if path_length <= MOVEMENT_TOLERANCE:
                continue

            average_speed = np.mean(step_lengths)  # Get mean pixels per frame.
            average_speed *= PIXEL_SIZE  # Multiply by µm per pixel to get µm per frame.
            average_speed /= FRAME_DURATION  # Divide by 2.5 minutes to get µm per minute.
            average_speeds.append(average_speed)

            # Get meander ratio:
            total_displacement = np.sqrt(np.sum(np.sum(step_differences, axis=0) ** 2))
            meander_ratio = total_displacement / path_length
            meander_ratios.append(meander_ratio)

    # Neither metric is defined if no track is long enough and moved:
    if not average_speeds:
        return np.nan, np.nan

    # We want a per-site geometric mean of the particle speed distribution:
    site_average_speed = np.exp(np.mean(np.log(average_speeds)))
    return np.mean(meander_ratios), site_average_speed


def find_motion_order(position_array):
    """Nematic order parameter and mean axis of the cells' frame-to-frame motion.

    The order parameter is, for each frame, the magnitude of the mean of exp(2i theta) over the
    cells that moved, theta being each cell's direction of motion, averaged over frames. It is 1
    when cells move along a common axis, whichever way along it, and 0 when their directions are
    isotropic. The mean axis is half the angle of the mean of exp(2i theta) over frames, in
    (-pi/2, pi/2]: the axis of motion, without a direction along it.
    """
    # Array shape: (CELL_NUMBER, TIMESTEPS, (X, Y)). Take frame-to-frame differences across the
    # periodic boundaries:
    differences = np.diff(position_array, axis=1)
    differences[differences > WORLD_SIZE / 2] -= WORLD_SIZE
    differences[differences < -(WORLD_SIZE / 2)] += WORLD_SIZE
    x_diff, y_diff = differences[:, :, 0], differences[:, :, 1]

    # Directions of motion, leaving out cells that did not move in a frame:
    headings = np.arctan2(y_diff, x_diff)
    moved = np.sqrt(x_diff**2 + y_diff**2) > MOVEMENT_TOLERANCE
    headings[~moved] = np.nan

    # Average over cells, then frames:
    nematic_x = np.nanmean(np.cos(2 * headings), axis=0)
    nematic_y = np.nanmean(np.sin(2 * headings), axis=0)
    order_parameter = np.mean(np.sqrt(nematic_x**2 + nematic_y**2))
    mean_axis = np.arctan2(np.mean(nematic_y), np.mean(nematic_x)) / 2
    return order_parameter, mean_axis


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
    coherency_fraction_array = []
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
            site_coherency_fraction = np.nan
        else:
            windowed_coherencies = []
            windowed_interactions = []
            windowed_coherency_fractions = []
            for window_index in range(len(window_limits) - 1):
                start_frame = window_limits[window_index]
                end_frame = window_limits[window_index+1]
                coherency, interaction, coherency_fraction = find_coherency_fraction(
                    interpolated_array[:, start_frame:end_frame, :]
                )
                windowed_coherencies.append(np.copy(coherency))
                windowed_interactions.append(np.copy(interaction))
                windowed_coherency_fractions.append(np.copy(coherency_fraction))
            site_coherency = np.mean(windowed_coherencies)
            site_interaction = np.mean(windowed_interactions)
            site_coherency_fraction = np.mean(windowed_coherency_fractions)

        # Loop through frames to get average ANNI:
        anni_timeseries = []
        for timepoint in range(interpolated_array.shape[1]):
            anni_timeseries.append(find_anni(interpolated_array[:, timepoint, :]))
        site_anni = np.mean(anni_timeseries)

        # Get average meander ratio across cells:
        meander_ratio, mean_speed = find_motion_metrics(interpolated_array)

        # Get the nematic order parameter and mean axis of cell motion (saved as mean_directions):
        order_parameter, mean_axis = find_motion_order(interpolated_array)

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
        coherency_fraction_array.append(site_coherency_fraction)
        interaction_array.append(site_interaction)
        ann_indices.append(site_anni)
        meander_ratios.append(meander_ratio)
        speeds.append(mean_speed)
        order_parameters.append(order_parameter)
        mean_directions.append(mean_axis)
        if args.com_analysis:
            cell_lengths.append(get_mean_length(trajectory_dataframe) / 2)


    if not args.com_analysis:
        interaction_array = np.array(interaction_array)
        np.save(os.path.join(run_folderpath, "interaction.npy"), interaction_array)
        coherency_array = np.array(coherency_array)
        np.save(os.path.join(run_folderpath, "coherency.npy"), coherency_array)
        np.save(os.path.join(run_folderpath, "coherency_fraction.npy"), np.array(coherency_fraction_array))
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
        np.save(os.path.join(run_folderpath, "com_coherency_fraction.npy"), np.array(coherency_fraction_array))
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
