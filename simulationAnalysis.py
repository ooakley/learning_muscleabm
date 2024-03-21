"""
Script to run analysis of model output data.

Outputs trajectory images, matrix images, and a large set of derived
statistics.
"""
import argparse
import os
import copy
import scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
import skimage.morphology as skmorph
import skimage.measure as skmeasure

SIMULATION_OUTPUTS_FOLDER = "./fileOutputs/"
CSV_COLUMN_NAMES = [
    "frame", "particle", "x", "y", "orientation",
    "polarity_extent", "percept_direction", "percept_intensity",
    "actin_flow", "movement_direction", "turning_angle", "sampled_angle", "contact_inhibition"
]

TIMESTEPS = 5760
TIMESTEP_WIDTH = 576
GRID_SIZE = 32


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a folder with a given integer name.')
    parser.add_argument('--folder_id', type=int, help='Integer corresponding to the folder name')
    args = parser.parse_args()
    return args


def read_matrix_into_numpy(filename, grid_size, timesteps):
    number_of_elements = (grid_size**2) * timesteps * 2
    flattened_matrix = np.loadtxt(filename, delimiter=',', usecols=range(number_of_elements))
    return np.reshape(flattened_matrix, (timesteps, 2, grid_size, grid_size), order='C')


def find_persistence_times(orientation_array):
    # Calculating all pairwise angular differences in orientation timeseries:
    direction_differences = np.subtract.outer(orientation_array, orientation_array)

    # Ensuring distances are bounded between pi and -pi:
    direction_differences[direction_differences < -np.pi] += 2 * np.pi
    direction_differences[direction_differences > np.pi] -= 2 * np.pi
    direction_differences = np.abs(direction_differences)

    # Calculating number of frames that it takes for the angle to change by pi/2 (90 degrees):
    pt_list = []
    for timestep in range(orientation_array.shape[0]):
        subsequent_directions = direction_differences[timestep, timestep:]
        if np.count_nonzero(subsequent_directions > np.pi / 2) == 0:
            continue
        persistence_time = np.argmax(subsequent_directions > np.pi / 2)
        pt_list.append(persistence_time)

    return pt_list


def analyse_particle(particle_dataframe):
    # Getting dx and dy, using these to calculate orientations and RMS displacements of
    # particle over time. We subsample by four as we simulate every timestep as 2.5 minutes,
    # but we capture image data every ten minutes:
    sampled_xpos = np.array(particle_dataframe['x'])[::4]
    sampled_ypos = np.array(particle_dataframe['y'])[::4]

    # Adding measurement noise:
    # noise_x = rng.normal(0, 1e-5, len(sampled_xpos))
    # noise_y = rng.normal(0, 1e-5, len(sampled_ypos))
    # sampled_xpos = sampled_xpos + noise_x
    # sampled_ypos = sampled_ypos + noise_y

    dx = np.diff(sampled_xpos)
    dy = np.diff(sampled_ypos)

    # Correcting for periodic boundaries:
    dx[dx > 1024] = dx[dx > 1024] - 2048
    dx[dx < -1024] = dx[dx < -1024] + 2048

    dy[dy > 1024] = dy[dy > 1024] - 2048
    dy[dy < -1024] = dy[dy < -1024] + 2048

    # Calculating particle's RMS displacement:
    instantaneous_speeds = np.sqrt(dx**2 + dy**2)
    instantaneous_speeds = instantaneous_speeds[~np.isnan(instantaneous_speeds)]
    particle_rmsd = np.mean(instantaneous_speeds)

    # Calculating the particle's average persistence time:
    orientation_timeseries = np.arctan2(dy, dx)
    persistence_times = find_persistence_times(orientation_timeseries)
    if len(persistence_times) == 0:
        particle_pt = np.inf
    else:
        particle_pt = np.mean(persistence_times)

    return particle_rmsd, particle_pt


def get_order_parameter(submatrix, presence_submatrix):
    central_val = submatrix[1, 1]
    comparators = submatrix.flatten()
    comparators = np.concatenate([comparators[0:5], comparators[6:]])
    angle_diff = comparators - central_val

    # Masking for areas where the matrix is not present:
    presence_mask = presence_submatrix.flatten()
    presence_mask = np.concatenate([presence_mask[0:5], presence_mask[6:]])
    angle_diff = angle_diff[presence_mask.astype(bool)]

    if len(angle_diff) == 0:
        return np.nan

    # Taking angle modulus:
    angle_diff[angle_diff > np.pi / 2] = angle_diff[angle_diff > np.pi / 2] - np.pi
    angle_diff[angle_diff < -np.pi / 2] = angle_diff[angle_diff < -np.pi / 2] + np.pi

    # Calculating order parameter:
    order_parameter = np.mean(np.abs(angle_diff))
    return order_parameter


def get_summary_order_parameter(matrix):
    order_parameters = []
    for i in range(1, GRID_SIZE - 1):
        for j in range(1, GRID_SIZE - 1):
            submatrix = matrix[0, i - 1:i + 2, j - 1:j + 2]
            presence_submatrix = matrix[1, i - 1:i + 2, j - 1:j + 2]
            order_parameter = get_order_parameter(submatrix, presence_submatrix)
            order_parameters.append(order_parameter)

    order_parameters = np.array(order_parameters)
    order_parameters = order_parameters[~np.isnan(order_parameters)]
    return order_parameters


def return_periodic_labels(binary_matrix):
    # Creating tiles
    side_length = binary_matrix.shape[0]  # Assuming square matrix.
    tiled_matrix = np.tile(binary_matrix, (3, 3))

    # Processing threshold map:
    tiled_matrix = skmorph.binary_closing(tiled_matrix)
    tiled_matrix = skmorph.remove_small_objects(tiled_matrix, min_size=8)

    # Labelling regions:
    labelled_tiles = skmorph.label(tiled_matrix)

    # Getting region properties:
    properties = skmeasure.regionprops(labelled_tiles)
    infinity_flag = False
    valid_gaps = []
    for area_property in properties:
        min_row, min_col, max_row, max_col = area_property.bbox
        box_length = max_row - min_row
        box_width = max_col - min_col

        # Filtering out maximum measurable size:
        if box_length >= side_length or box_width >= side_length:
            # Gap too large for measurement.
            infinity_flag = True
            continue

        # Bounding box conditions:
        upper_boundary_criterion = min_row >= side_length
        lower_boundary_criterion = max_row < side_length * 2
        left_boundary_criterion = min_col >= side_length
        right_boundary_criterion = max_col < side_length * 2

        boundary_criterion = [
            upper_boundary_criterion,
            lower_boundary_criterion,
            left_boundary_criterion,
            right_boundary_criterion
        ]

        # Check if no part of the object is within simulation frame:
        if np.all(~np.array(boundary_criterion)):
            continue

        # Checking if entirely within simulation frame:
        if np.all(np.array(boundary_criterion)):
            valid_gaps.append(area_property)
            continue

        # Checking for edge cases, crossing upper or left boundary:
        lower_in_frame = max_row > side_length and max_row < side_length * 2
        upper_out_frame = min_row < side_length
        right_in_frame = max_col > side_length and max_col < side_length * 2
        left_out_frame = min_col < side_length

        # Intersecting upper edge only:
        if right_in_frame and not left_out_frame:
            if lower_in_frame and upper_out_frame:
                valid_gaps.append(area_property)
                continue

        # Intersecting left edge only:
        if lower_in_frame and not upper_out_frame:
            if right_in_frame and left_out_frame:
                valid_gaps.append(area_property)
                continue

        # Intersecting corner:
        if right_in_frame and left_out_frame:
            if lower_in_frame and upper_out_frame:
                valid_gaps.append(area_property)
                continue

    # Returning centre tile:
    labelled_matrix = labelled_tiles[side_length:side_length * 2, side_length:side_length * 2]
    assert (labelled_matrix.shape == binary_matrix.shape)

    return labelled_matrix, labelled_tiles, valid_gaps, infinity_flag


def plot_superiteration(
    trajectory_list, matrix_list, area_size, grid_size, timestep, iteration, ax
):
    # Accessing and formatting relevant dataframe:
    trajectory_dataframe = trajectory_list[iteration]
    x_mask = (trajectory_dataframe["x"] > 50) & (trajectory_dataframe["x"] < area_size - 50)
    y_mask = (trajectory_dataframe["y"] > 50) & (trajectory_dataframe["y"] < area_size - 50)
    full_mask = x_mask & y_mask
    rollover_skipped_df = trajectory_dataframe[full_mask]
    timeframe_mask = \
        (rollover_skipped_df["frame"] > timestep - TIMESTEP_WIDTH) & \
        (rollover_skipped_df["frame"] <= timestep)
    timepoint_mask = rollover_skipped_df["frame"] == timestep
    ci_lookup = trajectory_dataframe[trajectory_dataframe["frame"] == 1]
    unstacked_dataframe = \
        rollover_skipped_df[timeframe_mask].set_index(['particle', 'frame'])[['x', 'y']].unstack()

    # Setting up matrix plotting:
    tile_width = area_size / grid_size
    X = np.arange(0, area_size, tile_width) + (tile_width / 2)
    Y = np.arange(0, area_size, tile_width) + (tile_width / 2)
    X, Y = np.meshgrid(X, Y)

    matrix = matrix_list[iteration][0, :, :]
    U = np.cos(matrix)
    V = -np.sin(matrix)

    # Setting up plot
    colour_list = ['r', 'g']

    # Plotting particle trajectories:
    for i, trajectory in unstacked_dataframe.iterrows():
        identity = int(ci_lookup[ci_lookup["particle"] == i]["contact_inhibition"].iloc[0])
        ax.plot(
            trajectory['x'], trajectory['y'],
            c=colour_list[identity], alpha=0.4, linewidth=0.5
        )

    # Plotting background matrix:
    ax.quiver(
        X, Y, U, V, [matrix],
        cmap='twilight', pivot='mid', scale=75,
        headwidth=0, headlength=0, headaxislength=0, alpha=0.5
    )

    # Plotting cells & their directions:
    type0_mask = rollover_skipped_df['contact_inhibition'][timepoint_mask] == 0
    type1_mask = rollover_skipped_df['contact_inhibition'][timepoint_mask] == 1
    x_pos = rollover_skipped_df['x'][timepoint_mask]
    y_pos = rollover_skipped_df['y'][timepoint_mask]
    x_heading = np.cos(rollover_skipped_df['orientation'][timepoint_mask])
    y_heading = - np.sin(rollover_skipped_df['orientation'][timepoint_mask])
    # heading_list = rollover_skipped_df['orientation'][timepoint_mask]

    ax.quiver(
        x_pos[type0_mask], y_pos[type0_mask], x_heading[type0_mask], y_heading[type0_mask],
        pivot='mid', scale=75, color='r'
    )
    ax.quiver(
        x_pos[type1_mask], y_pos[type1_mask], x_heading[type1_mask], y_heading[type1_mask],
        pivot='mid', scale=75, color='g'
    )
    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.invert_yaxis()
    ax.set_axis_off()


def plot_trajectories(subdirectory_path, trajectory_list, matrix_list, area_size, grid_size):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    count = 0
    for i in range(3):
        for j in range(3):
            plot_superiteration(
                trajectory_list, matrix_list, area_size, grid_size, TIMESTEPS-1, count, axs[i, j]
            )
            count += 1

    fig.tight_layout()
    fig.savefig(os.path.join(subdirectory_path, "trajectories.png"))


# Feature matrix extraction:
def get_particle_positions_list(site_trajectory_df):
    particle_list = list(set(list(site_trajectory_df["particle"])))
    particle_positions_list = []
    for particle in particle_list:
        particle_mask = site_trajectory_df["particle"] == particle
        x_pos = np.array(site_trajectory_df[particle_mask]["x"])[::4]
        y_pos = np.array(site_trajectory_df[particle_mask]["y"])[::4]
        particle_positions_list.append((x_pos, y_pos))
    return particle_positions_list


def get_features_from_positions(particle_positions_list):
    # Extracted features:
    path_lengths = []
    maximal_excursions = []
    net_displacements = []
    outreach_ratios = []
    meandering_ratios = []
    maximum_densities = []
    angle_entropies = []

    for particle_positions in particle_positions_list:
        x_pos, y_pos = particle_positions
        dx = np.diff(x_pos)
        dy = np.diff(y_pos)

        # Accounting for rollover positions:
        dx[dx > 1024] = dx[dx > 1024] - 2048
        dx[dx < -1024] = dx[dx < -1024] + 2048

        dy[dy > 1024] = dy[dy > 1024] - 2048
        dy[dy < -1024] = dy[dy < -1024] + 2048

        # Calculating path statistics:
        displacements = np.sqrt(dx**2 + dy**2)

        x_excursion = np.cumsum(dx)
        y_excursion = np.cumsum(dy)
        total_excursion = np.sqrt(x_excursion**2 + y_excursion**2)
        max_excursion = np.max(total_excursion)

        net_dx = x_excursion[-1]
        net_dy = y_excursion[-1]
        net_displacement = np.sqrt(net_dx**2 + net_dy**2)
        path_length = np.sum(displacements)

        path_lengths.append(path_length)
        maximal_excursions.append(max_excursion)
        net_displacements.append(net_displacement)

        outreach_ratios.append(max_excursion / path_length)
        meandering_ratios.append(net_displacement / path_length)

        # Calculating directional statistics:
        directions = np.arctan2(dy, dx)
        bins = np.linspace(-np.pi, np.pi, 12)
        hist_densities, _ = np.histogram(directions, bins=bins, density=True, weights=displacements)
        maximum_density = np.max(hist_densities)
        maximum_densities.append(maximum_density)

        # Calculating entropy:
        hist_densities = hist_densities + 1e-4
        hist_densities = hist_densities / np.sum(hist_densities)
        entropy = np.sum(-hist_densities * np.log(hist_densities))
        angle_entropies.append(entropy)

    feature_list = [
        np.array(path_lengths),
        np.array(maximal_excursions),
        np.array(net_displacements),
        np.array(outreach_ratios),
        np.array(meandering_ratios),
        np.array(maximum_densities),
        np.array(angle_entropies)
    ]

    feature_matrix = np.stack(feature_list, axis=1)

    return feature_matrix


def main():
    # Parsing command line arguments:
    args = parse_arguments()
    print(f"Processing folder with name: {args.folder_id}")

    # Getting simulation parameters:
    gridseach_dataframe = pd.read_csv("fileOutputs/gridsearch.txt", delimiter="\t")
    gridsearch_row = gridseach_dataframe[gridseach_dataframe["array_id"] == args.folder_id]
    GRID_SIZE = int(gridsearch_row["gridSize"].iloc[0])
    AREA_SIZE = int(gridsearch_row["worldSize"].iloc[0])
    SUPERITERATION_NUM = int(gridsearch_row["superIterationCount"].iloc[0])

    # Finding specified simulation output files:
    subdirectory_path = os.path.join(SIMULATION_OUTPUTS_FOLDER, str(args.folder_id))

    # Defining and seeding RNG for measurement noise simulation:
    # rng = np.random.default_rng(0)

    matrix_list = []
    particle_rmsd_list = []
    particle_pt_list = []
    particle_neighbour_list = []
    particle_seed_list = []
    trajectory_list = []
    sub_dataframes = []
    for seed in range(SUPERITERATION_NUM):
        print(f"Reading subiteration {seed}...")
        # Reading trajectory information:
        csv_filename = f"positions_seed{seed:03d}.csv"
        csv_filepath = os.path.join(subdirectory_path, csv_filename)
        trajectory_dataframe = pd.read_csv(
            csv_filepath, index_col=None, header=None, names=CSV_COLUMN_NAMES
        )
        trajectory_list.append(trajectory_dataframe)

        # Analysing all particles:
        particles = list(set(list(trajectory_dataframe["particle"])))
        rmsd_list = []
        pt_list = []
        for particle in particles:
            particle_dataframe = trajectory_dataframe[trajectory_dataframe["particle"] == particle]
            rmsd, persistence_time = analyse_particle(particle_dataframe)
            rmsd_list.append(rmsd), pt_list.append(persistence_time)

        # Getting nearest-neighbour distance distribution:
        frames = np.array(list(set(list(trajectory_dataframe["frame"]))))
        final_frame = np.max(frames)
        final_frame_mask = trajectory_dataframe["frame"] == final_frame
        final_frame_df = trajectory_dataframe[final_frame_mask]
        x = np.array(final_frame_df['x'])
        y = np.array(final_frame_df['y'])
        positions = np.stack([x, y], axis=1)
        neighbour_tree = KDTree(positions)
        nn_distances, _ = neighbour_tree.query(positions, k=[2])
        nn_distances = [distance[0] for distance in nn_distances]

        # Reading final-step matrix information into list:
        matrix_filename = f"matrix_seed{seed:03d}.txt"
        matrix_filepath = os.path.join(subdirectory_path, matrix_filename)
        matrix = read_matrix_into_numpy(matrix_filepath, GRID_SIZE, TIMESTEPS)
        matrix_list.append(matrix[-1, :, :, :])

        # Getting final-step order parameter:
        order_parameter = np.mean(get_summary_order_parameter(matrix[-1, :, :, :]))

        # Saving to dataframes:
        particle_rmsd_list.extend(rmsd_list)
        particle_pt_list.extend(pt_list)
        particle_neighbour_list.extend(nn_distances)
        particle_seed_list.extend([seed] * len(rmsd_list))

        if np.any([element == np.inf for element in pt_list]):
            persistence_speed_corrcoef = np.nan
            p_val = np.nan
        else:
            persistence_speed_corrcoef, p_val = scipy.stats.pearsonr(rmsd_list, pt_list)

        # Saving to dataframe row:
        new_info = pd.DataFrame(
            {
                "seed_number": [seed],
                "mean_rmsd": [np.mean(rmsd_list)],
                "mean_persistence_time": [np.mean(pt_list)],
                "speed_persistence_correlation": [persistence_speed_corrcoef],
                "speed_persistence_correlation_pval": [p_val],
                "order_parameter": [order_parameter]
            }
        )
        simulation_properties = copy.deepcopy(
            gridseach_dataframe[gridseach_dataframe["array_id"] == args.folder_id]
        )
        simulation_properties = simulation_properties.reset_index()
        iteration_row = pd.concat([simulation_properties, new_info], axis=1)
        sub_dataframes.append(iteration_row)

    extracted_feature_matrix = []
    for trajectory_dataframe in trajectory_list:
        particle_positions_list = get_particle_positions_list(trajectory_dataframe)
        feature_matrix = get_features_from_positions(particle_positions_list)
        extracted_feature_matrix.append(feature_matrix)
    extracted_feature_matrix = np.concatenate(extracted_feature_matrix, axis=0)
    np.save(os.path.join(subdirectory_path, "feature_matrix.npy"), extracted_feature_matrix)

    # Generating full dataframe and saving to subdirectory:
    summary_dataframe = pd.concat(sub_dataframes).reset_index().drop(columns=["index", "level_0"])
    summary_dataframe.to_csv(os.path.join(subdirectory_path, "summary.csv"))

    # Saving particle speed distributions to subdirectory:
    particle_dataframe = pd.DataFrame(
        {
            "particle_rmsd": particle_rmsd_list,
            "particle_persistence_time": particle_pt_list,
            "particle_nn_distance": particle_neighbour_list,
            "seed": particle_seed_list
        }
    )
    particle_dataframe.to_csv(os.path.join(subdirectory_path, "particle_data.csv"))

    # Calculating collagen vortex number, size and shape:
    seed_list = []
    area_list = []
    eccentricity_list = []

    for seed, matrix in enumerate(matrix_list):
        matrix_density = matrix[1, :, :]
        labelled_matrix, labelled_tiles, valid_gaps, infinity_flag = \
            return_periodic_labels(matrix_density < 0.25)

        if infinity_flag:
            seed_list.append(seed)
            area_list.append(-1)
            eccentricity_list.append(-1)
            continue

        for gap_properties in valid_gaps:
            seed_list.append(seed)
            area_list.append(gap_properties.area)
            eccentricity_list.append(gap_properties.eccentricity)

    collagen_vortex_dataframe = pd.DataFrame(
        {
            "seed": seed_list,
            "area": area_list,
            "eccentricity": eccentricity_list
        }
    )
    collagen_vortex_dataframe.to_csv(os.path.join(subdirectory_path, "collagen_vortex_data.csv"))

    # Plotting final orientations of the matrix:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0, np.pi, 0.05)
    for matrix in matrix_list:
        secreted_matrix = matrix[0, :, :][np.nonzero(matrix[1, :, :])]
        ax.hist(secreted_matrix.flatten(), bins=bins, alpha=0.3, density=True)
    fig.tight_layout()
    fig.savefig(os.path.join(subdirectory_path, "matrix_orientation.png"))

    # Plotting trajectories:
    plot_trajectories(subdirectory_path, trajectory_list, matrix_list, AREA_SIZE, GRID_SIZE)


if __name__ == "__main__":
    main()
