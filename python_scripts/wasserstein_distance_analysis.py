"""Perform only basic order parameter calculations."""
import time
import argparse
import itertools
import json
import os

import scipy.spatial

import numpy as np
import pandas as pd

from ot import emd2_1d
# from ot.bregman import empirical_sinkhorn_divergence

SIMULATION_OUTPUTS_FOLDER = "./fileOutputs/"
DATA_DIRECTORY = "./wetlab_data/OEO20241206"
ROWS = ["A", "B", "C"]
COLUMNS = ["1", "2", "3", "4", "5", "6"]

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

TIME_WEIGHTING = 2880 / 1440
TRAJECTORY_INTERACTION_DISTANCE = 75


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a folder with a given integer name.')
    parser.add_argument('--folder_id', type=int, help='Integer corresponding to the folder name')
    args = parser.parse_args()
    return args


def get_trajectory_list_from_site(site_dataframe):
    # Looping through all particles, first finding all valid particle IDs:
    particles = list(set(list(site_dataframe["tree_id"])))
    trajectory_list = []

    for particle in particles:
        # Constructing dataframe masks:
        particle_mask = site_dataframe["tree_id"] == particle
        x_pos = np.array(site_dataframe[particle_mask]["x"])
        y_pos = np.array(site_dataframe[particle_mask]["y"])
        frame_array = np.array(site_dataframe[particle_mask]["frame"])
        if len(frame_array) < 500:
            continue
        trajectory_list.append((frame_array, x_pos, y_pos))

    return trajectory_list


def analyse_site(trajectory_list, column):
    # Find (approximate) valid position list from trajectories:
    total_x_positions = []
    total_y_positions = []
    total_frames = []
    trajectory_ids = []
    for id, trajectory in enumerate(trajectory_list):
        frame_list, x_positions, y_positions = trajectory
        total_x_positions.extend(x_positions * 2)
        total_y_positions.extend(y_positions * 2)
        total_frames.extend((frame_list * 2.5) * TIME_WEIGHTING)
        trajectory_ids.extend([id] * len(frame_list))

    # Generate positions array for easier lookup of frame data:
    positions_array = np.stack(
        [total_x_positions, total_y_positions, total_frames],
        axis=1
    )

    # Find (approximately) valid position list from trajectories:
    positions_dictionary = {}  # keys: frame, values: list of positions
    for frame_list, x_positions, y_positions in trajectory_list:
        if len(frame_list) < 500:
            continue
        for frame_index, frame in enumerate(frame_list):
            position = [x_positions[frame_index], y_positions[frame_index]]
            # Instantiate list if new frame:
            if frame not in list(positions_dictionary.keys()):
                positions_dictionary[frame] = []
            positions_dictionary[frame].append(position)

    # Generate KD-Tree for each frame:
    kd_tree_dictionary = {}
    for frame in positions_dictionary:
        positions_array = np.stack(positions_dictionary[frame], axis=0)
        tree = scipy.spatial.KDTree(positions_array)
        kd_tree_dictionary[frame] = tree

    # Get speed, MR, and NN & TI distances for each cell across its
    # trajectory:
    site_speeds = []
    site_meander_ratios = []
    site_mean_nn_distances = []
    site_minimum_nn_distances = []
    site_maximum_nn_distances = []
    site_ti_distances = []
    site_interaction_times = []

    for id, trajectory in enumerate(trajectory_list):
        # Check if trajectory is long enough to be valid:
        length = trajectory[0][-1] - trajectory[0][0]
        if length < 500:
            continue

        # Get query trajectory tree:
        trajectory_mask = np.array(trajectory_ids) == id
        trajectory_positions = np.stack(
            [
                np.array(total_x_positions)[trajectory_mask],
                np.array(total_y_positions)[trajectory_mask],
                np.array(total_frames)[trajectory_mask],
            ],
            axis=1
        )
        trajectory_tree = scipy.spatial.KDTree(trajectory_positions)

        # Get tree of all other trajectories:
        environment_positions = np.stack(
            [
                np.array(total_x_positions)[~trajectory_mask],
                np.array(total_y_positions)[~trajectory_mask],
                np.array(total_frames)[~trajectory_mask]
            ],
            axis=1
        )
        environment_tree = scipy.spatial.KDTree(environment_positions)

        # Query current trajectory against all other trajectories:
        point_pairs = trajectory_tree.query_ball_tree(
            environment_tree, TRAJECTORY_INTERACTION_DISTANCE
        )

        # Get valid points, skip if none present:
        empty_mask = [points != [] for points in point_pairs]
        idx_point_pairs = list(
            zip(range(trajectory_positions.shape[0]), point_pairs)
        )
        valid_point_pairs = list(
            itertools.compress(idx_point_pairs, empty_mask)
        )
        if len(valid_point_pairs) == 0:
            continue

        # Iterate through valid points on query trajectory:
        trajectory_interactions = []
        time_interactions = []
        for trajectory_index, neighbouring_indices in valid_point_pairs:
            # Calculate distances from queries in space-time:
            source_position = trajectory_positions[trajectory_index, :]
            neighbour_positions = \
                environment_positions[neighbouring_indices, :]
            interactions = np.sqrt(
                np.sum((neighbour_positions - source_position)**2, axis=1)
            )
            trajectory_interactions.append(interactions)

            # Find time differences:
            time_differences = \
                np.abs(neighbour_positions[:, 2] - source_position[2])
            time_interactions.append(time_differences)

        trajectory_interactions = np.concatenate(trajectory_interactions)
        time_interactions = np.concatenate(time_interactions)

        # Find minimum, and associated time difference:
        minimum_index = np.argmin(trajectory_interactions)
        trajectory_interaction_distance = \
            trajectory_interactions[minimum_index]
        trajectory_interaction_time = \
            time_interactions[minimum_index]

        # Unpack trajectory:
        frame_list, x_positions, y_positions = trajectory

        # Get speed and meander ratio, doubling to account for binning:
        x = x_positions * 2
        y = y_positions * 2
        dx = x[-1] - x[0]
        dy = y[-1] - y[0]
        total_displacement = np.sqrt(dx**2 + dy**2)

        x_displacements = np.diff(x)
        y_displacements = np.diff(y)
        path_length = np.sum(np.sqrt(
            x_displacements**2 + y_displacements**2
        ))

        speed = path_length / (length * 2.5)
        meander_ratio = total_displacement / path_length

        # Get nearest neighbour distance metrics:
        nn_distance_list = []
        for frame_index, frame in enumerate(frame_list):
            query_position = \
                [x_positions[frame_index], y_positions[frame_index]]
            distances, _ = \
                kd_tree_dictionary[frame].query(query_position, k=2)
            nn_distance_list.append(distances[1:] * 2)  # Double for binning.

        mean_nn_distance = np.mean(nn_distance_list)
        minimum_nn_distance = np.min(nn_distance_list)
        maximum_nn_distance = np.max(nn_distance_list)

        # Accumulate to lists:
        site_speeds.append(speed)
        site_meander_ratios.append(meander_ratio)
        site_mean_nn_distances.append(mean_nn_distance)
        site_minimum_nn_distances.append(minimum_nn_distance)
        site_maximum_nn_distances.append(maximum_nn_distance)
        site_ti_distances.append(trajectory_interaction_distance)
        site_interaction_times.append(trajectory_interaction_time)

    site_data_dictionary = {
        "speed": site_speeds,
        "meander_ratio": site_meander_ratios,
        "mean_nn_distance": site_mean_nn_distances,
        "min_nn_distance": site_minimum_nn_distances,
        "max_nn_distance": site_maximum_nn_distances,
        "ti_distance": site_ti_distances,
        "time_interaction": site_interaction_times,
        "column": [column] * len(site_speeds)
    }
    site_dataframe = pd.DataFrame(data=site_data_dictionary)
    return site_dataframe


def analyse_simulation_site(trajectory_dataframe, superiteration, cell_number):
    # Sort by cell and then by frame:
    sorted_dataframe = trajectory_dataframe.sort_values(
        by=["particle", "frame"]
    )
    positions_dataframe = sorted_dataframe.loc[:, ('x', 'y')]
    position_array = \
        np.array(positions_dataframe).reshape(cell_number, 2880, 2)

    # Generate KD-Tree for each frame:
    kd_tree_list = {}
    for timestep in range(1440, 2880):
        tree = scipy.spatial.KDTree(position_array[:, timestep, :])
        kd_tree_list[timestep] = tree

    # Get speed, MR, and NN & TI distances for each cell across its
    # trajectory:
    site_speeds = []
    site_meander_ratios = []
    site_mean_nn_distances = []
    site_minimum_nn_distances = []
    site_maximum_nn_distances = []
    site_ti_distances = []
    site_interaction_times = []

    for cell_index in range(position_array.shape[0]):
        # Get positions:
        x = position_array[cell_index, 1440:, 0]
        y = position_array[cell_index, 1440:, 1]
        x_displacements = np.diff(x)
        y_displacements = np.diff(y)

        # Account for periodicity:
        x_displacements[x_displacements > 1024] -= 2048
        x_displacements[x_displacements < -1024] += 2048
        y_displacements[y_displacements > 1024] -= 2048
        y_displacements[y_displacements < -1024] += 2048

        total_dx = np.sum(x_displacements)
        total_dy = np.sum(y_displacements)
        total_displacement = np.sqrt(total_dx**2 + total_dy**2)

        path_length = np.sum(np.sqrt(
            x_displacements**2 + y_displacements**2
        ))

        # Get nearest neighbour distances:
        timestep_distances = []
        for timestep in range(1440, 2880):
            query_position = [
                position_array[cell_index, timestep, 0],
                position_array[cell_index, timestep, 1]
            ]
            distances, _ = kd_tree_list[timestep].query(query_position, k=2)
            timestep_distances.append(distances[1])

        # Get trajectory interaction distances:
        weighted_timepoints = np.arange(1440) * TIME_WEIGHTING
        weighted_timepoints = np.expand_dims(weighted_timepoints, axis=1)
        environment_timepoints = np.repeat(weighted_timepoints, cell_number - 1, 0)

        # --- Get query trajectory tree:
        trajectory_positions = np.concatenate(
            [position_array[cell_index, 1440:, :], weighted_timepoints],
            axis=1
        )
        trajectory_tree = scipy.spatial.KDTree(trajectory_positions)

        # --- Get tree of all other trajectories:
        environment_positions = np.concatenate(
            [
                position_array[:cell_index, 1440:, :],
                position_array[cell_index + 1:, 1440:, :]
            ],
            axis=0
        )
        environment_positions = environment_positions.reshape(-1, 2)
        environment_positions = np.concatenate(
            [environment_positions, environment_timepoints],
            axis=1
        )
        environment_tree = scipy.spatial.KDTree(environment_positions)

        # --- Query current trajectory against all other trajectories:
        point_pairs = trajectory_tree.query_ball_tree(
            environment_tree, TRAJECTORY_INTERACTION_DISTANCE
        )

        # --- Get valid points, skip if none present:
        empty_mask = [points != [] for points in point_pairs]
        idx_point_pairs = list(
            zip(range(trajectory_positions.shape[0]), point_pairs)
        )
        valid_point_pairs = list(
            itertools.compress(idx_point_pairs, empty_mask)
        )
        if len(valid_point_pairs) == 0:
            continue

        # -- Iterate through valid points on query trajectory:
        trajectory_interactions = []
        time_interactions = []
        for trajectory_index, neighbouring_indices in valid_point_pairs:
            # Calculate distances from queries in space-time:
            source_position = trajectory_positions[trajectory_index, :]
            neighbour_positions = \
                environment_positions[neighbouring_indices, :]
            interactions = np.sqrt(
                np.sum((neighbour_positions - source_position)**2, axis=1)
            )
            trajectory_interactions.append(interactions)

            # Find time differences:
            time_differences = \
                np.abs(neighbour_positions[:, 2] - source_position[2])
            time_interactions.append(time_differences)

        trajectory_interactions = np.concatenate(trajectory_interactions)
        time_interactions = np.concatenate(time_interactions)

        # --- Find minimum, and the time difference associated with the
        # minimum:
        minimum_index = np.argmin(trajectory_interactions)
        trajectory_interaction_distance = \
            trajectory_interactions[minimum_index]
        trajectory_interaction_time = \
            time_interactions[minimum_index]

        site_speeds.append(path_length / 1440)
        site_meander_ratios.append(total_displacement / path_length)
        site_mean_nn_distances.append(np.mean(timestep_distances))
        site_minimum_nn_distances.append(np.min(timestep_distances))
        site_maximum_nn_distances.append(np.max(timestep_distances))
        site_ti_distances.append(trajectory_interaction_distance)
        site_interaction_times.append(trajectory_interaction_time)

    simulation_data_dictionary = {
        "speed": site_speeds,
        "meander_ratio": site_meander_ratios,
        "mean_nn_distance": site_mean_nn_distances,
        "min_nn_distance": site_minimum_nn_distances,
        "max_nn_distance": site_maximum_nn_distances,
        "ti_distance": site_ti_distances,
        "time_interaction": site_interaction_times,
        "superiteration": superiteration
    }
    simulation_dataframe = pd.DataFrame(data=simulation_data_dictionary)
    return simulation_dataframe


def generate_dataframe(trajectory_dictionary):
    full_dataframe_list = []
    for column in COLUMNS:
        for site_index in range(12):
            # Get trajectories:
            trajectory_list = get_trajectory_list_from_site(
                trajectory_dictionary[column][site_index]
            )

            # Get the analysis:
            site_dataframe = analyse_site(trajectory_list, column)

            # Add to lists:
            full_dataframe_list.append(site_dataframe)

        dataframe = pd.concat(full_dataframe_list)

    return dataframe


def whiten_distributions(experiment_distribution, simulation_distribution):
    # Determine whitening transform:
    exp_means = np.mean(experiment_distribution, axis=0, keepdims=True)
    exp_variances = np.std(experiment_distribution, axis=0, keepdims=True)

    # Transform experiment data:
    transformed_experiment = (experiment_distribution - exp_means) / exp_variances
    transformed_simulation = (simulation_distribution - exp_means) / exp_variances

    return transformed_experiment, transformed_simulation


def get_wd_estimate(experiment_distribution, simulation_distribution, reg=1):
    # Whiten data:
    transformed_experiment, transformed_simulation = \
        whiten_distributions(experiment_distribution, simulation_distribution)

    # Determining sampling strategy:
    sample_size = transformed_experiment.shape[0]
    rng = np.random.default_rng(0)

    ss_distances = []
    for ss_index in range(experiment_distribution.shape[1]):
        distances = []
        for resample_index in range(25):
            sample_indices = rng.choice(len(transformed_simulation), sample_size)
            sampled_simulation_data = transformed_simulation[sample_indices, ss_index]
            experiment_data = transformed_experiment[:, ss_index]

            # Ensure that the data is interpreted as one dimensional:
            sampled_simulation_data = np.expand_dims(sampled_simulation_data, axis=1)
            experiment_data = np.expand_dims(experiment_data, axis=1)

            # Estimate distance with Sinkhorn:
            loss = emd2_1d(experiment_data, sampled_simulation_data)
            distances.append(loss)
        ss_distances.append(np.mean(distances))

    return np.array(ss_distances)


def main():
    """Run basic script logic."""
    print("Modules loaded...")
    # Parse arguments:
    args = parse_arguments()

    # Defining output of experimental data analysis:
    processed_dataset_filepath = os.path.join(DATA_DIRECTORY, "fitting_dataset.csv")
    if not os.path.exists(processed_dataset_filepath):
        # Get wet lab data:
        print("Loading and processing wet lab trajectory data...")
        trajectory_folderpath = os.path.join(DATA_DIRECTORY, "trajectories")
        trajectory_dictionary = {column: [] for column in COLUMNS}

        # Each different cell type is contained in different columns of 3 wells,
        # with four sites each:
        for row in ROWS:
            for column in COLUMNS:
                for site in range(4):
                    csv_filename = f"{row}{column}-Site_{site}.csv"
                    site_dataframe = pd.read_csv(
                        os.path.join(trajectory_folderpath, csv_filename), index_col=0
                    )
                    trajectory_dictionary[column].append(site_dataframe)

        # Extract relevant statistics from trajectory data:
        experiment_dataframe = generate_dataframe(trajectory_dictionary)
        experiment_dataframe.to_csv(processed_dataset_filepath)
    else:
        # Read in processed dataframe:
        experiment_dataframe = pd.read_csv(processed_dataset_filepath, index_col=0)

    # Find specified simulation output files:
    subdirectory_path = os.path.join(SIMULATION_OUTPUTS_FOLDER, str(args.folder_id))

    # Get arguments to simulation:
    json_filepath = os.path.join(subdirectory_path, f"{args.folder_id}_arguments.json")
    with open(json_filepath) as json_file:
        simulation_arguments = json.load(json_file)

    # Define variables needed for matrix reshaping later on:
    # timesteps = simulation_arguments["timestepsToRun"]
    cell_number = simulation_arguments["numberOfCells"]
    superiteration_number = simulation_arguments["superIterationCount"]

    # Loop through subiterations:
    simulation_dataframes = []
    for seed in range(superiteration_number):
        # Read dataframe into memory:
        print(f"Reading and analysing subiteration {seed}...")
        filename = f"positions_seed{seed:03d}.csv"
        filepath = os.path.join(subdirectory_path, filename)
        trajectory_dataframe = pd.read_csv(
            filepath, index_col=None, header=None, names=OUTPUT_COLUMN_NAMES
        )
        simulation_dataframe = analyse_simulation_site(trajectory_dataframe, seed, cell_number)
        simulation_dataframes.append(simulation_dataframe)
    full_simulation_dataframe = pd.concat(simulation_dataframes)

    # Get individual column datasets:
    print("Comparing wet lab data to simulation data...")
    list_of_comparators = [
        "speed",
        "meander_ratio",
        "mean_nn_distance",
        "min_nn_distance",
        "max_nn_distance",
        "ti_distance"
    ]
    simulation_distribution = np.array(full_simulation_dataframe.loc[:, list_of_comparators])

    distances_array = []  # (CELL TYPE) X (SS DISTANCE)
    for column in COLUMNS:
        print(f"--- Comparing to column {column}...")
        selection_mask = experiment_dataframe["column"] == int(column)
        experiment_distribution = np.array(
            experiment_dataframe.loc[
                selection_mask,
                list_of_comparators
            ]
        )

        column_distances = get_wd_estimate(experiment_distribution, simulation_distribution)
        distances_array.append(column_distances)
    distances_array = np.stack(distances_array, axis=0)

    # Save to .npy files as distance arrays:
    distances_savepath = os.path.join(subdirectory_path, "distributional_distances.npy")
    np.save(distances_savepath, distances_array)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"--- {time.time() - start_time} seconds taken ---")
