"""Perform only basic order parameter calculations."""
import argparse
import json
import os

import numpy as np
import pandas as pd

from scipy.spatial import KDTree
from ot.bregman import empirical_sinkhorn_divergence

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
    "turning_angle", "sampled_angle"
]


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
        trajectory_list.append((frame_array, x_pos, y_pos))

    return trajectory_list


def analyse_site(trajectory_list):
    # Find (approximate) valid position list from trajectories:
    positions_dictionary = {}  # keys: frame, values: list of positions
    for frame_list, x_positions, y_positions in trajectory_list:
        if len(frame_list) < 250:
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
        tree = KDTree(positions_array)
        kd_tree_dictionary[frame] = tree

    # Get average nearest-neighbour distances and trajectory characteristics:
    trajectory_distances = []
    speed = []
    meander_ratio = []
    for frame_list, x_positions, y_positions in trajectory_list:
        # Skip if likely to be noise:
        length = frame_list[-1] - frame_list[0]
        if length < 500:
            continue

        # Iterate through valid frames:
        mean_neighbour_distances = []
        for frame_index, frame in enumerate(frame_list):
            query_position = [x_positions[frame_index], y_positions[frame_index]]
            distances, _ = kd_tree_dictionary[frame].query(query_position, k=2)
            # Take mean of final three entries - first entry will be the query position
            # and so give a distance of 0.
            # Have to double because of pixel binning in experiments:
            mean_neighbour_distances.append(np.mean(distances[1:]) * 2)

        # Take average over trajectory:
        trajectory_distances.append(np.mean(mean_neighbour_distances))

        # Get trajectory characteristics:
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

        speed.append(path_length / (length * 2.5))
        meander_ratio.append(total_displacement / path_length)

    return speed, meander_ratio, trajectory_distances


def generate_dataframe(trajectory_dictionary):
    dataframe = pd.DataFrame(
        columns=["Speed", "Meander Ratio", "NN Distance", "Column"]
    )
    for column in COLUMNS:
        speeds = []
        meander_ratios = []
        nn_distances = []
        for site_index in range(12):
            # Get trajectories:
            trajectory_list = get_trajectory_list_from_site(
                trajectory_dictionary[column][site_index]
            )

            # Get the analysis:
            speed, meander_ratio, nn_distance = analyse_site(
                trajectory_list
            )

            # Add to lists:
            speeds.extend(speed)
            meander_ratios.extend(meander_ratio)
            nn_distances.extend(nn_distance)

        data_dictionary = {
            "Speed": speeds,
            "Meander Ratio": meander_ratios,
            "NN Distance": nn_distances,
            "Column": [int(column)] * len(speeds)
        }
        column_dataframe = pd.DataFrame(data=data_dictionary)
        dataframe = pd.concat([dataframe, column_dataframe])

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

    distances = []
    for i in range(25):
        sample_indices = rng.choice(len(transformed_simulation), sample_size)
        sampled_sim = transformed_simulation[sample_indices, :]

        # Estimate distance with Sinkhorn:
        distances.append(
            empirical_sinkhorn_divergence(
                transformed_experiment, sampled_sim, reg
            )
        )

    print(np.mean(distances))
    return np.mean(distances)


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
    timesteps = simulation_arguments["timestepsToRun"]
    cell_number = simulation_arguments["numberOfCells"]
    superiteration_number = simulation_arguments["superIterationCount"]

    # Loop through subiterations:
    collated_simulation_dataset = []
    for seed in range(superiteration_number):
        # Read dataframe into memory:
        print(f"Reading subiteration {seed}...")
        filename = f"positions_seed{seed:03d}.csv"
        filepath = os.path.join(subdirectory_path, filename)
        trajectory_dataframe = pd.read_csv(
            filepath, index_col=None, header=None, names=OUTPUT_COLUMN_NAMES
        )

        # Sort by cell and then by frame:
        sorted_dataframe = trajectory_dataframe.sort_values(by=["particle", "frame"])
        positions_dataframe = sorted_dataframe.loc[:, ('x', 'y')]
        position_array = np.array(positions_dataframe).reshape(cell_number, timesteps, 2)

        # Generate KD-Tree for each frame:
        kd_tree_list = {}
        for timestep in range(1440, 2880):
            tree = KDTree(position_array[:, timestep, :])
            kd_tree_list[timestep] = tree

        sim_nn_distance = []
        sim_speed = []
        sim_mr = []
        for cell_index in range(position_array.shape[0]):
            # Get nearest neighbour distances
            timestep_distances = []
            for timestep in range(1440, 2880):
                query_position = [
                    position_array[cell_index, timestep, 0],
                    position_array[cell_index, timestep, 1]
                ]
                distances, _ = kd_tree_list[timestep].query(query_position, k=2)
                timestep_distances.append(np.mean(distances[1:]))
            sim_nn_distance.append(np.mean(timestep_distances))

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
            sim_speed.append(path_length / 1440)
            sim_mr.append(total_displacement / path_length)

        simulated_dataset = np.stack([sim_speed, sim_mr, sim_nn_distance], axis=1)
        collated_simulation_dataset.append(simulated_dataset)

    collated_simulation_dataset = np.concatenate(collated_simulation_dataset, axis=0)

    # Get individual column datasets:
    print("Comparing wet lab data to simulation data...")
    mean_wd_distances = []
    for column in COLUMNS:
        print(f"--- Comparing to column {column}...")
        selection_mask = experiment_dataframe["Column"] == int(column)
        experiment_distribution = np.array(
            experiment_dataframe.loc[
                selection_mask,
                ["Speed", "Meander Ratio", "NN Distance"]
            ]
        )
        print(experiment_distribution.shape)
        mean_wd_distances.append(
            get_wd_estimate(experiment_distribution, collated_simulation_dataset)
        )
    superiteration_distances = np.array(mean_wd_distances)

    # Save to .npy files as distance arrays:
    np.save(
        os.path.join(
            subdirectory_path, "distributional_distances.npy"
        ), superiteration_distances
    )


if __name__ == "__main__":
    main()
