import marimo

__generated_with = "0.12.10"
app = marimo.App(width="columns")


@app.cell
def _():
    import os
    import json
    import itertools

    import scipy.stats
    import scipy.spatial

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import colorcet as cc

    import matplotlib.pyplot as plt

    DATA_DIRECTORY = "./wetlab_data/OEO20241206"
    ROWS = ["A", "B", "C"]
    COLUMNS = ["1", "2", "3", "4", "5", "6"]

    TIME_WEIGHTING = 2048/1440

    trajectory_folderpath = os.path.join(DATA_DIRECTORY, "trajectories")
    return (
        COLUMNS,
        DATA_DIRECTORY,
        ROWS,
        TIME_WEIGHTING,
        cc,
        itertools,
        json,
        np,
        os,
        pd,
        plt,
        scipy,
        sns,
        trajectory_folderpath,
    )


@app.cell
def _(COLUMNS, ROWS, os, pd, trajectory_folderpath):
    trajectory_dictionary = {column: [] for column in COLUMNS}

    # Each different cell type is contained in different columns of 3 wells, with four sites each:
    for row in ROWS:
        for column in COLUMNS:
            for site in range(4):
                csv_filename = f"{row}{column}-Site_{site}.csv"
                site_dataframe = pd.read_csv(
                    os.path.join(trajectory_folderpath, csv_filename),
                    index_col=0
                )
                trajectory_dictionary[column].append(site_dataframe)
    return (
        column,
        csv_filename,
        row,
        site,
        site_dataframe,
        trajectory_dictionary,
    )


@app.cell
def _(np):
    def get_trajectory_list_from_site(site_dataframe):
        # Looping through all particles, first finding all valid particle IDs:
        particles = list(set(list(site_dataframe["tree_id"])))
        site_particle_count = len(particles)
        trajectory_list = []

        for particle in particles:
            # Constructing dataframe masks:
            particle_mask = site_dataframe["tree_id"] == particle
            x_pos = np.array(site_dataframe[particle_mask]["x"])
            y_pos = np.array(site_dataframe[particle_mask]["y"])
            frame_array = np.array(site_dataframe[particle_mask]["frame"])
            if (frame_array[-1] - frame_array[0]) < 250:
                continue
            trajectory_list.append((frame_array, x_pos, y_pos))

        return trajectory_list
    return (get_trajectory_list_from_site,)


@app.cell
def _(TIME_WEIGHTING, itertools, np, pd, scipy):
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
            trajectory_positions = np.stack([
                np.array(total_x_positions)[trajectory_mask],
                np.array(total_y_positions)[trajectory_mask],
                np.array(total_frames)[trajectory_mask],
                ], axis=1
            )
            trajectory_tree = scipy.spatial.KDTree(trajectory_positions)

            # Get tree of all other trajectories:
            environment_positions = np.stack([
                np.array(total_x_positions)[~trajectory_mask],
                np.array(total_y_positions)[~trajectory_mask],
                np.array(total_frames)[~trajectory_mask]
                ], axis=1
            )
            environment_tree = scipy.spatial.KDTree(environment_positions)

            # Query current trajectory against all other trajectories:
            point_pairs = trajectory_tree.query_ball_tree(environment_tree, 75)

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
                nn_distance_list.append(distances[1:] * 2) # Double for binning.

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
    return (analyse_site,)


@app.cell
def _(
    COLUMNS,
    analyse_site,
    get_trajectory_list_from_site,
    pd,
    trajectory_dictionary,
):
    dataframe_list = []
    for _column in COLUMNS:
        for _site in range(1):
            # Retrieve data:
            trajectory_list = get_trajectory_list_from_site(
                trajectory_dictionary[_column][_site]
            )

            # Analyse data:
            results_dataframe = analyse_site(trajectory_list, _column)
            dataframe_list.append(results_dataframe)

    experiment_dataframe = pd.concat(dataframe_list)
    return (
        dataframe_list,
        experiment_dataframe,
        results_dataframe,
        trajectory_list,
    )


@app.cell
def _(experiment_dataframe):
    experiment_dataframe
    return


@app.cell
def _(experiment_dataframe, sns):
    sns.jointplot(
        data=experiment_dataframe, x="speed", y="meander_ratio",
        hue="column"
    )
    return


@app.cell
def _():
    """
    What we want:
    --> 2D Wasserstein distance:
    - Speed
    - Meander Ratio

    --> Dimensional combinations query:
    - Minimum TT distance
    - Minimum approach
    - Minimum contemporaneous distance
    - Time difference at minimum TT distance
    - Distance at minimum approach
    - Time difference at minimum TT distance

    For a total of 8 different per-cell metrics.
    """;
    return


@app.cell
def _():
    # Get similar metrics for simulations:
    return


@app.cell
def _(
    TIME_WEIGHTING,
    TRAJECTORY_INTERACTION_DISTANCE,
    itertools,
    np,
    pd,
    scipy,
):
    def analyse_simulation_site(
        trajectory_dataframe, superiteration, cell_number
    ):
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
                timestep_distances.append(distances[1:])

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
            point_pairs = trajectory_tree.query_ball_tree(environment_tree, TRAJECTORY_INTERACTION_DISTANCE)

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
    return (analyse_simulation_site,)


@app.cell
def _(os, pd):
    DATA_DIR_PATH = "./fileOutputs/"
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
    TRAJECTORY_INTERACTION_DISTANCE = 75

    def get_trajectory_data(job_id, subiteration):
        # Determine filepaths:
        filename = f"positions_seed{subiteration:03d}.csv"
        directory_path = os.path.join(DATA_DIR_PATH, str(job_id))
        filepath = os.path.join(directory_path, filename)

        # Get cell trajectory data:
        trajectory_dataframe = pd.read_csv(
            filepath, index_col=None, header=None,
            names=OUTPUT_COLUMN_NAMES
        )
        return trajectory_dataframe

    trajectory_dataframe = get_trajectory_data(0, 3)
    return (
        DATA_DIR_PATH,
        OUTPUT_COLUMN_NAMES,
        TRAJECTORY_INTERACTION_DISTANCE,
        get_trajectory_data,
        trajectory_dataframe,
    )


@app.cell
def _(analyse_simulation_site, trajectory_dataframe):
    analysed_dataframe = analyse_simulation_site(trajectory_dataframe, 0, 123)
    return (analysed_dataframe,)


@app.cell
def _(analysed_dataframe):
    analysed_dataframe
    return


@app.cell
def _(analysed_dataframe, sns):
    sns.jointplot(
        data=analysed_dataframe, x="speed", y="mean_nn_distance",
        hue="superiteration"
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
