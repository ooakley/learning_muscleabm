import marimo

__generated_with = "0.12.10"
app = marimo.App(width="columns")


@app.cell
def _():
    import gpytorch
    import torch
    import json
    import skimage

    import time
    import itertools
    import os

    import numpy as np
    import matplotlib.pyplot as plt

    from scipy import stats
    from torch.utils.data import TensorDataset, DataLoader

    import scipy.spatial

    import pandas as pd

    from ot import emd2_1d
    return (
        DataLoader,
        TensorDataset,
        emd2_1d,
        gpytorch,
        itertools,
        json,
        np,
        os,
        pd,
        plt,
        scipy,
        skimage,
        stats,
        time,
        torch,
    )


@app.cell
def _():
    # ANNI calculation taken from:
    # https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-statistics/h-how-average-nearest-neighbor-distance-spatial-st.htm
    return


@app.cell
def _(emd2_1d, itertools, np, pd, scipy):
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
    TRAJECTORY_INTERACTION_DISTANCE = 150

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
    return (
        COLUMNS,
        DATA_DIRECTORY,
        OUTPUT_COLUMN_NAMES,
        ROWS,
        SIMULATION_OUTPUTS_FOLDER,
        TIME_WEIGHTING,
        TRAJECTORY_INTERACTION_DISTANCE,
        analyse_simulation_site,
        analyse_site,
        generate_dataframe,
        get_trajectory_list_from_site,
        get_wd_estimate,
        whiten_distributions,
    )


@app.cell
def _(COLUMNS, DATA_DIRECTORY, ROWS, os, pd):
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
    return (
        column,
        csv_filename,
        row,
        site,
        site_dataframe,
        trajectory_dictionary,
        trajectory_folderpath,
    )


@app.cell
def _(np, skimage):
    def find_coherency_fraction(site_data):
        # Get valid cell indices:
        cell_index_list = list(set(list(site_data["tree_id"])))

        # Estimate from trajectory dataframe as test:
        line_array = np.zeros((1024, 1024))
        for cell_index in cell_index_list:
            particle_mask = site_data["tree_id"] == cell_index
            particle_data = site_data[particle_mask].sort_values("frame")
            xy_data = np.array(particle_data.loc[:, ["x", "y"]])
            for frame_index in range(len(xy_data) - 1):
                # Get indices of line:
                xy_t = xy_data[frame_index, :].astype(int)
                xy_t1 = xy_data[frame_index+1, :].astype(int)
    
                # Account for periodic boundaries:
                distance = np.sqrt(np.sum((xy_t - xy_t1)**2, axis=0))
                if distance > 1024:
                    continue
    
                # Plot line indices on matrix:
                _rr, _cc = skimage.draw.line(*xy_t, *xy_t1)
                line_array[_rr, _cc] += 1

        # Find orientations of lines:
        structure_tensor = skimage.feature.structure_tensor(
            line_array, sigma=32,
            mode='constant', cval=0,
            order='rc'
        )

        eigenvalues = skimage.feature.structure_tensor_eigenvalues(structure_tensor)
        coherency_numerator = eigenvalues[0, :, :] - eigenvalues[1, :, :]
        coherency_denominator = eigenvalues[0, :, :] + eigenvalues[1, :, :]
        coherency = coherency_numerator / coherency_denominator

        line_array_mask = line_array > 0
        coherency_fraction = np.sum(coherency[line_array_mask]) / np.sum(line_array)
        return coherency_fraction
    return (find_coherency_fraction,)


@app.cell
def _(find_coherency_fraction, trajectory_dictionary):
    cf_dictionary = {}
    for _column in ["1", "2", "3", "4", "5", "6"]:
        coherency_fractions = []
        for _i in range(12):
            site_data = trajectory_dictionary[_column][_i]
            coherency_fractions.append(find_coherency_fraction(site_data))
        cf_dictionary[_column] = coherency_fractions
    return cf_dictionary, coherency_fractions, site_data


@app.cell
def _(cf_dictionary, json, np):
    for _column in ["1", "2", "3", "4", "5", "6"]:
        print(np.mean(cf_dictionary[_column]))

    print(json.dumps(cf_dictionary, indent=4))
    return


@app.cell
def _(np, skimage, trajectory_dictionary):
    _site_data = trajectory_dictionary["1"][0]

    # Get valid cell indices:
    cell_index_list = list(set(list(_site_data["tree_id"])))

    # Estimate from trajectory dataframe as test:
    line_array = np.zeros((1024, 1024))
    for cell_index in cell_index_list:
        particle_mask = _site_data["tree_id"] == cell_index
        particle_data = _site_data[particle_mask].sort_values("frame")
        xy_data = np.array(particle_data.loc[:, ["x", "y"]])
        for frame_index in range(len(xy_data) - 1):
            # Get indices of line:
            xy_t = xy_data[frame_index, :].astype(int)
            xy_t1 = xy_data[frame_index+1, :].astype(int)

            # Account for periodic boundaries:
            distance = np.sqrt(np.sum((xy_t - xy_t1)**2, axis=0))
            if distance > 1024:
                continue

            # Plot line indices on matrix:
            _rr, _cc = skimage.draw.line(*xy_t, *xy_t1)
            line_array[_rr, _cc] += 1

    # Find orientations of lines:
    structure_tensor = skimage.feature.structure_tensor(
        line_array, sigma=32,
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
    return (
        cell_index,
        cell_index_list,
        coherency,
        coherency_denominator,
        coherency_fraction,
        coherency_numerator,
        distance,
        eigenvalues,
        frame_index,
        idling_factor,
        line_array,
        line_array_mask,
        particle_data,
        particle_mask,
        structure_tensor,
        xy_data,
        xy_t,
        xy_t1,
    )


@app.cell
def _(idling_factor):
    idling_factor
    return


@app.cell
def _(coherency_fraction):
    coherency_fraction
    return


@app.cell
def _(coherency_fraction):
    coherency_fraction
    return


@app.cell
def _(line_array, plt):
    plt.imshow(line_array, cmap='gray', interpolation='bilinear', vmin=0, vmax=1)
    return


@app.cell
def _(coherency, line_array_mask, plt):
    coherency[~line_array_mask] = 0

    plt.imshow(coherency, cmap='gray', interpolation='bilinear', vmin=0, vmax=1)
    return


@app.cell
def _():
    # coherency_map = np.zeros_like(line_array)
    # coherency_map[line_array] = coherency[line_array]
    return


@app.cell
def _(coherency, plt):
    plt.imshow(coherency)
    return


@app.cell
def _(structure_tensor):
    structure_tensor
    return


@app.cell
def _(DATA_DIRECTORY, os, pd):
    # Defining output of experimental data analysis:
    processed_dataset_filepath = os.path.join(DATA_DIRECTORY, "fitting_dataset.csv")
    experiment_dataframe = pd.read_csv(processed_dataset_filepath, index_col=0)
    return experiment_dataframe, processed_dataset_filepath


@app.cell
def _(experiment_dataframe):
    experiment_dataframe
    return


@app.cell
def _(OUTPUT_COLUMN_NAMES, os, pd):
    # Constant display variables:
    MESH_NUMBER = 64
    CELL_NUMBER = 250
    TIMESTEPS = 2880
    TIMESTEP_WIDTH = 1440
    WORLD_SIZE = 2048
    DATA_DIR_PATH = "./fileOutputs/"

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
    return (
        CELL_NUMBER,
        DATA_DIR_PATH,
        MESH_NUMBER,
        TIMESTEPS,
        TIMESTEP_WIDTH,
        WORLD_SIZE,
        get_trajectory_data,
    )


@app.cell
def _(np, skimage):
    def find_sim_coherency_fraction(positions_array):
        # Estimate from trajectory dataframe as test:
        line_array = np.zeros((1024, 1024))
        for cell_index in range(positions_array.shape[0]):
            xy_data = positions_array[cell_index, :, :] / 2
            for frame_index in range(len(xy_data) - 1):
                # Get indices of line:
                xy_t = np.floor(xy_data[frame_index, :]).astype(int)
                xy_t1 = np.floor(xy_data[frame_index+1, :]).astype(int)
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
    return (find_sim_coherency_fraction,)


@app.cell
def _(CELL_NUMBER, TIMESTEPS, get_trajectory_data, np):
    trajectory_dataframe = get_trajectory_data(0, 0)
    positions = trajectory_dataframe.sort_values(['particle', 'frame']).loc[:, ('x', 'y')]
    position_array = np.array(positions).reshape(CELL_NUMBER, TIMESTEPS, 2)
    position_array = position_array[:, 1440:, :]
    return position_array, positions, trajectory_dataframe


@app.cell
def _(np):
    def interpolate_to_wetlab_frames(position_array):
        differences = np.diff(position_array, axis=1)
        differences[differences > 1024] -= 2048
        differences[differences < -1024] += 2048

        interpolants = position_array[:, :-1, :] + (differences / 2)
        interpolated_array = []
        for _idx in range(288):
            interpolated_array.append(position_array[:, _idx*5, :])
            interpolated_array.append(interpolants[:, (_idx*5) + 2, :])

        interpolated_array = np.stack(interpolated_array, axis=1)
        interpolated_array[interpolated_array > 2048] -= 2048
        interpolated_array[interpolated_array < 0] += 2048
        return interpolated_array
    return (interpolate_to_wetlab_frames,)


@app.cell
def _(
    find_sim_coherency_fraction,
    interpolate_to_wetlab_frames,
    position_array,
):
    interpolated_array = interpolate_to_wetlab_frames(position_array)
    _cf, _if, larr = find_sim_coherency_fraction(interpolated_array)
    print(_cf)
    return interpolated_array, larr


@app.cell
def _(larr, plt):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(larr, vmin=0, vmax=2, cmap='gray', interpolation='bilinear')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()
    return ax, fig


@app.cell
def _(np):
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
    return (find_anni,)


@app.cell
def _(find_anni, position_array):
    anni_timeseries = []
    for _i in range(1440, 2880):
        anni_timeseries.append(find_anni(position_array[:, _i, :]))
    return (anni_timeseries,)


@app.cell
def _(anni_timeseries, plt):
    plt.plot(anni_timeseries)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
