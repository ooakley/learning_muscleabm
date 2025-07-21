import marimo

__generated_with = "0.12.10"
app = marimo.App(width="columns")


@app.cell
def _():
    import os
    import ot

    import scipy.stats
    import scipy.spatial

    import numpy as np
    import pandas as pd
    import seaborn as sns

    import matplotlib.pyplot as plt 

    DATA_DIRECTORY = "./wetlab_data/OEO20241206"
    ROWS = ["A", "B", "C"]
    COLUMNS = ["1", "2", "3", "4", "5", "6"]

    trajectory_folderpath = os.path.join(DATA_DIRECTORY, "trajectories")
    return (
        COLUMNS,
        DATA_DIRECTORY,
        ROWS,
        np,
        os,
        ot,
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
    )


@app.cell
def _(trajectory_dictionary):
    trajectory_dictionary["1"]
    return


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
def _(np, scipy):
    def get_mma_data(trajectory_list):
        # Find (approximate) valid position list from trajectories:
        total_x_positions = []
        total_y_positions = []
        total_frames = []
        trajectory_ids = []
        for id, trajectory in enumerate(trajectory_list):
            frame_list, x_positions, y_positions = trajectory
            total_x_positions.extend(x_positions)
            total_y_positions.extend(y_positions)
            total_frames.extend(frame_list)
            trajectory_ids.extend([id] * len(frame_list))

        mean_minimum_approach = []
        for id, trajectory in enumerate(trajectory_list):
            length = trajectory[0][-1] - trajectory[0][0]
            if length < 500:
                continue

            # Results list:
            approaches = []

            # Get trajectory tree:
            trajectory_mask = np.array(trajectory_ids) == id
            trajectory_positions = np.stack([
                np.array(total_x_positions)[trajectory_mask],
                np.array(total_y_positions)[trajectory_mask]
                ], axis=1
            )
            trajectory_tree = scipy.spatial.KDTree(trajectory_positions)

            # Get tree of other positions:
            environment_positions = np.stack([
                np.array(total_x_positions)[~trajectory_mask],
                np.array(total_y_positions)[~trajectory_mask]
                ], axis=1
            )
            environment_tree = scipy.spatial.KDTree(environment_positions)

            # Find nearest neighbours:
            point_pairs = trajectory_tree.query_ball_tree(environment_tree, 75)
            for trajectory_index, neighbouring_indices in enumerate(point_pairs):
                if len(neighbouring_indices) == 0:
                    continue
                source_position = trajectory_positions[trajectory_index, :]
                neighbour_positions = environment_positions[neighbouring_indices, :]
                distances = np.sqrt(
                    np.sum((neighbour_positions - source_position)**2, axis=1)
                )
                # Double the distances to correct for pixel binning:
                distances *= 2
                approaches.append(np.sum(1 / (1 + distances)))

            mean_minimum_approach.append(np.mean(approaches))

        return mean_minimum_approach
    return (get_mma_data,)


@app.cell
def _(
    COLUMNS,
    get_mma_data,
    get_trajectory_list_from_site,
    np,
    pd,
    scipy,
    trajectory_dictionary,
):
    # Iterate through sites:
    def analyse_site(column_key, site_index):
        # Get trajectories:
        trajectory_list = get_trajectory_list_from_site(
            trajectory_dictionary[column_key][site_index]
        )

        # Get MMA data:
        mma = get_mma_data(trajectory_list)

        # Find (approximate) valid position list from trajectories:
        positions_dictionary = {} # keys: frame, values: list of positions
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
            tree = scipy.spatial.KDTree(positions_array)
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

            speed.append(path_length / (length*2.5))
            meander_ratio.append(total_displacement / path_length)

        return speed, meander_ratio, trajectory_distances, mma

    def generate_dataframe(trajectory_dictionary):
        dataframe = pd.DataFrame(
            columns=["Speed", "Meander Ratio", "NN Distance", "MMA", "Column"]
        )
        for column in COLUMNS:
            print(f"Processing column {column}...")
            speeds = []
            meander_ratios = []
            nn_distances = []
            mmas = []
            for site_index in range(12):
                print(f"Processing site {site_index}...")
                # Get the analysis:
                speed, meander_ratio, nn_distance, mma = analyse_site(
                    column, site_index
                )
                # Add to lists:
                speeds.extend(speed)
                meander_ratios.extend(meander_ratio)
                nn_distances.extend(nn_distance)
                mmas.extend(mma)

            data_dictionary = {
                "Speed": speeds,
                "Meander Ratio": meander_ratios,
                "NN Distance":  nn_distances,
                "MMA": mmas,
                "Column": [column] * len(speeds)
            }
            column_dataframe = pd.DataFrame(data=data_dictionary)
            dataframe = pd.concat([dataframe, column_dataframe])

        return dataframe
    return analyse_site, generate_dataframe


@app.cell
def _(DATA_DIRECTORY, generate_dataframe, os, pd, trajectory_dictionary):
    processed_dataset_filepath = os.path.join(DATA_DIRECTORY, "fitting_dataset.csv")

    # experiment_dataframe = generate_dataframe(trajectory_dictionary)
    # experiment_dataframe.to_csv(processed_dataset_filepath)

    if not os.path.exists(processed_dataset_filepath):
        experiment_dataframe = generate_dataframe(trajectory_dictionary)
        experiment_dataframe.to_csv(processed_dataset_filepath)
    else:
        experiment_dataframe = pd.read_csv(processed_dataset_filepath, index_col=0)
    return experiment_dataframe, processed_dataset_filepath


@app.cell
def _(COLUMNS, experiment_dataframe, np):
    for _column in COLUMNS:
        print(np.count_nonzero(experiment_dataframe["Column"] == int(_column)))
    return


@app.cell
def _(experiment_dataframe):
    experiment_dataframe
    return


@app.cell
def _(experiment_dataframe, plt, sns):
    g = sns.jointplot(
        data=experiment_dataframe.reset_index(),
        x="Meander Ratio", y="MMA", hue='Column',
        kind="kde",
    )
    plt.show()
    return (g,)


@app.cell
def _(experiment_dataframe, plt, sns):
    _g = sns.histplot(
        data=experiment_dataframe.reset_index(),
        x="MMA", hue='Column', element="step" 
    )
    plt.show()
    return


@app.cell
def _(experiment_dataframe, plt, sns):
    _g = sns.histplot(
        data=experiment_dataframe.reset_index(),
        x="NN Distance", hue='Column', element="step" 
    )
    plt.show()
    return


@app.cell
def _(np, os, pd):
    # Constant display variables:
    MESH_NUMBER = 64
    CELL_NUMBER = 150
    TIMESTEPS = 2880
    TIMESTEP_WIDTH = 100
    WORLD_SIZE = 2048
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
        "stadium_x", "stadium_y",
        "turning_angle", "sampled_angle"
    ]

    def read_matrix_into_numpy(job_id, subiteration):
        # Determine filepaths:
        filename = f"matrix_seed{subiteration:03d}.txt"
        directory_path = os.path.join(DATA_DIR_PATH, str(job_id))
        filepath = os.path.join(directory_path, filename)

        # Calculate the exact number of matrix mesh elements:
        number_of_elements = (MESH_NUMBER**2)*TIMESTEPS*3
        flattened_matrix = np.loadtxt(
            filepath, delimiter=',',
            usecols=range(number_of_elements)
        )

        # Reshape into (timesteps, [density/orientation/anisotropy], grid, grid):
        dimensions = (TIMESTEPS, 3, MESH_NUMBER, MESH_NUMBER)
        return np.reshape(flattened_matrix, dimensions, order='C')


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
        OUTPUT_COLUMN_NAMES,
        TIMESTEPS,
        TIMESTEP_WIDTH,
        WORLD_SIZE,
        get_trajectory_data,
        read_matrix_into_numpy,
    )


@app.cell
def _(get_trajectory_data):
    trajectory_dataframe = get_trajectory_data(0, 0)
    return (trajectory_dataframe,)


@app.cell
def _(trajectory_dataframe):
    positions = trajectory_dataframe.sort_values(['particle', 'frame']).loc[:, ('x', 'y')]
    return (positions,)


@app.cell
def _(positions):
    positions.tail()
    return


@app.cell
def _(CELL_NUMBER, TIMESTEPS, np, positions):
    position_array = np.array(positions).reshape(CELL_NUMBER, TIMESTEPS, 2)
    return (position_array,)


@app.cell
def _(np, scipy):
    def get_model_mma(position_array):
        mean_minimum_approach = []
    
        for id in range(position_array.shape[0]):
            # Results list:
            approaches = []

            # Get trajectory tree:
            trajectory_positions = position_array[id, :, :]
            trajectory_tree = scipy.spatial.KDTree(trajectory_positions)

            # Get tree of other positions:
            environment_positions = np.concatenate([
                position_array[:id, :, :],
                position_array[id+1:, :, :]
                ], axis=0
            )
            total_positions = \
                environment_positions.shape[0] * environment_positions.shape[1]
            environment_positions = np.reshape(environment_positions, (total_positions, 2))
            environment_tree = scipy.spatial.KDTree(environment_positions)

            # Find nearest neighbours:
            point_pairs = trajectory_tree.query_ball_tree(environment_tree, 150)
            for trajectory_index, neighbouring_indices in enumerate(point_pairs):
                if len(neighbouring_indices) == 0:
                    continue
                source_position = trajectory_positions[trajectory_index, :]
                neighbour_positions = environment_positions[neighbouring_indices, :]
                distances = np.sqrt(
                    np.sum((neighbour_positions - source_position)**2, axis=1)
                ) 
                distances /= 2
                approaches.append(np.sum(1 / (1 + distances)))

            mean_minimum_approach.append(np.mean(approaches) * 2)

        return mean_minimum_approach
    return (get_model_mma,)


@app.cell
def _(np, position_array, scipy):
    # Generate KD-Tree for each frame:
    kd_tree_list = {}
    for timestep in range(1440, 2880):
        tree = scipy.spatial.KDTree(position_array[:, timestep, :])
        kd_tree_list[timestep] = tree

    sim_nn_distance = []
    sim_speed = []
    sim_mr = []
    for cell_index in range(position_array.shape[0]):
        # Get nearest neighbour distances
        timestep_distances = []
        for timestep in range(1440, 2880):
            query_position  = [
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
    return (
        cell_index,
        distances,
        kd_tree_list,
        path_length,
        query_position,
        sim_mr,
        sim_nn_distance,
        sim_speed,
        timestep,
        timestep_distances,
        total_displacement,
        total_dx,
        total_dy,
        tree,
        x,
        x_displacements,
        y,
        y_displacements,
    )


@app.cell
def _(get_model_mma, position_array):
    mmas = get_model_mma(position_array[:, 1440:2880, :])
    return (mmas,)


@app.cell
def _(experiment_dataframe, mmas, np, plt):
    plt.hist(np.array(mmas), density=True, alpha=0.5)
    plt.hist(experiment_dataframe["MMA"], density=True, alpha=0.5)
    plt.show()
    return


@app.cell
def _(plt, sim_nn_distance):
    plt.hist(sim_nn_distance, bins=50);
    plt.show()
    return


@app.cell
def _(experiment_dataframe, mmas, np, sim_mr, sim_nn_distance, sim_speed):
    a_mask = experiment_dataframe["Column"] == 5
    experiment_distribution = np.array(
        experiment_dataframe.loc[a_mask, ["Speed", "Meander Ratio", "NN Distance", "MMA"]]
    )
    simulation_distribution = np.stack([
        sim_speed, sim_mr, sim_nn_distance, np.array(mmas) / 2
        ], axis=1
    )
    return a_mask, experiment_distribution, simulation_distribution


@app.cell
def _(experiment_distribution, np):
    np.count_nonzero(np.isnan(experiment_distribution))
    return


@app.cell
def _(experiment_distribution, plt, simulation_distribution):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    _ax.scatter(experiment_distribution[:, 1], experiment_distribution[:, 2])
    _ax.scatter(simulation_distribution[:, 1], simulation_distribution[:, 2])

    # _ax.set_xlim(0, 1)
    # _ax.set_ylim(0, 1)

    plt.show()
    return


@app.cell
def _():
    # from ot.bregman import empirical_sinkhorn_divergence

    # def whiten_distributions(experiment_distribution, simulation_distribution):
    #     # Determine whitening transform:
    #     means = np.mean(experiment_distribution, axis=0, keepdims=True)
    #     vars = np.std(experiment_distribution, axis=0, keepdims=True)

    #     # Transform experiment data:
    #     transformed_experiment = (experiment_distribution - means) / vars
    #     transformed_simulation = (simulation_distribution - means) / vars

    #     return transformed_experiment, transformed_simulation

    # def get_wd_estimate(experiment_distribution, simulation_distribution, reg=1):
    #     # Whiten data:
    #     transformed_experiment, transformed_simulation = \
    #         whiten_distributions(experiment_distribution, simulation_distribution)

    #     # Determining sampling strategy:
    #     sample_size = len(experiment_distribution)
    #     weights = np.ones(sample_size) / sample_size
    #     rng = np.random.default_rng(0)

    #     distances = []
    #     for i in range(50):
    #         sample_indices = rng.choice(len(transformed_simulation), sample_size)
    #         sampled_sim = transformed_simulation[sample_indices, :]

    #         # Estimate distance with Sinkhorn:
    #         distances.append(
    #             ot.bregman.empirical_sinkhorn_divergence(
    #                 transformed_experiment, sampled_sim, reg
    #             )
    #         )

        # return distances
    return


@app.cell
def _(np):
    from ot.bregman import empirical_sinkhorn_divergence

    def whiten_distributions(experiment_distribution, simulation_distribution):
        # Determine whitening transform:
        exp_means = np.mean(experiment_distribution, axis=0, keepdims=True)
        print(exp_means)
        exp_variances = np.std(experiment_distribution, axis=0, keepdims=True)
        print(exp_variances)

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
    return empirical_sinkhorn_divergence, get_wd_estimate, whiten_distributions


@app.cell
def _(transformed_experiment):
    transformed_experiment[:, 3]
    return


@app.cell
def _(
    experiment_distribution,
    plt,
    simulation_distribution,
    whiten_distributions,
):
    transformed_experiment, transformed_simulation = \
        whiten_distributions(experiment_distribution, simulation_distribution)

    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    _ax.scatter(transformed_experiment[:, 0], transformed_experiment[:, 2], alpha=1)
    _ax.scatter(transformed_simulation[:, 0], transformed_simulation[:, 2], alpha=1)

    # _ax.set_xlim(-4.5, 4.5)
    # _ax.set_ylim(-4.5, 4.5)

    plt.show()
    return transformed_experiment, transformed_simulation


@app.cell
def _(experiment_distribution, get_wd_estimate, simulation_distribution):
    wd_distances = get_wd_estimate(experiment_distribution, simulation_distribution, reg=1)
    return (wd_distances,)


@app.cell
def _(np, plt, wd_distances):
    print(np.mean(wd_distances))
    plt.hist(wd_distances);
    plt.show()
    return


@app.cell
def _(experiment_distribution, get_wd_estimate, simulation_distribution):
    reg_distances = get_wd_estimate(
        experiment_distribution, simulation_distribution, reg=1
    )
    return (reg_distances,)


@app.cell
def _(np, plt, reg_distances):
    # plt.hist(distances);
    print(np.mean(reg_distances))
    plt.hist(reg_distances);
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
