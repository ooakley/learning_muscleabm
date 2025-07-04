import marimo

__generated_with = "0.12.10"
app = marimo.App(width="columns")


@app.cell
def _():
    import os

    import scipy.stats

    import numpy as np
    import pandas as pd
    import seaborn as sns

    import matplotlib.pyplot as plt 

    DATA_DIRECTORY = "/Users/eloaklo/Documents/Data/OEO20241206"
    ROWS = ["A", "B", "C"]
    COLUMNS = ["1", "2", "3", "4", "5", "6"]

    trajectory_folderpath = os.path.join(DATA_DIRECTORY, "trajectories")
    return (
        COLUMNS,
        DATA_DIRECTORY,
        ROWS,
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
            trajectory_list.append((frame_array, x_pos, y_pos))

        return trajectory_list
    return (get_trajectory_list_from_site,)


@app.cell
def _(COLUMNS, get_trajectory_list_from_site, np, pd, trajectory_dictionary):
    def get_column(column_key):
        dataset = []
        for _i in range(12):
            dataset.extend(
                get_trajectory_list_from_site(
                    trajectory_dictionary[column_key][_i]
                )
            )
        return dataset

    def generate_dataframe():
        dataframe = pd.DataFrame(columns=["Speed", "Meander Ratio", "Column"])
        for column in COLUMNS:
            dataset = get_column(column)
            speed = []
            meander_ratio = []
            for trajectory in dataset: 
                frames = trajectory[0]
                length = frames[-1] - frames[0]

                if length == 576:
                    x = trajectory[1]
                    y = trajectory[2]
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

            data_dictionary = {
                "Speed": speed,
                "Meander Ratio": meander_ratio,
                "Column": [column] * len(speed)
            }
            column_dataframe = pd.DataFrame(data=data_dictionary)
            dataframe = pd.concat([dataframe, column_dataframe])

        return dataframe
    return generate_dataframe, get_column


@app.cell
def _(generate_dataframe):
    experiment_dataframe = generate_dataframe()
    return (experiment_dataframe,)


@app.cell
def _(experiment_dataframe):
    experiment_dataframe
    return


@app.cell
def _(experiment_dataframe, plt, sns):
    g = sns.jointplot(
        data=experiment_dataframe,
        x="Speed", y="Meander Ratio", hue="Column",
        kind="kde",
    )
    plt.show()
    return (g,)


@app.cell
def _(np, scipy):
    def get_wd_estimate(a_distribution, b_distribution):
        distances = []
        rng = np.random.default_rng(0)
    
        for i in range(200):
            a_indices = rng.choice(len(a_distribution), 100)
            a_sampled = np.array(a_distribution)[a_indices, :]

            b_indices = rng.choice(len(b_distribution), 100)
            b_sampled = np.array(b_distribution)[b_indices, :]
    
            distance = scipy.stats.wasserstein_distance_nd(
                a_sampled, b_sampled
            )
            distances.append(distance)

        return distances
    return (get_wd_estimate,)


@app.cell
def _(distances, plt):
    plt.hist(distances);
    plt.show()
    return


@app.cell
def _(distances, np):
    np.mean(distances)
    return


@app.cell
def _(distances, plt):
    plt.hist(distances);
    plt.show()
    return


@app.cell
def _(np, os, pd):
    # Constant display variables:
    MESH_NUMBER = 64
    CELL_NUMBER = 250
    TIMESTEPS = 1440
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
def _(np, plt, position_array):
    sim_speed = []
    sim_mr = []
    for cell_index in range(position_array.shape[0]):
        x = position_array[cell_index, :, 0]
        y = position_array[cell_index, :, 1]
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

    plt.hist(x_displacements, bins=100);
    plt.show()
    return (
        cell_index,
        path_length,
        sim_mr,
        sim_speed,
        total_displacement,
        total_dx,
        total_dy,
        x,
        x_displacements,
        y,
        y_displacements,
    )


@app.cell
def _(a_distribution, plt, sim_mr, sim_speed):
    _fig, _ax = plt.subplots()

    _ax.scatter(sim_speed, sim_mr)
    _ax.scatter(a_distribution[::10, 0], a_distribution[::10, 1])

    # _ax.set_xlim(0, 25)
    _ax.set_ylim(0, 1)

    plt.show()
    return


@app.cell
def _(experiment_dataframe, get_wd_estimate, np, sim_mr, sim_speed):
    a_mask = experiment_dataframe["Column"] == "1"
    a_distribution = np.array(experiment_dataframe.loc[a_mask, "Speed":"Meander Ratio"])
    b_distribution = np.stack([sim_speed, sim_mr], axis=1)

    distances = get_wd_estimate(a_distribution, b_distribution)
    return a_distribution, a_mask, b_distribution, distances


@app.cell
def _(a_distribution):
    a_distribution
    return


@app.cell
def _(distances, plt):
    plt.hist(distances);
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
