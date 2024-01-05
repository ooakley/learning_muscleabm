import argparse
import os
import copy
import scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree

SIMULATION_OUTPUTS_FOLDER = "./fileOutputs/"
CSV_COLUMN_NAMES = [
    "frame", "particle", "x", "y", "orientation",
    "polarity_extent", "percept_direction", "percept_intensity",
    "actin_flow", "movement_direction", "turning_angle", "sampled_angle", "contact_inhibition"
]

TIMESTEP_WIDTH = 576


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a folder with a given integer name.')
    parser.add_argument('--folder_id', type=int, help='Integer corresponding to the folder name')
    args = parser.parse_args()
    return args


def read_matrix_into_numpy(filename, grid_size, timesteps):
    number_of_elements = (grid_size**2)*timesteps*2
    flattened_matrix = np.loadtxt(filename, delimiter=',', usecols=range(number_of_elements))
    return np.reshape(flattened_matrix, (timesteps, 2, grid_size, grid_size), order='C')


def find_persistence_times(orientation_array):
    # Calculating all pairwise angular differences in orientation timeseries:
    direction_differences = np.subtract.outer(orientation_array, orientation_array)

    # Ensuring distances are bounded between pi and -pi:
    direction_differences[direction_differences < -np.pi] += 2*np.pi
    direction_differences[direction_differences > np.pi] -= 2*np.pi
    direction_differences = np.abs(direction_differences)

    # Calculating number of frames that it takes for the angle to change by pi/2 (90 degrees):
    pt_list = []
    for timestep in range(orientation_array.shape[0]):
        subsequent_directions = direction_differences[timestep, timestep:]
        persistence_time = np.argmax(subsequent_directions > np.pi/2)
        if persistence_time == 0:
            continue
        pt_list.append(persistence_time)

    return pt_list


def analyse_particle(particle_dataframe):
    # Getting dx and dy, using these to calculate orientations and RMS displacements of
    # particle over time. We shift by four as we simulate every timestep as 2.5 minutes,
    # but we capture image data every ten minutes:
    dx = np.array(particle_dataframe.shift(4)['x'] - particle_dataframe['x'])
    dy = np.array(particle_dataframe.shift(4)['y'] - particle_dataframe['y'])

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


def plot_superiteration(
    trajectory_list, matrix_list, area_size, grid_size, timestep, iteration, ax
):
    # Accessing and formatting relevant dataframe:
    trajectory_dataframe = trajectory_list[iteration]
    x_mask = (trajectory_dataframe["x"] > 10) & (trajectory_dataframe["x"] < area_size - 10)
    y_mask = (trajectory_dataframe["y"] > 10) & (trajectory_dataframe["y"] < area_size - 10)
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
            c=colour_list[identity], alpha=0.4
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
    fig, axs = plt.subplots(3, 3, figsize=(7, 7))
    count = 0
    for i in range(3):
        for j in range(3):
            plot_superiteration(
                trajectory_list, matrix_list, area_size, grid_size, 575, count, axs[i, j]
            )
            count += 1

    fig.tight_layout()
    fig.savefig(os.path.join(subdirectory_path, "trajectories.png"))


def main():
    # Parsing command line arguments:
    args = parse_arguments()
    print(f"Processing folder with name: {args.folder_id}")

    # Getting simulation parameters:
    gridseach_dataframe = pd.read_csv("fileOutputs/gridsearch.txt", delimiter="\t")
    gridsearch_row = gridseach_dataframe[gridseach_dataframe["array_id"] == args.folder_id]
    GRID_SIZE = int(gridsearch_row["gridSize"])
    AREA_SIZE = int(gridsearch_row["worldSize"])
    TIMESTEPS = 576

    # Finding specified simulation output files:
    subdirectory_path = os.path.join(SIMULATION_OUTPUTS_FOLDER, str(args.folder_id))

    matrix_list = []
    particle_rmsd_list = []
    particle_pt_list = []
    particle_neighbour_list = []
    particle_seed_list = []
    trajectory_list = []
    sub_dataframes = []
    for seed in range(10):
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
                "speed_persistence_correlation_pval": [p_val]
            }
        )
        simulation_properties = copy.deepcopy(
            gridseach_dataframe[gridseach_dataframe["array_id"] == args.folder_id]
        )
        simulation_properties = simulation_properties.reset_index()
        iteration_row = pd.concat([simulation_properties, new_info], axis=1)
        sub_dataframes.append(iteration_row)

        # Reading final-step matrix information into list:
        matrix_filename = f"matrix_seed{seed:03d}.txt"
        matrix_filepath = os.path.join(subdirectory_path, matrix_filename)
        matrix = read_matrix_into_numpy(matrix_filepath, GRID_SIZE, TIMESTEPS)
        matrix_list.append(matrix[-1, :, :, :])

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
