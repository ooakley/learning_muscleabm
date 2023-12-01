import argparse
import os
import copy
import scipy

import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

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


def find_average_rmsd(trajectory_dataframe):
    particles = list(set(list(trajectory_dataframe["particle"])))
    rmsd_list = []
    for particle in particles:
        particle_dataframe = trajectory_dataframe[trajectory_dataframe["particle"] == particle]
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

        # orientation = np.arctan2(dy, dx)
        instantaneous_speeds = np.sqrt(dx**2, dy**2)
        instantaneous_speeds = instantaneous_speeds[~np.isnan(instantaneous_speeds)]

        rmsd_list.append(np.mean(instantaneous_speeds))

    return rmsd_list


def find_persistence_time(trajectory_dataframe):
    particles = list(set(list(trajectory_dataframe["particle"])))
    pt_list = []

    for particle in particles:
        particle_directions = np.array(
            trajectory_dataframe[trajectory_dataframe["particle"] == particle]["movement_direction"]
        )

        direction_differences = np.subtract.outer(particle_directions, particle_directions)
        direction_differences[direction_differences < -np.pi] += 2*np.pi
        direction_differences[direction_differences > np.pi] -= 2*np.pi
        direction_differences = np.abs(direction_differences)

        persistence_time_list = []
        for timestep in range(len(particle_directions)):
            subsequent_directions = direction_differences[timestep, timestep:]
            persistence_time = np.argmax(subsequent_directions > np.pi/2)
            if persistence_time == 0:
                continue
            persistence_time_list.append(persistence_time)

        pt_list.append(np.mean(np.log(persistence_time_list)))

    return pt_list


def plot_superiteration(trajectory_list, matrix_list, area_size, grid_size, timestep, iteration, ax):
    # Accessing and formatting relevant dataframe:
    trajectory_dataframe = trajectory_list[iteration]
    x_mask = (trajectory_dataframe["x"] > 10) & (trajectory_dataframe["x"] < area_size - 10)
    y_mask = (trajectory_dataframe["y"] > 10) & (trajectory_dataframe["y"] < area_size - 10)
    full_mask = x_mask & y_mask
    rollover_skipped_df = trajectory_dataframe[full_mask]
    timeframe_mask = (rollover_skipped_df["frame"] > timestep - TIMESTEP_WIDTH) & (rollover_skipped_df["frame"] <= timestep)
    timepoint_mask = rollover_skipped_df["frame"] == timestep
    ci_lookup = trajectory_dataframe[trajectory_dataframe["frame"] == 1]
    unstacked_dataframe = rollover_skipped_df[timeframe_mask].set_index(['particle', 'frame'])[['x', 'y']].unstack()

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
        ax.plot(trajectory['x'], trajectory['y'],
            c=colour_list[identity], alpha=0.4
        )

    # Plotting background matrix:
    ax.quiver(X, Y, U, V, [matrix], cmap='twilight', pivot='mid', scale=75, headwidth=0, headlength=0, headaxislength=0, alpha=0.5)

    # Plotting cells & their directions:
    type0_mask = rollover_skipped_df['contact_inhibition'][timepoint_mask] == 0
    type1_mask = rollover_skipped_df['contact_inhibition'][timepoint_mask] == 1
    x_pos = rollover_skipped_df['x'][timepoint_mask]
    y_pos = rollover_skipped_df['y'][timepoint_mask]
    x_heading = np.cos(rollover_skipped_df['orientation'][timepoint_mask])
    y_heading = - np.sin(rollover_skipped_df['orientation'][timepoint_mask])
    heading_list = rollover_skipped_df['orientation'][timepoint_mask]
    
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
            plot_superiteration(trajectory_list, matrix_list, area_size, grid_size, 999, count, axs[i, j])
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
    TIMESTEPS = 1000

    # Finding specified simulation output files:
    subdirectory_path = os.path.join(SIMULATION_OUTPUTS_FOLDER, str(args.folder_id))
    output_files = os.listdir(subdirectory_path)

    matrix_list = []
    particle_rmsd_list = []
    particle_pt_list = []
    trajectory_list = []
    sub_dataframes = []
    for seed in range(10):
        # Reading trajectory information:
        csv_filename = f"positions_seed{seed:03d}.csv"
        csv_filepath = os.path.join(subdirectory_path, csv_filename)
        trajectory_dataframe = pd.read_csv(csv_filepath, index_col=None, header=None, names=CSV_COLUMN_NAMES)
        trajectory_list.append(trajectory_dataframe)

        # Calculating derived statistics:
        rmsd_list = find_average_rmsd(trajectory_dataframe)
        particle_rmsd_list.extend(rmsd_list)

        pt_list = find_persistence_time(trajectory_dataframe)
        particle_pt_list.extend(pt_list)
        try:
            persistence_speed_corrcoef, p_val = scipy.stats.pearsonr(rmsd_list, pt_list)
        except:
            persistence_speed_corrcoef = np.nan
            p_val = np.nan

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
        simulation_properties = copy.deepcopy(gridseach_dataframe[gridseach_dataframe["array_id"] == args.folder_id])
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
    print(summary_dataframe)
    summary_dataframe.to_csv(os.path.join(subdirectory_path, "summary.csv"))

    # Saving particle speed distributions to subdirectory:
    particle_rmsd_df = pd.DataFrame({"particle_rmsd": particle_rmsd_list})
    particle_rmsd_df.to_csv(os.path.join(subdirectory_path, "speeds.csv"))

    particle_pt_df = pd.DataFrame({"particle_persistence_time": particle_pt_list})
    particle_pt_df.to_csv(os.path.join(subdirectory_path, "persistence_times.csv"))

    # Plotting final orientations of the matrix:
    fig, ax = plt.subplots(figsize=(7, 4))
    distributions = []
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