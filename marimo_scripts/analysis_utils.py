import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
colors_list = list(mcolors.TABLEAU_COLORS.values())

DATA_DIR_PATH = "/Users/eloaklo/Documents/GitHub/learning_muscleabm/fileOutputs/"
OUTPUT_COLUMN_NAMES = [
    "frame", "particle", "x", "y",
    "shapeDirection",
    "orientation", "polarity_extent",
    "percept_direction", "percept_intensity",
    "actin_flow", "actin_mag",
    "movement_direction",
    "turning_angle", "sampled_angle"
]

MESH_NUMBER = 32
TIMESTEPS = 576

TIMESTEP_WIDTH = 500
WORLD_SIZE = 2048

def read_matrix_into_numpy(filename, mesh_number, timesteps):
    number_of_elements = (mesh_number**2)*timesteps*3
    flattened_matrix = np.loadtxt(filename, delimiter=',', usecols=range(number_of_elements))
    return np.reshape(flattened_matrix, (timesteps, 3, mesh_number, mesh_number), order='C')

def get_model_outputs(job_id, output_directory_path=DATA_DIR_PATH, mesh_number=MESH_NUMBER, timesteps=TIMESTEPS, superiterations=10):
    directory_path = os.path.join(output_directory_path, str(job_id))
    output_files = os.listdir(directory_path)
    output_files.sort()

    # Getting matrix data:
    matrix_list = []
    for i in range(superiterations):
        filename = f"matrix_seed{i:03d}.txt"
        filepath = os.path.join(directory_path, filename)
        matrix_list.append(read_matrix_into_numpy(filepath, mesh_number, timesteps))

    # Getting cell trajectory data:
    trajectory_list = []
    for i in range(superiterations):
        filename = f"positions_seed{i:03d}.csv"
        filepath = os.path.join(directory_path, filename)
        trajectory_dataframe = pd.read_csv(filepath, index_col=None, header=None, names=OUTPUT_COLUMN_NAMES)
        trajectory_list.append(trajectory_dataframe)

    return matrix_list, trajectory_list

def plot_orientation_timepoint(matrix, timepoint, return_array=False):
    fig, ax = plt.subplots(figsize=(5, 5), layout='constrained')
    ax.imshow(matrix[timepoint, 0, :, :], cmap='hsv', alpha=matrix[timepoint, 1, :, :], vmin=0, vmax=np.pi)
    ax.set_axis_off()

    if return_array:
        fig.canvas.draw()
        array_plot = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        return array_plot

def plot_anisotropy_timepoint(matrix, timepoint, return_array=False):
    fig, ax = plt.subplots(figsize=(5, 5), layout='constrained')
    # , vmin=0, vmax=1
    ax.imshow(matrix[timepoint, 2, :, :], cmap='gray_r', vmin=0, vmax=1)
    ax.set_axis_off()

    if return_array:
        fig.canvas.draw()
        array_plot = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        return array_plot

def plot_superiteration(trajectory_dataframe, matrix_series, timestep, ax, cell_size):
    # Accessing and formatting relevant dataframe:
    x_mask = (trajectory_dataframe["x"] > 0) & (trajectory_dataframe["x"] < WORLD_SIZE)
    y_mask = (trajectory_dataframe["y"] > 0) & (trajectory_dataframe["y"] < WORLD_SIZE)
    full_mask = x_mask & y_mask
    rollover_skipped_df = trajectory_dataframe[full_mask]
    timeframe_mask = (rollover_skipped_df["frame"] > timestep - TIMESTEP_WIDTH) & (rollover_skipped_df["frame"] <= timestep)
    timepoint_mask = rollover_skipped_df["frame"] == timestep
    ci_lookup = trajectory_dataframe[trajectory_dataframe["frame"] == 1]
    unstacked_dataframe = rollover_skipped_df[timeframe_mask].set_index(['particle', 'frame'])[['x', 'y']].unstack()

    # Setting up matrix plotting:
    tile_width = WORLD_SIZE / MESH_NUMBER
    X = np.arange(0, WORLD_SIZE, tile_width) + (tile_width / 2)
    Y = np.arange(0, WORLD_SIZE, tile_width) + (tile_width / 2)
    X, Y = np.meshgrid(X, Y)

    matrix = matrix_series[timestep, 0, :, :]
    U = np.cos(matrix) * matrix_series[timestep, 1, :, :]
    V = -np.sin(matrix) * matrix_series[timestep, 1, :, :]

    # Setting up plot
    # colour_list = ['r', 'g']

    # Plotting particle trajectories:
    for i, trajectory in unstacked_dataframe.iterrows():
        identity = int(ci_lookup[ci_lookup["particle"] == i]["contact_inhibition"].iloc[0])

        rollover_x = np.abs(np.diff(np.array(trajectory['x'])[::4])) > (WORLD_SIZE/2)
        rollover_y = np.abs(np.diff(np.array(trajectory['y'])[::4])) > (WORLD_SIZE/2)
        rollover_mask = rollover_x | rollover_y
        
        color = colors_list[i % len(colors_list)]

        if np.count_nonzero(rollover_mask) == 0:
            ax.plot(np.array(trajectory['x'])[::4], np.array(trajectory['y'])[::4],
                alpha=0.8, linewidth=1, c=color
            )
        else:
            plot_separation_indices = np.argwhere(rollover_mask)
            prev_index = 0
            for separation_index in plot_separation_indices:
                separation_index = separation_index[0]
                x_array = np.array(trajectory['x'])[::4][prev_index:separation_index]
                y_array = np.array(trajectory['y'])[::4][prev_index:separation_index]
                ax.plot(x_array, y_array, alpha=0.8, linewidth=1, c=color)
                prev_index = separation_index+1

            # Plotting final segment:
            x_array = np.array(trajectory['x'])[::4][prev_index:]
            y_array = np.array(trajectory['y'])[::4][prev_index:]
            ax.plot(x_array, y_array, alpha=0.8, linewidth=1, c=color)

    # Plotting background matrix:
    try:
        speed = np.sqrt(U**2 + V**2)
        alpha = speed / speed.max()
        ax.quiver(X, Y, U, V, [matrix], cmap='twilight', pivot='mid', scale=50, headwidth=0, headlength=0, headaxislength=0, alpha=alpha)
    except:
        print("Low Matrix Density")
    # ax.streamplot(
    #     X, Y, U, -V, linewidth=0.5, arrowsize=1e-5, density=2
    # )

    # Plotting cells & their directions:
    type0_mask = rollover_skipped_df['contact_inhibition'][timepoint_mask] == 0
    type1_mask = rollover_skipped_df['contact_inhibition'][timepoint_mask] == 1
    x_pos = rollover_skipped_df['x'][timepoint_mask]
    y_pos = rollover_skipped_df['y'][timepoint_mask]
    x_heading = np.cos(rollover_skipped_df['orientation'][timepoint_mask]) * rollover_skipped_df["polarity_extent"][timepoint_mask] * 2
    y_heading = - np.sin(rollover_skipped_df['orientation'][timepoint_mask]) * rollover_skipped_df["polarity_extent"][timepoint_mask] * 2
    heading_list = rollover_skipped_df['orientation'][timepoint_mask]
    # , color='r',
    ax.quiver(
        x_pos[type0_mask], y_pos[type0_mask], x_heading[type0_mask], y_heading[type0_mask],
        pivot='tail', scale=50, headwidth=5, headlength=5, headaxislength=3, width=0.002, alpha=0.8
    )
    ax.quiver(
        x_pos[type1_mask], y_pos[type1_mask], x_heading[type1_mask], y_heading[type1_mask],
        pivot='tail', scale=50, color='g', headwidth=5, headlength=5, headaxislength=3, width=0.002, alpha=0.8
    )
    ax.scatter(x_pos, y_pos, color='k', alpha=0.5, s=cell_size)
    ax.set_xlim(0, WORLD_SIZE)
    ax.set_ylim(0, WORLD_SIZE)
    ax.invert_yaxis()
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(2)
    ax.set_xticks([])
    ax.set_yticks([])