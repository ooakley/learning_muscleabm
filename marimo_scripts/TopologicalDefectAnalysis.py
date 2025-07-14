import marimo

__generated_with = "0.12.10"
app = marimo.App(width="columns")


@app.cell
def _():
    import cv2
    import matplotlib
    import os
    import copy

    import numpy as np
    import pandas as pd
    import colorcet as cc

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    colors_list = list(mcolors.TABLEAU_COLORS.values())
    return cc, colors_list, copy, cv2, matplotlib, mcolors, np, os, pd, plt


@app.cell
def _():
    # Constant display variables:
    MESH_NUMBER = 64
    CELL_NUMBER = 293
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
    return (
        CELL_NUMBER,
        DATA_DIR_PATH,
        MESH_NUMBER,
        OUTPUT_COLUMN_NAMES,
        TIMESTEPS,
        TIMESTEP_WIDTH,
        WORLD_SIZE,
    )


@app.cell
def _(DATA_DIR_PATH, MESH_NUMBER, OUTPUT_COLUMN_NAMES, TIMESTEPS, np, os, pd):
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
    return get_trajectory_data, read_matrix_into_numpy


@app.cell
def _(read_matrix_into_numpy):
    ecm_matrix = read_matrix_into_numpy(0, 0)
    return (ecm_matrix,)


@app.cell
def _(TIMESTEPS, cc, ecm_matrix, np, plt):
    timestep = TIMESTEPS - 1
    plt.imshow(
        ecm_matrix[timestep, 0, :, :], alpha=ecm_matrix[timestep, 1, :, :],
        cmap=cc.cm.CET_C6, clim=(0, np.pi)
    )
    return (timestep,)


@app.cell
def _(cc, ecm_matrix, np, plt, timestep):
    tiled_orientation = np.tile(ecm_matrix[timestep, 0, :, :], (3, 3))
    tiled_alpha = np.tile(ecm_matrix[timestep, 1, :, :], (3, 3))

    _fig, _ax = plt.subplots(figsize=(8, 8))
    _ax.imshow(
        tiled_orientation, alpha=tiled_alpha,
        cmap=cc.cm.CET_C6, clim=(0, np.pi)
    )
    return tiled_alpha, tiled_orientation


@app.cell
def _(MESH_NUMBER, np):
    def roll_index(index, mesh_number=MESH_NUMBER):
        while index >= MESH_NUMBER:
            index -= MESH_NUMBER
        while index < 0:
            index += MESH_NUMBER
        return index

    def nematic_modulus(angle):
        return angle - (np.pi * np.floor(0.5 + angle/np.pi))
    return nematic_modulus, roll_index


@app.cell
def _(MESH_NUMBER, nematic_modulus, np, roll_index):
    def get_td_map(orientation_matrix, coherence_matrix):
        # Instantiate output matrix:
        td_out = np.zeros_like(orientation_matrix)
        robustness_out = np.zeros_like(orientation_matrix)
        for i_index in range(MESH_NUMBER):
            for j_index in range(MESH_NUMBER):
                # Get counterclockwise path around 3x3 area:
                i_1, i1 = roll_index(i_index - 1), roll_index(i_index + 1)
                j_1, j1 = roll_index(j_index - 1), roll_index(j_index + 1)
                path_indices = [
                    (i1, j1),
                    (i1, j_index),
                    (i1, j_1),
                    (i_index, j_1),
                    (i_1, j_1),
                    (i_1, j_index),
                    (i_1, j1),
                    (i_index, j1),
                    (i1, j1)
                ]

                # Cycle through path to get TCE:
                tce_sum = 0
                robustness_of_edges = []
                for path_index in range(len(path_indices)):
                    # Calculate smallest net azimuth change:
                    azimuth_n_prev = orientation_matrix[*path_indices[path_index-1]]
                    azimuth_n = orientation_matrix[*path_indices[path_index]]
                    edge_modulus = nematic_modulus(azimuth_n - azimuth_n_prev)
                    # edge_robustness = (np.pi / 2) - np.abs(edge_modulus)

                    # Get local matrix density:
                    density_n_prev = coherence_matrix[*path_indices[path_index-1]]
                    robustness = density_n_prev*np.cos(edge_modulus)

                    # Accumulate:
                    tce_sum += edge_modulus
                    robustness_of_edges.append(robustness)

                td_out[i_index, j_index] = tce_sum / (2*np.pi)
                robustness_out[i_index, j_index] = np.min(robustness_of_edges)

        return td_out, robustness_out
    return (get_td_map,)


@app.cell
def _(cc, copy, ecm_matrix, get_td_map, np, plt):
    _td_map, _rob_map = get_td_map(ecm_matrix[-1, 0, :, :], ecm_matrix[-1, 1, :, :])

    _fig, _axs = plt.subplots(1, 2, figsize=(8, 4), layout='constrained')
    masked_td = copy.deepcopy(_td_map)
    masked_td[_rob_map < 0.1] = 0

    _axs[0].imshow(
        masked_td,
        cmap='gray', clim=(-1, 1)
    )
    _axs[0].set_xticks([])
    _axs[0].set_yticks([])

    _axs[1].imshow(
        ecm_matrix[-1, 0, :, :], alpha=ecm_matrix[-1, 1, :, :],
        cmap=cc.cm.CET_C6, clim=(0, np.pi)
    )
    _axs[1].set_xticks([])
    _axs[1].set_yticks([])

    plt.show()
    return (masked_td,)


@app.cell
def _(TIMESTEPS, cc, cv2, ecm_matrix, get_td_map, np, plt):
    size = 400, 800
    fps = 60
    out = cv2.VideoWriter(
        './td_video.mp4', cv2.VideoWriter_fourcc(*'avc1'),
        fps, (size[1], size[0]), True
    )

    for timeframe in list(range(TIMESTEPS))[::25]:
        if (timeframe) % 1000 == 0:
            print(timeframe)

        # Get topological defects:
        td_map, rob_map = get_td_map(
            ecm_matrix[timeframe, 0, :, :], ecm_matrix[timeframe, 1, :, :]
        )
        td_map[rob_map > 0.05] = 0

        # # Plot defects:
        # _fig, _ax = plt.subplots(figsize=(4, 4), layout='constrained')
        # _ax.imshow(
        #     td_map,
        #     cmap='gray', clim=(-0.5, 0.5)
        # )
        # _ax.set_xticks([])
        # _ax.set_yticks([])

        _fig, _axs = plt.subplots(1, 2, figsize=(8, 4), layout='constrained')

        _axs[0].imshow(
            td_map,
            cmap='gray', clim=(-0.5, 0.5)
        )
        _axs[0].set_xticks([])
        _axs[0].set_yticks([])
    
        _axs[1].imshow(
            ecm_matrix[timeframe, 0, :, :], alpha=ecm_matrix[timeframe, 1, :, :],
            cmap=cc.cm.CET_C6, clim=(0, np.pi)
        )
        _axs[1].set_xticks([])
        _axs[1].set_yticks([])

        # Export image to array:
        _fig.canvas.draw()
        array_plot = np.array(_fig.canvas.renderer.buffer_rgba())
        plt.close(_fig)

        # Save array plot to opencv file:
        bgr_data = cv2.cvtColor(array_plot, cv2.COLOR_RGB2BGR)
        out.write(bgr_data)

    out.release() 
    return array_plot, bgr_data, fps, out, rob_map, size, td_map, timeframe


@app.cell
def _(ecm_matrix, get_td_map, plt):
    _td_map, _rob_map = get_td_map(ecm_matrix[2500, 0, :, :])

    _fig, _ax = plt.subplots(figsize=(4, 4))

    _ax.imshow(
        _rob_map,
        cmap='gray', clim=(0, 1.5)
    )
    _ax.set_xticks([])
    _ax.set_yticks([])

    plt.show()
    return


@app.cell
def _(TIMESTEPS, ecm_matrix, get_td_map, np):
    defection_history = []
    for _timeframe in list(range(TIMESTEPS))[::25]:
        if (_timeframe) % 200 == 0:
            print(_timeframe)

        # Get topological defects:
        _td_map, _rob_map = get_td_map(
            ecm_matrix[_timeframe, 0, :, :], ecm_matrix[_timeframe, 1, :, :]
        )

        # Get robustness-weighted sum of defects:
        defection = np.sum(np.abs(_td_map) * _rob_map)
        defection_history.append(defection)
    return defection, defection_history


@app.cell
def _(defection_history, plt):
    plt.plot(defection_history)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
