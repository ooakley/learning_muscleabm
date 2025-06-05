import marimo

__generated_with = "0.12.10"
app = marimo.App(width="columns")


@app.cell
def _():
    import cv2
    import matplotlib
    import os

    import numpy as np
    import pandas as pd
    import colorcet as cc

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    colors_list = list(mcolors.TABLEAU_COLORS.values())
    return cc, colors_list, cv2, matplotlib, mcolors, np, os, pd, plt


@app.cell
def _():
    # Constant display variables:
    MESH_NUMBER = 64
    CELL_NUMBER = 61
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
def _(
    MESH_NUMBER,
    TIMESTEP_WIDTH,
    WORLD_SIZE,
    cc,
    colors_list,
    matplotlib,
    np,
):
    DIVISOR = 1
    def plot_superiteration(
        trajectory_dataframe, matrix_series, timestep, ax, cell_size,
        plot_trajectories=True, plot_matrix=True, plot_ellipses=True
        ):
        # Accessing and formatting relevant dataframe:
        x_mask = (trajectory_dataframe["x"] > 0) & (trajectory_dataframe["x"] < WORLD_SIZE)
        y_mask = (trajectory_dataframe["y"] > 0) & (trajectory_dataframe["y"] < WORLD_SIZE)
        full_mask = x_mask & y_mask
        rollover_skipped_df = trajectory_dataframe[full_mask]
        timeframe_mask = \
            (rollover_skipped_df["frame"] > timestep - TIMESTEP_WIDTH) & \
            (rollover_skipped_df["frame"] <= timestep)
        timepoint_mask = rollover_skipped_df["frame"] == timestep
        ci_lookup = trajectory_dataframe[trajectory_dataframe["frame"] == 10]
        unstacked_dataframe = rollover_skipped_df[timeframe_mask].set_index(
            ['particle', 'frame']
        )[['x', 'y']].unstack()

        # Setting up matrix plotting:
        tile_width = WORLD_SIZE / MESH_NUMBER
        X = np.arange(0, WORLD_SIZE, tile_width) + (tile_width / 2)
        Y = np.arange(0, WORLD_SIZE, tile_width) + (tile_width / 2)
        X, Y = np.meshgrid(X, Y)

        matrix = matrix_series[timestep, 0, :, :]
        U = np.cos(matrix) * matrix_series[int(timestep / DIVISOR), 1, :, :]
        V = -np.sin(matrix) * matrix_series[int(timestep / DIVISOR), 1, :, :]

        # Setting up plot
        # colour_list = ['r', 'g']

        # Plotting particle trajectories:
        if plot_trajectories:
            alpha=0.4
            linewidth=1
            for i, trajectory in unstacked_dataframe.iterrows():
                identity = 1
    
                rollover_x = np.abs(np.diff(np.array(trajectory['x'])[::4])) > (WORLD_SIZE/2)
                rollover_y = np.abs(np.diff(np.array(trajectory['y'])[::4])) > (WORLD_SIZE/2)
                rollover_mask = rollover_x | rollover_y
            
                color = colors_list[i % len(colors_list)]
    
                if np.count_nonzero(rollover_mask) == 0:
                    ax.plot(np.array(trajectory['x'])[::4], np.array(trajectory['y'])[::4],
                        alpha=alpha, linewidth=linewidth, c=color
                    )
                else:
                    plot_separation_indices = np.argwhere(rollover_mask)
                    prev_index = 0
                    for separation_index in plot_separation_indices:
                        separation_index = separation_index[0]
                        x_array = np.array(trajectory['x'])[::4][prev_index:separation_index]
                        y_array = np.array(trajectory['y'])[::4][prev_index:separation_index]
                        ax.plot(x_array, y_array, alpha=alpha, linewidth=linewidth, c=color)
                        prev_index = separation_index+1
    
                    # Plotting final segment:
                    x_array = np.array(trajectory['x'])[::4][prev_index:]
                    y_array = np.array(trajectory['y'])[::4][prev_index:]
                    ax.plot(x_array, y_array, alpha=alpha, linewidth=linewidth, c=color)

        # Plotting background matrix:
        if plot_matrix:
            speed = np.sqrt(U**2 + V**2)
            if speed.max() != 0:
                alpha = speed / speed.max()
                ax.quiver(
                    X, Y, U, V, [matrix],
                    cmap=cc.cm.CET_C6, clim=(0, np.pi),
                    pivot='mid', scale=50, headwidth=0, headlength=0, headaxislength=0,
                    width=0.005, alpha=alpha
                )


        arrow_scaling = 0.25

        # # Plotting cells & their directions:
        # x_pos = rollover_skipped_df['x'][timepoint_mask]
        # y_pos = rollover_skipped_df['y'][timepoint_mask]
        # x_heading = np.cos(rollover_skipped_df['orientation'][timepoint_mask]) \
        #     * rollover_skipped_df["polarity_extent"][timepoint_mask] * arrow_scaling
        # y_heading = - np.sin(rollover_skipped_df['orientation'][timepoint_mask]) \
        #     * rollover_skipped_df["polarity_extent"][timepoint_mask] * arrow_scaling
        # heading_list = rollover_skipped_df['orientation'][timepoint_mask]
        # ax.quiver(
        #     x_pos, y_pos, x_heading, y_heading, np.array(heading_list).flatten(),
        #     pivot='tail', scale=1/100, scale_units='x', color='k', headwidth=3,
        #     headlength=3, headaxislength=3, width=0.004, alpha=0.75,
        #     cmap=cc.cm.CET_C6
        # )

        # Plot cell positions:
        x_pos = rollover_skipped_df['x'][timepoint_mask]
        y_pos = rollover_skipped_df['y'][timepoint_mask]

        # Plot cell actin directions:
        x_heading = np.cos(rollover_skipped_df['actin_flow'][timepoint_mask]) \
            * rollover_skipped_df["actin_mag"][timepoint_mask] * arrow_scaling
        y_heading = - np.sin(rollover_skipped_df['actin_flow'][timepoint_mask]) \
            * rollover_skipped_df["actin_mag"][timepoint_mask] * arrow_scaling
        heading_list = rollover_skipped_df['actin_flow'][timepoint_mask]

        # Run direction plot:
        ax.quiver(
            x_pos, y_pos, x_heading, y_heading, np.array(heading_list).flatten(),
            pivot='tail', scale=1/100, scale_units='x',
            headwidth=3, headlength=3, headaxislength=3, width=0.004, alpha=1,
            cmap=cc.cm.CET_C6
        )

        # Plot cell CIL directions:
        x_cil_heading = rollover_skipped_df['cil_x'][timepoint_mask] * arrow_scaling
        y_cil_heading = -rollover_skipped_df['cil_y'][timepoint_mask] * arrow_scaling
        # heading_list = np.arctan2(y_heading, x_heading)

        # Run CIL plot:
        ax.quiver(
            x_pos, y_pos, x_cil_heading, y_cil_heading,
            pivot='tail', scale=1/100, scale_units='x',
            headwidth=3, headlength=3, headaxislength=3, width=0.004, alpha=0.5,
            color='k'
        )

        aspect_ratio = 1

        # Plotting cell shape:
        if plot_ellipses:
            orientations = np.asarray(
                rollover_skipped_df['shapeDirection'][timepoint_mask]
            )
            collision_state = np.asarray(
                rollover_skipped_df['collision_number'][timepoint_mask]
            )
            for index, xy in enumerate(zip(x_pos, y_pos)):
                # Get data:
                major_axis = 2*cell_size*np.sqrt(aspect_ratio)
                minor_axis = 2*cell_size*np.sqrt(1/aspect_ratio)
                angle = (orientations[index] * 180) / np.pi

                # Determine collision state color:
                color_index = int(collision_state[index] > 0)
                collision_color = ['k', 'r'][color_index]

                # Plot ellipse:
                ellipse = matplotlib.patches.Ellipse(
                    xy, major_axis, minor_axis, angle=angle,
                    alpha=0.25, color=collision_color
                )
                ax.add_patch(ellipse)

                # Plot extra ellipses if close to edge:
                x = xy[0]
                y = xy[1]
                if x < cell_size:
                    ellipse = matplotlib.patches.Ellipse(
                        (x+2048, y), major_axis, minor_axis, angle=angle,
                        alpha=0.1, color='k'
                    )
                    ax.add_patch(ellipse)  
                if 2048 - x < cell_size:
                    ellipse = matplotlib.patches.Ellipse(
                        (x-2048, y), major_axis, minor_axis, angle=angle,
                        alpha=0.1, color='k'
                    )
                    ax.add_patch(ellipse)
                if y < cell_size:
                    ellipse = matplotlib.patches.Ellipse(
                        (x, y+2048), major_axis, minor_axis, angle=angle,
                        alpha=0.1, color='k'
                    )
                    ax.add_patch(ellipse)  
                if 2048 - y < cell_size:
                    ellipse = matplotlib.patches.Ellipse(
                        (x, y-2048), major_axis, minor_axis, angle=angle,
                        alpha=0.1, color='k'
                    )
                    ax.add_patch(ellipse)  

        # ax.scatter(x_pos, y_pos, color='k', alpha=0.4, s=cell_size)
        ax.set_xlim(0, WORLD_SIZE)
        ax.set_ylim(0, WORLD_SIZE)
        ax.invert_yaxis()
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])
    return DIVISOR, plot_superiteration


@app.cell
def _(get_trajectory_data, read_matrix_into_numpy):
    ecm_matrix = read_matrix_into_numpy(0, 0)
    trajectory_dataframe = get_trajectory_data(0, 0)
    return ecm_matrix, trajectory_dataframe


@app.cell
def _(TIMESTEPS, ecm_matrix, plot_superiteration, plt, trajectory_dataframe):
    _fig, _ax = plt.subplots(figsize=(7, 7))
    plot_superiteration(
        trajectory_dataframe, ecm_matrix, TIMESTEPS-1, _ax, 75,
        plot_matrix=True, plot_trajectories=True, plot_ellipses=False
    )
    plt.show()
    return


@app.cell
def _(CELL_NUMBER, TIMESTEPS, np, trajectory_dataframe):
    # Test out order parameter calculations:

    # Sort by cell and then by frame:
    sorted_dataframe = trajectory_dataframe.sort_values(by=["particle", "frame"])

    # Get speed and angle data:
    actin_magnitude = np.asarray(sorted_dataframe["actin_mag"])
    actin_direction = np.asarray(sorted_dataframe["actin_flow"])
    collisions = np.asarray(sorted_dataframe["collision_number"])

    actin_magnitude = np.reshape(actin_magnitude, (CELL_NUMBER, TIMESTEPS))
    actin_direction = np.reshape(actin_direction, (CELL_NUMBER, TIMESTEPS))
    collisions = np.reshape(collisions, (CELL_NUMBER, TIMESTEPS))
    mean_collisions = np.mean(collisions, axis=0)

    # Get x and y components across cell populations for each timestep:
    x_components = np.cos(actin_direction) * actin_magnitude
    y_components = np.sin(actin_direction) * actin_magnitude

    # Sum components:
    summed_x = np.mean(x_components, axis=0)
    summed_y = np.mean(y_components, axis=0)
    mean_magnitude = np.mean(actin_magnitude, axis=0)

    # Get order parameter:
    order_parameter = np.sqrt(summed_x**2 + summed_y**2) / mean_magnitude
    return (
        actin_direction,
        actin_magnitude,
        collisions,
        mean_collisions,
        mean_magnitude,
        order_parameter,
        sorted_dataframe,
        summed_x,
        summed_y,
        x_components,
        y_components,
    )


@app.cell
def _(TIMESTEPS, order_parameter, plt):
    _fig, _ax = plt.subplots()
    _ax.plot(order_parameter)
    _ax.set_ylim(0, 1)
    _ax.set_xlim(0, TIMESTEPS)

    plt.show()
    return


@app.cell
def _(TIMESTEPS, mean_collisions, plt):
    _fig, _ax = plt.subplots()
    _ax.plot(mean_collisions)
    _ax.set_ylim(0, 6)
    _ax.set_xlim(0, TIMESTEPS)

    plt.show()
    return


@app.cell
def _(
    TIMESTEPS,
    cv2,
    ecm_matrix,
    np,
    plot_superiteration,
    plt,
    trajectory_dataframe,
):
    size = 500, 500
    fps = 30
    out = cv2.VideoWriter(
        './basic_video.mp4', cv2.VideoWriter_fourcc(*'avc1'),
        fps, (size[1], size[0]), True
    )

    for timeframe in list(range(TIMESTEPS))[::2]:
        if (timeframe) % 200 == 0:
            print(timeframe)

        _fig, _ax = plt.subplots(figsize=(5, 5), layout='constrained')
        plot_superiteration(
            trajectory_dataframe, ecm_matrix, timeframe, _ax, 75,
            plot_matrix=False, plot_trajectories=False, plot_ellipses=True
        )

        # Export to array:
        _fig.canvas.draw()
        array_plot = np.array(_fig.canvas.renderer.buffer_rgba())
        plt.close(_fig)
 
        # Save array plot to opencv file:
        bgr_data = cv2.cvtColor(array_plot, cv2.COLOR_RGB2BGR)
        out.write(bgr_data)

    out.release()
    return array_plot, bgr_data, fps, out, size, timeframe


@app.cell
def _(out):
    out.release()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
