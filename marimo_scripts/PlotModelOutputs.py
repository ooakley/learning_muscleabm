import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    import os

    import matplotlib
    import cv2
    import scipy

    import numpy as np
    import pandas as pd
    import colorcet as cc

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    colors_list = list(mcolors.TABLEAU_COLORS.values())
    return cc, colors_list, cv2, matplotlib, np, os, pd, plt, scipy


@app.cell
def _():
    DATA_DIR_PATH = "model_parameter_json/matrix_test"
    return (DATA_DIR_PATH,)


@app.cell
def _():
    # Constant display variables:
    MESH_NUMBER = 128
    CELL_NUMBER = 120
    TIMESTEPS = 2880
    TIMESTEP_WIDTH = 1440
    WORLD_SIZE = 2048
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
    return (
        CELL_NUMBER,
        MESH_NUMBER,
        OUTPUT_COLUMN_NAMES,
        TIMESTEPS,
        TIMESTEP_WIDTH,
        WORLD_SIZE,
    )


@app.cell
def _(DATA_DIR_PATH, OUTPUT_COLUMN_NAMES, np, os, pd):
    def read_matrix_into_numpy(subiteration):
        # Determine filepaths:
        filename = f"matrix_seed{subiteration:03d}.txt"
        filepath = os.path.join(DATA_DIR_PATH, filename)

        array_list = []
        with open(filepath, "r") as f:
            for line in f:
                heading_string = str(line.rstrip())
                array_list.append(np.fromstring(heading_string, sep=","))

        return array_list
        # # Calculate the exact number of matrix mesh elements:
        # number_of_elements = 3 * (MESH_NUMBER**2)
        # flattened_matrix = np.loadtxt(
        #     filepath, delimiter=',',
        #     usecols=range(number_of_elements)
        # )

        # # Reshape into (timesteps, [density/orientation/anisotropy], grid, grid):
        # dimensions = (MESH_NUMBER, MESH_NUMBER, 3)
        # return np.reshape(flattened_matrix, dimensions, order='C')


    def get_trajectory_data(subiteration):
        # Determine filepaths:
        filename = f"positions_seed{subiteration:03d}.csv"
        # directory_path = os.path.join(DATA_DIR_PATH, str(job_id))
        filepath = os.path.join(DATA_DIR_PATH, filename)

        # Get cell trajectory data:
        trajectory_dataframe = pd.read_csv(
            filepath, index_col=None, header=None,
            names=OUTPUT_COLUMN_NAMES
        )
        return trajectory_dataframe
    return get_trajectory_data, read_matrix_into_numpy


@app.cell
def _(trajectory_dataframe):
    trajectory_dataframe
    return


@app.cell
def _(WORLD_SIZE, np):
    def plot_center_of_mass(trajectory_dataframe, timestep, ax):
        # Plotting information:
        alpha=0.65
        linewidth=1.5

        # Get list of particle indices:
        plot_dataframe = trajectory_dataframe.copy(deep=True)
        timestep_mask = np.logical_and(plot_dataframe["frame"] > (timestep - 1440), plot_dataframe["frame"] <= timestep)
        plot_dataframe = plot_dataframe.loc[timestep_mask]
        particle_indices = np.unique(plot_dataframe["particle"])

        for particle_index in particle_indices:
            # Get particle information:
            particle_mask = plot_dataframe["particle"] == particle_index
            particle_dataframe = plot_dataframe.loc[particle_mask]
            x_front = np.array(particle_dataframe['x'])
            y_front = np.array(particle_dataframe['y'])
            x_back = np.array(particle_dataframe['stadium_x'])
            y_back = np.array(particle_dataframe['stadium_y'])

            # Get stadium span and account for periodic boundaries:
            x_span = x_back - x_front
            y_span = y_back - y_front
            x_span[x_span < -1024] += 2048
            x_span[x_span > +1024] -= 2048
            y_span[y_span < -1024] += 2048
            y_span[y_span > +1024] -= 2048

            # Get centers of mass:
            x_com = x_front + (x_span / 2)
            x_com[x_com > 2048] -= 2048
            x_com[x_com < 0]    += 2048
            y_com = y_front + (y_span / 2)
            y_com[y_com > 2048] -= 2048
            y_com[y_com < 0]    += 2048

            # Get points of rollover:
            rollover_x = \
                np.abs(np.diff(x_com)) > (WORLD_SIZE/2)
            rollover_y = \
                np.abs(np.diff(y_com)) > (WORLD_SIZE/2)
            rollover_mask = rollover_x | rollover_y

            # color = colors_list[particle_index % len(colors_list)]
            color = "k"
            if np.count_nonzero(rollover_mask) == 0:
                ax.plot(
                    x_com, y_com,
                    alpha=alpha, linewidth=linewidth, c=color
                )
            else:
                # Plot separated segments:
                plot_separation_indices = np.argwhere(rollover_mask)
                prev_index = 0
                for separation_index in plot_separation_indices:
                    separation_index = separation_index[0]
                    sub_x = x_com[prev_index:separation_index + 1]
                    sub_y = y_com[prev_index:separation_index + 1]
                    ax.plot(
                        sub_x, sub_y,
                        alpha=alpha, linewidth=linewidth,c=color
                    )
                    prev_index = separation_index + 1

                # Plotting final segment:
                x_array = x_com[prev_index:]
                y_array = y_com[prev_index:]
                ax.plot(
                    x_array, y_array,
                    alpha=alpha, linewidth=linewidth, c=color
                )

        ax.set_xlim(0, WORLD_SIZE)
        ax.set_ylim(0, WORLD_SIZE)
        ax.invert_yaxis()
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])
    return (plot_center_of_mass,)


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
        trajectory_dataframe, matrix_data, timestep, ax, cell_size,
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

        # U = np.cos(matrix_data[:, :, 0]) * matrix_data[:, :, 1]
        # V = -np.sin(matrix_data[:, :, 0]) * matrix_data[:, :, 1]
        U = np.cos(matrix_data[:, :, 0]) * 1
        V = -np.sin(matrix_data[:, :, 0]) * 1

        # Setting up plot
        # colour_list = ['r', 'g']

        # Plotting particle trajectories:
        if plot_trajectories:
            alpha=0.65
            linewidth=1.5
            for i, trajectory in unstacked_dataframe.iterrows():
                identity = 1

                rollover_x = \
                    np.abs(np.diff(np.array(trajectory['x']))) > (WORLD_SIZE/2)
                rollover_y = \
                    np.abs(np.diff(np.array(trajectory['y']))) > (WORLD_SIZE/2)
                rollover_mask = rollover_x | rollover_y

                color = colors_list[i % len(colors_list)]

                if np.count_nonzero(rollover_mask) == 0:
                    ax.plot(
                        np.array(trajectory['x']),
                        np.array(trajectory['y']),
                        alpha=alpha, linewidth=linewidth, c=color
                    )
                else:
                    plot_separation_indices = np.argwhere(rollover_mask)
                    prev_index = 0
                    for separation_index in plot_separation_indices:
                        separation_index = separation_index[0]
                        x_array = np.array(trajectory['x'])\
                            [prev_index:separation_index]
                        y_array = np.array(trajectory['y'])\
                            [prev_index:separation_index]
                        ax.plot(
                            x_array, y_array, alpha=alpha,
                            linewidth=linewidth,
                            c=color
                        )
                        prev_index = separation_index+1

                    # Plotting final segment:
                    x_array = np.array(trajectory['x'])[prev_index:]
                    y_array = np.array(trajectory['y'])[prev_index:]
                    ax.plot(x_array, y_array, alpha=alpha, linewidth=linewidth, c=color)

        # Plotting background matrix:
        if plot_matrix:
            speed = np.sqrt(U**2 + V**2)
            if speed.max() != 0:
                fibre_count = matrix_data[:, :, 1]
                ax.quiver(
                    X, Y, U, V, [matrix_data[:, :, 0]],
                    cmap=cc.cm.CET_C6, clim=(-np.pi/2, np.pi/2),
                    pivot='mid', scale=50, headwidth=0, headlength=0, headaxislength=0,
                    width=0.003, alpha=0.5
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

        # Plot cell stadia:
        stadia_x = np.array(rollover_skipped_df['stadium_x'][timepoint_mask])
        stadia_y = np.array(rollover_skipped_df['stadium_y'][timepoint_mask])

        # Plot cell actin directions:
        x_heading = np.cos(rollover_skipped_df['actin_flow'][timepoint_mask]) \
            * rollover_skipped_df["actin_mag"][timepoint_mask] * arrow_scaling
        y_heading = -np.sin(rollover_skipped_df['actin_flow'][timepoint_mask]) \
            * rollover_skipped_df["actin_mag"][timepoint_mask] * arrow_scaling
        heading_list = rollover_skipped_df['actin_flow'][timepoint_mask]

        # Run direction plot:
        ax.quiver(
            x_pos, y_pos, x_heading, y_heading, np.array(heading_list).flatten(),
            pivot='tail', scale=1/300, scale_units='x',
            headwidth=3, headlength=3, headaxislength=3, width=0.004, alpha=1,
            cmap=cc.cm.CET_C6
        )

        # # Plot cell CIL directions:
        # x_cil_heading = rollover_skipped_df['cil_x'][timepoint_mask] * arrow_scaling
        # y_cil_heading = -rollover_skipped_df['cil_y'][timepoint_mask] * arrow_scaling
        # ax.quiver(
        #     x_pos, y_pos, x_cil_heading, y_cil_heading,
        #     pivot='tail', scale=1/250, scale_units='x',
        #     headwidth=3, headlength=3, headaxislength=3, width=0.004, alpha=0.5,
        #     color='k'
        # )

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

                # Plot stadia:
                ellipse = matplotlib.patches.Ellipse(
                    (stadia_x[index], stadia_y[index]),
                    major_axis, minor_axis, angle=angle,
                    alpha=0.25, color='b'
                )
                ax.add_patch(ellipse)

                stadia_mask = \
                    np.abs(xy[0] - stadia_x[index]) > 1024 or \
                    np.abs(xy[1] - stadia_y[index]) > 1024

                if not stadia_mask:
                    ax.plot(
                        [xy[0], stadia_x[index]], [xy[1], stadia_y[index]],
                        alpha=0.4, linewidth=10,
                        c='r'
                    )

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
    return (plot_superiteration,)


@app.cell
def _(get_trajectory_data, read_matrix_into_numpy):
    ecm_matrix = read_matrix_into_numpy(0)
    trajectory_dataframe = get_trajectory_data(0)
    return ecm_matrix, trajectory_dataframe


@app.cell
def _(MESH_NUMBER, ecm_matrix, np):
    average_heading = np.empty(MESH_NUMBER**2)
    angular_variance = np.empty(MESH_NUMBER**2)
    fibre_count = np.empty(MESH_NUMBER**2)
    for index, heading_array in enumerate(ecm_matrix):
        fibre_count[index] = len(heading_array)
        if len(heading_array) == 0:
            angular_variance[index] = 1
            average_heading[index] = 0
            continue
        x_component = np.cos(heading_array * 2)
        y_component = np.sin(heading_array * 2)
        angular_variance[index] = 1 - np.linalg.norm([np.mean(x_component), np.mean(y_component)])
        average_heading[index] = np.atan2(np.mean(y_component), np.mean(x_component)) / 2

    angular_variance = np.reshape(angular_variance, (MESH_NUMBER, MESH_NUMBER))
    average_heading = np.reshape(average_heading, (MESH_NUMBER, MESH_NUMBER))
    fibre_count = np.reshape(fibre_count, (MESH_NUMBER, MESH_NUMBER))
    return angular_variance, average_heading, fibre_count


@app.cell
def _(angular_variance, average_heading, fibre_count, np):
    ecm_array = np.stack([average_heading, fibre_count, angular_variance], axis=-1)
    return (ecm_array,)


@app.cell
def _(CELL_NUMBER, TIMESTEPS):
    averaged_count = (CELL_NUMBER * TIMESTEPS) / (512 * 512)
    return


@app.cell
def _(CELL_NUMBER, TIMESTEPS, np, scipy):
    def get_density_threshold():
        mean, var = scipy.stats.binom.stats(TIMESTEPS, CELL_NUMBER / (512 * 512), moments='mv')
        return mean - np.sqrt(var)
    return (get_density_threshold,)


@app.cell
def _(MESH_NUMBER, NEIGHBOURHOOD_SIZES, np):
    def get_order_parameter(submatrix):
        # Getting central values:
        central_index = int(np.floor(submatrix.shape[0] / 2))
        central_val = submatrix[central_index, central_index]

        # Getting values in window:
        central_cutoff = int(np.ceil(submatrix.size / 2))
        comparators = submatrix.flatten()
        comparators = np.concatenate([comparators[0:central_cutoff], comparators[central_cutoff + 1:]])
        angle_diff = comparators * 2 - central_val * 2

        # Calculating order parameter:
        order_parameter = np.nanmean(np.cos(angle_diff * 2))
        return order_parameter


    def roll_indices(index, half_index):
        # Need to roll matrix to ensure that the order parameter captures the
        # periodic boundaries - calculating the amount of rolling is a bit
        # fiddly however:
        roll_index = 0
        index_start = index - half_index
        index_end = index + (half_index + 1)
        if index_start < 0:
            roll_index -= index_start
            index_start += roll_index
            index_end += roll_index
        if index_end > MESH_NUMBER:
            roll_index = -(index_end - MESH_NUMBER)
            index_start += roll_index
            index_end += roll_index

        return roll_index, index_start, index_end


    def get_order_parameter_distribution(matrix, neighbourhood_size=3):
        order_parameters = []
        half_index = int(np.floor(neighbourhood_size / 2))
        for i in range(MESH_NUMBER):
            # Determining amount of rolling required along row:
            roll_i, i_start, i_end = roll_indices(i, half_index)
            for j in range(MESH_NUMBER):
                # Determining amount of rolling required along column:
                roll_j, j_start, j_end = roll_indices(j, half_index)

                # Rolling matrix:
                rolled_matrix = np.roll(matrix, roll_i, axis=0)
                rolled_matrix = np.roll(rolled_matrix, roll_j, axis=1)

                # Getting submatrix:
                orientation_submatrix = rolled_matrix[i_start:i_end, j_start:j_end]
                order_parameter = get_order_parameter(orientation_submatrix)
                order_parameters.append(order_parameter)
        return np.array(order_parameters)


    def generate_order_parameter_scale_curve(matrix):
        order_parameters = []
        for neighbourhood_size in NEIGHBOURHOOD_SIZES:
            mean_order_parameter = np.nanmean(
                get_order_parameter_distribution(matrix, neighbourhood_size=neighbourhood_size)
            )
            order_parameters.append(mean_order_parameter)
        return np.array(order_parameters)
    return (get_order_parameter_distribution,)


@app.cell
def _(get_density_threshold):
    density_threshold = get_density_threshold()
    return (density_threshold,)


@app.cell
def _(average_heading, fibre_count, np):
    nan_heading = np.copy(average_heading)
    nan_heading[fibre_count == 0] = np.nan
    return (nan_heading,)


@app.cell
def _(get_order_parameter_distribution, nan_heading):
    op_distribution = get_order_parameter_distribution(nan_heading, 31)
    return (op_distribution,)


@app.cell
def _(np, op_distribution):
    np.nanmean(op_distribution)
    return


@app.cell
def _(np, op_distribution):
    np.count_nonzero(np.isnan(op_distribution)) / len(op_distribution)
    return


@app.cell
def _(op_distribution, plt):
    plt.hist(op_distribution, bins=100, range=(-1, 1));
    plt.show()
    return


@app.cell
def _(cc, nan_heading, np, plt):
    plt.imshow(nan_heading, cmap=cc.m_CET_C6, clim=(-np.pi/2, np.pi/2))
    return


@app.cell
def _(MESH_NUMBER, cc, op_distribution, plt):
    op_array = op_distribution.reshape((MESH_NUMBER, MESH_NUMBER))
    plt.imshow(op_array, vmin=-1, vmax=1, cmap=cc.m_CET_D1)
    return (op_array,)


@app.cell
def _(cc, op_array, plt):
    plt.imshow(op_array > 0.15, vmin=-1, vmax=1, cmap=cc.m_CET_D1)
    return


@app.cell
def _(cc, nan_heading, np, op_array, plt):
    masked_headings = np.copy(nan_heading)
    masked_headings[op_array < 0.15] = np.nan
    plt.imshow(masked_headings, cmap=cc.m_CET_C6, clim=(-np.pi/2, np.pi/2))
    return


@app.cell
def _(angular_variance, cc, plt):
    plt.imshow(angular_variance, cmap=cc.m_CET_L1, clim=(0, 1))
    return


@app.cell
def _(cc, fibre_count, plt):
    plt.imshow(fibre_count, cmap=cc.m_CET_L1, clim=(0, None))
    return


@app.cell
def _(cc, fibre_count, np, plt, scipy):
    averaged_fc = scipy.ndimage.gaussian_filter(fibre_count, 1, mode='wrap')
    print(averaged_fc.min())
    print(averaged_fc.mean())
    print("IDR:", np.quantile(averaged_fc, 0.9) - np.quantile(averaged_fc, 0.1))
    plt.imshow(averaged_fc, interpolation="none", cmap=cc.m_CET_L1, clim=(0, None))
    return (averaged_fc,)


@app.cell
def _(averaged_fc, cc, density_threshold, plt):
    plt.imshow(averaged_fc < density_threshold, interpolation="none", cmap=cc.m_CET_L1, clim=(0, None))
    return


@app.cell
def _(angular_variance, plt, scipy):
    blurred = scipy.ndimage.gaussian_filter(angular_variance, 5, mode='wrap')
    print(blurred.min())
    print(blurred.mean())
    plt.imshow(blurred, vmin=0, interpolation="none")
    return (blurred,)


@app.cell
def _(blurred, np, plt):
    plt.hist(np.log(blurred.flatten() + 1), bins=100);
    print(np.std(np.log(1 + blurred)))
    plt.show()
    return


@app.cell
def _(TIMESTEPS, ecm_array, plot_superiteration, plt, trajectory_dataframe):
    _fig, _ax = plt.subplots(figsize=(10, 10))
    plot_superiteration(
        trajectory_dataframe, ecm_array, TIMESTEPS - 1, _ax, 20,
        plot_matrix=False, plot_trajectories=False, plot_ellipses=True
    )
    # plot_center_of_mass(trajectory_dataframe, TIMESTEPS - 1, _ax)
    plt.show()
    return


@app.cell
def _(TIMESTEPS, plot_center_of_mass, plt, trajectory_dataframe):
    def get_com_plot():
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_center_of_mass(trajectory_dataframe, TIMESTEPS-1, ax)
        plt.show()

    get_com_plot()
    return


@app.cell
def _():
    # _fig, _ax = plt.subplots(figsize=(10, 10))
    # plot_superiteration(
    #     trajectory_dataframe, ecm_matrix, TIMESTEPS-1, _ax, 50,
    #     plot_matrix=True, plot_trajectories=True, plot_ellipses=False
    # )
    # plt.show()
    return


@app.cell
def _():
    # import skimage

    # _ax = plt.figure(layout='constrained').add_subplot(projection='3d')
    # _ax.view_init(elev=30, azim=45, roll=15)

    # positions = trajectory_dataframe.sort_values(['particle', 'frame']).loc[:, ('x', 'y')]
    # position_array = np.array(positions).reshape(CELL_NUMBER, TIMESTEPS, 2)

    # # Estimate from trajectory dataframe as test:
    # # line_array = np.zeros((2048, 2048, 1440))

    # for _cell_index in range(position_array.shape[0]):
    #     relevant_trajectory = position_array[_cell_index, 1440:, :]
    #     coords = []
    #     for t in range(1440 - 1):
    #         # Get indices of line:
    #         xy_t = relevant_trajectory[t, :].astype(int)
    #         xy_t1 = relevant_trajectory[t+1, :].astype(int)

    #         # Account for periodic boundaries:
    #         distance = np.sqrt(np.sum((xy_t - xy_t1)**2, axis=0))
    #         if distance > 1024:
    #             continue

    #         st_t = [*xy_t, t]
    #         st_t1 = [*xy_t, t+1]
    #         coords.append(st_t)

    #         # Plot line indices on matrix:
    #         # _ii, _jj, _kk = skimage.draw.line_nd(st_t, st_t1)
    #         # # line_array[_ii, _jj, _kk] = 1
    #     trajectory_array = np.stack(coords, axis=0)
    #     _ax.plot(
    #         trajectory_array[:, 0],
    #         trajectory_array[:, 1],
    #         trajectory_array[:, 2],
    #         lw=0.5
    #     )

    # plt.show()
    return


@app.cell
def _(trajectory_dataframe):
    # Test out order parameter calculations:

    # Sort by cell and then by frame:
    sorted_dataframe = trajectory_dataframe.sort_values(by=["particle", "frame"])

    # # Get speed and angle data:
    # actin_magnitude = np.asarray(sorted_dataframe[\"actin_mag\"])
    # actin_direction = np.asarray(sorted_dataframe[\"actin_flow\"])
    # collisions = np.asarray(sorted_dataframe[\"collision_number\"])

    # actin_magnitude = np.reshape(actin_magnitude, (CELL_NUMBER, TIMESTEPS))
    # actin_direction = np.reshape(actin_direction, (CELL_NUMBER, TIMESTEPS))
    # collisions = np.reshape(collisions, (CELL_NUMBER, TIMESTEPS))
    # mean_collisions = np.mean(collisions, axis=0)

    # # Get x and y components across cell populations for each timestep:
    # x_components = np.cos(actin_direction) * actin_magnitude
    # y_components = np.sin(actin_direction) * actin_magnitude

    # # Sum components:
    # summed_x = np.mean(x_components, axis=0)
    # summed_y = np.mean(y_components, axis=0)
    # mean_magnitude = np.mean(actin_magnitude, axis=0)

    # # Get order parameter:
    # order_parameter = np.sqrt(summed_x**2 + summed_y**2) / mean_magnitude
    return


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
    ecm_array,
    os,
    plot_superiteration,
    plt,
    trajectory_dataframe,
):
    def write_video_to_file():
        size = 750, 750

        if not os.path.exists("img_tmp"):
            os.mkdir("img_tmp")

        count = 0
        for timeframe in list(range(TIMESTEPS))[1:2880:50]:
            if (timeframe) % 200 == 0:
                print(timeframe)

            fig, ax = plt.subplots(figsize=(7.5, 7.5), layout='constrained')
            plot_superiteration(
                trajectory_dataframe, ecm_array, timeframe, ax, 50,
                plot_matrix=False, plot_trajectories=False, plot_ellipses=True
            )
            # plot_center_of_mass(trajectory_dataframe, timeframe, ax)
            plt.savefig(os.path.join("img_tmp", f"frame_{count}.png"))
            plt.close()
            count += 1

    write_video_to_file()
    return


@app.cell
def _():
    import subprocess
    # subprocess.run("ml load FFmpeg/7.1.1-GCCcore-14.2.0; ffmpeg -y -i img_tmp/frame_%d.png -r 24 -vcodec libx264 -crf 18 test_video.mp4", shell=True)
    subprocess.run("ffmpeg -y -i img_tmp/frame_%d.png -r 24 -vcodec libx264 -crf 18 test_video.mp4", shell=True)
    return


@app.cell
def _(os):
    temp_files = os.listdir("img_tmp")
    for temp_file in temp_files:
        os.remove(os.path.join("img_tmp", temp_file))
    return


@app.cell
def _(
    TIMESTEPS,
    cv2,
    ecm_array,
    np,
    plot_superiteration,
    plt,
    trajectory_dataframe,
):
    size = 750, 750
    fps = 30
    out = cv2.VideoWriter(
        './basic_video.mp4', cv2.VideoWriter_fourcc(*'avc1'),
        fps, (size[1], size[0]), True
    )
    # out = cv2.VideoWriter(
    #     './basic_video.avi', cv2.VideoWriter_fourcc(*'MJPG'),
    #     fps, (size[1], size[0]), True
    # )



    for timeframe in list(range(TIMESTEPS))[1440:1500:10]:
        if (timeframe) % 200 == 0:
            print(timeframe)

        _fig, _ax = plt.subplots(figsize=(7.5, 7.5), layout='constrained')
        plot_superiteration(
            trajectory_dataframe, ecm_array, timeframe, _ax, 63,
            plot_matrix=False, plot_trajectories=True, plot_ellipses=True
        )

        # Export to array:
        _fig.canvas.draw()
        array_plot = np.array(_fig.canvas.renderer.buffer_rgba())
        plt.close(_fig)

        # Save array plot to opencv file:
        bgr_data = cv2.cvtColor(array_plot, cv2.COLOR_RGB2BGR)
        out.write(bgr_data)

    out.release()
    return


@app.cell
def _():
    "ffmpeg -framerate 1 -pattern_type glob -i '*.png' -c:v libx264 -r 30 -pix_fmt yuv420p output.mp4"
    return


if __name__ == "__main__":
    app.run()
