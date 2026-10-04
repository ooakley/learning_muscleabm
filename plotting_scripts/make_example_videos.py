import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import subprocess

    import pandas as pd
    import numpy as np
    import colorcet as cc

    import matplotlib.pyplot as plt
    return cc, np, os, pd, plt, subprocess


@app.cell
def _():
    import matplotlib as mpl

    # Font formatting:
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = "cmr10"
    mpl.rcParams['font.size'] = 8
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams['axes.unicode_minus'] = False

    # Tick formating:
    mpl.rcParams['xtick.major.size'] = 2
    mpl.rcParams['xtick.major.pad'] = 1.5
    mpl.rcParams['ytick.major.size'] = 2
    mpl.rcParams['ytick.major.pad'] = 1.5
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7

    # Label formatting:
    mpl.rcParams['axes.labelpad'] = 2.5

    # Layout formatting:
    mpl.rcParams['figure.constrained_layout.hspace'] = 0.04
    mpl.rcParams['figure.constrained_layout.wspace'] = 0.04
    return (mpl,)


@app.cell
def _():
    MESH_NUMBER = 128
    TIMESTEPS = 2880
    VIDEO_DIRPATH = "configs/model_parameter_json/example_rd_csm_params"
    CELL_COUNT = 400
    MATRIX_TIMESERIES = False
    return CELL_COUNT, MESH_NUMBER, TIMESTEPS, VIDEO_DIRPATH


@app.cell
def _(MESH_NUMBER, TIMESTEPS, np):
    def read_matrix(filepath):
        matrix_list = []
        with open(filepath, "r") as f:
            for index, line in enumerate(f):
                values_string = str(line.rstrip())
                values = values_string.split(",")[:-1]
                matrix = np.asarray(values, dtype=float).reshape(MESH_NUMBER, MESH_NUMBER, -1)
                matrix_list.append(matrix)
                if index == TIMESTEPS - 1:
                    break
        return np.stack(matrix_list, axis=0)
    return (read_matrix,)


@app.cell
def _(MESH_NUMBER, np):
    def format_fibre_list(fibre_list):
        # Truncate list:
        print(f"Truncating to timestep: {len(fibre_list) % (MESH_NUMBER ** 2)}")
        truncate_index = (len(fibre_list) % (MESH_NUMBER ** 2)) * (MESH_NUMBER ** 2)
        fibre_list = fibre_list[:truncate_index]

        # Instantiate empty matrix:
        total_size = truncate_index * MESH_NUMBER**2
        average_heading = np.empty(total_size)
        fibre_count = np.empty(total_size)
        angular_variance = np.empty(total_size)

        # Iterate through each cell in matrix:
        for index, heading_array in enumerate(fibre_list):
            fibre_count[index] = len(heading_array)
            if len(heading_array) == 0:
                angular_variance[index] = np.nan
                average_heading[index] = np.nan
                continue
            x_component = np.cos(heading_array * 2)
            y_component = np.sin(heading_array * 2)
            angular_variance[index] = 1 - np.linalg.norm([np.mean(x_component), np.mean(y_component)])
            average_heading[index] = np.atan2(np.mean(y_component), np.mean(x_component)) / 2

        # Reshape into appropriately sized square matrices:
        average_heading = np.reshape(average_heading, (-1, MESH_NUMBER, MESH_NUMBER))
        fibre_count = np.reshape(fibre_count, (-1, MESH_NUMBER, MESH_NUMBER))
        angular_variance = np.reshape(angular_variance, (-1, MESH_NUMBER, MESH_NUMBER))
        return average_heading, fibre_count, angular_variance
    return


@app.cell
def _(VIDEO_DIRPATH, os, read_matrix):
    matrix_filepath = os.path.join(VIDEO_DIRPATH, "matrix_seed000.txt")
    matrix_timeseries = read_matrix(matrix_filepath)
    return (matrix_timeseries,)


@app.cell
def _(cc, matrix_timeseries, np, os, plt):
    def reverse_transpose(array):
        return array[:, :]

    def plot_matrix_heading(array):
        # Set up plot:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        cmap = cc.m_CET_CBC1
        # cmap = cc.m_CET_C1
        cmap.set_bad("#FFC0CB", 1.0)  # Plot NaNs as color outside of colorbar.
        alpha = np.clip(array[:, :, 2] / 250, 0, 1)
        alpha = reverse_transpose(alpha)
        heading = reverse_transpose(array[:, :, 0])
        image_object = ax.imshow(heading, vmin=-np.pi/2, vmax=np.pi/2, cmap=cmap, alpha=alpha, origin="lower")

        # Format colorbar:
        cbar = fig.colorbar(image_object, shrink=0.8)
        cbar.set_ticks([-np.pi/2, 0, np.pi/2])
        cbar.set_ticklabels(["$-\\pi/2$", 0, "$\\pi/2$"])

        # Format axes:
        ax.set_axis_off()
        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)

    def write_matrix_video_to_file():
        if not os.path.exists("img_tmp"):
            os.mkdir("img_tmp")

        count = 0
        for timestep in list(range(2880))[::8]:
            plot_matrix_heading(matrix_timeseries[timestep, :, :, :])
            plt.savefig(os.path.join("img_tmp", f"frame_{count}.png"), dpi=300)
            count += 1
            plt.close()
    return plot_matrix_heading, write_matrix_video_to_file


@app.cell
def _(matrix_timeseries, plot_matrix_heading, plt):
    plot_matrix_heading(matrix_timeseries[0, :, :, :])
    # plot_matrix_heading(matrix_timeseries[TIMESTEPS - 1, :, :, :])
    plt.show()
    return


@app.cell
def _(write_matrix_video_to_file):
    write_matrix_video_to_file()
    return


@app.cell
def _(VIDEO_DIRPATH, os, subprocess):
    MATRIX_OUT_DIRPATH = os.path.join(VIDEO_DIRPATH, "heading_video.mp4")
    matrix_command = f"ffmpeg -y -framerate 30 -i img_tmp/frame_%d.png -vcodec libx264 -crf 18 {MATRIX_OUT_DIRPATH}"
    subprocess.run(matrix_command, shell=True)
    return


@app.cell
def _():
    # subprocess.run("rm img_tmp/*", shell=True)
    return


@app.cell
def _(np):
    def interpolate_to_wetlab_frames(position_array):
        differences = np.diff(position_array, axis=1)
        differences[differences > 1024] -= 2048
        differences[differences < -1024] += 2048

        interpolants = position_array[:, :-1, :] + (differences / 2)
        interpolated_array = []
        for idx in range(288):
            interpolated_array.append(position_array[:, idx * 5, :])
            interpolated_array.append(interpolants[:, (idx * 5) + 2, :])

        interpolated_array = np.stack(interpolated_array, axis=1)
        interpolated_array[interpolated_array > 2048] -= 2048
        interpolated_array[interpolated_array < 0] += 2048

        # Halve to match the (1024, 1024) pixel field of the real data:
        interpolated_array /= 2
        return interpolated_array
    return


@app.cell
def _(CELL_COUNT, TIMESTEPS, VIDEO_DIRPATH, np, os, pd):
    OUTPUT_COLUMN_NAMES = [
        "frame",
        "particle",
        "x",
        "y",
        "stadium_x",
        "stadium_y",
    ]


    positions_filepath = os.path.join(VIDEO_DIRPATH, "positions_seed000.csv")
    trajectory_dataframe = pd.read_csv(
        positions_filepath, index_col=None, header=None, names=OUTPUT_COLUMN_NAMES
    )

    trajectory_dataframe = trajectory_dataframe.loc[trajectory_dataframe["frame"] <= TIMESTEPS]

    # Sort by cell and then by frame:
    positions = trajectory_dataframe.sort_values(['particle', 'frame']).loc[:, ('x', 'y')]
    position_array = np.array(positions).reshape(CELL_COUNT, TIMESTEPS, 2)
    return position_array, trajectory_dataframe


@app.cell
def _(trajectory_dataframe):
    trajectory_dataframe
    return


@app.cell
def _(CELL_COUNT, TIMESTEPS, np, trajectory_dataframe):
    stadium_array = trajectory_dataframe.sort_values(['particle', 'frame']).loc[:, ('x', 'y', 'stadium_x', 'stadium_y')]
    stadium_array = np.array(stadium_array).reshape(CELL_COUNT, TIMESTEPS, 4)
    return (stadium_array,)


@app.cell
def _(np, os, plt, position_array):
    def plot_trajectory(cell_trajectory, ax, c, alpha=0.75, lw=1):
        # Separate trajectories under crossing of torus boundary:
        movement = np.sqrt(np.sum(np.diff(cell_trajectory, axis=0) ** 2, axis=1))
        rollover_mask = movement > 512

        # Get vectors for effective boundary plotting:
        movement_vectors = np.diff(cell_trajectory, axis=0)

        # Plot trajectory as normal:
        if np.count_nonzero(rollover_mask) == 0:
            ax.plot(
                cell_trajectory[:, 0],
                cell_trajectory[:, 1],
                c=c, alpha=alpha, lw=lw
            )
        else:
            rollover_indices = np.argwhere(rollover_mask)
            prev_index = 0
            for rollover_index in rollover_indices:
                rollover_index = rollover_index[0]
                ax.plot(
                    cell_trajectory[prev_index:rollover_index, 0],
                    cell_trajectory[prev_index:rollover_index, 1],
                    c=c, alpha=alpha, lw=lw
                )
                prev_index = rollover_index + 1

            # Plot final segment
            ax.plot(
                cell_trajectory[prev_index:, 0],
                cell_trajectory[prev_index:, 1],
                c=c, alpha=alpha, lw=lw
            )

    def plot_csv(position_array):
        fig, ax = plt.subplots(figsize=(3, 3))
        for cell_index in range(position_array.shape[0]):
        # for cell_index in range(10):
            cell_trajectory = position_array[cell_index, :, :]
            plot_trajectory(cell_trajectory, ax, 'k')

        ax.set_xlim(0, 2048)
        ax.set_ylim(0, 2048)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)

    def write_trajectory_video_to_file():
        if not os.path.exists("img_tmp"):
            os.mkdir("img_tmp")

        count = 0
        for timestep in list(range(2880))[::8]:
            plot_csv(position_array[:, :timestep, :])
            plt.savefig(os.path.join("img_tmp", f"frame_{count}.png"), dpi=300)
            count += 1
            plt.close()
    return plot_csv, write_trajectory_video_to_file


@app.cell
def _(TIMESTEPS, plot_csv, plt, position_array):
    plot_csv(position_array[:, TIMESTEPS - 2048:TIMESTEPS, :])
    plt.show()
    return


@app.cell
def _(write_trajectory_video_to_file):
    write_trajectory_video_to_file()
    return


@app.cell
def _(position_array):
    position_array.shape
    return


@app.cell
def _(VIDEO_DIRPATH, os, subprocess):
    TRAJECTORY_OUT_DIRPATH = os.path.join(VIDEO_DIRPATH, "trajectory_video.mp4")
    trajectory_command = f"ffmpeg -y -framerate 30 -i img_tmp/frame_%d.png -vcodec libx264 -crf 18 {TRAJECTORY_OUT_DIRPATH}"
    subprocess.run(trajectory_command, shell=True)
    return


@app.cell
def _(CELL_COUNT, mpl, np, plt, stadium_array):
    CELL_BODY_RADIUS = 60 / 2

    def calculate_eff_radius(l, r_0):
        phi = 2 * l / np.pi
        det = phi**2 + 4*(r_0**2)
        effectiveRadius = (np.sqrt(det) - phi) / 2
        return effectiveRadius

    def plot_stadium(x1, x2, y1, y2, ax, base_radius=CELL_BODY_RADIUS, colour="k"):
        # Account for boundaries:
        x_diff = x1 - x2
        y_diff = y1 - y2

        # Get stadium characteristics:
        stadium_length = np.sqrt(x_diff**2 + y_diff**2)
        stadium_angle = np.atan2(y_diff, x_diff) * 180 / np.pi

        # Get effective radius:
        radius = base_radius

        # Set up patch collection:
        patches = []

        # Patch for leading edge:
        leading_patch = mpl.patches.Wedge(
            (x1, y1), radius, stadium_angle - 90, stadium_angle + 90,
            linewidth=0
        )
        patches.append(leading_patch)

        # Patch for trailing edge:
        trailing_patch = mpl.patches.Wedge(
            (x2, y2), radius, stadium_angle + 90, stadium_angle - 90,
            linewidth=0
        )
        patches.append(trailing_patch)

        # Patch for connection:
        # -- Get bottom left corner:
        shift_angle = (stadium_angle - 90) * np.pi / 180 
        corner_x = x2 + (radius * np.cos(shift_angle))
        corner_y = y2 + (radius * np.sin(shift_angle))
        # -- Set up rectangle:
        rectangle_patch = mpl.patches.Rectangle(
            (corner_x, corner_y), stadium_length, radius * 2, angle=stadium_angle, rotation_point='xy',
            linewidth=0
        )
        patches.append(rectangle_patch)

        # Plot patches:
        p = mpl.collections.PatchCollection(patches, alpha=0.15, color=['b'] + ([colour] * 2), linewidth=0)
        ax.add_collection(p)


    def plot_stadia(timestep):
        # Set up figure:
        fig, ax = plt.subplots(figsize=(3, 3))

        for cell_index in range(CELL_COUNT):
            # Points:
            x1, y1, x2, y2 = stadium_array[cell_index, timestep, :] / 2

            # Account for boundaries:
            x_diff = x1 - x2
            y_diff = y1 - y2
            if np.abs(x_diff) > 512 and np.abs(y_diff) > 512:
                x_correction = 1024 * np.sign(x_diff)
                y_correction = 1024 * np.sign(y_diff)
                plot_stadium(x1, x2 + x_correction, y1, y2 + y_correction, ax)
                plot_stadium(x1 - x_correction, x2, y1 - y_correction, y2, ax)
                plot_stadium(x1, x2 + x_correction, y1 - y_correction, y2, ax)
                plot_stadium(x1 - x_correction, x2, y1, y2 + y_correction, ax)
            elif np.abs(x_diff) > 512:
                x_correction = 1024 * np.sign(x_diff)
                plot_stadium(x1, x2 + x_correction, y1, y2, ax)
                plot_stadium(x1, x2 + x_correction, y1, y2, ax, CELL_BODY_RADIUS / 2, "g")
                plot_stadium(x1 - x_correction, x2, y1, y2, ax)
                plot_stadium(x1 - x_correction, x2, y1, y2, ax, CELL_BODY_RADIUS / 2, "g")
            elif np.abs(y_diff) > 512:
                y_correction = 1024 * np.sign(y_diff)
                plot_stadium(x1, x2, y1, y2 + y_correction, ax)
                plot_stadium(x1, x2, y1, y2 + y_correction, ax, CELL_BODY_RADIUS / 2, "g")
                plot_stadium(x1, x2, y1 - y_correction, y2, ax)
                plot_stadium(x1, x2, y1 - y_correction, y2, ax, CELL_BODY_RADIUS / 2, "g")
            else:
                plot_stadium(x1, x2, y1, y2, ax)
                plot_stadium(x1, x2, y1, y2, ax, CELL_BODY_RADIUS / 2, "g")

        ax.set_xlim(0, 1024)
        ax.set_ylim(0, 1024)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)
    return (plot_stadia,)


@app.cell
def _(TIMESTEPS, plot_stadia, plt):
    plot_stadia(TIMESTEPS - 1)
    plt.show()
    return


@app.cell
def _(os, plot_stadia, plt):
    def write_stadium_video_to_file():
        if not os.path.exists("img_tmp"):
            os.mkdir("img_tmp")

        count = 0
        for timestep in list(range(2880))[::16]:
            plot_stadia(timestep)
            plt.savefig(os.path.join("img_tmp", f"frame_{count}.png"), dpi=300)
            count += 1
            plt.close()

    write_stadium_video_to_file()
    return


@app.cell
def _(VIDEO_DIRPATH, os, subprocess):
    STADIUM_OUT_DIRPATH = os.path.join(VIDEO_DIRPATH, "stadium_video.mp4")
    stadium_command = f"ffmpeg -y -framerate 30 -i img_tmp/frame_%d.png -vcodec libx264 -crf 18 {STADIUM_OUT_DIRPATH}"
    subprocess.run(stadium_command, shell=True)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
