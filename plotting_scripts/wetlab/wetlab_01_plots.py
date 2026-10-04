import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import subprocess

    import skimage

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import colorcet as cc

    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    from datetime import datetime
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    return (
        AnchoredSizeBar,
        cc,
        datetime,
        fm,
        np,
        os,
        pd,
        plt,
        skimage,
        subprocess,
    )


@app.cell
def _():
    import matplotlib as mpl

    # Font formatting:
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = "cmr10"
    mpl.rcParams['font.size'] = 9
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['axes.labelsize'] = 9

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
    return


@app.cell
def _(datetime, subprocess):
    # PNG 300 dpi
    # A4 dimensions: 8.27 × 11.69 inches
    # Image dimensions: 160 x ? mm
    # Metadata: date, script, github branch id, og experiment source
    OUT_DIRPATH = "plotting_scripts/wetlab/out"
    CONTROL_PALETTE = "#1A85FF"
    RD_PALETTE = "#D41159"
    MINIMUM_FRAMES = 48
    PIXEL_SIZE = 0.3469 * 2  # Pixel size in µm
    SAMPLE_EXPERIMENT = "/camp/home/eloaklo/home/shared/eloaklo/analysed_data/OEO20260313"
    MM_UNIT = 1/25.4  # Millimeters in inches, for matplotlib

    TEXT_WIDTH = 135 * MM_UNIT
    TEXT_HEIGHT = 217 * MM_UNIT 

    FULL_WIDTH = 170 * MM_UNIT
    FULL_HEIGHT = TEXT_HEIGHT * 0.8

    # Get current commit hash:
    commit_hash = subprocess.run("git rev-parse --short HEAD", shell=True, capture_output=True)
    commit_hash = commit_hash.stdout.decode("utf-8")[:-1]

    METADATA_DICTIONARY = {
        "creator": "Omar El Oakley",
        "script": "wetlab_01_plots.py",
        "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_experiment": SAMPLE_EXPERIMENT,
        "current_commit_hash": commit_hash
    }
    return (
        CONTROL_PALETTE,
        METADATA_DICTIONARY,
        MINIMUM_FRAMES,
        OUT_DIRPATH,
        PIXEL_SIZE,
        RD_PALETTE,
        SAMPLE_EXPERIMENT,
        TEXT_WIDTH,
    )


@app.cell
def _(MINIMUM_FRAMES, PIXEL_SIZE, np, skimage):
    def filter_movement(site_csv, all_ids, min_frames):
        valid_ids = []
        for particle_index in all_ids:
            # Get relevant positions from global dataframe:
            particle_mask = site_csv["tree_id"] == particle_index

            # Render trajectory invalid if its number of constituent frames is less than the limit:
            frame_count = np.count_nonzero(particle_mask)
            if frame_count < min_frames:
                continue

            x_pos = np.array(site_csv.loc[particle_mask, "x"])
            y_pos = np.array(site_csv.loc[particle_mask, "y"])

            # Get speed:
            # -- Ensure we only estimate speeds from positions in adjacent frames:
            frame_array = np.array(site_csv.loc[particle_mask, "frame"])
            no_frame_break_mask = np.diff(frame_array) == 1

            # -- Mask position differences so we get adequate speed estimation:
            x_diff = np.diff(x_pos)
            y_diff = np.diff(y_pos)
            x_filter_diff = x_diff[no_frame_break_mask]
            y_filter_diff = y_diff[no_frame_break_mask]
            frame_length = len(x_filter_diff)

            # Get meander ratio:
            path_length = np.sum(np.sqrt(x_diff**2 + y_diff**2))
            final_displacement = np.sqrt(((x_pos[0] - x_pos[-1]) ** 2) + ((y_pos[0] - y_pos[-1]) ** 2))
            # Render trajectory invalid if its path length is 0:
            if path_length == 0:
                continue
            meander_ratio = final_displacement / path_length

            valid_ids.append(particle_index)

        return valid_ids


    def generate_history_plot(site_csv):
        # Filter trajectories:
        all_ids = np.unique(site_csv["tree_id"])
        valid_ids = filter_movement(site_csv, all_ids, MINIMUM_FRAMES)

        # Raster plot trajectories based on frame:
        line_array = np.zeros((768, 768))
        for valid_id in valid_ids:
            # Get position data for this particle:
            particle_mask = site_csv["tree_id"] == valid_id
            particle_csv = site_csv.loc[particle_mask].sort_values("frame")
            xy_data = np.array(particle_csv.loc[:, ["x", "y"]]) * (3/4)
            frame_labels = np.array(particle_csv.loc[:, "frame"])

            # Generate trajectory frame:
            for trajectory_index in range(len(xy_data) - 1):
                # Get indices of line:
                xy_t = xy_data[trajectory_index, :].astype(int)
                xy_t1 = xy_data[trajectory_index+1, :].astype(int)

                # Plot line indices on matrix:
                line_rr, line_cc, val = skimage.draw.line_aa(*xy_t, *xy_t1)
                frame_value = (frame_labels[trajectory_index] + frame_labels[trajectory_index + 1]) / 2
                line_array[line_rr, line_cc] = val * frame_value

        line_array = (line_array * 2.5) / 60

        return line_array


    def generate_speed_plot(site_csv):
        # Filter trajectories:
        all_ids = np.unique(site_csv["tree_id"])
        valid_ids = filter_movement(site_csv, all_ids, MINIMUM_FRAMES)

        # Raster plot trajectories based on frame:
        line_array = np.zeros((768, 768))
        for valid_id in valid_ids:
            # Get position data for this particle:
            particle_mask = site_csv["tree_id"] == valid_id
            particle_csv = site_csv.loc[particle_mask].sort_values("frame")
            xy_data = np.array(particle_csv.loc[:, ["x", "y"]]) * (3/4)
            frame_labels = np.array(particle_csv.loc[:, "frame"])

            # Generate trajectory frame:
            for trajectory_index in range(len(xy_data) - 1):
                # Get indices of line:
                xy_t = xy_data[trajectory_index, :]
                xy_t1 = xy_data[trajectory_index+1, :]
                distance_travelled = np.sqrt(np.sum((xy_t1 - xy_t) ** 2))
                time_elapsed = frame_labels[trajectory_index + 1] - frame_labels[trajectory_index]
                time_elapsed *= 2.5
                segment_speed = (distance_travelled * PIXEL_SIZE) /  time_elapsed

                # Plot line indices on matrix:
                line_rr, line_cc, val = skimage.draw.line_aa(*xy_t.astype(int), *xy_t1.astype(int))
                line_array[line_rr, line_cc] = val * segment_speed

        return line_array
    return generate_history_plot, generate_speed_plot


@app.cell
def _(SAMPLE_EXPERIMENT, generate_history_plot, generate_speed_plot, os, pd):
    # Get representative array of trajectories:
    representative_sites = [
        "A1-Site_0",
        "A2-Site_0",
        "A3-Site_0",
        "A4-Site_0",
        "A5-Site_0",
        "A6-Site_0"
    ]
    csv_list = []
    for site_name in representative_sites:
        csv_list.append(pd.read_csv(os.path.join(SAMPLE_EXPERIMENT, "trajectories", f"{site_name}.csv")))

    trajectory_array_list = []
    speed_array_list = []
    for site_csv in csv_list:
        speed_array_list.append(generate_speed_plot(site_csv))
        trajectory_array_list.append(generate_history_plot(site_csv))
    return representative_sites, speed_array_list, trajectory_array_list


@app.cell
def _(
    AnchoredSizeBar,
    CONTROL_PALETTE,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    PIXEL_SIZE,
    RD_PALETTE,
    TEXT_WIDTH,
    cc,
    datetime,
    fm,
    os,
    plt,
    representative_sites,
    trajectory_array_list,
):
    def plot_trajectory_grid():
        fig, axs = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 3.5))

        # Set up fontprops for scalebar:
        fontprops = fm.FontProperties(size=7)
        for index, ax in enumerate(axs.flatten()):
            # Set color:
            if index <= 2:
                color = CONTROL_PALETTE
            else:
                color = RD_PALETTE

            # Plot trajectories:
            extent=[0,100,0,100]
            image_object = ax.imshow(
                trajectory_array_list[index], vmin=0, vmax=24,
                extent=[0, PIXEL_SIZE * 1024, 0, PIXEL_SIZE * 1024],
                cmap=cc.m_CET_L16, interpolation="bilinear"
            )
            ax.text(25, 30, representative_sites[index], color="w", fontproperties=fontprops)

            # Format spines:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set(color=color, linewidth=3.5)

            # Add scalebar:
            scalebar = AnchoredSizeBar(
                ax.transData,
                100, '100 µm', 'upper right', 
                pad=1, color='white', frameon=False,
                size_vertical=1, fontproperties=fontprops
            )
            ax.add_artist(scalebar)    

        # Set condition labels:
        axs[0, 0].set_ylabel("Control", color=CONTROL_PALETTE,  labelpad=6.0)
        axs[1, 0].set_ylabel("RD", color=RD_PALETTE, labelpad=6.0)

        # Set cell density labels:
        axs[1, 0].set_xlabel("5.0x$10^3$ cells/well", labelpad=6.0)
        axs[1, 1].set_xlabel("3.3x$10^3$ cells/well", labelpad=6.0)
        axs[1, 2].set_xlabel("1.6x$10^3$ cells/well", labelpad=6.0)

        # Adjust layout:
        edging = 0.02
        fig.subplots_adjust(0.1, 0.1, 1 - 0.1, 1 - 0.1, hspace=0.05, wspace=0.05)

        # Add colorbar:
        fig.colorbar(
            image_object, ax=axs.flatten(),
            label="Time Elapsed (h)", ticks=[0, 12, 24],
            fraction=0.025, shrink=1.0
        )

        # Update metadata time:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(os.path.join(OUT_DIRPATH, "trajectory_grid.png"), dpi=300, metadata=METADATA_DICTIONARY, transparent=True)
        plt.show()

    plot_trajectory_grid()
    return


@app.cell
def _(
    AnchoredSizeBar,
    CONTROL_PALETTE,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    PIXEL_SIZE,
    RD_PALETTE,
    TEXT_WIDTH,
    cc,
    datetime,
    fm,
    os,
    plt,
    representative_sites,
    speed_array_list,
):
    def plot_speed_grid():
        fig, axs = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 3.5))

        # Set up fontprops for scalebar:
        fontprops = fm.FontProperties(size=7)
        for index, ax in enumerate(axs.flatten()):
            # Set color:
            if index <= 2:
                color = CONTROL_PALETTE
            else:
                color = RD_PALETTE

            # Plot trajectories:
            extent=[0,100,0,100]
            image_object = ax.imshow(
                speed_array_list[index], vmin=0, vmax=0.5,
                extent=[0, PIXEL_SIZE * 1024, 0, PIXEL_SIZE * 1024],
                cmap=cc.m_CET_L4, interpolation="bilinear"
            )
            ax.text(25, 30, representative_sites[index], color="w", fontproperties=fontprops)

            # Format spines:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set(color=color, linewidth=3.5)

            # Add scalebar:
            scalebar = AnchoredSizeBar(
                ax.transData,
                100, '100 $\\mu$m', 'upper right', 
                pad=1, color='white', frameon=False,
                size_vertical=1, fontproperties=fontprops
            )
            ax.add_artist(scalebar)

        # Set condition labels:
        axs[0, 0].set_ylabel("Control", color=CONTROL_PALETTE,  labelpad=6.0)
        axs[1, 0].set_ylabel("RD", color=RD_PALETTE, labelpad=6.0)

        # Set cell density labels:
        axs[1, 0].set_xlabel("5.0x$10^3$ cells/well", labelpad=6.0)
        axs[1, 1].set_xlabel("3.3x$10^3$ cells/well", labelpad=6.0)
        axs[1, 2].set_xlabel("1.6x$10^3$ cells/well", labelpad=6.0)

        # Adjust layout:
        edging = 0.02
        fig.subplots_adjust(0.1, 0.1, 1 - 0.1, 1 - 0.1, hspace=0.05, wspace=0.05)

        # Add colorbar:
        fig.colorbar(
            image_object, ax=axs.flatten(),
            label="Speed ($\\mu$m/min)",
            fraction=0.025, shrink=1.0
        )

        # Update metadata time:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(os.path.join(OUT_DIRPATH, "speed_grid.png"), dpi=300, metadata=METADATA_DICTIONARY, transparent=True)
        plt.show()

    plot_speed_grid()
    return


@app.cell
def _():
    # Generate roseplots:
    return


@app.cell
def _():
    # Generate LDA plot of trajectories (just for fun):
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
