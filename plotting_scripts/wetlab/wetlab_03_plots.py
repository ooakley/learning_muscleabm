import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import subprocess

    import skimage
    import sklearn

    import scipy.stats

    import numpy as np
    import pandas as pd
    import colorcet as cc
    import tifffile as tfl

    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    from datetime import datetime
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    return cc, datetime, np, os, pd, plt, skimage, subprocess


@app.cell
def _(datetime, subprocess):
    # PNG 300 dpi
    # A4 dimensions: 8.27 × 11.69 inches
    # Metadata: date, script, github branch id, og experiment source
    OUT_DIRPATH = "plotting_scripts/out"
    CONTROL_PALETTE = "#1A85FF"
    RD_PALETTE = "#D41159"
    PIXEL_SIZE = 0.3469 * 2  # Pixel size in µm
    SAMPLE_EXPERIMENT = "/camp/home/eloaklo/home/shared/eloaklo/analysed_data/OEO20260313"
    MM_UNIT = 1/25.4  # Millimeters in inches, for matplotlib

    # Get current commit hash:
    commit_hash = subprocess.run("git rev-parse --short HEAD", shell=True, capture_output=True)
    commit_hash = commit_hash.stdout.decode("utf-8")[:-1]

    METADATA_DICTIONARY = {
        "creator": "Omar El Oakley",
        "script": "wetlab_02_plots.py",
        "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_experiment": SAMPLE_EXPERIMENT,
        "current_commit_hash": commit_hash
    }

    seaborn_palette = [CONTROL_PALETTE, RD_PALETTE]
    return MM_UNIT, SAMPLE_EXPERIMENT


@app.cell
def _():
    COL_PHENOTYPE_DICT = {
        1: "CTL",
        2: "CTL",
        3: "CTL",
        4: "RD",
        5: "RD",
        6: "RD"
    }

    COL_DENSITY_DICT = {
        1: "High",
        2: "Medium",
        3: "Low",
        4: "High",
        5: "Medium",
        6: "Low"
    }

    FRAME_DURATION = 2.5  # Frame length in minutes
    return COL_DENSITY_DICT, COL_PHENOTYPE_DICT, FRAME_DURATION


@app.cell
def _(
    COL_DENSITY_DICT,
    COL_PHENOTYPE_DICT,
    FRAME_DURATION,
    np,
    os,
    pd,
    skimage,
):
    def get_movement(site_csv, all_ids, min_frames=25):
        particle_speeds = []
        meander_ratios = []
        frame_lengths = []
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
            average_speed = np.sum(np.sqrt(x_filter_diff**2 + y_filter_diff**2)) / (frame_length * FRAME_DURATION)

            # Get meander ratio:
            path_length = np.sum(np.sqrt(x_diff**2 + y_diff**2))
            final_displacement = np.sqrt(((x_pos[0] - x_pos[-1]) ** 2) + ((y_pos[0] - y_pos[-1]) ** 2))
            # Render trajectory invalid if its path length is 0:
            if path_length == 0:
                continue
            meander_ratio = final_displacement / path_length

            particle_speeds.append(average_speed)
            meander_ratios.append(meander_ratio)
            frame_lengths.append(frame_length)
            valid_ids.append(particle_index)

        return np.array(particle_speeds), np.array(meander_ratios), np.array(frame_lengths), valid_ids


    def get_average_particle_count(site_csv, valid_ids):
        counts = []
        for frame_index in range(np.max(site_csv["frame"])):
            id_csv = site_csv.loc[site_csv["frame"] == frame_index, "tree_id"]
            counts.append(np.count_nonzero(np.isin(id_csv, valid_ids)))
        return np.mean(counts), np.std(counts)


    def plot_trajectories(site_csv, id_list, ax):
        # Plot trajectory filtering:
        for id in id_list:
            particle_mask = site_csv["tree_id"] == id
            ax.plot(
                site_csv.loc[particle_mask, "x"],
                site_csv.loc[particle_mask, "y"],
                lw=1, c='k'
            )
        ax.set_aspect("equal")
        ax.set_xlim(0, 1024)
        ax.set_ylim(0, 1024)
        ax.set_xticks([])
        ax.set_yticks([])


    def get_coherency_fraction(site_csv, valid_ids):
        # Estimate from trajectory dataframe as test:
        line_array = []
        disk_array = []
        for valid_id in valid_ids:
            # Set up empty array (we downsample by 2x for speed & robustness):
            particle_line_array = np.zeros((512, 512))

            # Get position data for this particle:
            particle_mask = site_csv["tree_id"] == valid_id
            particle_csv = site_csv[particle_mask].sort_values("frame")
            xy_data = np.array(particle_csv.loc[:, ["x", "y"]]) / 2

            # Generate trajectory frame:
            for frame_index in range(len(xy_data) - 1):
                # Get indices of line:
                xy_t = xy_data[frame_index, :].astype(int)
                xy_t1 = xy_data[frame_index+1, :].astype(int)

                # Plot line indices on matrix:
                line_rr, line_cc = skimage.draw.line(*xy_t, *xy_t1)
                particle_line_array[line_rr, line_cc] += 1

            # Get frame of interaction area of trajectory:
            particle_line_array = np.clip(particle_line_array, 0, 1)
            particle_disk_array = skimage.morphology.isotropic_dilation(np.bool(particle_line_array), 16)
            line_array.append(particle_line_array)
            disk_array.append(particle_disk_array)

        line_array = np.stack(line_array, axis=0)
        disk_array = np.stack(disk_array, axis=0)
        trajectory_array = np.clip(np.sum(line_array, axis=0), 0, 1)

        # Find orientations of lines:
        structure_tensor = skimage.feature.structure_tensor(
            trajectory_array, sigma=16,
            mode='constant', cval=0,
            order='rc'
        )

        # Get local coherencies:
        eigenvalues = skimage.feature.structure_tensor_eigenvalues(structure_tensor)
        coherency_numerator = eigenvalues[0, :, :] - eigenvalues[1, :, :]
        coherency_denominator = eigenvalues[0, :, :] + eigenvalues[1, :, :]
        coherency = coherency_numerator / coherency_denominator

        # Get per-particle coherencies and interaction terms:
        filtered_coherency = np.copy(coherency)
        filtered_coherency[np.isnan(coherency)] = 0

        interaction_values = []
        coherency_values = []
        path_sum = np.sum(disk_array, axis=0)
        for particle_index in range(disk_array.shape[0]):
            indexed_path = disk_array[particle_index, :, :]
            comparator_paths = np.clip(path_sum - indexed_path, 0, 1)
            interaction_value = np.sum(comparator_paths * indexed_path) / np.sum(indexed_path)
            coherency_value = np.sum(filtered_coherency * indexed_path) / np.sum(indexed_path)
            interaction_values.append(interaction_value)
            coherency_values.append(coherency_value)

        return interaction_values, coherency_values, line_array, disk_array, coherency


    def analyse_sites(data_directory):
        ROWS = ['A', 'B', 'C']
        COLUMNS = ['1', '2', '3', '4', '5', '6']
        dataframe = []
        line_arrays = []
        disk_arrays = []
        coherency_arrays = []
        for row in ROWS:
            for column in COLUMNS:
                for site in range(4):
                    # Get filepath:
                    filepath = os.path.join(data_directory, "trajectories", f"{row}{column}-Site_{site}.csv")
                    site_csv = pd.read_csv(filepath)

                    # Analyse sites:
                    all_ids = np.unique(site_csv["tree_id"])
                    particle_speeds, meander_ratios, frame_lengths, valid_ids = get_movement(site_csv, all_ids)
                    particle_count, count_std = get_average_particle_count(site_csv, valid_ids)
                    print(f"{row}{column}-Site_{site} count: {particle_count}")
                    print(f"{row}{column}-Site_{site} stddev: {count_std}")

                    # # Plot filter outcome:
                    # fig, axs = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
                    # plot_trajectories(site_csv, all_ids, axs[0])
                    # plot_trajectories(site_csv, valid_ids, axs[1])
                    # figure_dirpath = os.path.join(data_directory, "trajectory_filters")
                    # if not os.path.exists(figure_dirpath):
                    #     os.mkdir(figure_dirpath)
                    # plt.savefig(os.path.join(figure_dirpath, f"{row}{column}-Site_{site}.png"))
                    # plt.close()

                    # Calculate coherency fraction:
                    interaction_values, coherency_values, line_array, disk_array, coherency \
                        = get_coherency_fraction(site_csv, valid_ids)

                    line_arrays.append(line_array)
                    disk_arrays.append(disk_array)
                    coherency_arrays.append(coherency)

                    # Put into dictionary:
                    partial_dataframe = {}

                    # Independent variables:
                    partial_dataframe["row"] = [row] * len(particle_speeds)
                    partial_dataframe["column"] = [column] * len(particle_speeds)
                    partial_dataframe["site"] = [site] * len(particle_speeds)
                    partial_dataframe["phenotype"] = [COL_PHENOTYPE_DICT[int(column)]] * len(particle_speeds)
                    partial_dataframe["density"] = [COL_DENSITY_DICT[int(column)]] * len(particle_speeds)

                    # Dependent variables:
                    partial_dataframe["speed"] = particle_speeds
                    partial_dataframe["meander_ratio"] = meander_ratios
                    partial_dataframe["frame_length"] = frame_lengths
                    partial_dataframe["particle_interaction"] = interaction_values
                    partial_dataframe["particle_coherency"] = coherency_values
                    partial_dataframe["particle_count"] = [particle_count] * len(particle_speeds)

                    # Append to total dataframe:
                    dataframe.append(pd.DataFrame.from_dict(partial_dataframe))

        return pd.concat(dataframe), line_arrays, disk_arrays, coherency_arrays
    return (analyse_sites,)


@app.cell
def _(SAMPLE_EXPERIMENT, analyse_sites):
    dataframe, line_arrays, disk_arrays, coherency_arrays = analyse_sites(SAMPLE_EXPERIMENT)
    return coherency_arrays, dataframe, disk_arrays, line_arrays


@app.cell
def _(coherency_arrays, dataframe, disk_arrays, line_arrays, np, os):
    # Set up scratch directory:
    scratch_dirpath = os.path.join("plotting_scripts", "scratch")
    if not os.path.exists(scratch_dirpath):
        os.mkdir(scratch_dirpath)
    dataframe, line_arrays, disk_arrays, coherency_arrays

    # Save to scratch directory:
    dataframe.to_csv(os.path.join(scratch_dirpath, "sample_df.csv"))
    np.savez(os.path.join(scratch_dirpath, "line_arrays.npz"), *line_arrays)
    np.savez(os.path.join(scratch_dirpath, "disk_arrays.npz"), *disk_arrays)
    np.savez(os.path.join(scratch_dirpath, "coherency_arrays.npz"), *coherency_arrays)
    return (scratch_dirpath,)


@app.cell
def _(np, os, scratch_dirpath):
    test = np.load(os.path.join(scratch_dirpath, "line_arrays.npz"))
    return


@app.cell
def _():
    list()
    return


@app.cell
def _(MM_UNIT, cc, coherency_arrays, line_arrays, np, plt):
    def plot_coherency_site():
        fig, axs = plt.subplots(3, 1, figsize=(80 * MM_UNIT, 160 * MM_UNIT), layout="constrained")

        # Plot lines:
        line_mask = np.clip(np.sum(line_arrays[0], axis=0), 0, 1)
        axs[0].imshow(line_mask, cmap='gray')
        axs[0].set_axis_off()

        # Plot coherency:
        axs[1].imshow(coherency_arrays[0], cmap=cc.m_CET_L3)
        axs[1].set_axis_off()

        # Plot line-masked coherency:
        coherency_line = np.copy(coherency_arrays[0])
        coherency_line[~np.bool(line_mask)] = 0
        axs[2].imshow(coherency_line, cmap=cc.m_CET_L3)
        axs[2].set_axis_off()
        plt.show()

    plot_coherency_site()
    return


@app.cell
def _(MM_UNIT, disk_array, disk_arrays, filtered_coherency, np, plt, skimage):
    def get_disk_outline(disk_array):
        binary_dilation = skimage.morphology.dilation(disk_array)
        outlined_image = binary_dilation ^ disk_array
        return outlined_image

    def plot_disk_array_site():
        fig, axs = plt.subplots(3, 1, figsize=(80 * MM_UNIT, 160 * MM_UNIT), layout="constrained")
        disk_sum = np.sum(disk_arrays[0], axis=0)

        # Plot disc sum:
        axs[0].imshow(disk_sum)
        axs[0].set_axis_off()

        # Plot outlines:
        for i in range(50):
            outline = get_disk_outline(disk_arrays[0][i])
            disk_sum[outline] = 50
        axs[1].imshow(disk_sum)
        axs[1].set_axis_off()

        # Plot trajectories colored by interaction:
        interaction_values = []
        coherency_values = []
        path_sum = np.sum(disk_array, axis=0)
        for particle_index in range(disk_array.shape[0]):
            indexed_path = disk_array[particle_index, :, :]
            comparator_paths = np.clip(path_sum - indexed_path, 0, 1)
            interaction_value = np.sum(comparator_paths * indexed_path) / np.sum(indexed_path)
            coherency_value = np.sum(filtered_coherency * indexed_path) / np.sum(indexed_path)
            interaction_values.append(interaction_value)
            coherency_values.append(coherency_value)
        plt.show()

    plot_disk_array_site()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
