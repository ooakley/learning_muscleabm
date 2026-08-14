import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import skimage

    import numpy as np
    import pandas as pd
    import seaborn as sns

    import matplotlib.pyplot as plt

    import colorcet as cc
    return cc, np, os, pd, plt, skimage, sns


@app.cell
def _():
    # Experiment folders:
    # OEO20260313
    # OEO20260317
    # OEO20260320 & OEO20260321
    # OEO20260324
    return


@app.cell
def _():
    column_phenotype_dictionary = {
        1: "CTL",
        2: "CTL",
        3: "CTL",
        4: "RD",
        5: "RD",
        6: "RD"
    }

    column_density_dictionary = {
        1: "High",
        2: "Medium",
        3: "Low",
        4: "High",
        5: "Medium",
        6: "Low"
    }
    return column_density_dictionary, column_phenotype_dictionary


@app.cell
def _():
    FRAME_DURATION = 2.5  # Duration of each frame in minutes.
    return (FRAME_DURATION,)


@app.cell
def _(
    FRAME_DURATION,
    column_density_dictionary,
    column_phenotype_dictionary,
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
            # Set up empty array (we downsample by 4x for speed & robustness):
            particle_line_array = np.zeros((256, 256))

            # Get position data for this particle:
            particle_mask = site_csv["tree_id"] == valid_id
            particle_csv = site_csv[particle_mask].sort_values("frame")
            xy_data = np.array(particle_csv.loc[:, ["x", "y"]]) / 4

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
            particle_disk_array = skimage.morphology.isotropic_dilation(np.bool(particle_line_array), 4)
            line_array.append(particle_line_array)
            disk_array.append(particle_disk_array)

        line_array = np.stack(line_array, axis=0)
        disk_array = np.stack(disk_array, axis=0)
        trajectory_array = np.clip(np.sum(line_array, axis=0), 0, 1)

        # Find orientations of lines:
        structure_tensor = skimage.feature.structure_tensor(
            trajectory_array, sigma=8,
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

        return interaction_values, coherency_values


    def analyse_sites(data_directory):
        ROWS = ['A', 'B', 'C']
        COLUMNS = ['1', '2', '3', '4', '5', '6']
        dataframe = []
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
                    interaction_values, coherency_values = get_coherency_fraction(site_csv, valid_ids)

                    # # Plot coherency:
                    # fig, axs = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
                    # # -- Direct coherency:
                    # axs[0].imshow(coherency, cmap='gray', vmin=0, vmax=1)
                    # axs[0].set_axis_off()
                    # # -- Blur cutoff:
                    # # masked_blur = np.copy(blurred_paths)
                    # # masked_blur[blurred_paths < 0.1] = 0
                    # # axs[1].hist(masked_blur.flatten())
                    # axs[1].imshow(interaction_map, cmap='gray')
                    # axs[1].set_axis_off()
                    # # -- Path shaded by coherency:
                    # masked_interaction = np.copy(interaction_map)
                    # masked_interaction[line_array < 1] = 0
                    # axs[2].imshow(masked_interaction, cmap='gray', vmin=0, vmax=1)
                    # axs[2].set_axis_off()

                    # Save image:
                    # figure_dirpath = os.path.join(data_directory, "coherency_images")
                    # if not os.path.exists(figure_dirpath):
                    #     os.mkdir(figure_dirpath)
                    # plt.savefig(os.path.join(figure_dirpath, f"{row}{column}-Site_{site}.png"))
                    # plt.close()

                    # Put into dictionary:
                    partial_dataframe = {}

                    # Independent variables:
                    partial_dataframe["row"] = [row] * len(particle_speeds)
                    partial_dataframe["column"] = [column] * len(particle_speeds)
                    partial_dataframe["site"] = [site] * len(particle_speeds)
                    partial_dataframe["phenotype"] = [column_phenotype_dictionary[int(column)]] * len(particle_speeds)
                    partial_dataframe["density"] = [column_density_dictionary[int(column)]] * len(particle_speeds)

                    # Dependent variables:
                    partial_dataframe["speed"] = particle_speeds
                    partial_dataframe["meander_ratio"] = meander_ratios
                    partial_dataframe["frame_length"] = frame_lengths
                    partial_dataframe["particle_interaction"] = interaction_values
                    partial_dataframe["particle_coherency"] = coherency_values
                    partial_dataframe["particle_count"] = [particle_count] * len(particle_speeds)

                    # Append to total dataframe:
                    dataframe.append(pd.DataFrame.from_dict(partial_dataframe))

        return pd.concat(dataframe)
    return (
        analyse_sites,
        get_average_particle_count,
        get_coherency_fraction,
        get_movement,
    )


@app.cell
def _(
    get_average_particle_count,
    get_coherency_fraction,
    get_movement,
    np,
    os,
    pd,
):
    def test_analysis():
        site_csv = pd.read_csv(
            os.path.join(
                "/camp/home/eloaklo/home/shared/eloaklo/analysed_data/OEO20260313",
                "trajectories",
                "A1-Site_0.csv"
            )
        )

        # Analyse sites:
        all_ids = np.unique(site_csv["tree_id"])
        particle_speeds, meander_ratios, frame_lengths, valid_ids = get_movement(site_csv, all_ids)
        particle_count, count_std = get_average_particle_count(site_csv, valid_ids)

        # Calculate coherency fraction:
        _, _, coherency, trajectory, disk = get_coherency_fraction(site_csv, valid_ids)

        return coherency, trajectory, disk

    coherency, trajectory, disk = test_analysis()
    return coherency, disk, trajectory


@app.cell
def _(cc, coherency, np, plt, trajectory):
    def plot_example_coherency():
        fig, ax = plt.subplots(layout="constrained")
        trajectory_coherency = np.copy(coherency)
        trajectory_coherency[~np.bool(trajectory)] = 0
        image_object = ax.imshow(trajectory_coherency.T, vmin=0, vmax=1, cmap=cc.m_CET_L20)
        ax.set_axis_off()
        fig.colorbar(image_object, label="Coherency")
        plt.show()

    plot_example_coherency()
    return


@app.cell
def _(cc, disk, np, plt):
    def plot_example_interaction():
        fig, ax = plt.subplots(layout="constrained")
        image_object = ax.imshow(np.sum(disk, axis=0).T, vmin=0, vmax=10, cmap=cc.m_CET_L20)
        ax.set_axis_off()
        fig.colorbar(image_object, label="Total Overlap")
        plt.show()

    plot_example_interaction()
    return


@app.cell
def _(
    get_average_particle_count,
    get_coherency_fraction,
    get_movement,
    np,
    os,
    pd,
):
    def get_coherency_timeseries():
        # Get sample site:
        site_csv = pd.read_csv(
            os.path.join(
                "/camp/home/eloaklo/home/shared/eloaklo/analysed_data/OEO20260313",
                "trajectories",
                "A6-Site_1.csv"
            )
        )

        # Subsample the site:
        disk_arrays = []
        coherency_arrays = []
        trajectory_arrays = []
        for subsample_index in range(7):
            # Get frame limits:
            half_frame = subsample_index / 2
            frame_start = half_frame * 144
            frame_end = (half_frame + 1) * 144
            print(frame_start, frame_end)
            frame_mask = np.logical_and(
                site_csv["frame"] >= frame_start,
                site_csv["frame"] < frame_end
            )

            # Restrict dataframe to one hour time window:
            window_csv = site_csv.loc[frame_mask]
            all_ids = np.unique(window_csv["tree_id"])
            particle_speeds, meander_ratios, frame_lengths, valid_ids = get_movement(window_csv, all_ids, min_frames=4)
            particle_count, count_std = get_average_particle_count(window_csv, valid_ids)

            # Calculate coherency fraction:
            _, _, disk_array, coherency, trajectory_array = get_coherency_fraction(window_csv, valid_ids)
            disk_arrays.append(disk_array)
            coherency_arrays.append(coherency)
            trajectory_arrays.append(trajectory_array)
        return disk_arrays, coherency_arrays, trajectory_arrays
    return (get_coherency_timeseries,)


@app.cell
def _(get_coherency_timeseries):
    disk_arrays, coherency_arrays, trajectory_arrays = get_coherency_timeseries()
    return coherency_arrays, disk_arrays, trajectory_arrays


@app.cell
def _(coherency_arrays, disk_arrays, np, plt, trajectory_arrays):
    def plot_frames(index):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
        coherency_trajectory = np.copy(coherency_arrays[index])
        coherency_trajectory[~np.bool(trajectory_arrays[index])] = 0
        axs[0].imshow(trajectory_arrays[index], vmin=0, vmax=1)
        axs[0].set_axis_off()
        axs[1].imshow(np.sum(disk_arrays[index], axis=0), vmin=0, vmax=5)
        axs[1].set_axis_off()
        axs[2].imshow(coherency_trajectory, vmin=0, vmax=1)
        axs[2].set_axis_off()
        plt.show()

    plot_frames(1)
    return


@app.cell
def _(analyse_sites):
    EXPERIMENT_FOLDERS = [
        "OEO20260313",
        "OEO20260317",
        "OEO20260410"
    ]

    trajectory_dataframes = []
    for experiment_folder in EXPERIMENT_FOLDERS:
        print(f"--- --- {experiment_folder} --- ---")
        data_directory = f"/camp/home/eloaklo/home/shared/eloaklo/analysed_data/{experiment_folder}"
        speed_dataframe = analyse_sites(data_directory)
        trajectory_dataframes.append(speed_dataframe)
    return (trajectory_dataframes,)


@app.cell
def _(pd, trajectory_dataframes):
    all_dataframe = pd.concat(trajectory_dataframes)
    return (all_dataframe,)


@app.cell
def _(all_dataframe, sns):
    sns.violinplot(data=all_dataframe, x="density", y="particle_count", hue="phenotype", inner="stick", fill=False)
    return


@app.cell
def _(all_dataframe, sns):
    sns.boxenplot(data=all_dataframe, x="density", y="speed", hue="phenotype", fill=False, gap=0.2, log_scale=False)
    return


@app.cell
def _(all_dataframe, sns):
    sns.scatterplot(data=all_dataframe, x="frame_length", y="meander_ratio", hue="phenotype", s=1)
    return


@app.cell
def _(all_dataframe, sns):
    sns.scatterplot(data=all_dataframe, x="frame_length", y="particle_interaction", hue="phenotype", s=1)
    return


@app.cell
def _(all_dataframe, sns):
    sns.scatterplot(data=all_dataframe, x="frame_length", y="particle_coherency", hue="phenotype", s=1)
    return


@app.cell
def _():
    # sns.boxenplot(data=all_dataframe, x="density", y="particle_coherency", hue="phenotype", fill=False, gap=0.2, log_scale=False)
    return


@app.cell
def _(all_dataframe, sns):
    sns.boxenplot(data=all_dataframe, x="density", y="meander_ratio", hue="phenotype", fill=False, gap=0.2, log_scale=False)
    return


@app.cell
def _(column_density_dictionary, column_phenotype_dictionary, np, pd):
    def get_geo_mean_speed(input_dataframe):
        dataframe = []
        # Estimate geometric mean per site:
        ROWS = ['A', 'B', 'C']
        COLUMNS = ['1', '2', '3', '4', '5', '6']
        for row in ROWS:
            row_mask = input_dataframe["row"] == row
            row_dataframe = input_dataframe.loc[row_mask]
            for column in COLUMNS:
                column_mask = row_dataframe["column"] == column
                column_dataframe = row_dataframe.loc[column_mask]
                for site in range(4):
                    # Get site data:
                    site_mask = column_dataframe["site"] == site
                    frame_mask = column_dataframe["frame_length"] > 400
                    full_mask = np.logical_and(site_mask, frame_mask)
                    print(np.count_nonzero(full_mask))
                    site_dataframe = column_dataframe.loc[full_mask]

                    # Get mean of log-normal distribution:
                    gm_speed = np.exp(np.mean(np.log(site_dataframe["speed"])))
                    mean_mr = np.mean(site_dataframe["meander_ratio"])
                    mean_coherency = np.mean(site_dataframe["particle_coherency"])
                    mean_interaction = np.mean(site_dataframe["particle_interaction"])

                    # Put into dictionary:
                    partial_dataframe = {}

                    # Independent variables:
                    partial_dataframe["row"] = row
                    partial_dataframe["column"] = column
                    partial_dataframe["site"] = site
                    partial_dataframe["phenotype"] = column_phenotype_dictionary[int(column)]
                    partial_dataframe["density"] = column_density_dictionary[int(column)]

                    # Dependent variables:
                    partial_dataframe["gm_speed"] = gm_speed
                    partial_dataframe["mean_mr"] = mean_mr
                    partial_dataframe["site_coherency"] = mean_coherency
                    partial_dataframe["site_interaction"] = mean_interaction
                    partial_dataframe["particle_count"] = site_dataframe.loc[:, "particle_count"].iloc[0]

                    # Append to total dataframe:
                    dataframe.append(pd.DataFrame(partial_dataframe, index=[0]))

        return pd.concat(dataframe)
    return (get_geo_mean_speed,)


@app.cell
def _(get_geo_mean_speed, pd, trajectory_dataframes):
    mean_dataframes = []
    for input_dataframe in trajectory_dataframes:
        mean_dataframe = get_geo_mean_speed(input_dataframe)
        mean_dataframes.append(mean_dataframe)

    all_mean_dataframe = pd.concat(mean_dataframes)

    # all_mean_dataframe = pd.read_csv("test_collated_dataframe.csv")
    return (all_mean_dataframe,)


@app.cell
def _(all_mean_dataframe):
    all_mean_dataframe.to_csv("test_collated_dataframe.csv")
    return


@app.cell
def _(all_mean_dataframe, sns):
    sns.swarmplot(data=all_mean_dataframe, x="density", y="particle_count", hue="phenotype", dodge=True)
    return


@app.cell
def _(all_mean_dataframe, sns):
    sns.swarmplot(data=all_mean_dataframe, x="density", y="gm_speed", hue="phenotype", dodge=True)
    return


@app.cell
def _(all_mean_dataframe, sns):
    sns.swarmplot(data=all_mean_dataframe,  x="density", y="mean_mr", hue="phenotype", dodge=True)
    return


@app.cell
def _(all_mean_dataframe, sns):
    sns.swarmplot(data=all_mean_dataframe,  x="density", y="site_coherency", hue="phenotype", dodge=True)
    return


@app.cell
def _():
    # sns.swarmplot(data=all_mean_dataframe, x="density", y="site_coherency", hue="phenotype", dodge=True)
    return


@app.cell
def _():
    # sns.swarmplot(data=all_mean_dataframe, x="density", y="site_interaction", hue="phenotype", dodge=True)
    return


@app.cell
def _(all_mean_dataframe, sns):
    sns.scatterplot(data=all_mean_dataframe, x="particle_count", y="gm_speed", hue="phenotype", alpha=0.75)
    return


@app.cell
def _(all_mean_dataframe, sns):
    sns.lmplot(data=all_mean_dataframe,  x="particle_count", y="gm_speed", hue="phenotype", order=2)
    return


@app.cell
def _(all_mean_dataframe, sns):
    sns.scatterplot(data=all_mean_dataframe, x="particle_count", y="mean_mr", hue="phenotype", alpha=0.75)
    return


@app.cell
def _(all_mean_dataframe, sns):
    sns.scatterplot(data=all_mean_dataframe, x="particle_count", y="site_coherency", hue="phenotype", alpha=0.75)
    return


@app.cell
def _(all_mean_dataframe, sns):
    sns.scatterplot(data=all_mean_dataframe, x="mean_mr", y="site_coherency", hue="phenotype", alpha=0.75)
    return


@app.cell
def _(all_mean_dataframe, sns):
    sns.lmplot(data=all_mean_dataframe, x="particle_count", y="gm_speed", hue="phenotype", order=2)
    return


@app.cell
def _(all_mean_dataframe, sns):
    sns.lmplot(data=all_mean_dataframe,  x="particle_count", y="mean_mr", hue="phenotype", order=2)
    return


@app.cell
def _(all_mean_dataframe, sns):
    sns.lmplot(data=all_mean_dataframe,  x="particle_count", y="site_coherency", hue="phenotype", order=1)
    return


@app.cell
def _(all_mean_dataframe, sns):
    sns.lmplot(data=all_mean_dataframe,  x="particle_count", y="site_interaction", hue="phenotype", order=2)
    return


@app.cell
def _(all_ids, plt, site_csv):
    def plot_all_trajectories():
        fig, ax = plt.subplots()
        for id in all_ids:
            particle_mask = site_csv["tree_id"] == id
            ax.plot(
                site_csv.loc[particle_mask, "x"],
                site_csv.loc[particle_mask, "y"],
                lw=1
            )

        # Format plot:
        ax.set_aspect("equal")
        ax.set_xlim(0, 1024)
        ax.set_ylim(0, 1024)
        ax.set_xticks([])
        ax.set_yticks([])

        # Show plot:
        plt.show()

    plot_all_trajectories()
    return


@app.cell
def _(plt, site_csv, valid_ids):
    def plot_valid_trajectories():
        fig, ax = plt.subplots()
        for id in valid_ids:
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
        plt.show()

    plot_valid_trajectories()
    return


@app.cell
def _(plt):
    def trajectory_filter_plot():
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    return


if __name__ == "__main__":
    app.run()
