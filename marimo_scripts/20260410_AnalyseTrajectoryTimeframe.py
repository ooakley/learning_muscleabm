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
    return np, os, pd, plt, skimage, sns


@app.cell
def _():
    FRAME_DURATION = 2.5
    TRAVEL_LIMIT = 75
    return FRAME_DURATION, TRAVEL_LIMIT


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
def _(FRAME_DURATION, np, os, pd, skimage):
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

        coherency_fraction = np.sum(trajectory_array * coherency) / np.sum(trajectory_array)
        trajectory_sum = np.sum(trajectory_array)
        return coherency_fraction, trajectory_sum


    def get_coherency_timeseries(site_csv, timeframe):
        # Calculate indices:
        subsamples = 576 / timeframe 
        subsamples = int((subsamples * 2) - 1)

        # Subsample the site:
        coherency_fractions = []
        trajectory_sums = []
        for subsample_index in range(subsamples):
            # Get frame limits:
            half_frame = subsample_index / 2
            frame_start = half_frame * timeframe
            frame_end = (half_frame + 1) * timeframe
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
            coherency_fraction, trajectory_sum = get_coherency_fraction(window_csv, valid_ids)
            coherency_fractions.append(coherency_fraction)
            trajectory_sums.append(trajectory_sum)

        return coherency_fractions, trajectory_sums


    def ranged_timeframe_analysis():
        # Get sample site:
        site_csv = pd.read_csv(
            os.path.join(
                "/camp/home/eloaklo/home/shared/eloaklo/analysed_data/OEO20260313",
                "trajectories",
                f"A1-Site_1.csv"
            )
        )

        timeframes = np.arange(3, 9) * 24
        timeframes = timeframes.astype(int)

        ranged_cf = []
        ranged_sum_trajectory = []
        for timeframe in timeframes:
            coherency_fractions, trajectory_sums = get_coherency_timeseries(site_csv, timeframe)
            ranged_cf.append(coherency_fractions)
            ranged_sum_trajectory.append(trajectory_sums)

        return ranged_cf, ranged_sum_trajectory
    return (get_average_particle_count,)


@app.cell
def _(
    FRAME_DURATION,
    TRAVEL_LIMIT,
    column_density_dictionary,
    column_phenotype_dictionary,
    get_average_particle_count,
    np,
    os,
    pd,
    skimage,
):
    def get_travel_frame_cutoffs(site_csv, all_ids, min_frames=25):
        particle_speeds = []
        meander_ratios = []
        frame_lengths = []
        valid_ids = []
        total_movement = np.zeros(576)
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

            # If valid, add particle movement to relevant frame index:
            total_movement[frame_array[:-1][no_frame_break_mask]] += np.sqrt(x_filter_diff**2 + y_filter_diff**2)

            particle_speeds.append(average_speed)
            meander_ratios.append(meander_ratio)
            frame_lengths.append(frame_length)
            valid_ids.append(particle_index)

        return np.array(particle_speeds), np.array(meander_ratios), np.array(frame_lengths), valid_ids, total_movement


    def get_windowed_coherency(site_csv, valid_ids, window_limits):
        window_coherencies = []
        for window_index in range(len(window_limits) - 1):
            windowed_csv = site_csv.copy(deep=True)
            lower_window_mask = windowed_csv["frame"] >= window_limits[window_index]
            upper_window_mask = windowed_csv["frame"] <  window_limits[window_index + 1]
            full_window_mask = np.logical_and(lower_window_mask, upper_window_mask)
            windowed_csv = windowed_csv.loc[full_window_mask]

            # Estimate from trajectory dataframe as test:
            line_array = []
            disk_array = []
            for valid_id in valid_ids:
                # Set up empty array (we downsample by 4x for speed & robustness):
                particle_line_array = np.zeros((256, 256))

                # Get position data for this particle:
                particle_mask = windowed_csv["tree_id"] == valid_id
                if np.count_nonzero(particle_mask) == 0:
                    continue
                particle_csv = windowed_csv[particle_mask].sort_values("frame")
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
                trajectory_array, sigma=16,
                mode='constant', cval=0,
                order='rc'
            )
    
            # Get local coherencies:
            eigenvalues = skimage.feature.structure_tensor_eigenvalues(structure_tensor)
            coherency_numerator = eigenvalues[0, :, :] - eigenvalues[1, :, :]
            coherency_denominator = eigenvalues[0, :, :] + eigenvalues[1, :, :]
            coherency = coherency_numerator / coherency_denominator
            coherency_fraction = np.sum(coherency[np.bool(trajectory_array)]) / np.sum(trajectory_array)
            window_coherencies.append(coherency_fraction)

        return window_coherencies, trajectory_array

    def get_speed_compensated_coherency(data_directory):
        ROWS = ['A', 'B', 'C']
        COLUMNS = ['1', '2', '3', '4', '5', '6']
        dataframe = []
        for row in ROWS:
            for column in COLUMNS:
                for site in range(4):
                    print(f"{row}{column}-Site_{site}")
                    # Get filepath:
                    filepath = os.path.join(data_directory, "trajectories", f"{row}{column}-Site_{site}.csv")
                    site_csv = pd.read_csv(filepath)

                    # Get full csv:
                    all_ids = np.unique(site_csv["tree_id"])
                    particle_speeds, meander_ratios, frame_lengths, valid_ids, total_movement = \
                        get_travel_frame_cutoffs(site_csv, all_ids, min_frames=4)
                    particle_count, _ = get_average_particle_count(site_csv, valid_ids)
                    total_movement /= particle_count
            
                    # Get cut-offs for travel control (approx. thirds):
                    window_limits = [0]
                    for i in range(3):
                        upper_limit = TRAVEL_LIMIT * (i + 1)
                        window_limits.append(np.argmax(np.cumsum(total_movement) > upper_limit))
                    print(window_limits)
                    if window_limits[-1] == 0:
                        window_limits = window_limits[:3]

                    # Get coherency:
                    windowed_coherencies, trajectory_array = get_windowed_coherency(site_csv, valid_ids, window_limits)

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
                    partial_dataframe["site_coherency"] = [np.mean(windowed_coherencies)] * len(particle_speeds)
                    partial_dataframe["particle_count"] = [particle_count] * len(particle_speeds)

                    # Append to total dataframe:
                    dataframe.append(pd.DataFrame.from_dict(partial_dataframe))

        return pd.concat(dataframe)
    return (get_speed_compensated_coherency,)


@app.cell
def _(get_speed_compensated_coherency):
    data_directory = f"/camp/home/eloaklo/home/shared/eloaklo/analysed_data/OEO20260313"
    windowed_coherency_df = get_speed_compensated_coherency(data_directory)
    return (windowed_coherency_df,)


@app.cell
def _(windowed_coherency_df):
    windowed_coherency_df
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
                    site_dataframe = column_dataframe.loc[full_mask]

                    # Get mean of log-normal distribution:
                    gm_speed = np.exp(np.mean(np.log(site_dataframe["speed"])))
                    mean_mr = np.mean(site_dataframe["meander_ratio"])

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
                    partial_dataframe["site_coherency"] = site_dataframe.loc[:, "site_coherency"].iloc[0]
                    partial_dataframe["particle_count"] = site_dataframe.loc[:, "particle_count"].iloc[0]

                    # Append to total dataframe:
                    dataframe.append(pd.DataFrame(partial_dataframe, index=[0]))

        return pd.concat(dataframe)
    return (get_geo_mean_speed,)


@app.cell
def _(get_geo_mean_speed, windowed_coherency_df):
    site_dataframe = get_geo_mean_speed(windowed_coherency_df)
    return (site_dataframe,)


@app.cell
def _(site_dataframe, sns):
    sns.swarmplot(data=site_dataframe, x="density", y="particle_count", hue="phenotype", dodge=True)
    return


@app.cell
def _(site_dataframe, sns):
    sns.swarmplot(data=site_dataframe, x="density", y="site_coherency", hue="phenotype", dodge=True)
    return


@app.cell
def _(site_dataframe, sns):
    sns.lmplot(data=site_dataframe,  x="particle_count", y="site_coherency", hue="phenotype", order=2)
    return


@app.cell
def _(site_dataframe, sns):
    sns.lmplot(data=site_dataframe,  x="particle_count", y="mean_mr", hue="phenotype", order=2)
    return


@app.cell
def _(site_dataframe, sns):
    sns.scatterplot(data=site_dataframe, x="gm_speed", y="site_coherency", hue="phenotype")
    return


@app.cell
def _(site_dataframe):
    site_dataframe
    return


@app.cell
def _(windowed_coherencies):
    windowed_coherencies
    return


@app.cell
def _(plt, trajectory_array):
    plt.imshow(trajectory_array)
    return


@app.cell
def _(np, plt, total_movement):
    plt.plot(np.cumsum(total_movement))
    return


@app.cell
def _():
    return


@app.cell
def _(os, pd):
    site_csv = pd.read_csv(
        os.path.join(
            "/camp/home/eloaklo/home/shared/eloaklo/analysed_data/OEO20260313",
            "trajectories",
            f"A1-Site_1.csv"
        )
    )
    return (site_csv,)


@app.cell
def _(site_csv):
    site_csv
    return


@app.cell
def _(ctl_cf, np):
    mean_range = []
    for cf_list in ctl_cf:
        mean_range.append(np.mean(cf_list))
    return (mean_range,)


@app.cell
def _(ctl_sum_trajectory, np):
    mean_sum = []
    for sum_list in ctl_sum_trajectory:
        mean_sum.append(np.mean(sum_list))
    return (mean_sum,)


@app.cell
def _(mean_range, mean_sum, plt):
    plt.scatter(mean_sum, mean_range)
    return


@app.cell
def _(mean_range, plt):
    plt.plot(mean_range)
    return


@app.cell
def _(mean_range, plt):
    plt.plot(mean_range)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
