import os
import argparse
import skimage

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors


EXPERIMENT_FOLDERS = [
    "OEO20260313",
    "OEO20260317",
    "OEO20260410",
    "OEO20260417",
    "OEO20260418"
]
SITE_EXCLUSION = {
    "OEO20260313": [
        "B1-Site_3",
        "B3-Site_3",
        "B5-Site_2",
    ],
    "OEO20260317": [],
    "OEO20260410": [
        "C3-Site_0"
    ],
    "OEO20260417": [
        "B1-Site_0",
        "B1-Site_1",
        "B1-Site_2",
        "B1-Site_3",
        "B3-Site_0",
        "B3-Site_1",
        "B3-Site_2",
        "B3-Site_3",
    ],
    "OEO20260418": [
        "B2-Site_0",
        "B6-Site_0",
        "B6-Site_1",
        "B6-Site_2",
        "B6-Site_3",
    ]
}

PIXEL_SIZE = 0.3469 * 2  # Pixel size in µm
FRAME_DURATION = 2.5  # Frame duration in minutes
TRAVEL_LIMIT = 75  # Cutoff for travel per particle for raster analysis
MIN_FRAMES = 48  # Cutoff in frames for incorporating trajectory into analysis (2 hours)

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


def get_movement_descriptors(site_csv, all_ids, min_frames):
    valid_ids = []
    data_dictionaries = []
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

        # -- Mask multiframe position differences so we get adequate speed estimation:
        x_diff = np.diff(x_pos)
        y_diff = np.diff(y_pos)
        x_filter_diff = x_diff[no_frame_break_mask]
        y_filter_diff = y_diff[no_frame_break_mask]
        frame_length = len(x_filter_diff)
        average_speed = np.mean(np.sqrt(x_filter_diff**2 + y_filter_diff**2))
        average_speed *= PIXEL_SIZE  # To get final speed estimate in µm/frame
        average_speed /= FRAME_DURATION  # To get final speed in µm/min
    
        # Get meander ratio:
        path_length = np.sum(np.sqrt(x_diff**2 + y_diff**2))
        final_displacement = np.sqrt(((x_pos[0] - x_pos[-1]) ** 2) + ((y_pos[0] - y_pos[-1]) ** 2))

        # Render trajectory invalid if its path length is 0:
        if path_length == 0:
            continue
        meander_ratio = final_displacement / path_length

        # Get outreach ratio:
        centred_x = x_pos - x_pos[0]
        centred_y = y_pos - y_pos[0]
        maximal_excursion = np.max(np.sqrt(centred_x**2 + centred_y**2))
        outreach_ratio = maximal_excursion / path_length

        # Record movement metrics:
        data_dictionaries.append({
            "average_speed": average_speed,
            "meander_ratio": meander_ratio,
            "outreach_ratio": outreach_ratio,
            "frame_length": frame_length
        })

        # Add particle movement to total movement per frame:
        total_movement[frame_array[:-1][no_frame_break_mask]] += np.sqrt(x_filter_diff**2 + y_filter_diff**2)

        # Add index to valid IDs if still valid:
        valid_ids.append(particle_index)

    return data_dictionaries, total_movement, valid_ids


def get_average_particle_count(site_csv, valid_ids):
    counts = []
    for frame_index in range(np.max(site_csv["frame"])):
        id_csv = site_csv.loc[site_csv["frame"] == frame_index, "tree_id"]
        counts.append(np.count_nonzero(np.isin(id_csv, valid_ids)))
    return np.mean(counts), np.std(counts)


def get_average_nearest_neighbour_distance(site_csv, valid_ids):
    annd_list = []
    anni_list = []
    for frame_index in range(np.max(site_csv["frame"])):
        # Get valid positions:
        frame_mask = site_csv["frame"] == frame_index
        frame_csv = site_csv.loc[frame_mask, :]
        validity_mask = np.isin(frame_csv["tree_id"], valid_ids)
        valid_csv = frame_csv.loc[validity_mask, :]
        x_positions = valid_csv.loc[:, "x"]
        y_positions = valid_csv.loc[:, "y"]
        positions = np.stack([x_positions, y_positions], axis=1)

        # Calculate average nearest neigbour distance:
        neighbours_analysis = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(positions)
        distances, _ = neighbours_analysis.kneighbors(positions)
        distances = distances[:, 1]
        annd = np.mean(distances) * PIXEL_SIZE
        annd_list.append(annd)

        # Calculate average nearest neighbour index:
        site_area = (PIXEL_SIZE * 1024) ** 2
        frame_particle_count = np.count_nonzero(validity_mask)
        expected_distance = 0.5 * np.sqrt(site_area / frame_particle_count)
        anni_list.append(annd / expected_distance)

    return np.mean(annd_list), np.mean(anni_list)


def get_windowed_coherency(site_csv, valid_ids, window_limits):
    window_particle_coherency = []
    window_particle_interaction = []
    window_coherencies = []
    window_interactions = []
    for window_index in range(len(window_limits) - 1):
        windowed_csv = site_csv.copy(deep=True)
        lower_window_mask = windowed_csv["frame"] >= window_limits[window_index]
        upper_window_mask = windowed_csv["frame"] < window_limits[window_index + 1]
        full_window_mask = np.logical_and(lower_window_mask, upper_window_mask)
        windowed_csv = windowed_csv.loc[full_window_mask]

        # Estimate from trajectory dataframe as test:
        line_array = []
        disk_array = []
        for valid_id in valid_ids:
            # Set up empty array (we downsample by 2x for speed & robustness):
            particle_line_array = np.zeros((512, 512))

            # Get position data for this particle:
            particle_mask = windowed_csv["tree_id"] == valid_id
            if np.count_nonzero(particle_mask) == 0:
                continue
            particle_csv = windowed_csv[particle_mask].sort_values("frame")
            # !!! Downsample:
            xy_data = np.copy(np.array(particle_csv.loc[:, ["x", "y"]])) / 2

            # Generate trajectory frame:
            for frame_index in range(len(xy_data) - 1):
                # Get indices of line:
                xy_t = xy_data[frame_index, :].astype(int)
                xy_t1 = xy_data[frame_index + 1, :].astype(int)

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
        coherency_fraction = np.sum(coherency[np.bool(trajectory_array)]) / np.sum(trajectory_array)
        window_coherencies.append(coherency_fraction)

        # Get interaction over window:
        interaction_fraction = np.sum(np.sum(disk_array, axis=0) > 1) / np.sum(np.sum(disk_array, axis=0) > 0)
        window_interactions.append(interaction_fraction)

        # Get collectivity estimates per-particle:
        # -- First filter coherency array for nan terms:
        filtered_coherency = np.copy(coherency)
        filtered_coherency[np.isnan(coherency)] = 0
        # -- Iterate through particle frames:
        particle_coherencies = []
        particle_interactions = []
        path_sum = np.sum(disk_array, axis=0)
        for particle_index in range(disk_array.shape[0]):
            indexed_path = disk_array[particle_index, :, :]
            comparator_paths = path_sum - indexed_path
            interaction_value = np.sum(comparator_paths * indexed_path) / np.sum(indexed_path)
            coherency_value = np.sum(filtered_coherency * indexed_path) / np.sum(indexed_path)
            particle_coherencies.append(coherency_value)
            particle_interactions.append(interaction_value)

        # Average over particles:
        window_particle_coherency.append(np.mean(particle_coherencies))
        window_particle_interaction.append(np.mean(particle_interactions))

    mean_particle_coherency = np.mean(window_particle_coherency)
    mean_particle_interaction = np.mean(window_particle_interaction)
    return np.mean(window_coherencies), np.mean(window_interactions), mean_particle_coherency, mean_particle_interaction


def analyse_site(site_csv, column, experiment):
    # Get movement data, and extract valid trajectories:
    all_ids = np.unique(site_csv["tree_id"])
    data_dictionaries, total_movement, valid_ids = get_movement_descriptors(site_csv, all_ids, MIN_FRAMES)
    count_mean, _ = get_average_particle_count(site_csv, valid_ids)
    annd_mean, anni_mean = get_average_nearest_neighbour_distance(site_csv, valid_ids)

    # Get coherency and interaction values:
    total_movement /= count_mean
    # -- Get cut-offs for travel control (approx. thirds):
    window_limits = [0]
    for i in range(3):
        upper_limit = TRAVEL_LIMIT * (i + 1)
        window_limits.append(np.argmax(np.cumsum(total_movement) > upper_limit))
    # -- If final travel window exceeds data length, remove it:
    while window_limits[-1] == 0:
        window_limits = window_limits[:-1]

    # -- Calculate coherency fraction over given windows:
    coherency_fraction, interaction_fraction, mean_particle_coherency, mean_particle_interaction \
        = get_windowed_coherency(site_csv, valid_ids, window_limits)

    # Construct particle dataframe:
    particle_dataframe = pd.DataFrame(data_dictionaries)
    particle_dataframe["phenotype"] = [COL_PHENOTYPE_DICT[column]] * len(particle_dataframe)
    particle_dataframe["density"] = [COL_DENSITY_DICT[column]] * len(particle_dataframe)
    particle_dataframe["experiment"] = [experiment] * len(particle_dataframe)

    # Construct site data dictionary:
    frame_length = np.array(particle_dataframe["frame_length"])
    # -- Filter particles based on a minimum frame length:
    frame_mask = frame_length > 250
    particle_speed = np.array(particle_dataframe["average_speed"])[frame_mask]
    particle_mr = np.array(particle_dataframe["meander_ratio"])[frame_mask]
    particle_or = np.array(particle_dataframe["outreach_ratio"])[frame_mask]
    site_data = {
        "particle_count": count_mean,
        "annd": annd_mean,
        "anni": anni_mean,
        "coherency_fraction": coherency_fraction,
        "interaction_fraction": interaction_fraction,
        "mean_particle_coherency": mean_particle_coherency,
        "mean_particle_interaction": mean_particle_interaction,
        "mean_speed": np.exp(np.mean(np.log(particle_speed))),
        "mean_mr": np.mean(particle_mr),
        "mean_or": np.mean(particle_or),
        "phenotype": COL_PHENOTYPE_DICT[column],
        "density": COL_DENSITY_DICT[column],
        "experiment": experiment
    }

    return site_data, particle_dataframe


def analyse_experiment(analysed_data_dirpath, experiment_folder):
    # Set up necessary paths:
    print(f"--- --- {experiment_folder} --- ---")
    exclusion_list = SITE_EXCLUSION[experiment_folder]
    data_directory = os.path.join(analysed_data_dirpath, experiment_folder)
    ROWS = ['A', 'B', 'C']
    COLUMNS = [1, 2, 3, 4, 5, 6]

    site_data_dictionaries = []
    particle_data_dataframes = []
    for row in ROWS:
        for column in COLUMNS:
            for site in range(4):
                # Get filepath:
                site_name = f"{row}{column}-Site_{site}"
                if site_name in exclusion_list:
                    print(f"Excluding {site_name}...")
                    continue
                filepath = os.path.join(data_directory, "trajectories", f"{site_name}.csv")
                site_csv = pd.read_csv(filepath)

                # Analyse site:
                print(f"-> Analysing {row}{column}-Site_{site}", flush=True)
                site_data, particle_dataframe = analyse_site(site_csv, column, experiment_folder)

                site_data_dictionaries.append(site_data)
                particle_data_dataframes.append(particle_dataframe)

    return pd.DataFrame(site_data_dictionaries), pd.concat(particle_data_dataframes)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysed_data_dirpath", required=True,
        help="Folder containing one folder of tracked trajectories per wet lab experiment."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    site_dataframes = []
    particle_dataframes = []
    for experiment_folder in EXPERIMENT_FOLDERS:
        site_dataframe, particle_dataframe = analyse_experiment(args.analysed_data_dirpath, experiment_folder)
        site_dataframes.append(site_dataframe)
        particle_dataframes.append(particle_dataframe)

    full_site_dataframe = pd.concat(site_dataframes)
    full_particle_dataframe = pd.concat(particle_dataframes)

    full_site_dataframe.to_csv("wetlab_data/site_dataframe.csv")
    full_particle_dataframe.to_csv("wetlab_data/particle_dataframe.csv")


if __name__ == "__main__":
    main()
