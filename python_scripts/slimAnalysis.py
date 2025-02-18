"""
Script to run analysis of model output data.

Outputs trajectory images, matrix images, and a small set of derived
statistics.
"""
import argparse
import os

import pandas as pd
import numpy as np

SIMULATION_OUTPUTS_FOLDER = "./fileOutputs/"
CSV_COLUMN_NAMES = [
    "frame", "particle", "x", "y", "orientation",
    "polarity_extent", "percept_direction", "percept_intensity",
    "actin_flow", "movement_direction", "turning_angle", "sampled_angle", "contact_inhibition"
]

TIMESTEP_WIDTH = 576
GRID_SIZE = 32


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a folder with a given integer name.')
    parser.add_argument('--folder_id', type=int, help='Integer corresponding to the folder name')
    args = parser.parse_args()
    return args


# Feature matrix extraction:
def get_particle_positions_list(site_trajectory_df):
    particle_list = list(set(list(site_trajectory_df["particle"])))
    particle_positions_list = []
    for particle in particle_list:
        particle_mask = site_trajectory_df["particle"] == particle
        x_pos = np.array(site_trajectory_df[particle_mask]["x"])[::4]
        y_pos = np.array(site_trajectory_df[particle_mask]["y"])[::4]
        particle_positions_list.append((x_pos, y_pos))
    return particle_positions_list


def get_features_from_positions(particle_positions_list):
    # Extracted features:
    path_lengths = []
    maximal_excursions = []
    net_displacements = []
    outreach_ratios = []
    meandering_ratios = []
    maximum_densities = []
    angle_entropies = []

    for particle_positions in particle_positions_list:
        x_pos, y_pos = particle_positions
        dx = np.diff(x_pos)
        dy = np.diff(y_pos)

        # Accounting for rollover positions:
        dx[dx > 1024] = dx[dx > 1024] - 2048
        dx[dx < -1024] = dx[dx < -1024] + 2048

        dy[dy > 1024] = dy[dy > 1024] - 2048
        dy[dy < -1024] = dy[dy < -1024] + 2048

        # Calculating path statistics:
        displacements = np.sqrt(dx**2 + dy**2)

        x_excursion = np.cumsum(dx)
        y_excursion = np.cumsum(dy)
        total_excursion = np.sqrt(x_excursion**2 + y_excursion**2)
        max_excursion = np.max(total_excursion)

        net_dx = x_excursion[-1]
        net_dy = y_excursion[-1]
        net_displacement = np.sqrt(net_dx**2 + net_dy**2)
        path_length = np.sum(displacements)

        path_lengths.append(path_length)
        maximal_excursions.append(max_excursion)
        net_displacements.append(net_displacement)

        outreach_ratios.append(max_excursion / path_length)
        meandering_ratios.append(net_displacement / path_length)

        # Calculating directional statistics:
        directions = np.arctan2(dy, dx)
        bins = np.linspace(-np.pi, np.pi, 12)
        hist_densities, _ = np.histogram(directions, bins=bins, density=True, weights=displacements)
        maximum_density = np.max(hist_densities)
        maximum_densities.append(maximum_density)

        # Calculating entropy:
        hist_densities = hist_densities + 1e-4
        hist_densities = hist_densities / np.sum(hist_densities)
        entropy = np.sum(-hist_densities * np.log(hist_densities))
        angle_entropies.append(entropy)

    feature_list = [
        np.array(path_lengths),
        np.array(maximal_excursions),
        np.array(net_displacements),
        np.array(outreach_ratios),
        np.array(meandering_ratios),
        np.array(maximum_densities),
        np.array(angle_entropies)
    ]

    feature_matrix = np.stack(feature_list, axis=1)

    return feature_matrix


def main():
    # Parsing command line arguments:
    args = parse_arguments()
    print(f"Processing folder with name: {args.folder_id}")

    # Getting simulation parameters:
    gridseach_dataframe = pd.read_csv("fileOutputs/gridsearch.txt", delimiter="\t")
    gridsearch_row = gridseach_dataframe[gridseach_dataframe["array_id"] == args.folder_id]
    GRID_SIZE = int(gridsearch_row["gridSize"].iloc[0])
    AREA_SIZE = int(gridsearch_row["worldSize"].iloc[0])
    SUPERITERATION_NUM = int(gridsearch_row["superIterationCount"].iloc[0])
    TIMESTEPS = 576

    # Finding specified simulation output files:
    subdirectory_path = os.path.join(SIMULATION_OUTPUTS_FOLDER, str(args.folder_id))

    trajectory_list = []
    for seed in range(SUPERITERATION_NUM):
        print(f"Reading subiteration {seed}...")
        # Reading trajectory information:
        csv_filename = f"positions_seed{seed:03d}.csv"
        csv_filepath = os.path.join(subdirectory_path, csv_filename)
        trajectory_dataframe = pd.read_csv(
            csv_filepath, index_col=None, header=None, names=CSV_COLUMN_NAMES
        )
        trajectory_list.append(trajectory_dataframe)

    extracted_feature_matrix = []
    for trajectory_dataframe in trajectory_list:
        particle_positions_list = get_particle_positions_list(trajectory_dataframe)
        feature_matrix = get_features_from_positions(particle_positions_list)
        extracted_feature_matrix.append(feature_matrix)
    extracted_feature_matrix = np.concatenate(extracted_feature_matrix, axis=0)
    np.save(os.path.join(subdirectory_path, "feature_matrix.npy"), extracted_feature_matrix)


if __name__ == "__main__":
    main()
