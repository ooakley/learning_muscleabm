"""Perform only basic order parameter calculations."""
import argparse
import os
import json
import scipy

import numpy as np
import colorcet as cc

import matplotlib.pyplot as plt

NEIGHBOURHOOD_SIZES = [3, 33, 65]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a folder with a given integer name.')
    parser.add_argument('--run_folderpath', type=str)
    parser.add_argument('--folder_id', type=int)
    args = parser.parse_args()
    return args


def read_matrix_into_list(filepath):
    fibre_list = []
    with open(filepath, "r") as f:
        for line in f:
            heading_string = str(line.rstrip())
            headings = heading_string.split(",")[:-1]
            fibre_list.append(np.asarray(headings, dtype=float))
    return fibre_list


def format_fibre_list(fibre_list):
    # Instantiate empty matrix:
    average_heading = np.empty(MESH_NUMBER**2)
    fibre_count = np.empty(MESH_NUMBER**2)
    angular_variance = np.empty(MESH_NUMBER**2)

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
    average_heading = np.reshape(average_heading, (MESH_NUMBER, MESH_NUMBER))
    fibre_count = np.reshape(fibre_count, (MESH_NUMBER, MESH_NUMBER))
    angular_variance = np.reshape(angular_variance, (MESH_NUMBER, MESH_NUMBER))
    return average_heading, fibre_count, angular_variance


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


def plot_matrix_heading(average_heading, filepath):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    cmap = cc.m_CET_CBC1
    cmap.set_bad("#FFC0CB", 1.)  # Plot NaNs as color outside of colorbar.
    ax.imshow(average_heading, vmin=-np.pi/2, vmax=np.pi/2, cmap=cmap, origin="lower")
    ax.set_axis_off()
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)
    plt.savefig(filepath, pad_inches=0.0, dpi=200)


def plot_matrix_density(fibre_count, filepath):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.imshow(fibre_count, vmin=0, cmap=cc.m_CET_L20, origin="lower")
    ax.text(1, 1, f"{int(np.max(fibre_count))}", c="w")
    ax.set_axis_off()
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)
    plt.savefig(filepath, pad_inches=0.0, dpi=200)


def plot_matrix_variance(angular_variance, filepath):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.imshow(angular_variance, vmin=0, vmax=1, cmap=cc.m_CET_L1, origin="lower")
    ax.set_axis_off()
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)
    plt.savefig(filepath, pad_inches=0.0, dpi=200)


def main():
    """Run basic script logic."""
    # Parse arguments:
    args = parse_arguments()
    run_folderpath = args.run_folderpath
    folder_id = args.folder_id

    # Get arguments to simulation:
    json_filepath = os.path.join(run_folderpath, f"{folder_id}_arguments.json")
    with open(json_filepath) as json_file:
        simulation_arguments = json.load(json_file)

    # Define variables needed for matrix reshaping later on:
    global MESH_NUMBER  # Don't judge me. It's honestly cleaner this way
    MESH_NUMBER = simulation_arguments["gridSize"]
    superiteration_number = simulation_arguments["superIterationCount"]

    # Loop through subiterations:
    order_parameters = []
    density_idr = []

    for seed in range(superiteration_number):
        # Read matrix into numpy:
        print(f"Reading subiteration {seed} for site analysis...")
        filename = f"matrix_seed{seed:03d}.txt"
        filepath = os.path.join(run_folderpath, filename)
        fibre_list = read_matrix_into_list(filepath)
        average_heading, fibre_count, angular_variance = format_fibre_list(fibre_list)

        # Get order parameter across neighbourhood sizes:
        order_parameters.append(generate_order_parameter_scale_curve(average_heading))

        # Get spatially averaged fiber count:
        spatial_average_fc = scipy.ndimage.gaussian_filter(fibre_count, 1, mode='wrap')
        density_distribution = spatial_average_fc.flatten()
        interdecile_range = np.quantile(density_distribution, 0.9) - np.quantile(density_distribution, 0.1)
        density_idr.append(interdecile_range)

        # Plot example matrix:
        if seed == 0:
            plot_matrix_heading(average_heading, os.path.join(run_folderpath, "matrix_heading.png"))
            plot_matrix_density(fibre_count, os.path.join(run_folderpath, "matrix_density.png"))
            plot_matrix_variance(angular_variance, os.path.join(run_folderpath, "angular_variance.png"))

    # Save to .npy files as (SUPERITERATIONS) arrays:
    order_parameters = np.stack(order_parameters, axis=0)
    np.save(os.path.join(run_folderpath, "matrix_order_parameters.npy"), order_parameters)

    density_idr = np.array(density_idr)
    np.save(os.path.join(run_folderpath, "density_idr.npy"), density_idr)


if __name__ == "__main__":
    main()
