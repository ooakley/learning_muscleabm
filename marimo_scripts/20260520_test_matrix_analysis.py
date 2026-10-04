import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    """Perform only basic order parameter calculations."""
    import argparse
    import os
    import json
    import scipy

    import numpy as np
    import colorcet as cc

    import matplotlib.pyplot as plt

    NEIGHBOURHOOD_SIZES = [3, 33, 65]

    MESH_NUMBER = 128

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
    return (
        cc,
        format_fibre_list,
        generate_order_parameter_scale_curve,
        np,
        os,
        plt,
        read_matrix_into_list,
        scipy,
    )


@app.cell
def _(format_fibre_list, os, read_matrix_into_list):
    run_folderpath = "configs/model_parameter_json/matrix_test"
    filename = "matrix_seed000.txt"
    filepath = os.path.join(run_folderpath, filename)
    fibre_list = read_matrix_into_list(filepath)
    average_heading, fibre_count, angular_variance = format_fibre_list(fibre_list)
    return angular_variance, average_heading, fibre_count, run_folderpath


@app.cell
def _(fibre_count, np):
    np.count_nonzero(fibre_count == 0)
    return


@app.cell
def _(average_heading):
    average_heading.shape
    return


@app.cell
def _(average_heading, np):
    x_mean = np.nanmean(np.cos(2 * average_heading))
    y_mean = np.nanmean(np.sin(2 * average_heading))
    global_order_parameter = np.sqrt(x_mean**2 + y_mean**2)
    global_matrix_direction = np.atan2(y_mean, x_mean) / 2
    return global_matrix_direction, global_order_parameter


@app.cell
def _(global_order_parameter):
    global_order_parameter
    return


@app.cell
def _(global_matrix_direction):
    global_matrix_direction / 2
    return


@app.cell
def _(angular_variance):
    angular_variance.max()
    return


@app.cell
def _(angular_variance):
    angular_variance.min()
    return


@app.cell
def _(angular_variance, cc, plt):
    plt.imshow(angular_variance, vmin=0, vmax=1, cmap=cc.m_CET_L1)
    return


@app.cell
def _(average_heading, cc, np, os, plt, run_folderpath):
    def plot_matrix(average_heading, filepath):
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        cmap = cc.m_CET_CBC1
        cmap.set_bad("#FFC0CB", 1.)
        ax.imshow(average_heading, vmin=-np.pi/2, vmax=np.pi/2, cmap=cmap, origin="lower")
        ax.set_axis_off()
        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)
        plt.savefig(filepath, pad_inches=0.0, dpi=200)
        plt.show()

    plot_matrix(average_heading, os.path.join(run_folderpath, "matrix_plot.png"))
    return


@app.cell
def _(cc, fibre_count, np, plt):
    def plot_matrix_density(fibre_count, filepath):
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        cmap = cc.m_CET_CBC1
        # cmap.set_bad("#FFC0CB", 1.)  # Plot NaNs as color outside of colorbar.
        ax.imshow(fibre_count, vmin=0, cmap=cc.m_CET_L20, origin="lower")
        ax.set_axis_off()
        ax.text(1, 1, f"{int(np.max(fibre_count))}", c="w")
        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)
        # plt.savefig(filepath, pad_inches=0.0, dpi=200)
        plt.show() 

    plot_matrix_density(fibre_count, None)
    return


@app.cell
def _(average_heading, generate_order_parameter_scale_curve):
    scale_curve = generate_order_parameter_scale_curve(average_heading * 2)
    return (scale_curve,)


@app.cell
def _(scale_curve):
    scale_curve
    return


@app.cell
def _(fibre_count, np, scipy):
    spatial_average_fc = scipy.ndimage.gaussian_filter(fibre_count, 1, mode='wrap')
    density_distribution = spatial_average_fc.flatten()
    interdecile_range = np.quantile(density_distribution, 0.9) - np.quantile(density_distribution, 0.1)
    interdecile_range
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
