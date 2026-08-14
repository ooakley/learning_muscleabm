import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import math

    import numpy as np

    import matplotlib.pyplot as plt
    import imageio.v3 as iio
    return iio, math, np, os, plt


@app.cell
def _(np, os):
    experiment_dirpath = "model_experiments/2026-06-02-matrix_shape"

    parameter_matrix = np.load(os.path.join(experiment_dirpath, "sample_matrix.npy"))

    matrix_ops = np.load(os.path.join(experiment_dirpath, "summary_data", "matrix_order_parameters.npy"))
    matrix_ops = np.mean(matrix_ops[:, :, -1], axis=1)

    speeds = np.load(os.path.join(experiment_dirpath, "summary_data", "speeds.npy"))
    speeds = np.mean(speeds, axis=1)

    mean_directions = np.load(os.path.join(experiment_dirpath, "summary_data", "mean_directions.npy"))

    # no_mat_order_parameters = np.load("model_experiments/2026-05-20-collisions_shape/summary_data/order_parameters.npy")
    # no_mat_mean_directions = np.load("model_experiments/2026-05-31-collisions_shape/summary_data/mean_directions.npy")
    return experiment_dirpath, matrix_ops, mean_directions, speeds


@app.cell
def _(matrix_ops):
    matrix_ops.shape
    return


@app.cell
def _(mean_directions, plt):
    plt.hist(mean_directions.flatten(), bins=100)
    return


@app.cell
def _(matrix_ops, plt, speeds):
    plt.scatter(speeds, matrix_ops, s=1)
    return


@app.cell
def _(experiment_dirpath, iio, math, matrix_ops, np, os, plt, speeds):
    data_dirpath = os.path.join(experiment_dirpath, "run_data")

    def plot_organisation():
        # Get images:
        reasonable_organisation_indices = np.argwhere(
            np.logical_and(speeds < 0.15, matrix_ops > 0.075)
        )
        print(len(reasonable_organisation_indices))
        print("--- --- --- ---")
        matrix_images = []
        for index in reasonable_organisation_indices[:16]:
            index = index[0]
            print(index)
            hash_index = math.floor(index / 1000)
            matrix_image = iio.imread(os.path.join(data_dirpath, f"{hash_index}", f"{index}", "matrix_heading.png"))
            matrix_images.append(matrix_image)

        # Plot images:
        grid_size = 4
        fig, axs = plt.subplots(grid_size, grid_size, layout="constrained", figsize=(7, 7))
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                axs[i, j].imshow(matrix_images[count])
                axs[i, j].set_axis_off()
                axs[i, j].set_aspect("equal")
                count += 1

        # Show image:
        plt.show()

    plot_organisation()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
