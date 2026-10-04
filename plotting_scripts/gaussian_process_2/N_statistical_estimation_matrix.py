import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import json
    import subprocess

    import pandas as pd
    import numpy as np
    import colorcet as cc

    import scipy.stats
    import scipy.spatial

    import matplotlib.pyplot as plt

    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.ticker import AutoLocator, MaxNLocator

    from datetime import datetime
    return cc, datetime, np, os, plt, subprocess


@app.cell
def _():
    import matplotlib as mpl

    # Font formatting:
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = "cmr10"
    mpl.rcParams['font.size'] = 9
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
    return


@app.cell
def _(datetime, os, subprocess):
    # PNG 300 dpi
    # A4 dimensions: 8.27 × 11.69 inches
    # Image dimensions: 160 x ? mm
    # Metadata: date, script, github branch id, og experiment source
    OUT_DIRPATH = "plotting_scripts/gaussian_process/out"
    CONTROL_PALETTE = "#1A85FF"
    RD_PALETTE = "#D41159"
    ADJ_RD_PALETTE = "#FFB000"

    PIXEL_SIZE = 0.3469 * 2  # Pixel size in µm
    MM_UNIT = 1/25.4  # Millimeters in inches, for matplotlib

    TEXT_WIDTH = 135 * MM_UNIT
    TEXT_HEIGHT = 217 * MM_UNIT 

    FULL_WIDTH = 170 * MM_UNIT
    FULL_HEIGHT = TEXT_HEIGHT * 0.8

    if not os.path.exists(OUT_DIRPATH):
        os.mkdir(OUT_DIRPATH)

    # Get current commit hash:
    commit_hash = subprocess.run("git rev-parse --short HEAD", shell=True, capture_output=True)
    commit_hash = commit_hash.stdout.decode("utf-8")[:-1]

    METADATA_DICTIONARY = {
        "creator": "Omar El Oakley",
        "script": "matrix_hessian_analysis.py",
        "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_commit_hash": commit_hash
    }
    return


@app.cell
def _(np, os):
    EXPERIMENT_DIRPATH = "model_experiments/2026-09-19-matrix_shape"

    LOG_DICT = {
        "speeds": False,
        "meander_ratios": False,
        "ann_indices": True,
        "coherency": True,
        "op65": True
    }

    METRICS_TO_LOAD = [
        "speeds",
        "meander_ratios",
        "ann_indices",
        "coherency",
        "op65"
    ]

    def load_metrics():
        metric_arrays = []
        for metric_name in METRICS_TO_LOAD:
            if metric_name == "op65":
                # Get order parameter:
                metric_array = np.load(os.path.join(EXPERIMENT_DIRPATH, "summary_data", "matrix_order_parameters.npy"))
                if LOG_DICT[metric_name]:
                    metric_arrays.append(np.log(metric_array[:, :, 2]))
                else:
                    metric_arrays.append(metric_array[:, :, 2])
            else:
                metric_array = np.load(os.path.join(
                    EXPERIMENT_DIRPATH, "summary_data", f"{metric_name}.npy"
                ))
                if LOG_DICT[metric_name]:
                    metric_arrays.append(np.log(metric_array))
                else:
                    metric_arrays.append(metric_array)
        return np.stack(metric_arrays, axis=2)

    metrics_array = load_metrics()
    inpca_coordinates = np.load(os.path.join(EXPERIMENT_DIRPATH, "inpca", "coordinates.npy"))
    filtered_indices = np.load(os.path.join(EXPERIMENT_DIRPATH, "inpca", "filtered_indices.npy")).astype(int)
    filtered_metrics = metrics_array[filtered_indices, :, :][::4]
    return filtered_metrics, inpca_coordinates


@app.cell
def _(filtered_metrics, np, plt):
    plt.hist(filtered_metrics[:, :, 0].flatten(), bins=100)
    speed_filter = np.mean(filtered_metrics[:, :, 0], axis=1) > 0.05
    sf_metric_std = filtered_metrics[speed_filter, :, :].std(axis=1, ddof=1)
    return sf_metric_std, speed_filter


@app.cell
def _(filtered_metrics, speed_filter):
    print(filtered_metrics.shape)
    print(filtered_metrics[speed_filter, :, :].shape)
    return


@app.cell
def _(filtered_metrics, np, plt, speed_filter):
    plt.hist(np.mean(filtered_metrics[speed_filter, :, 0], axis=1), bins=100)
    return


@app.cell
def _(np, sf_metric_std):
    np.max(sf_metric_std, axis=0) / np.min(sf_metric_std, axis=0)
    return


@app.cell
def _(filtered_metrics, np):
    metric_std = filtered_metrics[:, :, :].std(axis=1, ddof=1)
    np.max(metric_std, axis=0) / np.min(metric_std, axis=0)
    return


@app.cell
def _(inpca_coordinates, np):
    scaled_coordinates = inpca_coordinates / np.std(inpca_coordinates, axis=0)
    return (scaled_coordinates,)


@app.cell
def _(scaled_coordinates):
    import sklearn
    isomapper = sklearn.manifold.Isomap(n_neighbors=5, n_components=2)
    isomap_inpca_embeddings = isomapper.fit_transform(scaled_coordinates)
    return (isomap_inpca_embeddings,)


@app.cell
def _(filtered_metrics, isomap_inpca_embeddings, np, plt):
    def plot_isomap(metric_index):
        fig, ax = plt.subplots()
        color_values = np.mean(filtered_metrics[:, :, metric_index], axis=1)
        color_sort = np.argsort(color_values)
        ax.scatter(isomap_inpca_embeddings[color_sort, 0], isomap_inpca_embeddings[color_sort, 1], c=color_values[color_sort], s=1)
        plt.show()
    return (plot_isomap,)


@app.cell
def _(plot_isomap):
    plot_isomap(0)
    return


@app.cell
def _(plot_isomap):
    plot_isomap(4)
    return


@app.cell
def _(cc, filtered_metrics, inpca_coordinates, np, plt):
    def plot_inpca_coordinates(metric_index, grid_count=7):
        fig, axs = plt.subplots(grid_count, grid_count, figsize=(5, 5))
        color_values = np.mean(filtered_metrics[:, :, metric_index], axis=1)
        color_sort = np.argsort(color_values)

        for i in range(grid_count):
            for j in range(grid_count):
                if i == j:
                    axs[i, j].scatter(
                        inpca_coordinates[color_sort, i],
                        color_values[color_sort],
                        c=color_values[color_sort],
                        edgecolors="none", s=0.5, alpha=0.5,
                        cmap=cc.m_CET_L9
                    )
                else:
                    axs[i, j].scatter(
                        inpca_coordinates[color_sort, i],
                        inpca_coordinates[color_sort, j], 
                        edgecolors="none", s=0.5, alpha=0.5,
                        c=color_values[color_sort], cmap=cc.m_CET_L8,
                        vmin=np.quantile(color_values, 0.02), 
                        vmax=np.quantile(color_values, 0.98),
                    )

        plt.show()
    return (plot_inpca_coordinates,)


@app.cell
def _(plot_inpca_coordinates):
    plot_inpca_coordinates(0)
    return


@app.cell
def _(plot_inpca_coordinates):
    plot_inpca_coordinates(1)
    return


@app.cell
def _(plot_inpca_coordinates):
    plot_inpca_coordinates(3)
    return


@app.cell
def _(metric_array, np):
    def test_std():
        metric_std = metric_array.std(axis=1, ddof=1)
        print(np.max(metric_std, axis=0))
        print(np.min(metric_std, axis=0))
        print(np.max(metric_std, axis=0) / np.min(metric_std, axis=0))

    test_std()
    return


@app.cell
def _(metric_array, np):
    norm_array = metric_array - np.mean(metric_array, axis=1, keepdims=True)
    norm_array = norm_array / np.std(norm_array, axis=1, keepdims=True, ddof=1)
    all_residuals = norm_array.reshape(-1, 5)
    return (all_residuals,)


@app.cell
def _(all_residuals, np):
    global_covariance = np.cov(all_residuals, rowvar=False)
    global_correlation = global_covariance / np.sqrt(np.outer(np.diag(global_covariance), np.diag(global_covariance)))
    # global_correlation = np.eye(5)
    return (global_correlation,)


@app.cell
def _(global_correlation):
    global_correlation
    return


@app.cell
def _(global_correlation, metric_array, np):
    def bhattacharyya_gaussian(set_a, set_b):
        set_a = set_a.reshape(4, 5)
        set_b = set_b.reshape(4, 5)

        # Get relevant descriptors:
        a_mean = np.mean(set_a, axis=0)
        a_std = np.std(set_a, axis=0, ddof=1)
        b_mean = np.mean(set_b, axis=0)
        b_std = np.std(set_b, axis=0, ddof=1)

        # Get standard deviations:
        joint_std = np.sqrt(0.5 * (a_std**2 + b_std**2))

        # Do necessary covariance calculations:
        covariance_a = np.outer(a_std, a_std) * global_correlation
        covariance_b = np.outer(b_std, b_std) * global_correlation
        averaged_covariance = 0.5 * (covariance_a + covariance_b)
        rescaled_covariance = averaged_covariance / np.outer(joint_std, joint_std)

        # Get mean difference term:
        scaled_mean_difference = (a_mean - b_mean) / joint_std

        # Scaling both the mean difference and the covariance allows us
        # to avoid some potentially destructive determinant calculations:
        solved = np.linalg.solve(rescaled_covariance, scaled_mean_difference)
        mean_component = 0.125 * np.dot(solved, scaled_mean_difference)

        # Some determinant identities are required to retrieve
        # this from the original Bhattacharyya formulation 
        # (see https://en.wikipedia.org/wiki/Bhattacharyya_distance#Gaussian_case)
        # Average sigma:
        _, rescaled_logdet = np.linalg.slogdet(rescaled_covariance)
        log_det = 2 * np.sum(np.log(joint_std)) + rescaled_logdet
        # Sigma comparisons:
        _, log_det_global_covar = np.linalg.slogdet(global_correlation)
        pair_average = np.log(a_std).sum() + np.log(b_std).sum() + log_det_global_covar
        covariance_component = 0.5 * (log_det - pair_average)

        # The full distance is effectively how close the means are, and how close the standard deviations are:
        return mean_component + covariance_component

    bhattacharyya_gaussian(metric_array[0, :, :], metric_array[1, :, :])
    return


@app.cell
def _(metric_array, np):
    def bg_vectorised(metric_set, global_correlation=None):
        # Set covariance matrix to be diagonal if none provided:
        if global_correlation is None:
            global_correlation = np.eye(metric_set.shape[-1])

        # Get dimensions:
        set_count, repeat_count, metric_count = metric_set.shape
        distance_matrix = np.zeros((set_count, set_count))

        # Calculate Bhattacharyya distance, iterating over rows (only calculate upper triangle):
        for set_index in range(set_count - 1):
            # Get comparator sets, to allow us to vectorise calculation:
            indexed_set = metric_set[set_index, :, :]
            comparator_sets = metric_set[set_index + 1:, :, :]

            # Get mean and standard deviations of summary statistics at each parameter set:
            indexed_mean = np.mean(indexed_set, axis=0)
            comparator_means = np.mean(comparator_sets, axis=1)
            indexed_std = np.std(indexed_set, axis=0, ddof=1)
            comparator_stds = np.std(comparator_sets, axis=1, ddof=1)

            # Broadcast std averaging:
            joint_stds = np.sqrt(0.5 * (indexed_std**2 + comparator_stds**2))

            # Broadcast covariance averaging:
            outer_indexed_std = indexed_std[None, :, None] * indexed_std[None, None, :]
            outer_comparator_stds = comparator_stds[:, :, None] * comparator_stds[:, None, :]
            averaged_covariances = 0.5 * (outer_indexed_std + outer_comparator_stds) * global_correlation[None]
            rescaled_covariances = averaged_covariances / (joint_stds[:, :, None] * joint_stds[:, None, :])

            # Get mean differences:
            scaled_mean_differences = (indexed_mean - comparator_means) / joint_stds
            scaled_solution = np.linalg.solve(rescaled_covariances, scaled_mean_differences[..., None])[..., 0]
            # Get dot product of solution with the scaled mean difference:
            mean_terms = 0.125 * np.einsum('nk,nk->n', scaled_mean_differences, scaled_solution)

            # Decompose upper part of fraction into 2.logdet + rescaled:
            _, rescaled_logdets = np.linalg.slogdet(rescaled_covariances)
            log_det = 2 * np.sum(np.log(joint_stds), axis=1) + rescaled_logdets

            # Decompose lower part of fraction into logdet[indexed] + logdet[comparators] + logdet[global]:
            _, log_det_global_covar = np.linalg.slogdet(global_correlation)
            pair_averages = np.log(indexed_std).sum() + np.log(comparator_stds).sum(axis=1) + log_det_global_covar
            covariance_terms = 0.5 * (log_det - pair_averages)

            distance_matrix[set_index, set_index + 1:] = mean_terms + covariance_terms

        return distance_matrix + distance_matrix.T

    test_vectorised = bg_vectorised(metric_array[::8, :, :], global_correlation=None)
    return (test_vectorised,)


@app.cell
def _(test_vectorised):
    test_vectorised[0, 1]
    return


@app.cell
def _(test_vectorised):
    test_vectorised.shape
    return


@app.cell
def _(np, test_vectorised):
    log_distances = np.log1p(test_vectorised)
    # log_distances = test_vectorised
    return (log_distances,)


@app.cell
def _(log_distances, np):
    n = log_distances.shape[0]
    centering_matrix = np.eye(n) - (np.ones(n) / n)
    centred_matrix = centering_matrix @ (log_distances) @ centering_matrix
    eigenvalues, eigenvectors = np.linalg.eigh(centred_matrix)
    order = np.argsort(-np.abs(eigenvalues))
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    coordinates = eigenvectors * np.sqrt(np.abs(eigenvalues))
    return coordinates, eigenvalues


@app.cell
def _(eigenvalues, np, plt):
    plt.plot(np.log(np.abs(eigenvalues))[:100])
    return


@app.cell
def _(metric_array, np):
    test_speeds = metric_array[::8, :, 0].mean(axis=1)
    test_speeds = np.clip(test_speeds, np.quantile(test_speeds, 0.05), np.quantile(test_speeds, 0.95))

    test_mr = metric_array[::8, :, 1].mean(axis=1)
    test_mr = np.clip(test_mr, np.quantile(test_mr, 0.05), np.quantile(test_mr, 0.95))

    test_coherency = metric_array[::8, :, 3].mean(axis=1)
    test_coherency = np.clip(test_coherency, np.quantile(test_coherency, 0.05), np.quantile(test_coherency, 0.95))

    test_op65 = metric_array[::8, :, 4].mean(axis=1)
    test_op65 = np.clip(test_op65, np.quantile(test_op65, 0.05), np.quantile(test_op65, 0.95))
    return test_coherency, test_op65, test_speeds


@app.cell
def _(np, plt, test_op65):
    plt.hist(np.exp(test_op65))
    return


@app.cell
def _(coordinates, np, plt):
    def plot_inpca(colors):
        fig, ax = plt.subplots()
        color_sort = np.argsort(colors)
        ax.scatter(coordinates[color_sort, 0], coordinates[color_sort, 2], s=0.1, c=colors[color_sort])
        ax.set_aspect("equal")
        plt.show()
    return (plot_inpca,)


@app.cell
def _(plot_inpca, test_speeds):
    plot_inpca(test_speeds)
    return


@app.cell
def _(plot_inpca, test_coherency):
    plot_inpca(test_coherency)
    return


@app.cell
def _(plot_inpca, test_op65):
    plot_inpca(test_op65)
    return


@app.cell
def _(coordinates, np, plt):
    def plot_3d_inpca(colors):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        color_sort = np.argsort(colors)
        ax.scatter(
            coordinates[color_sort, 0],
            coordinates[color_sort, 1],
            coordinates[color_sort, 2],
            s=1, alpha=0.25, c=colors[color_sort]
        )
        plt.show()
    return (plot_3d_inpca,)


@app.cell
def _(plot_3d_inpca, test_speeds):
    plot_3d_inpca(test_speeds)
    return


@app.cell
def _(plot_3d_inpca, test_op65):
    plot_3d_inpca(test_op65)
    return


@app.cell
def _(filtered_inputs, plot_3d_inpca):
    plot_3d_inpca(filtered_inputs[::8, -1])
    return


@app.cell
def _():
    # def get_inpca(distance_matrix):
    #     n = distance_matrix.shape[0]
    #     overlap_matrix = -4 * distance_matrix
    #     centering_matrix = np.eye(n) - (np.ones(n) / n)
    #     centred_matrix = centering_matrix @ overlap_matrix @ centering_matrix
    #     eigenvalues, eigenvectors = np.linalg.eigh(centred_matrix)
    #     order = np.argsort(-np.abs(eigenvalues))
    #     eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    #     coordinates = eigenvectors * np.sqrt(np.abs(eigenvalues))
    #     return coordinates, eigenvalues, eigenvectors
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
