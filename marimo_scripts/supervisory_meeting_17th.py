import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import json

    import numpy as np

    import matplotlib.pyplot as plt
    return json, np, os, plt


@app.cell
def _(json, np, os):
    def load_gridsearch_data(experiment_folderpath):
        # Load numpy data:
        parameter_matrix = np.load(
            os.path.join(experiment_folderpath, "sample_matrix.npy")
        )
        coherency_fractions = np.load(
            os.path.join(experiment_folderpath, "summary_data", "coherency_fractions.npy")
        )
        ann_indices = np.load(
            os.path.join(experiment_folderpath, "summary_data", "ann_indices.npy")
        )
        speeds = np.load(
            os.path.join(experiment_folderpath, "summary_data", "magnitude_cellmeans.npy")
        )

        # Generate mask over NaNs:
        cf_mask = np.any(np.isnan(coherency_fractions), axis=1)
        ann_mask = np.any(np.isnan(ann_indices), axis=1)
        speed_mask = np.any(np.isnan(speeds), axis=1)
        nan_mask = ~(cf_mask & ann_mask & speed_mask)

        parameter_matrix = parameter_matrix[nan_mask, :]
        coherency_fractions = coherency_fractions[nan_mask, :]
        ann_indices = ann_indices[nan_mask, :]
        speeds = speeds[nan_mask, :]

        # Get gridsearch configuration:
        with open(os.path.join(experiment_folderpath, "config.json")) as json_filestream:
            config_dictionary  = json.load(json_filestream)
        gridsearch_parameters = config_dictionary["gridsearch_parameters"]

        return parameter_matrix, coherency_fractions, ann_indices, speeds, gridsearch_parameters
    return (load_gridsearch_data,)


@app.cell
def _(load_gridsearch_data):
    parameter_matrix, coherency_fractions, ann_indices, speeds, gridsearch_parameters = load_gridsearch_data(
        "model_experiments/2025-12-02-collisions_shape"
    )
    return (
        ann_indices,
        coherency_fractions,
        gridsearch_parameters,
        parameter_matrix,
        speeds,
    )


@app.cell
def _(speeds):
    speeds
    return


@app.cell
def _(gridsearch_parameters, np, parameter_matrix, plt):
    def get_mean_and_variance(i_values, summary_metric, bin_count=50):
        # Set up output arrays:
        mean_summary_array = []
        std_summary_array = []
        bin_boundaries = np.linspace(0, 1, bin_count + 1)

        for bin_index in range(bin_count):
            low_threshold = bin_boundaries[bin_index]
            low_mask = i_values > low_threshold
            high_threshold = bin_boundaries[bin_index + 1]
            high_mask = i_values < high_threshold
            bin_mask = low_mask & high_mask
            mean_summary_array.append(np.mean(summary_metric[bin_mask]))
            std_summary_array.append(np.std(summary_metric[bin_mask]))

        return mean_summary_array, std_summary_array

    def plot_mean_and_variance(parameter_index, summary_metric, y_label):
        mean_array, std_array = get_mean_and_variance(parameter_matrix[:, parameter_index], summary_metric)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(np.linspace(0, 1, 51)[:50] + 1/50, mean_array)
        ax.set_xlabel(list(gridsearch_parameters.keys())[parameter_index])
        ax.set_ylabel(y_label)
        plt.show()

    def plot_binscatter_metric(metric_values, metric_label):
        fig, axs = plt.subplots(4, 3, figsize=(12, 8))

        for axis_index, ax in enumerate(axs.flatten()):
            mean_array, std_array = get_mean_and_variance(parameter_matrix[:, axis_index], metric_values)
            ax.scatter(np.linspace(0, 1, 51)[:50] + 1/50, mean_array, s=1)
            ax.set_xlabel(list(gridsearch_parameters.keys())[axis_index])
            ax.set_ylabel(metric_label)

        fig.suptitle(metric_label)
        fig.tight_layout()
        plt.show()
    return (plot_binscatter_metric,)


@app.cell
def _(ann_indices, plot_binscatter_metric):
    plot_binscatter_metric(ann_indices[:, 0], "Mean ANNI")
    return


@app.cell
def _(plot_binscatter_metric, speeds):
    plot_binscatter_metric(speeds[:, 0], "Mean Speed")
    return


@app.cell
def _(coherency_fractions, np, plot_binscatter_metric):
    plot_binscatter_metric(np.mean(coherency_fractions, axis=1), "Mean CF")
    return


@app.cell
def _(gridsearch_parameters, np, parameter_matrix, plt):
    def generate_phase_matrix(i_values, j_values, summary_metric, mesh_count=20):
        # Set up array:
        grid_boundaries =  np.linspace(0, 1, mesh_count + 1)
        phase_array = np.zeros((mesh_count, mesh_count))

        # Loop through indices of array:
        for grid_index_i in range(mesh_count):
            # Get row parameter mask:
            i_threshold_low = grid_boundaries[grid_index_i]
            i_low_mask = i_values >= i_threshold_low
            i_threshold_high = grid_boundaries[grid_index_i+1]
            i_high_mask = i_values < i_threshold_high
            i_mask = np.logical_and(i_low_mask, i_high_mask)

            for grid_index_j in range(mesh_count):
                # Get column parameter mask:
                j_threshold_low = grid_boundaries[grid_index_j]
                j_low_mask = j_values >= j_threshold_low
                j_threshold_high = grid_boundaries[grid_index_j+1]
                j_high_mask = j_values < j_threshold_high
                j_mask = np.logical_and(j_low_mask, j_high_mask)

                total_mask = np.logical_and(i_mask, j_mask)
                phase_array[grid_index_i, grid_index_j] = \
                    np.min(summary_metric[total_mask])

        return phase_array

    def phaseplot_matrix_plot(summary_metric, title):
        fig, axs = plt.subplots(10, 10, figsize=(15, 15), sharex=True, sharey=True)

        for i in range(10):
            for j in range(10):
                if i == j:
                    axs[i, j].set_xlabel(list(gridsearch_parameters.keys())[i], fontsize=8)
                    axs[i, j].set_aspect("equal")
                    continue
                if bool(axs[i, j].get_images()):
                    continue

                phase_matrix = generate_phase_matrix(parameter_matrix[:, i], parameter_matrix[:, j], summary_metric)

                # Set upper triangle:
                axs[i, j].imshow(phase_matrix, origin="lower")
                # axs[i, j].set_xlabel(list(gridsearch_parameters.keys())[i], fontsize=8)
                # axs[i, j].set_ylabel(list(gridsearch_parameters.keys())[j])

                # Set lower triangle:
                axs[j, i].imshow(phase_matrix.T, origin="lower")
                # axs[j, i].set_xlabel(list(gridsearch_parameters.keys())[j])
                # axs[j, i].set_ylabel(list(gridsearch_parameters.keys())[i])

        fig.suptitle(title)
        fig.tight_layout()
        plt.show()
    return generate_phase_matrix, phaseplot_matrix_plot


@app.cell
def _(coherency_fractions, np, phaseplot_matrix_plot):
    phaseplot_matrix_plot(np.mean(coherency_fractions, axis=1), "Coherency Fractions")
    return


@app.cell
def _(phaseplot_matrix_plot, speeds):
    phaseplot_matrix_plot(speeds[:, 0], "Speeds")
    return


@app.cell
def _(generate_phase_matrix, gridsearch_parameters, parameter_matrix, plt):
    def plot_contours(index_i, index_j, summary_metric, title):
        # Calculate bins:
        phase_matrix = generate_phase_matrix(parameter_matrix[:, index_i], parameter_matrix[:, index_j], summary_metric)

        # Plot contours:
        fig, ax = plt.subplots(figsize=(5, 5))
        contourplot = ax.contour(phase_matrix, 5, alpha=1.0)
        ax.clabel(contourplot, fontsize=10)
        ax.set_xlabel(list(gridsearch_parameters.keys())[index_i])
        ax.set_ylabel(list(gridsearch_parameters.keys())[index_j])
        ax.set_aspect("equal")
        fig.suptitle(title)
        fig.tight_layout()
        plt.show()
    return (plot_contours,)


@app.cell
def _(coherency_fractions, np, plot_contours):
    plot_contours(2, 3, np.mean(coherency_fractions, axis=1), "Coherency Fraction")
    return


@app.cell
def _(plot_contours, speeds):
    plot_contours(2, 3, speeds[:, 0], "Speed")
    return


@app.cell
def _(coherency_fractions, np):
    WT_1_CF_MEAN = 0.03646833938262533
    WT_1_CF_VAR = 1.289952251152215e-05

    mean_var_mask = np.logical_and(
        (WT_1_CF_MEAN - np.sqrt(WT_1_CF_VAR) < np.mean(coherency_fractions, axis=1)) & \
        (np.mean(coherency_fractions, axis=1) < WT_1_CF_MEAN + np.sqrt(WT_1_CF_VAR)),
        np.var(coherency_fractions, axis=1) < WT_1_CF_VAR
    )
    return (mean_var_mask,)


@app.cell
def _(mean_var_mask, np):
    np.count_nonzero(mean_var_mask)
    return


@app.cell
def _(mean_var_mask, parameter_matrix, plt):
    plt.scatter(parameter_matrix[mean_var_mask, 5], parameter_matrix[mean_var_mask, 6], s=1)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
