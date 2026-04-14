import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import json

    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt
    return json, np, os, pd, plt


@app.cell
def _(json, os, pd):
    def load_wetlab_data(experiment_folderpath):
        # Get average nearest neighbour index:
        with open(os.path.join(experiment_folderpath, "anni_dictionary.json")) as f:
            anni_dictionary = json.load(f)

        # Get coherency fraction:
        with open(os.path.join(experiment_folderpath, "cf_dictionary.json")) as f:
            cf_dictionary = json.load(f)

        # Get speeds:
        fitting_dataframe = pd.read_csv(os.path.join(experiment_folderpath, "fitting_dataset.csv"))

        return anni_dictionary, cf_dictionary, fitting_dataframe
    return (load_wetlab_data,)


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

        # Get gridsearch configuration:
        with open(os.path.join(experiment_folderpath, "config.json")) as json_filestream:
            config_dictionary  = json.load(json_filestream)
        gridsearch_parameters = config_dictionary["gridsearch_parameters"]

        return parameter_matrix, coherency_fractions, ann_indices, speeds, gridsearch_parameters
    return (load_gridsearch_data,)


@app.cell
def _(load_wetlab_data):
    anni_dictionary, cf_dictionary, fitting_dataframe = load_wetlab_data("wetlab_data/OEO20241206")
    return anni_dictionary, cf_dictionary, fitting_dataframe


@app.cell
def _(anni_dictionary, cf_dictionary, fitting_dataframe, np):
    ANNI_MEAN = np.mean(anni_dictionary["5"])
    ANNI_SEM = np.std(anni_dictionary["5"]) / np.sqrt(12)

    CF_MEAN = np.mean(cf_dictionary["5"])
    CF_SEM = np.std(cf_dictionary["5"]) / np.sqrt(12)

    column_mask = fitting_dataframe["column"] == 5
    SPEED_MEAN = np.mean(fitting_dataframe.loc[column_mask, "speed"])
    SPEED_SEM = np.std(fitting_dataframe.loc[column_mask, "speed"]) / np.sqrt(np.count_nonzero(column_mask))
    return ANNI_MEAN, ANNI_SEM, CF_MEAN, CF_SEM, SPEED_MEAN, SPEED_SEM


@app.cell
def _(load_gridsearch_data):
    parameter_matrix, coherency_fractions, ann_indices, speeds, gridsearch_parameters = load_gridsearch_data(
        "model_experiments/2025-12-01-collisions_only"
    )
    return (
        ann_indices,
        coherency_fractions,
        gridsearch_parameters,
        parameter_matrix,
        speeds,
    )


@app.cell
def _(
    ANNI_MEAN,
    ANNI_SEM,
    CF_MEAN,
    CF_SEM,
    SPEED_MEAN,
    SPEED_SEM,
    ann_indices,
    coherency_fractions,
    np,
    speeds,
):
    # Get standard errors in the mean:
    model_cf_sem = np.std(coherency_fractions, axis=1) / np.sqrt(12)
    model_anni_sem = np.std(coherency_fractions, axis=1) / np.sqrt(12)
    model_speed_sem = speeds[:, 1] / np.sqrt(12)

    # Get implausibility metrics:
    cf_implausibility_metric = \
        (np.abs(CF_MEAN - np.mean(coherency_fractions, axis=1))) \
        / np.sqrt(model_cf_sem**2 + CF_SEM**2)

    anni_implausibility_metric = \
        (np.abs(ANNI_MEAN - np.mean(ann_indices, axis=1))) \
        / np.sqrt(model_anni_sem**2 + ANNI_SEM**2)

    speed_implausibility_metric = \
        (np.abs(SPEED_MEAN - speeds[:, 0])) \
        / np.sqrt(model_speed_sem**2 + SPEED_SEM**2)

    max_implausibility = np.max(
        np.stack([cf_implausibility_metric, speed_implausibility_metric, anni_implausibility_metric], axis=1),
        axis=1
    )
    return (
        cf_implausibility_metric,
        max_implausibility,
        speed_implausibility_metric,
    )


@app.cell
def _(max_implausibility, plt, speed_implausibility_metric):
    plt.hist(speed_implausibility_metric, bins=100);
    plt.hist(max_implausibility, bins=100);
    plt.show()
    return


@app.cell
def _(max_implausibility, np):
    np.argwhere(max_implausibility < 3)
    return


@app.cell
def _(max_implausibility, parameter_matrix):
    parameter_matrix[max_implausibility < 3, :]
    return


@app.cell
def _(WT_1_CF_MEAN, WT_1_CF_VAR, coherency_fractions, np):
    mean_var_mask = np.logical_and(
        np.mean(coherency_fractions, axis=1) > WT_1_CF_MEAN,
        np.var(coherency_fractions, axis=1) < WT_1_CF_VAR
    )
    return (mean_var_mask,)


@app.cell
def _(max_implausibility, plt):
    plt.hist(max_implausibility, bins=100);
    plt.show()
    return


@app.cell
def _(max_implausibility, parameter_matrix, plt):
    plt.hist(parameter_matrix[max_implausibility < 3, 9], bins=50)
    plt.show()
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
    return (plot_mean_and_variance,)


@app.cell
def _(plt, speeds):
    plt.hist(speeds[:, 0], bins=100);
    plt.show()
    return


@app.cell
def _(ann_indices, np, plot_mean_and_variance):
    plot_mean_and_variance(1, np.mean(ann_indices, axis=1), "Mean ANNI")
    return


@app.cell
def _(coherency_fractions, np, plot_mean_and_variance):
    plot_mean_and_variance(5, np.mean(coherency_fractions, axis=1), "Mean Coherency Fraction")
    return


@app.cell
def _(coherency_fractions, np, plot_mean_and_variance):
    plot_mean_and_variance(1, np.mean(coherency_fractions, axis=1), "Mean Coherency Fraction")
    return


@app.cell
def _(coherency_fractions, np, plot_mean_and_variance):
    plot_mean_and_variance(4, np.mean(coherency_fractions, axis=1), "Mean Coherency Fraction")
    return


@app.cell
def _(np):
    def generate_phase_matrix(i_values, j_values, summary_metric, mesh_count=10):
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
    return (generate_phase_matrix,)


@app.cell
def _(generate_phase_matrix, max_implausibility, parameter_matrix, plt):
    cf_implausibility_array = generate_phase_matrix(parameter_matrix[:, 5], parameter_matrix[:, 6], max_implausibility, 25)
    plt.imshow(cf_implausibility_array)
    plt.colorbar()
    plt.show()
    return


@app.cell
def _(coherency_fractions, generate_phase_matrix, np, parameter_matrix, plt):
    cf_array = generate_phase_matrix(
        parameter_matrix[:, 0], parameter_matrix[:, 1], np.mean(coherency_fractions, axis=1), 25
    )
    plt.imshow(cf_array)
    return (cf_array,)


@app.cell
def _(cf_array, plt):
    def plot_contours(array):
        fig, ax = plt.subplots(figsize=(5, 5))
        contourplot = ax.contour(array, 5, alpha=1.0)
        ax.clabel(contourplot, fontsize=10)
        ax.set_aspect("equal")
        plt.show()

    plot_contours(cf_array)
    return


@app.cell
def _(coherency_fractions, plt):
    plt.hist(coherency_fractions[:, 0], bins=100);
    plt.show()
    return


@app.cell
def _(np):
    def generate_nroy_matrix(i_values, j_values, summary_metric, low_thresh, high_thresh, mesh_count=10):
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

                # Check if any realisations fall into NROY:
                total_mask = np.logical_and(i_mask, j_mask)
                summary_low_mask = summary_metric[total_mask] >= low_thresh
                summary_high_mask = summary_metric[total_mask] < high_thresh
                summary_mask = summary_low_mask & summary_high_mask
                phase_array[grid_index_i, grid_index_j] = np.count_nonzero(summary_mask) / len(summary_mask)

        return phase_array
    return (generate_nroy_matrix,)


@app.cell
def _(cf_implausibility_metric, generate_nroy_matrix, parameter_matrix, plt):
    cf_nroy_array = generate_nroy_matrix(
        parameter_matrix[:, 3], parameter_matrix[:, 4],
        cf_implausibility_metric,
        0.00, 3.00, 40
    )
    plt.imshow(cf_nroy_array, vmin=0, vmax=0.3)
    plt.colorbar()
    plt.show()
    return (cf_nroy_array,)


@app.cell
def _(cf_nroy_array):
    cf_nroy_array
    return


@app.cell
def _(generate_nroy_matrix, parameter_matrix, plt, speeds):
    speed_nroy_array = generate_nroy_matrix(
        parameter_matrix[:, 3], parameter_matrix[:, 4],
        speeds[:, 0],
        0.2, 0.4, 50
    )
    plt.imshow(speed_nroy_array, vmin=0, vmax=0.3)
    return (speed_nroy_array,)


@app.cell
def _(speed_nroy_array):
    speed_nroy_array
    return


@app.cell
def _(np):
    def generate_mask_histogram(i_values, j_values, mask, mesh_count=10):
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

                # Check if any realisations fall into NROY:
                grid_mask = np.logical_and(i_mask, j_mask)
                full_mask = np.logical_and(mask, grid_mask)
                phase_array[grid_index_i, grid_index_j] = np.count_nonzero(full_mask) / len(full_mask)

        return phase_array
    return


@app.cell
def _(mean_var_mask, parameter_matrix, plot_mask_histogram, plt):
    mean_var_hist = plot_mask_histogram(parameter_matrix[:, 0], parameter_matrix[:, 1], mean_var_mask, mesh_count=25)
    plt.imshow(mean_var_hist > 0)
    return


@app.cell
def _():
    # # # Discard failed simulations:
    # # nan_mask = np.any(np.isnan(distances), axis=(1, 2))
    # # nan_parameters = parameters[nan_mask, :]

    # # Temporary nan mask for overconfluent simulations:
    # cell_number = parameters[:, 5]
    # cell_radius = parameters[:, 7]
    # cell_area = np.pi * (cell_radius ** 2)
    # packing_fraction = (cell_area * cell_number) / (2048**2)
    # pf_mask = packing_fraction > 0.8
    # nan_mask = pf_mask
    # # nan_mask = np.logical_or(
    # #     nan_mask, pf_mask
    # # )

    # # Apply to remaining matrices:
    # parameters = parameters[~nan_mask, :]
    # # distances = distances[~nan_mask, :]
    # order_parameters = order_parameters[~nan_mask, 0]
    # speeds = speeds[~nan_mask, 0]
    # coherency_fractions = coherency_fractions[~nan_mask, :]
    # ann_indices = ann_indices[~nan_mask, :]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
