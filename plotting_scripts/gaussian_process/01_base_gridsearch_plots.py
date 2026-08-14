import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import json
    import subprocess
    import functools

    import sklearn

    import numpy as np
    import colorcet as cc

    import scipy.stats

    import matplotlib.pyplot as plt

    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.ticker import AutoLocator, MaxNLocator

    from datetime import datetime
    return (
        MaxNLocator,
        cc,
        datetime,
        json,
        np,
        os,
        plt,
        scipy,
        sklearn,
        subprocess,
    )


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
        "script": "base_gridsearch_plots.py",
        "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_commit_hash": commit_hash
    }
    return (
        FULL_HEIGHT,
        FULL_WIDTH,
        METADATA_DICTIONARY,
        OUT_DIRPATH,
        TEXT_HEIGHT,
        TEXT_WIDTH,
    )


@app.cell
def _(TEXT_HEIGHT, TEXT_WIDTH):
    print(TEXT_WIDTH)
    print(TEXT_HEIGHT)
    return


@app.cell
def _():
    METRICS_TO_PLOT = [
        "speeds",
        "meander_ratios",
        "ann_indices",
        "coherency"
    ]

    METRICS_LABELS = [
        "Speed ($\mu$m/min)",
        "MR",
        "ANNI",
        "Coherency"
    ]

    Q_LABELS = [
        "Speed Quantile",
        "MR Quantile",
        "ANNI Quantile",
        "Coherency Quantile"
    ]

    EXPERIMENT_DIRPATH = "model_experiments/2026-05-31-collisions_shape"
    return EXPERIMENT_DIRPATH, METRICS_LABELS, METRICS_TO_PLOT, Q_LABELS


@app.cell
def _(EXPERIMENT_DIRPATH, METRICS_TO_PLOT, json, np, os):
    # Load in gridsearch data:
    parameter_values = np.load(os.path.join(EXPERIMENT_DIRPATH, "sample_matrix.npy"))
    with open(os.path.join(EXPERIMENT_DIRPATH, "config.json")) as json_file:
        config_dict = json.load(json_file)

    # Get metrics:
    def load_metrics():
        metrics_dict = {}
        for metric_name in METRICS_TO_PLOT:
            metric_array = np.load(os.path.join(
                EXPERIMENT_DIRPATH, "summary_data", f"{metric_name}.npy"
            ))
            metrics_dict[metric_name] = np.nanmean(metric_array, axis=1)
        return metrics_dict

    metrics_dict = load_metrics()
    return config_dict, metrics_dict, parameter_values


@app.cell
def _(config_dict):
    import pandas as pd
    config_dataframe = [{"Parameter Name": name, "Lower Bound": range_def[0], "Upper Bound": range_def[1]} for name, range_def in  config_dict["gridsearch_parameters"]]
    config_dataframe = pd.DataFrame(config_dataframe)
    return (config_dataframe,)


@app.cell
def _(config_dataframe):
    print(config_dataframe.to_latex(index=False, float_format="%0.6g"))
    return


@app.cell
def _(parameter_values):
    print(parameter_values.shape)
    return


@app.cell
def _(
    METADATA_DICTIONARY,
    METRICS_LABELS,
    OUT_DIRPATH,
    TEXT_WIDTH,
    cc,
    datetime,
    metrics_dict,
    np,
    os,
    plt,
    scipy,
):
    def plot_joint_metric_distributions():
        # Get subsampling for KDE estimation:
        rng = np.random.default_rng(0)

        # Set up plot:
        fig, axs = plt.subplots(4, 4, figsize=(TEXT_WIDTH, TEXT_WIDTH))
        for i in range(4):
            for j in range(4):
                # Plot histogram if on diagonal:
                if i == j:
                    metric_i = list(metrics_dict.values())[i]
                    axs[i, j].hist(
                        metric_i, bins=50, density=True,
                        color='k', histtype="step"
                    )
                    axs[i, j].text(
                        0.95, 0.95, METRICS_LABELS[i], 
                        horizontalalignment="right",
                        verticalalignment="top",
                        transform=axs[i, j].transAxes
                    )
                    if j == 3:
                        axs[i, j].set_xticks([0.2, 0.8])
                    continue

                # Ignore if below diagonal:
                if i > j:
                    axs[i, j].set_axis_off()
                    continue

                # Do joint plot otherwise:
                metric_i = list(metrics_dict.values())[i]
                metric_j = list(metrics_dict.values())[j]

                # Estimate KDE, filtering for NaNs:
                density_dataset = np.stack([metric_i, metric_j], axis=1)
                nan_mask = np.any(np.isnan(density_dataset), axis=1)
                density_dataset = density_dataset[~nan_mask, :]
                random_indices = rng.choice(density_dataset.shape[0], size=(2048))
                kde = scipy.stats.gaussian_kde(density_dataset[random_indices, :].T)
                densities = kde.evaluate(density_dataset.T)
                density_sort = np.argsort(densities)

                # Plot points coloured by density:
                axs[i, j].scatter(
                    density_dataset[density_sort, 0],
                    density_dataset[density_sort, 1],
                    c=densities[density_sort], s=0.5, cmap=cc.m_CET_L20
                )

                # Fix ticks for coherency metric:
                if j == 3:
                    axs[i, j].set_yticks([0.2, 0.8])

        edging = 0.05
        fig.subplots_adjust(edging, edging, 1 - edging, 1 - edging, hspace=0.25, wspace=0.25)

        # Update metadata time:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(os.path.join(OUT_DIRPATH, "metric_scattergrid.png"), dpi=300, metadata=METADATA_DICTIONARY, transparent=True)
        plt.show()

    plot_joint_metric_distributions()
    return


@app.cell
def _(
    FULL_HEIGHT,
    FULL_WIDTH,
    METADATA_DICTIONARY,
    METRICS_TO_PLOT,
    MaxNLocator,
    OUT_DIRPATH,
    Q_LABELS,
    cc,
    config_dict,
    datetime,
    metrics_dict,
    np,
    os,
    parameter_values,
    plt,
    scipy,
):
    def plot_binscatter(parameter_index, metric_index, ax):
        # Format ticks:
        ax.xaxis.set_major_locator(MaxNLocator(3))

        # Get relevant data:
        parameter_array = parameter_values[:, parameter_index]
        metric_values = list(metrics_dict.values())[metric_index]
        parameter_name = config_dict["gridsearch_parameters"][parameter_index][0]
        scaling = config_dict["gridsearch_parameters"][parameter_index][1]

        # Get binned statistic:
        bins = np.linspace(0, 1, 51)
        bin_centers = bins[:-1] + 0.01
        bin_median, _, _ = scipy.stats.binned_statistic(parameter_array, metric_values, statistic=np.nanmedian, bins=bins)
        bin_mean, _, _ = scipy.stats.binned_statistic(parameter_array, metric_values, statistic=np.nanmean, bins=bins)
        bin_std, _, _ = scipy.stats.binned_statistic(parameter_array, metric_values, statistic=np.nanstd, bins=bins)
        # bin_sem = bin_std / np.sqrt(2**17)

        ecdf = scipy.stats.ecdf(metric_values[~np.isnan(metric_values)])
        std_quantile = ecdf.cdf.evaluate(bin_median + bin_std)
        bin_quantile = ecdf.cdf.evaluate(bin_median)

        stderr = np.abs(bin_quantile - std_quantile)

        # Plot data:
        scaled_x = (bin_centers * (scaling[1] - scaling[0])) + scaling[0]
        vmin, vmax = np.nanquantile(metric_values, [0.15, 0.85])
        vmin, vmax = [0.15, 0.85]
        ax.errorbar(scaled_x, bin_quantile, stderr, c='k', alpha=0.1)
        ax.scatter(scaled_x, bin_quantile, s=1, c=bin_quantile, vmin=vmin, vmax=vmax, cmap=cc.m_CET_D7)

        # Format plot:
        ax.set_xlim(scaling[0], scaling[1])
        ax.set_ylim(0, 1)

        # Label:
        ax.set_xlabel(parameter_name, fontsize=7)
        # ax.set_ylabel(METRICS_LABELS[metric_index])

    def plot_all_binscatter(metric_index):
        fig, axs = plt.subplots(6, 2, figsize=(FULL_WIDTH, FULL_HEIGHT), sharey=True)

        count = 0
        for i in range(6):
            for j in range(2):
                plot_binscatter(count, metric_index, axs[i, j])
                count += 1

        # Adjust subplots:
        fig.subplots_adjust(0.15, 0.05, 0.85, 0.95, wspace=0.15, hspace=0.55)

        # Add global y-label:
        fig.text(0.075, 0.5, Q_LABELS[metric_index], va='center', rotation='vertical', in_layout=True)

        # Update metadata time:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, f"binscatter_{METRICS_TO_PLOT[metric_index]}.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True, pad_inches=0
        )
        plt.show()

    for p_index in range(4):
        plot_all_binscatter(p_index)
    return


@app.cell
def _(EXPERIMENT_DIRPATH, np, os):
    import imageio.v3 as iio

    def load_image(index, image_type):
        npz_archive = np.load(os.path.join(EXPERIMENT_DIRPATH, f"{image_type}.npz"))
        image_array = npz_archive[str(index)]
        return image_array
    return (load_image,)


@app.cell
def _(metrics_dict, np, parameter_values, sklearn):
    # Get PCA of model summary statistics:
    metric_dataset = np.stack(list(metrics_dict.values()) + [parameter_values[:, 5]], axis=1)
    nan_mask = np.any(np.isnan(metric_dataset), axis=1)
    nan_mask = np.logical_or(nan_mask, parameter_values[:, 5] < 0.0)
    metric_dataset = metric_dataset[~nan_mask]
    metric_dataset[:, :3] = np.log(metric_dataset[:, :3])

    metric_stddev = np.std(metric_dataset, axis=0, keepdims=True)
    metric_mean = np.mean(metric_dataset, axis=0, keepdims=True)
    print(metric_mean.shape)
    print(metric_stddev.shape)
    whitened_input = (metric_dataset - metric_mean) / metric_stddev
    pca = sklearn.decomposition.PCA(n_components=2, whiten=True)
    metric_embeddings = pca.fit_transform(whitened_input)
    return metric_embeddings, nan_mask, pca


@app.cell
def _(pca):
    pca.components_
    return


@app.cell
def _(metric_embeddings, plt):
    plt.hist(metric_embeddings[:, 0], bins=100)
    return


@app.cell
def _(
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    TEXT_WIDTH,
    datetime,
    metric_embeddings,
    os,
    pca,
    plt,
):
    PCA_METRICS = [
        "Speed",
        "MR",
        "ANNI",
        "Coherency",
        "Cell Density"
    ]

    def plot_pca_results():
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH, TEXT_WIDTH))
        # color_values = metric_dataset[:, 4]
        # color_sort = np.argsort(color_values)
        ax.scatter(
            metric_embeddings[:, 0], metric_embeddings[:, 1],
            c='k', s=1, edgecolors="none"
        )

        for metric_index in range(5):
            x, y = pca.components_[:, metric_index]
            scatter = ax.scatter(x, y, s=10, alpha=0.7)

            # Label loading:
            if metric_index == 1:
                ax.text(
                    x + 0.04, y - 0.04, PCA_METRICS[metric_index],
                    fontsize=10, fontweight="bold",
                    color=scatter.get_facecolor(), va="top", alpha=1
                )
            else:
                ax.text(
                    x + 0.04, y + 0.04, PCA_METRICS[metric_index],
                    fontsize=10, fontweight="bold",
                    color=scatter.get_facecolor(), alpha=1
                )

        # Plot guidelines:
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        ax.hlines(0, *xlims, color='r', ls="--", alpha=0.6)
        ax.vlines(0, *ylims, color='r', ls="--", alpha=0.6)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        # Label axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        edging = 0.07
        fig.subplots_adjust(edging, edging, 1 - edging, 1 - edging)

        # Save:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, f"gridsearch_metric_pca_plot.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        ax.set_aspect("equal")
        plt.show()

    plot_pca_results()
    return


@app.cell
def _():
    # from prinpy.local_curves import ConstrainedFitter, GreedyFit

    # # Fit a principal curve
    # curve = ConstrainedFitter(algorithm=GreedyFit(), tolerance=0.05).fit(data)
    return


@app.cell
def _(metric_embeddings):
    EMBEDDINGS = metric_embeddings
    return (EMBEDDINGS,)


@app.cell
def _(EMBEDDINGS, np, plt):
    def plot_cq_scatterplot(grid_size):
        # Instantiate RNG for image selection:
        rng = np.random.default_rng(0)

        # Extract bins of primary component:
        base_bins = np.linspace(-1.5, 1.5, grid_size + 1)

        # Inspect plots:
        fig, ax = plt.subplots()

        # Iterate through bins, retrieving example indexes:
        collage_array = []
        for i in range(grid_size):
            collage_row = []
            for j in range(grid_size):
                x_low = base_bins[j]
                x_high = base_bins[j+1]
                x_mask = np.logical_and(
                    EMBEDDINGS[:, 0] <= x_high,
                    EMBEDDINGS[:, 0] > x_low
                )
                y_low = base_bins[i]
                y_high = base_bins[i+1]
                y_mask = np.logical_and(
                    EMBEDDINGS[:, 1] <= y_high,
                    EMBEDDINGS[:, 1] > y_low
                )

                joint_mask = np.logical_and(x_mask, y_mask)
                ax.scatter(EMBEDDINGS[joint_mask, 0], EMBEDDINGS[joint_mask, 1], s=0.1,)

                # Grid mid-point:
                mid_x = (x_low + x_high) / 2
                mid_y = (y_low + y_high) / 2
                pca_diff = EMBEDDINGS - np.expand_dims(np.array([mid_x, mid_y]), axis=0)
                center_point = np.argmin(np.sqrt(np.sum(pca_diff**2, axis=1)))
                ax.scatter(EMBEDDINGS[center_point, 0], EMBEDDINGS[center_point, 1])

        ax.set_aspect("equal")

        plt.show()

    plot_cq_scatterplot(6)
    return


@app.cell
def _(EMBEDDINGS, load_image, nan_mask, np):
    def generate_cq_image_matrix(grid_size, image_type):
        # Instantiate RNG for image selection:
        rng = np.random.default_rng(0)

        # Extract bins of primary component:
        base_bins = np.linspace(-1.5, 1.5, grid_size + 1)

        # Iterate through bins, retrieving example indexes:
        collage_array = []
        for i in range(grid_size):
            collage_row = []
            for j in range(grid_size):
                x_low = base_bins[j]
                x_high = base_bins[j+1]
                x_mask = np.logical_and(
                    EMBEDDINGS[:, 0] <= x_high,
                    EMBEDDINGS[:, 0] > x_low
                )
                y_low = base_bins[i]
                y_high = base_bins[i+1]
                y_mask = np.logical_and(
                    EMBEDDINGS[:, 1] <= y_high,
                    EMBEDDINGS[:, 1] > y_low
                )
                joint_mask = np.logical_and(x_mask, y_mask)

                # Get grid mid-point:
                mid_x = (x_low + x_high) / 2
                mid_y = (y_low + y_high) / 2
                pca_diff = EMBEDDINGS - np.expand_dims(np.array([mid_x, mid_y]), axis=0)
                center_point = np.argmin(np.sqrt(np.sum(pca_diff**2, axis=1)))
                center_index = np.arange(2**17)[~nan_mask][center_point]
                image_array = load_image(center_index, image_type)

                # indices = np.arange(2**17)[~nan_mask][np.argwhere(joint_mask)]
                # sorted_indices = indices[np.argsort(parameter_values[indices, 5])]
                # min_index = sorted_indices[0][0][0]
                # image_array = load_image(min_index)

                collage_row.append(image_array)
            collage_array.append(np.concatenate(collage_row, axis=1))

        collage_array = np.concatenate(collage_array[::-1], axis=0)
        return collage_array
    return (generate_cq_image_matrix,)


@app.cell
def _(
    FULL_WIDTH,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    datetime,
    generate_cq_image_matrix,
    os,
    plt,
):
    def plot_collage(image_type):
        # Get array:
        collage_array = generate_cq_image_matrix(6, image_type)

        # Set up and label plot:
        fig, ax = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH))
        ax.imshow(collage_array, extent=(-1.5, 1.5, -1.5, 1.5))
        ax.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        edging = 0.06
        fig.subplots_adjust(edging, edging, 1 - edging, 1 - edging)

        # Save:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, f"pc_{image_type}_plot.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()
    return (plot_collage,)


@app.cell
def _(plot_collage):
    plot_collage("trajectory")
    return


@app.cell
def _(plot_collage):
    plot_collage("stadia")
    return


@app.cell
def _(plot_collage):
    plot_collage("com_trajectory")
    return


@app.cell
def _(load_image, metrics_dict, np, plt):
    def get_metric_images(metric_index):
        metric = list(metrics_dict.values())[metric_index]
        metric = metric[~np.isnan(metric)]
        # metric = parameter_values[:, 11]
        metric_bins = np.linspace(np.quantile(metric, 0.1), np.quantile(metric, 0.9), 5)

        joint_array = []
        for bin_index in range(len(metric_bins) - 1):
            lower_bound = metric_bins[bin_index]
            upper_bound = metric_bins[bin_index + 1]
            run_indices = np.argwhere(np.logical_and(metric >= lower_bound, metric <= upper_bound))
            selected_index = run_indices[0][0]
            image_array = load_image(selected_index)
            joint_array.append(image_array)

        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.imshow(np.concatenate(joint_array, axis=1))
        ax.set_axis_off()
        plt.show()

    get_metric_images(0)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
