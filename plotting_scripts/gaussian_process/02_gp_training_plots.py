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

    import matplotlib.pyplot as plt

    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.ticker import AutoLocator, MaxNLocator

    from datetime import datetime
    return MaxNLocator, cc, datetime, json, np, os, pd, plt, scipy, subprocess


@app.cell
def _():
    import matplotlib as mpl

    # Font formatting:
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = "cmr10"
    mpl.rcParams['font.size'] = 8
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
        "script": "gp_training_plots.py",
        "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_commit_hash": commit_hash
    }
    return (
        FULL_HEIGHT,
        FULL_WIDTH,
        METADATA_DICTIONARY,
        OUT_DIRPATH,
        TEXT_WIDTH,
    )


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

    EXPERIMENT_DIRPATH = "model_experiments/2026-05-20-collisions_shape"
    return EXPERIMENT_DIRPATH, METRICS_LABELS, METRICS_TO_PLOT


@app.cell
def _(EXPERIMENT_DIRPATH, METRICS_TO_PLOT, json, np, os):
    # Load in gridsearch data:
    parameter_values = np.load(os.path.join(EXPERIMENT_DIRPATH, "sample_matrix.npy"))
    with open(os.path.join(EXPERIMENT_DIRPATH, "config.json")) as json_file:
        config_dict = json.load(json_file)

    # Get metrics:
    def load_metrics():
        metrics_dict = {}
        noise_dict = {}
        for metric_name in METRICS_TO_PLOT:
            metric_array = np.load(os.path.join(
                EXPERIMENT_DIRPATH, "summary_data", f"{metric_name}.npy"
            ))
            metrics_dict[metric_name] = np.nanmean(metric_array, axis=1)
            noise_dict[metric_name] = np.nanstd(metric_array, axis=1)
        return metrics_dict, noise_dict

    # Load predictions:
    def load_predictions():
        preds_dict = {}
        for metric_name in METRICS_TO_PLOT:
            preds_array = np.load(os.path.join(
                EXPERIMENT_DIRPATH, "gaussian_process_models", f"{metric_name}", "parameter_predictions.npy"
            ))
            preds_dict[metric_name] = preds_array
        return preds_dict

    metrics_dict, noise_dict = load_metrics()
    preds_dict = load_predictions()
    return metrics_dict, noise_dict, preds_dict


@app.cell
def _(
    FULL_HEIGHT,
    FULL_WIDTH,
    METADATA_DICTIONARY,
    METRICS_LABELS,
    MaxNLocator,
    OUT_DIRPATH,
    cc,
    datetime,
    metrics_dict,
    noise_dict,
    np,
    os,
    plt,
    preds_dict,
    scipy,
):
    def plot_predictions(metric_index, ax):
        # Set up plot:
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))

        x = list(metrics_dict.values())[metric_index]
        x = x[~np.isnan(x)]
        y = list(preds_dict.values())[metric_index][:, 0]
        full_dataset = np.stack([x, y], axis=1)

        # Estimate density for scatter plot colouring:
        rng = np.random.default_rng(0)
        random_indices = rng.choice(full_dataset.shape[0], size=(1024))
        density = scipy.stats.gaussian_kde(full_dataset[random_indices, :].T)(full_dataset.T)
        density_sort = np.argsort(density)
        ax.scatter(
            full_dataset[density_sort, 0],
            full_dataset[density_sort, 1],
            c=density[density_sort],
            s=1, alpha=0.25, cmap=cc.m_CET_L20
        )
        # Plot linear guideline:
        limits = [np.min(full_dataset), np.max(full_dataset)]
        ax.plot(limits, limits, c='r', ls='--')

        # Format axes:
        ax.set_xlim(*limits)
        ax.set_xlabel(f"True {METRICS_LABELS[metric_index]}")
        ax.set_ylim(*limits)
        ax.set_ylabel(f"Predicted {METRICS_LABELS[metric_index]}")
        ax.set_aspect("equal")


    def plot_noise_predictions(metric_index, ax):
        # Set up plot:
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Get data:
        test_metric = list(metrics_dict.values())[metric_index]
        test_std = np.nanstd(test_metric)

        # Retrieve sample estimate of the SEM:
        x = list(noise_dict.values())[metric_index]
        x = x[~np.isnan(x)]
        x /= np.sqrt(16)  # The superiteration count, and therefore sample count.

        # x = x[~np.isnan(x)] 
        # x = np.sqrt(x)
        # x *= test_std
        y = list(preds_dict.values())[metric_index][:, 1]
        y /= test_std
        y = np.sqrt(y)
        y *= test_std

        full_dataset = np.stack([x, y], axis=1) + 1e-5

        # Estimate density for scatter plot colouring:
        rng = np.random.default_rng(0)
        random_indices = rng.choice(full_dataset.shape[0], size=(1024))
        density = scipy.stats.gaussian_kde(full_dataset[random_indices, :].T)(full_dataset.T)
        density_sort = np.argsort(density)
        ax.scatter(
            full_dataset[density_sort, 0],
            full_dataset[density_sort, 1],
            c=density[density_sort],
            s=1, alpha=0.25, cmap=cc.m_CET_L20
        )
        # Plot linear guideline:
        limits = [np.min(full_dataset), np.max(full_dataset)]
        ax.plot(limits, limits, c='r', ls='--')

        # Format axes:
        ax.set_xlim(*limits)
        ax.set_xlabel(f"Sample {METRICS_LABELS[metric_index]} SEM")
        ax.set_ylim(*limits)
        ax.set_ylabel(f"Predicted {METRICS_LABELS[metric_index]} $\sigma$")
        ax.set_aspect("equal")


    # def plot_all_noise():
    #     fig, axs = plt.subplots(2, 2, figsize=(5, 5), layout="constrained")

    #     count = 0
    #     for i in range(2):
    #         for j in range(2):
    #             plot_noise_predictions(count, axs[i, j])
    #             count += 1

    #     # Update metadata time:
    #     METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     plt.savefig(
    #         os.path.join(OUT_DIRPATH, f"gp_noise.png"),
    #         dpi=300, metadata=METADATA_DICTIONARY, transparent=True
    #     )
    #     plt.show()

    # plot_all_noise()


    def plot_all_predictions():
        fig, axs = plt.subplots(4, 2, figsize=(FULL_WIDTH * 0.6, FULL_HEIGHT))

        # Plot predictions:
        for row_index in range(4):
            plot_predictions(row_index, axs[row_index, 0])
            plot_noise_predictions(row_index, axs[row_index, 1])

        # Adjust subplots:
        fig.subplots_adjust(0.0, 0.05, 1.0, 0.975, wspace=0.0, hspace=0.3)

        # Update metadata time:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, f"gp_predictions.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_all_predictions()
    return


@app.cell
def _(EXPERIMENT_DIRPATH, METRICS_LABELS, METRICS_TO_PLOT, json, np, os, pd):
    def collate_cross_validation_metrics():
        dataframe = []
        for metric_index, metric_name in enumerate(METRICS_TO_PLOT):
            # Load CV data:
            json_filepath = os.path.join(
                EXPERIMENT_DIRPATH, "gaussian_process_models", f"{metric_name}", "cv_metrics.json"
            )
            with open(json_filepath) as json_file:
                cv_dict = json.load(json_file)

            # Process CV data:
            cv_data = {}
            cv_data["Metric"] = METRICS_LABELS[metric_index]
            cv_data["MAE"] = np.mean(cv_dict["mae"])
            cv_data["MSE"] = np.mean(cv_dict["mse"])
            cv_data["SLL"] = np.mean(cv_dict["sll"])
            dataframe.append(cv_data)

        return pd.DataFrame(dataframe)

    cv_dataframe = collate_cross_validation_metrics()
    return (cv_dataframe,)


@app.cell
def _(cv_dataframe):
    print(cv_dataframe.to_latex(index=False))
    return


@app.cell
def _(
    EXPERIMENT_DIRPATH,
    FULL_HEIGHT,
    METADATA_DICTIONARY,
    METRICS_LABELS,
    METRICS_TO_PLOT,
    MaxNLocator,
    OUT_DIRPATH,
    TEXT_WIDTH,
    datetime,
    metrics_dict,
    np,
    os,
    plt,
):
    def plot_training_loss(metric_index, ax):
        # Load loss:
        loss_history = np.load(
            os.path.join(
                EXPERIMENT_DIRPATH, "gaussian_process_models",
                METRICS_TO_PLOT[metric_index], "loss_history.npy"
            )
        )

        # Plot data:
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.set_xlabel("Training Epoch")
        ax.set_ylabel(f"{METRICS_LABELS[metric_index]} PLL")
        ax.set_xlim(0, 15)
        dataset_size = np.count_nonzero(~np.isnan(list(metrics_dict.values())[metric_index]))
        batches_per_epoch = dataset_size / 512
        fractional_epochs = np.arange(len(loss_history)) / batches_per_epoch
        ax.plot(fractional_epochs, loss_history, lw=0.25)


    def plot_all_losses():
        fig, axs = plt.subplots(4, 1, figsize=(TEXT_WIDTH * 0.6, FULL_HEIGHT * 0.75), sharey=True)

        for i in range(4):
            plot_training_loss(i, axs[i])

        fig.subplots_adjust(0.15, 0.075, 0.85, 0.925, wspace=0.0, hspace=0.45)
        # Update metadata time:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, f"gp_loss_history.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_all_losses()
    return


@app.cell
def _(
    METADATA_DICTIONARY,
    METRICS_LABELS,
    MaxNLocator,
    OUT_DIRPATH,
    cc,
    datetime,
    metrics_dict,
    noise_dict,
    np,
    os,
    plt,
    preds_dict,
    scipy,
):
    def plot_noise_predictions(metric_index, ax):
        # Set up plot:
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Get data:
        test_metric = list(metrics_dict.values())[metric_index]
        test_std = np.nanstd(test_metric)

        # Retrieve sample estimate of the SEM:
        x = list(noise_dict.values())[metric_index]
        x = x[~np.isnan(x)]
        x /= np.sqrt(16)  # The superiteration count, and therefore sample count.

        # x = x[~np.isnan(x)] 
        # x = np.sqrt(x)
        # x *= test_std
        y = list(preds_dict.values())[metric_index][:, 1]
        y /= test_std
        y = np.sqrt(y)
        y *= test_std

        full_dataset = np.stack([x, y], axis=1) + 1e-5

        # Estimate density for scatter plot colouring:
        rng = np.random.default_rng(0)
        random_indices = rng.choice(full_dataset.shape[0], size=(1024))
        density = scipy.stats.gaussian_kde(full_dataset[random_indices, :].T)(full_dataset.T)
        density_sort = np.argsort(density)
        ax.scatter(
            full_dataset[density_sort, 0],
            full_dataset[density_sort, 1],
            c=density[density_sort],
            s=1, alpha=0.25, cmap=cc.m_CET_L20
        )
        # Plot linear guideline:
        limits = [np.min(full_dataset), np.max(full_dataset)]
        ax.plot(limits, limits, c='r', ls='--')

        # Format axes:
        ax.set_xlim(*limits)
        ax.set_xlabel(f"Sample {METRICS_LABELS[metric_index]} SEM")
        ax.set_ylim(*limits)
        ax.set_ylabel(f"Predicted {METRICS_LABELS[metric_index]} $\sigma$")
        ax.set_aspect("equal")


    def plot_all_noise():
        fig, axs = plt.subplots(2, 2, figsize=(5, 5), layout="constrained")

        count = 0
        for i in range(2):
            for j in range(2):
                plot_noise_predictions(count, axs[i, j])
                count += 1

        # Update metadata time:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, f"gp_noise.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_all_noise()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
