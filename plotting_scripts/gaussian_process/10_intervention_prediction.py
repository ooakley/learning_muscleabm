import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import json
    import subprocess

    import arviz_stats

    import pandas as pd
    import numpy as np
    import colorcet as cc

    import scipy.stats

    import matplotlib.pyplot as plt

    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.ticker import AutoLocator, MaxNLocator

    from datetime import datetime
    return datetime, json, np, os, plt, scipy, subprocess


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
        "script": "intervention_prediction.py",
        "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_commit_hash": commit_hash
    }
    return FULL_WIDTH, METADATA_DICTIONARY, OUT_DIRPATH, TEXT_WIDTH


@app.cell
def _():
    BASE_INTERVENTION = "#D55E00"
    GEE_INTERVENTION = "#009E73"
    return BASE_INTERVENTION, GEE_INTERVENTION


@app.cell
def _(np):
    # Load MLE data:
    ctl_chain = np.load("model_experiments/2026-05-31-collisions_shape/mcmc_results/wt_mcmc_chain.npy")
    rd_chain = np.load("model_experiments/2026-05-31-collisions_shape/mcmc_results/rd_mcmc_chain.npy")
    ctl_likelihoods = np.load("model_experiments/2026-05-31-collisions_shape/mcmc_results/wt_mcmc_likelihoods.npy")
    rd_likelihoods = np.load("model_experiments/2026-05-31-collisions_shape/mcmc_results/rd_mcmc_likelihoods.npy")

    THIN_FACTOR = 64
    ctl_mle_idx = np.argsort(ctl_likelihoods[::THIN_FACTOR, :, 0].flatten())[-1024:]
    rd_mle_idx = np.argsort(rd_likelihoods[::THIN_FACTOR, :, 0].flatten())[-1024:]

    ctl_mle = ctl_chain[::THIN_FACTOR, :, 0, :].reshape(-1, 11)[ctl_mle_idx, :]
    rd_mle = rd_chain[::THIN_FACTOR, :, 0, :].reshape(-1, 11)[rd_mle_idx, :]

    ctl_mle = np.concatenate([ctl_mle, np.ones((ctl_mle.shape[0], 3)) * 0.5], axis=1)
    rd_mle =  np.concatenate([rd_mle, np.ones((rd_mle.shape[0], 3)) * 0.5], axis=1)
    return (ctl_mle,)


@app.cell
def _(np):
    log_transforms = np.load("model_experiments/2026-06-03-matrix_shape/gaussian_process_models/op65/global_eigenparameter_estimation/log_transforms.npy")
    loss_histories = np.load("model_experiments/2026-06-03-matrix_shape/gaussian_process_models/op65/global_eigenparameter_estimation/loss_histories.npy")
    log_scale_factors = np.load("model_experiments/2026-06-03-matrix_shape/gaussian_process_models/op65/global_eigenparameter_estimation/scale_factors.npy")
    log_scale_factors = np.squeeze(log_scale_factors)

    def regularise_transforms(log_transforms, log_scale_factors):
        # Ensure we can make appropriate comparisons:
        unit_transforms = log_transforms / np.linalg.norm(log_transforms, axis=1, keepdims=True)
        mean_loss = np.mean(loss_histories[:, 4096:], axis=1)

        # Make base transform have the best plotting order:
        base_transform = np.copy(unit_transforms[np.argmin(mean_loss), :, :])
        base_transform = base_transform[:, [2, 0, 1]]
        base_transform[:, 1] *= -1
        base_transform[:, 2] *= -1

        # Set up collation:
        regularised_transforms = []
        regularised_scale_factors = []

        # Iterate through rest of dataset:
        for run_index in range(unit_transforms.shape[0]):
            run_transform = unit_transforms[run_index, :, :]
            cosine_matrix = base_transform.T @ run_transform

            arranged_transform = []
            arranged_sf = []
            for t_index in range(run_transform.shape[1]):
                choice_index = np.nanargmax(np.abs(cosine_matrix[t_index, :]))
                choice_sign = np.sign(cosine_matrix[t_index, choice_index])
                arranged_transform.append(run_transform[:, choice_index] * choice_sign)
                arranged_sf.append(log_scale_factors[run_index, choice_index])
                cosine_matrix[:, choice_index] = np.nan


            arranged_transform = np.stack(arranged_transform, axis=0)
            arranged_sf = np.stack(arranged_sf)
            assert(~np.any(np.isnan(arranged_transform)))
            regularised_transforms.append(arranged_transform)
            regularised_scale_factors.append(arranged_sf)

        return np.stack(regularised_transforms, axis=0), np.stack(regularised_scale_factors, axis=0)

    regularised_transforms, regularised_scale_factors = regularise_transforms(log_transforms, log_scale_factors)

    # Get mean loss:
    mean_loss = np.mean(loss_histories[:, 4096:], axis=1)
    gee_transform = np.copy(regularised_transforms[np.argmin(mean_loss), :, :]).T
    gee_transform /= np.linalg.norm(gee_transform, axis=0, keepdims=True)
    return (gee_transform,)


@app.cell
def _(ctl_mle, gee_transform, np):
    ctl_gee = np.log(ctl_mle) @ gee_transform
    return (ctl_gee,)


@app.cell
def _(np):
    base_intervention = np.load("model_experiments/2026-07-01-intervention-experiment/base_intervention.npz")["intervention"]
    conditioned_intervention = np.load("model_experiments/2026-07-01-intervention-experiment/gee_intervention.npz")["intervention"]
    return base_intervention, conditioned_intervention


@app.cell
def _(json, os):
    EXPERIMENT_DIRPATH = "model_experiments/2026-05-31-collisions_shape"

    with open(os.path.join(EXPERIMENT_DIRPATH, "config.json")) as json_file:
        config_dict = json.load(json_file)

    parameter_list = [parameter_range[0] for parameter_range in config_dict["gridsearch_parameters"]]
    return (parameter_list,)


@app.cell
def _(base_intervention):
    len(base_intervention)
    return


@app.cell
def _(base_intervention, np):
    print(np.round(np.exp(base_intervention), 2))
    return


@app.cell
def _(conditioned_intervention, np):
    print(np.round(np.exp(conditioned_intervention), 2))
    return


@app.cell
def _(parameter_list):
    parameter_list
    return


@app.cell
def _(gee_transform, np):
    intervention_inputs = np.load("model_experiments/2026-07-01-csm_posterior/sample_matrix.npy")
    base_int_inputs = intervention_inputs[:1024, :]
    base_gee_inputs = np.log(base_int_inputs) @ gee_transform
    cond_int_inputs = intervention_inputs[1024:, :]
    cond_gee_inputs = np.log(cond_int_inputs) @ gee_transform
    return base_gee_inputs, cond_gee_inputs


@app.cell
def _(np, scipy):
    def get_kde_grid(x, y, bin_count, clip):
        # Get meshgrid:
        x_edges = np.linspace(*np.quantile(x, [clip, 1 - clip]), bin_count+1)
        x_center = x_edges[:-1] + np.diff(x_edges)
        y_edges = np.linspace(*np.quantile(y, [clip, 1 - clip]), bin_count+1)
        y_center = y_edges[:-1] + np.diff(y_edges)

        # Estimate meshgrid on KDE:
        x_grid, y_grid = np.meshgrid(x_edges[:-1], y_edges[:-1])
        positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
        kernel = scipy.stats.gaussian_kde(np.vstack([x, y]), bw_method=0.5)
        density_estimate = kernel.evaluate(positions)

        # Return necessary information for contour plotting:
        return x_edges[:-1], y_edges[:-1], density_estimate
    return (get_kde_grid,)


@app.cell
def _(
    BASE_INTERVENTION,
    FULL_WIDTH,
    GEE_INTERVENTION,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    base_gee_inputs,
    cond_gee_inputs,
    ctl_gee,
    datetime,
    get_kde_grid,
    np,
    os,
    plt,
    scipy,
):
    def comparative_transform_kde_plot(ax, i, j, bin_count=50):
        # Plot control distribution:
        ctl_x, ctl_y, ctl_density = get_kde_grid(ctl_gee[:, i], ctl_gee[:, j], bin_count, 0.02)
        ax.contour(ctl_x, ctl_y, ctl_density.reshape(bin_count, bin_count), 5, colors="k", alpha=0.2)

        # Plot base intervention distribution:
        base_x, base_y, base_density = get_kde_grid(base_gee_inputs[:, i], base_gee_inputs[:, j], bin_count, 0.005)
        ax.contour(base_x, base_y, base_density.reshape(bin_count, bin_count), 4, alpha=0.5, colors=BASE_INTERVENTION)

        # Plot conditioned intervention distribution:
        cond_x, cond_y, cond_density = get_kde_grid(cond_gee_inputs[:, i], cond_gee_inputs[:, j], bin_count, 0.005)
        ax.contour(cond_x, cond_y, cond_density.reshape(bin_count, bin_count), 4, alpha=0.5, colors=GEE_INTERVENTION)

        # Label axes:
        ax.set_xlabel(f"$\\vartheta_{i + 1}$")
        ax.set_ylabel(f"$\\vartheta_{j + 1}$")


    def comparative_transform_histogram(ax, i):
        # Plot control KDE:
        ctl_kde = scipy.stats.gaussian_kde(ctl_gee[:, i])
        ctl_input = np.linspace(ctl_gee[:, i].min(), ctl_gee[:, i].max(), 50)
        ax.plot(ctl_input, ctl_kde(ctl_input), color='k', alpha=0.2, label="Control")

        # Plot histograms:
        bins = 30
        # ---> Plot base transform:
        ax.hist(
            base_gee_inputs[:, i], bins=bins,
            alpha=0.75,
            histtype="step", density=True, label="Base",
            color=BASE_INTERVENTION
        )
        # ---> Plot conditioned transform:
        ax.hist(
            cond_gee_inputs[~np.isnan(cond_gee_inputs[:, i]), i], bins=bins,
            alpha=0.75,
            histtype="step", density=True, label="$\\vartheta$ Conditioned",
            color=GEE_INTERVENTION
        )

        ax.set_xlabel(f"$\\vartheta_{i + 1}$")
        ax.set_ylim(0, None)

        if i == 0:
            ax.legend()


    def comparative_transform_scattergrid():
        fig, axs = plt.subplots(3, 3, figsize=(FULL_WIDTH, FULL_WIDTH))
        for i in range(3):
            for j in range(3):
                if i == j:
                    comparative_transform_histogram(axs[i, j], i)
                else:
                    comparative_transform_kde_plot(axs[i, j], i, j, 50)

        # Adjust subplot configuration:
        clip = 0.05
        fig.subplots_adjust(clip, clip, 1 - clip, 1 - clip, wspace=0.25, hspace=0.25)

        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "transform_comparative_plot.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()


    comparative_transform_scattergrid()
    return


@app.cell
def _(np, os):
    MLE_DIRPATH = os.path.join("model_experiments", "2026-06-12-csm_posterior")
    mle_ops = np.load(os.path.join(MLE_DIRPATH, "summary_data", "matrix_order_parameters.npy"))
    mle_ops = np.mean(mle_ops[:, :, 2], axis=1)
    mle_ops.shape

    BASE_DIRPATH = os.path.join("model_experiments", "2026-07-01-csm_posterior")
    INTERVENTION_DIRPATH = os.path.join("model_experiments", "2026-07-01-csm_posterior")
    adjusted_ops = np.load(os.path.join(INTERVENTION_DIRPATH, "summary_data", "matrix_order_parameters.npy"))
    adjusted_ops = np.mean(adjusted_ops[:, :, 2
        ], axis=1)
    adjusted_ops.shape
    return INTERVENTION_DIRPATH, MLE_DIRPATH, adjusted_ops, mle_ops


@app.cell
def _(INTERVENTION_DIRPATH, MLE_DIRPATH, np, os):
    LOAD_METRICS = [
        "speeds",
        "meander_ratios",
        "ann_indices",
        "coherency",
        "cell_lengths",
        "density_idr",
        "order_parameters",
        "interaction"
    ]

    def load_metrics(dirpath):
        metric_list = []
        for metric_name in LOAD_METRICS:
            adjusted_metric = np.load(os.path.join(dirpath, "summary_data", f"{metric_name}.npy"))
            adjusted_metric = np.mean(adjusted_metric, axis=1)
            metric_list.append(adjusted_metric)
        return metric_list

    mle_metrics = load_metrics(MLE_DIRPATH)
    intervention_metrics = load_metrics(INTERVENTION_DIRPATH)
    return intervention_metrics, mle_metrics


@app.cell
def _(adjusted_ops, mle_ops, scipy):
    scipy.stats.ttest_ind(mle_ops[:1024], adjusted_ops[1024:]).pvalue
    return


@app.cell
def _(
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    TEXT_WIDTH,
    adjusted_ops,
    datetime,
    mle_ops,
    os,
    plt,
    scipy,
):
    def plot_order_parameter_comparison():
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.75))
        data = [mle_ops[:1024], mle_ops[1024:], adjusted_ops[:1024], adjusted_ops[1024:]]
        ax.boxplot(data, widths=0.4);
        ax.set_xticks([1, 2, 3, 4], ["Control", "RD", "Base Transform", "Conditioned Transform"])

        # Plot error bars:
        heights = [0.11, 0.125, 0.14]

        # CTL - RD:
        ax.plot([1, 2], [heights[0], heights[0]], c='k')
        ctl_rd_pval = scipy.stats.ttest_ind(mle_ops[:1024], mle_ops[1024:]).pvalue
        print(ctl_rd_pval)
        ax.text(1.5, heights[0] + 0.004, "p $<$ 0.001", ha="center")

        # CTL - Base:
        ax.plot([1, 3], [heights[1], heights[1]], c='k')
        ctl_base_pval = scipy.stats.ttest_ind(mle_ops[:1024], adjusted_ops[:1024]).pvalue
        ax.text(2, heights[1] + 0.004, f"ns, p={ctl_base_pval:.2f}", ha="center")

        # CTL - Conditioned:
        ax.plot([1, 4], [heights[2], heights[2]], c='k')
        ctl_cond_pval = scipy.stats.ttest_ind(mle_ops[:1024], adjusted_ops[1024:]).pvalue
        ax.text(2.5, heights[2] + 0.004, f"ns, p={ctl_cond_pval:.2f}", ha="center")

        # Label plot
        ax.set_ylim(None, 0.155)
        ax.set_ylabel("$S_{64}$")

        # Save figures:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "transform_op_boxplot_comparison.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_order_parameter_comparison()
    return


@app.cell
def _(intervention_metrics, mle_metrics, np, plt):
    def plot_metric_differences():
        fig, axs = plt.subplots(4, figsize=(3.5, 9), sharex=True)
        for i in range(4): 
            base_difference = np.abs(np.mean(mle_metrics[i][:1024]) - np.mean(intervention_metrics[i][:1024])) \
                / np.mean(mle_metrics[i][:1024])
            base_difference = np.round(base_difference, 3)
            gee_difference = np.abs(np.mean(mle_metrics[i][:1024]) - np.mean(intervention_metrics[i][1024:])) \
                / np.mean(mle_metrics[i][:1024])
            gee_difference = np.round(gee_difference, 3)
            # base_difference = scipy.stats.wasserstein_distance(mle_metrics[i][:1024], intervention_metrics[i][:1024])
            # gee_difference = scipy.stats.wasserstein_distance(mle_metrics[i][:1024], intervention_metrics[i][1024:])
            print(base_difference, gee_difference)
            axs[i].boxplot([mle_metrics[i][:1024], intervention_metrics[i][:1024], intervention_metrics[i][1024:]])
        plt.show()

    plot_metric_differences()
    return


@app.cell
def _(np, os):
    import imageio.v3 as iio

    def load_image(index, image_type):
        npz_archive = np.load(os.path.join("model_experiments/2026-06-03-matrix_shape", f"{image_type}.npz"))
        image_array = npz_archive[str(index)]
        return image_array
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
