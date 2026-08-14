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
    return MaxNLocator, cc, datetime, json, np, os, plt, scipy, subprocess


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
        "script": "compare_mcmc_fits.py",
        "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_commit_hash": commit_hash
    }
    return (
        CONTROL_PALETTE,
        FULL_HEIGHT,
        FULL_WIDTH,
        METADATA_DICTIONARY,
        OUT_DIRPATH,
        RD_PALETTE,
        TEXT_WIDTH,
    )


@app.cell
def _(np, os):
    EXPERIMENT_DIRPATH = "model_experiments/2026-05-31-collisions_shape"
    mcmc_results = "wide_mcmc_results"
    ctl_mcmc_chain = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "wt_mcmc_chain.npy"))
    rd_mcmc_chain = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "rd_mcmc_chain.npy"))
    return EXPERIMENT_DIRPATH, ctl_mcmc_chain, mcmc_results, rd_mcmc_chain


@app.cell
def _(ctl_mcmc_chain, mcmc_results, np, rd_mcmc_chain):
    ctl_likelihoods = np.load(f"model_experiments/2026-05-31-collisions_shape/{mcmc_results}/wt_mcmc_likelihoods.npy")
    rd_likelihoods = np.load(f"model_experiments/2026-05-31-collisions_shape/{mcmc_results}/rd_mcmc_likelihoods.npy")

    THIN_FACTOR = 64

    ctl_mle_idx = np.argsort(ctl_likelihoods[2048::THIN_FACTOR, :, 0].flatten())[-1024:]
    rd_mle_idx = np.argsort(rd_likelihoods[2048::THIN_FACTOR, :, 0].flatten())[-1024:]

    ctl_mle = ctl_mcmc_chain[2048::THIN_FACTOR, :, 0, :].reshape(-1, 11)[ctl_mle_idx, :]
    rd_mle = rd_mcmc_chain[2048::THIN_FACTOR, :, 0, :].reshape(-1, 11)[rd_mle_idx, :]
    return THIN_FACTOR, ctl_likelihoods, ctl_mle, rd_likelihoods, rd_mle


@app.cell
def _(
    THIN_FACTOR,
    ctl_likelihoods,
    ctl_mcmc_chain,
    np,
    rd_likelihoods,
    rd_mcmc_chain,
):
    distribution_size = len(ctl_likelihoods[2048::THIN_FACTOR, :, 0].flatten())
    cutoff_95 = int(np.floor(distribution_size * 0.95))

    ctl_ci95_idx = np.argsort(ctl_likelihoods[2048::THIN_FACTOR, :, 0].flatten())[-cutoff_95:]
    rd_ci95_idx = np.argsort(rd_likelihoods[2048::THIN_FACTOR, :, 0].flatten())[-cutoff_95:]

    ctl_ci95 = ctl_mcmc_chain[2048::THIN_FACTOR, :, 0, :].reshape(-1, 11)[ctl_ci95_idx, :]
    rd_ci95 = rd_mcmc_chain[2048::THIN_FACTOR, :, 0, :].reshape(-1, 11)[rd_ci95_idx, :]
    return ctl_ci95, rd_ci95


@app.cell
def _(ctl_likelihoods):
    print(ctl_likelihoods[::2048, :, 0].flatten().shape)
    return


@app.cell
def _(ctl_mle, np):
    print(ctl_mle.shape)
    print(np.unique(ctl_mle, axis=0).shape)
    return


@app.cell
def _(EXPERIMENT_DIRPATH, json, os):
    with open(os.path.join(EXPERIMENT_DIRPATH, "config.json")) as json_file:
        config_dict = json.load(json_file)

    parameter_list = [parameter_range[0] for parameter_range in config_dict["gridsearch_parameters"]]
    parameter_list.remove("numberOfCells")

    parameter_scaling = [parameter_range[1] for parameter_range in config_dict["gridsearch_parameters"]]
    parameter_scaling.pop(5);
    return parameter_list, parameter_scaling


@app.cell
def _(
    FULL_WIDTH,
    METADATA_DICTIONARY,
    cc,
    datetime,
    np,
    parameter_list,
    plt,
    scipy,
):
    def plot_mc_distribution(i, j, kde, ax):
        # Get mesh over which to evaluate KDE:
        points = np.arange(0.01, 1.0, 0.02)
        X, Y = np.meshgrid(points, points)
        eval_points = np.stack([X.ravel(), Y.ravel()])
        marginal_kde = kde.marginal([i, j])
        densities = marginal_kde(eval_points)
        densities = np.reshape(densities, (50, 50))
        densities = densities / np.sum(densities)

        # Get effective display range:
        stddev = np.std(densities)
        uniform_density = 1 / 50**2
        vmin = uniform_density * 0
        vmax = uniform_density * 3

        map = np.unravel_index(np.argmax(densities), shape=densities.shape)
        map_y = points[map[0]]
        map_x = points[map[1]]

        # Plot sample:
        ax.imshow(
            densities,
            extent=[0, 1, 0, 1],
            vmin=vmin, vmax=vmax,
            cmap=cc.m_CET_R3, origin="lower"
        )
        # ax.scatter(mle[:, i], mle[:, j], s=10/3, c='tab:blue')
        # ax.scatter(estimates[:, i], estimates[:, j], s=10/3, c='tab:blue')
        ax.scatter(map_x, map_y, s=10, c='k')
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()


    def plot_scatter_density(i, j, kde, mle, posterior_distribution, color, ax):
        # Get evaluate KDE at posterior samples:
        marginal_kde = kde.marginal([i, j])
        densities = marginal_kde(posterior_distribution[:, [i, j]].T)
        density_sort = np.argsort(densities)

        # Plot sample:
        ax.scatter(
            posterior_distribution[density_sort, i],
            posterior_distribution[density_sort, j],
            s=0.5, c=densities[density_sort],
            cmap=cc.m_CET_L20, edgecolors="none"
        )
        ax.scatter(mle[:, i], mle[:, j], s=0.75, c="#F92A53", alpha=0.5, edgecolors="none")
        ax.set_aspect("equal")

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.25)
            ax.spines[axis].set_color(color)

        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_axis_off()


    def global_jointplot(posterior_distribution, mle, title, image_filename, color):
        # Calculate global KDE:
        kde = scipy.stats.gaussian_kde(posterior_distribution.T, bw_method=0.33)

        # Set up subplots:
        fig, axs = plt.subplots(11, 11, figsize=(FULL_WIDTH, FULL_WIDTH), sharex=True, sharey=True)

        # Iterate through indices:
        for i in range(11):
            for j in range(11):
                # Label diagonals:
                if i == j:
                    axs[i, j].text(
                        0.98, 0.98, parameter_list[i],
                        c=color, fontsize=6,
                        ha='right', va='top',
                        rotation='vertical',
                        in_layout=False,
                        transform=axs[i, j].transAxes
                    )
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
                    axs[i, j].set_xlim(0, 1)
                    axs[i, j].set_ylim(0, 1)
                    axs[i, j].set_aspect("equal")
                    axs[i, j].set_axis_off()
                    continue

                # Set off diagonals blank:
                if j < i:
                    axs[i, j].remove()
                    continue

                # Plot density scatter:
                plot_scatter_density(i, j, kde, mle, posterior_distribution, color, axs[i, j])

        edging = 0.01
        fig.subplots_adjust(edging, edging, 1 - edging, 1 - edging, wspace=0.025, hspace=0.025)
        fig.text(0.01, 0.01, title, ha="left", va="baseline", fontsize=12, style="italic", color=color)

        # Show in cell:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # plt.savefig(
        #     os.path.join(OUT_DIRPATH, f"{image_filename}.png"),
        #     dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        # )
        plt.show()
    return (global_jointplot,)


@app.cell
def _(ctl_mcmc_chain, rd_mcmc_chain):
    ctl_kde_posterior = ctl_mcmc_chain[2048::1024, :, 0, :].reshape(-1, 11)
    rd_kde_posterior = rd_mcmc_chain[2048::1024, :, 0, :].reshape(-1, 11)
    return ctl_kde_posterior, rd_kde_posterior


@app.cell
def _(ctl_kde_posterior):
    print(ctl_kde_posterior.shape)
    return


@app.cell
def _(ctl_mcmc_chain, np, rd_mcmc_chain):
    ctl_covariance = np.cov(ctl_mcmc_chain[2048:, :, 0, :].reshape(-1, 11), rowvar=False)
    ctl_precision = np.linalg.inv(ctl_covariance)

    rd_covariance = np.cov(rd_mcmc_chain[2048:, :, 0, :].reshape(-1, 11), rowvar=False)
    rd_precision = np.linalg.inv(rd_covariance)
    return ctl_precision, rd_covariance, rd_precision


@app.cell
def _(ctl_precision, np, rd_covariance):
    joint_eigvals, joint_eigenparameters = np.linalg.eigh(ctl_precision @ rd_covariance)
    joint_eigvals = joint_eigvals[::-1]
    joint_eigenparameters = joint_eigenparameters.T[::-1, :]
    return


@app.cell
def _(ctl_precision, np, rd_precision):
    ctl_eigvals, ctl_eigenparameters = np.linalg.eigh(ctl_precision)
    ctl_eigvals = ctl_eigvals[::-1]
    ctl_eigenparameters = ctl_eigenparameters.T[::-1, :]

    rd_eigvals, rd_eigenparameters = np.linalg.eigh(rd_precision)
    rd_eigvals = rd_eigvals[::-1]
    rd_eigenparameters = rd_eigenparameters.T[::-1, :]
    return ctl_eigenparameters, ctl_eigvals


@app.cell
def _(ctl_eigvals):
    ctl_eigvals
    return


@app.cell
def _(ctl_eigenparameters, np):
    simplified_eigenparameters = np.copy(ctl_eigenparameters[-1, :])
    simplified_eigenparameters[np.abs(simplified_eigenparameters) < 0.1] = 0
    print(np.round(simplified_eigenparameters, 2))
    return


@app.cell
def _(ctl_eigenparameters, ctl_kde_posterior, rd_kde_posterior):
    ctl_reparams = ctl_kde_posterior @ ctl_eigenparameters
    rd_reparams = rd_kde_posterior @ ctl_eigenparameters
    return ctl_reparams, rd_reparams


@app.cell
def _(ctl_reparams, plt, rd_reparams):
    plt.hist(ctl_reparams[:, 0], bins=25);
    plt.hist(rd_reparams[:, 0], bins=25);
    plt.show()
    return


@app.cell
def _(CONTROL_PALETTE, ctl_kde_posterior, ctl_mle, global_jointplot):
    global_jointplot(ctl_kde_posterior, ctl_mle, "Control Fit Parameter Distribution", "scattergrid_control", CONTROL_PALETTE)
    return


@app.cell
def _(RD_PALETTE, global_jointplot, rd_kde_posterior, rd_mle):
    global_jointplot(rd_kde_posterior, rd_mle, "RD Fit Parameter Distribution", "scattergrid_rd", RD_PALETTE)
    return


@app.cell
def _(
    CONTROL_PALETTE,
    FULL_HEIGHT,
    FULL_WIDTH,
    METADATA_DICTIONARY,
    MaxNLocator,
    RD_PALETTE,
    ctl_ci95,
    ctl_mcmc_chain,
    datetime,
    np,
    parameter_list,
    parameter_scaling,
    plt,
    rd_ci95,
    rd_mcmc_chain,
):
    ctl_hist_posterior = ctl_mcmc_chain[8192:, :, 0, :].reshape(-1, 11)
    rd_hist_posterior = rd_mcmc_chain[8192:, :, 0, :].reshape(-1, 11)

    ctl_hist_posterior = ctl_ci95
    rd_hist_posterior = rd_ci95

    def scale_parameter(x, index):
        scale_min, scale_max = parameter_scaling[index]
        return (x * (scale_max - scale_min)) + scale_min

    def plot_comparative_hist(index, ax):
        bins = np.linspace(parameter_scaling[index][0], parameter_scaling[index][1], 40)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.hist(
            scale_parameter(ctl_hist_posterior[:, index], index),
            alpha=0.75, label="Control Fit",
            color=CONTROL_PALETTE, histtype="step", bins=bins, density=True
        )
        ax.hist(
            scale_parameter(rd_hist_posterior[:, index], index),
            alpha=0.75, label="RD Fit",
            color=RD_PALETTE, histtype="step", bins=bins, density=True
        )
        ax.set_xlim(*parameter_scaling[index])
        ax.set_xlabel(parameter_list[index])

    def plot_all_marginals():
        fig, axs = plt.subplots(6, 2, figsize=(FULL_WIDTH, FULL_HEIGHT))

        count = 0
        for i in range(6):
            for j in range(2):
                if count == 11:
                    axs[i, j].remove()
                    continue
                plot_comparative_hist(count, axs[i, j])
                if count == 1:
                    axs[i, j].legend(fontsize=6)
                count += 1

        fig.subplots_adjust(0.15, 0.05, 0.85, 0.95, wspace=0.15, hspace=0.55)

        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # plt.savefig(
        #     os.path.join(OUT_DIRPATH, "marginal_fit_comparison.png"),
        #     dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        # )
        plt.show()

    plot_all_marginals()
    return (scale_parameter,)


@app.cell
def _(
    CONTROL_PALETTE,
    FULL_HEIGHT,
    FULL_WIDTH,
    METADATA_DICTIONARY,
    MaxNLocator,
    RD_PALETTE,
    ctl_mle,
    datetime,
    np,
    parameter_list,
    parameter_scaling,
    plt,
    rd_mle,
    scale_parameter,
):
    def plot_mle_hist(index, ax):
        bins = np.linspace(parameter_scaling[index][0], parameter_scaling[index][1], 30)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.hist(
            scale_parameter(ctl_mle[:, index], index),
            alpha=0.75, label="Control Fit",
            color=CONTROL_PALETTE, histtype="step", density=True, bins=bins
        )
        ax.hist(
            scale_parameter(rd_mle[:, index], index),
            alpha=0.75, label="RD Fit",
            color=RD_PALETTE, histtype="step", density=True, bins=bins
        )
        ax.set_xlim(*parameter_scaling[index])
        ax.set_xlabel(parameter_list[index])

    def plot_mle_marginals():
        fig, axs = plt.subplots(6, 2, figsize=(FULL_WIDTH, FULL_HEIGHT))

        count = 0
        for i in range(6):
            for j in range(2):
                if count == 11:
                    axs[i, j].remove()
                    continue
                plot_mle_hist(count, axs[i, j])
                if count == 0:
                    axs[i, j].legend(fontsize=6)
                count += 1

        fig.subplots_adjust(0.15, 0.05, 0.85, 0.95, wspace=0.15, hspace=0.55)
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # plt.savefig(
        #     os.path.join(OUT_DIRPATH, "marginal_fit_mle_comparison.png"),
        #     dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        # )
        plt.show()

    plot_mle_marginals()
    return


@app.cell
def _(
    CONTROL_PALETTE,
    FULL_HEIGHT,
    FULL_WIDTH,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    RD_PALETTE,
    ctl_mle,
    datetime,
    np,
    os,
    parameter_list,
    plt,
    rd_mle,
    scale_parameter,
):
    def plot_violin(index, ax):
        # Plot violins:
        scaled_data = [
            scale_parameter(ctl_mle[:, index], index),
            scale_parameter(rd_mle[:, index], index)
        ]
        xy_data = np.stack(scaled_data, axis=1)
        violin_parts = ax.violinplot(
            xy_data,
            positions=[0, 1],
            widths=0.8,
            bw_method="silverman",
            showextrema=False
        )

        for part_index, violin_part in enumerate(violin_parts['bodies']):
            violin_part.set_facecolor([CONTROL_PALETTE, RD_PALETTE][part_index])
            violin_part.set_edgecolor([CONTROL_PALETTE, RD_PALETTE][part_index])
            violin_part.set_alpha(0.5)

        # Plot internal boxes:
        low_quartiles, medians, high_quartiles = np.percentile(xy_data, [25, 50, 75], axis=0)
        for p_index in range(2):
            p_color = [CONTROL_PALETTE, RD_PALETTE][p_index]
            ax.scatter(p_index, medians[p_index], marker='o', color=p_color, s=10)
            ax.vlines(
                p_index,
                low_quartiles[p_index], high_quartiles[p_index],
                color=p_color, linestyle='-', lw=1
            )

        # Label xticks:
        ax.set_xticks([0, 1], ["CTL", "RD"])
        ax.set_ylabel(parameter_list[index])


    def plot_mle_violins():
        fig, axs = plt.subplots(3, 4, figsize=(FULL_WIDTH, FULL_HEIGHT), sharex=True)

        count = 0
        for i in range(3):
            for j in range(4):
                # Remove corner plot:
                if count == 11:
                    axs[i, j].set_axis_off()
                    continue

                # Plot violins:
                plot_violin(count, axs[i, j])
                count += 1

        fig.subplots_adjust(0.075, 0.025, 0.925, 0.975, wspace=0.5, hspace=0.075)

        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "marginal_fit_mle_violins.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_mle_violins()
    return


@app.cell
def _():
    # Python script; run matrix inference on fits:
    return


@app.cell
def _(EXPERIMENT_DIRPATH, np, os):
    matrix_inference_dirpath = os.path.join(EXPERIMENT_DIRPATH, "mcmc_results", "matrix_inference")
    wt_matrix_inference = np.load(os.path.join(matrix_inference_dirpath, "wt_mle_array.npy"))
    rd_matrix_inference = np.load(os.path.join(matrix_inference_dirpath, "rd_mle_array.npy"))
    return rd_matrix_inference, wt_matrix_inference


@app.cell
def _(np):
    advection_rates = np.linspace(0, 3, 50)
    sample_rates = np.linspace(0.5, 15, 10)
    cell_counts = np.linspace(300, 400, 3)
    return advection_rates, cell_counts


@app.cell
def _(wt_matrix_inference):
    wt_matrix_inference.shape
    return


@app.cell
def _(
    CONTROL_PALETTE,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    RD_PALETTE,
    TEXT_WIDTH,
    advection_rates,
    cell_counts,
    datetime,
    np,
    os,
    plt,
    rd_matrix_inference,
    wt_matrix_inference,
):
    def plot_short_inference():
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.5), sharex=True, sharey=True)

        cell_count_index = 2
        sr_index = 5

        # WT response:
        wt_line, = ax.plot(
            advection_rates,
            wt_matrix_inference[cell_count_index, sr_index, :],
            alpha=0.8, color=CONTROL_PALETTE
        )
        rd_line, = ax.plot(
            advection_rates,
            rd_matrix_inference[cell_count_index, sr_index, :],
            alpha=0.8, color=RD_PALETTE
        )

        # Fill in with full confidence interval:
        wt_plot = wt_matrix_inference[cell_count_index, sr_index, :]
        wt_stddev = np.std(wt_matrix_inference[cell_count_index, sr_index, :])
        wt_upper_bound = wt_plot + (1.96 * wt_stddev)
        wt_lower_bound = wt_plot - (1.96 * wt_stddev)
        ax.plot(advection_rates, wt_upper_bound, ls='--', alpha=0.5, color=CONTROL_PALETTE)
        ax.plot(advection_rates, wt_lower_bound, ls='--', alpha=0.5, color=CONTROL_PALETTE)
        ax.fill_between(
            advection_rates, wt_upper_bound, wt_lower_bound, 
            color=CONTROL_PALETTE, alpha=0.05
        )

        # Fill in with full confidence interval:
        rd_plot = rd_matrix_inference[cell_count_index, sr_index, :]
        rd_stddev = np.std(rd_matrix_inference[cell_count_index, sr_index, :])
        rd_upper_bound = rd_plot + (1.96 * rd_stddev)
        rd_lower_bound = rd_plot - (1.96 * rd_stddev)
        ax.plot(advection_rates, rd_upper_bound, ls='--', alpha=0.5, color=RD_PALETTE)
        ax.plot(advection_rates, rd_lower_bound, ls='--', alpha=0.5, color=RD_PALETTE)
        ax.fill_between(
            advection_rates, rd_upper_bound, rd_lower_bound, 
            color=RD_PALETTE, alpha=0.05
        )

        wt_line.set_label("Control")
        rd_line.set_label("RD")

        # Format axes:
        ax.set_xlim(0, 3)
        ax.text(
            0.98, 0.04, f'Cell Count: {int(cell_counts[cell_count_index])}', horizontalalignment='right',
            verticalalignment='baseline', transform=ax.transAxes
        )
        ax.set_ylabel("Matrix Order Parameter")
        ax.set_xlabel("Matrix Advection Rate")

        ax.legend(loc='best')

        fig.subplots_adjust(0.1, 0.15, 0.9, 0.85)

        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "single_matrix_inference_on_fit.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_short_inference()
    return


@app.cell
def _(
    CONTROL_PALETTE,
    FULL_HEIGHT,
    FULL_WIDTH,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    RD_PALETTE,
    advection_rates,
    cell_counts,
    datetime,
    np,
    os,
    plt,
    rd_matrix_inference,
    wt_matrix_inference,
):
    def plot_matrix_inference():
        fig, axs = plt.subplots(3, 1, figsize=(FULL_WIDTH, FULL_HEIGHT), sharex=True, sharey=True)
        alpha_array = np.linspace(0.3, 0.75, 10)

        # Iterate through gridsearch:
        for cell_count_index in range(3):
            for sr_index in list(range(10))[::-1]:
                wt_line, = axs[cell_count_index].plot(
                    advection_rates,
                    wt_matrix_inference[cell_count_index, sr_index, :],
                    alpha=alpha_array[sr_index], color=CONTROL_PALETTE
                )
                rd_line, = axs[cell_count_index].plot(
                    advection_rates,
                    rd_matrix_inference[cell_count_index, sr_index, :],
                    alpha=alpha_array[sr_index], color=RD_PALETTE
                )

                if cell_count_index == 0 and sr_index == 9:
                    wt_line.set_label("Control")
                    rd_line.set_label("RD")

            # Format axes:
            axs[cell_count_index].set_xlim(0, 3)
            axs[cell_count_index].text(
                0.98, 0.04, f'Cell Count: {int(cell_counts[cell_count_index])}', horizontalalignment='right',
                verticalalignment='baseline', transform=axs[cell_count_index].transAxes
            )
            axs[cell_count_index].set_ylabel("Matrix Order Parameter")

            if cell_count_index == 2:
                axs[cell_count_index].set_xlabel("Matrix Advection Rate")

        # fig.legend(loc='best')

        fig.subplots_adjust(0.2, 0.075, 0.8, 0.925, wspace=0.5, hspace=0.075)

        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "matrix_inference_on_fit.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_matrix_inference()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
