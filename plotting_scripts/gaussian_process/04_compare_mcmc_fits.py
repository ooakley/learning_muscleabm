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
    return (mpl,)


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
    SIM_UNIT_SIZE = 0.3469   # Simulation distance unit in µm
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
        SIM_UNIT_SIZE,
        TEXT_WIDTH,
    )


@app.cell
def _(np, os):
    PARAMETER_DIMENSION = 12
    EXPERIMENT_DIRPATH = "model_experiments/2026-10-02-collisions_shape"
    # mcmc_results = "wide_mcmc_results"
    hm_index = 2
    mcmc_results = f"hm{hm_index}/disc_cov_mcmc_results"


    ctl_mcmc_chain = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "wt_mcmc_chain.npy"))
    rd_mcmc_chain = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "rd_mcmc_chain.npy"))

    ctl_likelihoods = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "wt_mcmc_likelihoods.npy"))
    rd_likelihoods = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "rd_mcmc_likelihoods.npy"))
    return (
        EXPERIMENT_DIRPATH,
        PARAMETER_DIMENSION,
        ctl_mcmc_chain,
        hm_index,
        rd_mcmc_chain,
    )


@app.cell
def _():
    THIN_FACTOR = 32
    return


@app.cell
def _():
    PARAMETER_SYMBOLS = [
        "D^{\\ast}",
        "K_A",
        "K",
        "\\tau",
        "\\beta",
        "\\alpha",
        "R",
        "r_{col}",
        "\\alpha_{CIL}",
        "k",
        "r",
        "\\rho_{AR}"
    ]
    return (PARAMETER_SYMBOLS,)


@app.cell
def _():
    # distribution_size = len(ctl_likelihoods[2048::THIN_FACTOR, :, 0].flatten())
    # cutoff_95 = int(np.floor(distribution_size * 0.95))

    # ctl_ci95_idx = np.argsort(ctl_likelihoods[2048::THIN_FACTOR, :, 0].flatten())[-cutoff_95:]
    # rd_ci95_idx = np.argsort(rd_likelihoods[2048::THIN_FACTOR, :, 0].flatten())[-cutoff_95:]

    # ctl_ci95 = ctl_mcmc_chain[2048::THIN_FACTOR, :, 0, :].reshape(-1, PARAMETER_DIMENSION)[ctl_ci95_idx, :]
    # rd_ci95 = rd_mcmc_chain[2048::THIN_FACTOR, :, 0, :].reshape(-1, PARAMETER_DIMENSION)[rd_ci95_idx, :]
    return


@app.cell
def _(EXPERIMENT_DIRPATH, SIM_UNIT_SIZE, json, np, os):
    with open(os.path.join(EXPERIMENT_DIRPATH, "config.json")) as json_file:
        config_dict = json.load(json_file)

    parameter_list = [parameter_range[0] for parameter_range in config_dict["gridsearch_parameters"]]
    parameter_list.remove("numberOfCells")

    parameter_scaling = [parameter_range[1] for parameter_range in config_dict["gridsearch_parameters"]]
    parameter_scaling.pop(-1);

    PARAM_CONVERSION = {
        "cueDiffusionRate": 1,
        "cueKa": 1,
        "fluctuationAmplitude": SIM_UNIT_SIZE**2,
        "fluctuationTimescale": 1,
        "maximumSteadyStateActinFlow": SIM_UNIT_SIZE,
        "actinAdvectionRate": 1,
        "cellBodyRadius": SIM_UNIT_SIZE,
        "collisionFlowReductionRate": 1,
        "collisionAdvectionRate": 1,
        "cellStiffness": 1 / SIM_UNIT_SIZE,
        "surfaceStickiness": 1,
        "adhesionReductionRate": 1
    }

    conversion_array = np.array(list(PARAM_CONVERSION.values()))
    conversion_array = np.expand_dims(conversion_array, axis=1)
    parameter_scaling *= conversion_array
    return parameter_list, parameter_scaling


@app.cell
def _(EXPERIMENT_DIRPATH, hm_index, np, os):
    METRICS_TO_PLOT = [
        "speeds",
        "meander_ratios",
        "ann_indices",
        "coherency"
    ]

    # Get metrics:
    def load_metrics():
        metrics_dict = {}
        for metric_name in METRICS_TO_PLOT:
            metric_array = np.load(os.path.join(
                EXPERIMENT_DIRPATH, "global_dataset", "summary_data", f"{metric_name}.npy"
            ))
            metrics_dict[metric_name] = np.nanmean(metric_array, axis=1)
        return metrics_dict

    metrics_dict = load_metrics()
    sobol_inputs = np.load(os.path.join(EXPERIMENT_DIRPATH, f"hm{hm_index}", "sample_matrix.npy"))
    return


@app.cell
def _(
    FULL_WIDTH,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    PARAMETER_DIMENSION,
    cc,
    datetime,
    np,
    os,
    parameter_list,
    plot_scatter_density,
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


    # def plot_scatter_density(i, j, kde, posterior_distribution, color, ax):
    #     # Get evaluate KDE at posterior samples:
    #     marginal_kde = kde.marginal([i, j])
    #     densities = marginal_kde(posterior_distribution[:, [i, j]].T)
    #     density_sort = np.argsort(densities)

    #     # Plot sample:
    #     ax.scatter(
    #         posterior_distribution[density_sort, i],
    #         posterior_distribution[density_sort, j],
    #         s=0.5, alpha=0.7, c=densities[density_sort],
    #         cmap=cc.m_CET_L20, edgecolors="none"
    #     )
    #     ax.set_aspect("equal")

    #     for axis in ['top','bottom','left','right']:
    #         ax.spines[axis].set_linewidth(0.25)
    #         ax.spines[axis].set_color(color)

    #     # ax.set_xticks([])
    #     # ax.set_yticks([])
    #     # ax.set_axis_off()


    def global_jointplot(posterior_distribution, title, image_filename, color):
        # Calculate global KDE:
        kde = scipy.stats.gaussian_kde(posterior_distribution.T, bw_method=0.33)

        # Set up subplots:
        fig, axs = plt.subplots(PARAMETER_DIMENSION, PARAMETER_DIMENSION, figsize=(FULL_WIDTH, FULL_WIDTH), sharex=True, sharey=True)

        # Iterate through indices:
        for i in range(PARAMETER_DIMENSION):
            for j in range(PARAMETER_DIMENSION):
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
                    # axs[i, j].set_xlim(-3, 0)
                    # axs[i, j].set_ylim(-3, 0)
                    axs[i, j].set_aspect("equal")
                    axs[i, j].set_axis_off()
                    continue

                # Set off diagonals blank:
                if j < i:
                    axs[i, j].remove()
                    continue

                # Plot density scatter:
                plot_scatter_density(i, j, kde, posterior_distribution, color, axs[i, j])

        edging = 0.01
        fig.subplots_adjust(edging, edging, 1 - edging, 1 - edging, wspace=0.025, hspace=0.025)
        fig.text(0.01, 0.01, title, ha="left", va="baseline", fontsize=12, style="italic", color=color)

        # Show in cell:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, f"{image_filename}.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()
    return


@app.cell
def _(
    CONTROL_PALETTE,
    FULL_WIDTH,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    PARAMETER_SYMBOLS,
    RD_PALETTE,
    datetime,
    mpl,
    np,
    os,
    parameter_scaling,
    plt,
    scipy,
):
    def format_tick(x, pos):
        if abs(x) < 0.01 and x != 0:
            return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        return f"{x:g}"


    def plot_scatter_density(i, j, kde, posterior_distribution, color, ax):
        # Get evaluate KDE at posterior samples:
        marginal_kde = kde.marginal([i, j])
        densities = marginal_kde(posterior_distribution[:, [i, j]].T)
        density_sort = np.argsort(densities)

        if j < i:
            color = RD_PALETTE
        else:
            color = CONTROL_PALETTE

        # Plot sample:
        ax.scatter(
            posterior_distribution[density_sort, j],
            posterior_distribution[density_sort, i],
            s=0.5, alpha=0.4, edgecolors="none",
            c=color

            # c=densities[density_sort],
            # cmap=cc.m_CET_L20
        )
        # ax.set_aspect("equal")

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1)
            ax.spines[axis].set_color(color)

        ax.set_xlim(*parameter_scaling[j, :])
        ax.set_ylim(*parameter_scaling[i, :])

        ax.set_box_aspect(1)

        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_axis_off()

    def plot_posterior_histogram(i, posterior_distributions, ax):
        bins = np.linspace(*parameter_scaling[i, :], 20)
        ax.hist(posterior_distributions[0][:, i], bins=bins, color=CONTROL_PALETTE, histtype="step", density=True)
        ax.hist(posterior_distributions[1][:, i], bins=bins, color=RD_PALETTE, histtype="step", density=True)
        ax.set_xlim(*parameter_scaling[i, :])
        ax.set_box_aspect(1)

    def double_gridplot(posterior_distributions):
        # Do scaling:
        valrange = np.diff(parameter_scaling, axis=1)

        posterior_distributions[0] *= np.squeeze(valrange)
        posterior_distributions[0] += np.squeeze(parameter_scaling[:, 0])

        posterior_distributions[1] *= np.squeeze(valrange)
        posterior_distributions[1] += np.squeeze(parameter_scaling[:, 0])

        print(posterior_distributions[1][:, 3].max())

        # Calculate global KDE:
        ctl_kde = scipy.stats.gaussian_kde(posterior_distributions[0].T, bw_method=0.33)
        rd_kde = scipy.stats.gaussian_kde(posterior_distributions[1].T, bw_method=0.33)

        # For testing:
        PARAMETER_DIMENSION = 12

        # Set up subplots:
        fig, axs = plt.subplots(PARAMETER_DIMENSION, PARAMETER_DIMENSION, figsize=(FULL_WIDTH, FULL_WIDTH))

        for ax in axs.flat:
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(format_tick))
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(format_tick))
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2, prune="both"))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2, prune="both"))

        # Iterate through indices:
        for i in range(PARAMETER_DIMENSION):
            for j in range(PARAMETER_DIMENSION):
                # Plot comparative histograms:
                if i == j:
                    plot_posterior_histogram(i, posterior_distributions, axs[i, j])
                elif j < i:
                    # Lower triangle is RD:
                    plot_scatter_density(i, j, ctl_kde, posterior_distributions[1], RD_PALETTE, axs[i, j])
                else:
                    # Upper triangle is CTL:
                    plot_scatter_density(i, j, ctl_kde, posterior_distributions[0], CONTROL_PALETTE, axs[i, j])

                # Set up axis ticks:
                labelpad = 6
                top = False; bottom = False
                left = False; right = False
                if i == 0:
                    top = True
                    axs[i, j].xaxis.set_label_position("top")
                    axs[i, j].set_xlabel(f"${PARAMETER_SYMBOLS[j]}$", labelpad=labelpad)
                    axs[i, j].set_xticks([])
                if i == PARAMETER_DIMENSION - 1:
                    bottom = True
                    axs[i, j].set_xlabel(f"${PARAMETER_SYMBOLS[j]}$", labelpad=labelpad)

                if j == 0:
                    left = True
                    axs[i, j].set_ylabel(f"${PARAMETER_SYMBOLS[i]}$", labelpad=labelpad)
                if j == PARAMETER_DIMENSION - 1:
                    right = True
                    axs[i, j].yaxis.set_label_position("right")
                    axs[i, j].set_ylabel(f"${PARAMETER_SYMBOLS[i]}$", labelpad=labelpad)
                    axs[i, j].set_yticks([])

                axs[i, j].tick_params(
                    top=top, labeltop=top,
                    bottom=bottom, labelbottom=bottom,
                    right=right, labelright=right,
                    left=left, labelleft=left,
                    length=0, labelsize=5
                )


        edging = 0.08
        fig.subplots_adjust(edging, edging, 1 - edging, 1 - edging, wspace=0.1, hspace=0.1)
        # fig.text(0.01, 0.01, title, ha="left", va="baseline", fontsize=12, style="italic", color=color)

        # Show in cell:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, f"posterior_scattergrid.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()
    return double_gridplot, plot_scatter_density


@app.cell
def _(PARAMETER_DIMENSION, ctl_mcmc_chain, rd_mcmc_chain):
    chain_length = ctl_mcmc_chain.shape[0]
    quarter_index = int(chain_length // 4)
    ctl_kde_posterior = ctl_mcmc_chain[quarter_index::256, :, 0, :].reshape(-1, PARAMETER_DIMENSION)
    rd_kde_posterior = rd_mcmc_chain[quarter_index::256, :, 0, :].reshape(-1, PARAMETER_DIMENSION)
    return ctl_kde_posterior, quarter_index, rd_kde_posterior


@app.cell
def _(ctl_kde_posterior):
    print(ctl_kde_posterior.shape)
    return


@app.cell
def _(ctl_kde_posterior, double_gridplot, np, rd_kde_posterior):
    double_gridplot([np.copy(ctl_kde_posterior), np.copy(rd_kde_posterior)])
    return


@app.cell
def _():
    # global_jointplot(ctl_kde_posterior, "Control Fit Parameter Distribution", "scattergrid_control", CONTROL_PALETTE)
    return


@app.cell
def _():
    # global_jointplot(rd_kde_posterior, "RD Fit Parameter Distribution", "scattergrid_rd", RD_PALETTE)
    return


@app.cell
def _(
    CONTROL_PALETTE,
    FULL_HEIGHT,
    FULL_WIDTH,
    METADATA_DICTIONARY,
    MaxNLocator,
    OUT_DIRPATH,
    PARAMETER_DIMENSION,
    RD_PALETTE,
    ctl_mcmc_chain,
    datetime,
    np,
    os,
    parameter_list,
    parameter_scaling,
    plt,
    quarter_index,
    rd_mcmc_chain,
):
    ctl_hist_posterior = ctl_mcmc_chain[quarter_index::64, :, 0, :].reshape(-1, PARAMETER_DIMENSION)
    rd_hist_posterior = rd_mcmc_chain[quarter_index::64, :, 0, :].reshape(-1, PARAMETER_DIMENSION)

    # ctl_hist_posterior = ctl_ci95
    # rd_hist_posterior = rd_ci95

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
                if count == PARAMETER_DIMENSION:
                    axs[i, j].remove()
                    continue
                plot_comparative_hist(count, axs[i, j])
                if count == 3:
                    axs[i, j].legend(fontsize=6)
                count += 1

        fig.subplots_adjust(0.15, 0.05, 0.85, 0.95, wspace=0.15, hspace=0.55)

        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "marginal_fit_comparison.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_all_marginals()
    return


@app.cell
def _(
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    PARAMETER_SYMBOLS,
    TEXT_WIDTH,
    ctl_kde_posterior,
    datetime,
    np,
    os,
    plt,
    rd_kde_posterior,
):
    np_rng = np.random.default_rng(0)

    def plot_delta_distribution(parameter_index, ax=None):
        # Get uniform distance distribution:
        sample_size = ctl_kde_posterior.shape[0]
        uniform_a = np_rng.uniform(0, 1, sample_size)
        uniform_b = np_rng.uniform(0, 1, sample_size)
        uniform_delta = uniform_a - uniform_b

        # Get parameter distance distributiob:
        if ax is None:
            fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2))

        delta_dist = rd_kde_posterior[:, parameter_index] - ctl_kde_posterior[:, parameter_index]
        bins = np.linspace(-1, 1, 15)

        # Plot null distribution:
        ax.axvline(0, ls="--", c='k', alpha=0.25)
        ax.hist(uniform_delta, bins=bins, color='k', alpha=0.25, density=True, label="Prior difference")

        # Plot posterior distribution:
        ax.axvline(np.quantile(delta_dist, 0.1), ls="--", c="gold", alpha=0.75, label="0.8 ETI")
        ax.axvline(np.quantile(delta_dist, 0.9), ls="--", c="gold", alpha=0.75)
        ax.axvline(np.mean(delta_dist), ls="--", c="tab:blue", alpha=0.5)
        ax.hist(delta_dist, bins=bins, alpha=0.5, density=True, label="Posterior difference [CTL - RD]")
        ax.set_xlim(-1, 1)

        ax.set_xlabel(f"${PARAMETER_SYMBOLS[parameter_index]}$")



    def plot_all_delta_distributions():
        fig, axs = plt.subplots(6, 2, figsize=(TEXT_WIDTH, 7))

        parameter_index = 0
        for i in range(6):
            for j in range(2):
                plot_delta_distribution(parameter_index, ax=axs[i, j])
                if parameter_index == 0:
                    axs[i, j].legend(
                        loc='center right',
                        bbox_to_anchor=(0.675, 0.96), fontsize=7,
                        ncol=1, bbox_transform=fig.transFigure
                    )
                parameter_index += 1

        fig.subplots_adjust(0.15, 0.1, 0.85, 0.9, 0.2, 0.5)
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "parameter_delta_comparison.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_all_delta_distributions()
    return


@app.cell
def _():
    # def plot_mle_hist(index, ax):
    #     bins = np.linspace(parameter_scaling[index][0], parameter_scaling[index][1], 30)
    #     ax.xaxis.set_major_locator(MaxNLocator(3))
    #     ax.yaxis.set_major_locator(MaxNLocator(3))
    #     ax.hist(
    #         scale_parameter(ctl_mle[:, index], index),
    #         alpha=0.75, label="Control Fit",
    #         color=CONTROL_PALETTE, histtype="step", density=True, bins=bins
    #     )
    #     ax.hist(
    #         scale_parameter(rd_mle[:, index], index),
    #         alpha=0.75, label="RD Fit",
    #         color=RD_PALETTE, histtype="step", density=True, bins=bins
    #     )
    #     ax.set_xlim(*parameter_scaling[index])
    #     ax.set_xlabel(parameter_list[index])

    # def plot_mle_marginals():
    #     fig, axs = plt.subplots(6, 2, figsize=(FULL_WIDTH, FULL_HEIGHT))

    #     count = 0
    #     for i in range(6):
    #         for j in range(2):
    #             if count == PARAMETER_DIMENSION:
    #                 axs[i, j].remove()
    #                 continue
    #             plot_mle_hist(count, axs[i, j])
    #             if count == 0:
    #                 axs[i, j].legend(fontsize=6)
    #             count += 1

    #     fig.subplots_adjust(0.15, 0.05, 0.85, 0.95, wspace=0.15, hspace=0.55)
    #     METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     plt.savefig(
    #         os.path.join(OUT_DIRPATH, "marginal_fit_mle_comparison.png"),
    #         dpi=300, metadata=METADATA_DICTIONARY, transparent=True
    #     )
    #     plt.show()

    # plot_mle_marginals()
    return


@app.cell
def _():
    # def plot_violin(index, ax):
    #     # Plot violins:
    #     scaled_data = [
    #         scale_parameter(ctl_hist_posterior[:, index], index),
    #         scale_parameter(rd_hist_posterior[:, index], index)
    #     ]
    #     xy_data = np.stack(scaled_data, axis=1)
    #     violin_parts = ax.violinplot(
    #         xy_data,
    #         positions=[0, 1],
    #         widths=0.8,
    #         bw_method="silverman",
    #         showextrema=False
    #     )

    #     for part_index, violin_part in enumerate(violin_parts['bodies']):
    #         violin_part.set_facecolor([CONTROL_PALETTE, RD_PALETTE][part_index])
    #         violin_part.set_edgecolor([CONTROL_PALETTE, RD_PALETTE][part_index])
    #         violin_part.set_alpha(0.5)

    #     # Plot internal boxes:
    #     low_quartiles, medians, high_quartiles = np.percentile(xy_data, [25, 50, 75], axis=0)
    #     for p_index in range(2):
    #         p_color = [CONTROL_PALETTE, RD_PALETTE][p_index]
    #         ax.scatter(p_index, medians[p_index], marker='o', color=p_color, s=10)
    #         ax.vlines(
    #             p_index,
    #             low_quartiles[p_index], high_quartiles[p_index],
    #             color=p_color, linestyle='-', lw=1
    #         )

    #     # Label xticks:
    #     ax.set_xticks([0, 1], ["CTL", "RD"])
    #     ax.set_ylabel(parameter_list[index])


    # def plot_mle_violins():
    #     fig, axs = plt.subplots(3, 4, figsize=(FULL_WIDTH, FULL_HEIGHT), sharex=True)

    #     count = 0
    #     for i in range(3):
    #         for j in range(4):
    #             # Remove corner plot:
    #             if count == 11:
    #                 axs[i, j].set_axis_off()
    #                 continue

    #             # Plot violins:
    #             plot_violin(count, axs[i, j])
    #             count += 1

    #     fig.subplots_adjust(0.075, 0.025, 0.925, 0.975, wspace=0.5, hspace=0.075)

    #     METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     plt.savefig(
    #         os.path.join(OUT_DIRPATH, "marginal_fit_mle_violins.png"),
    #         dpi=300, metadata=METADATA_DICTIONARY, transparent=True
    #     )
    #     plt.show()

    # plot_mle_violins()
    return


@app.cell
def _():
    # Python script; run matrix inference on fits:
    return


@app.cell
def _():
    # def get_basal_confluency(r, count):
    #     return ((np.pi * r**2) * count) / (2048**2)
    return


@app.cell
def _():
    # print(get_basal_confluency(72.5, 300))
    # print(get_basal_confluency(62.5, 300))
    return


@app.cell
def _(EXPERIMENT_DIRPATH, np, os):
    matrix_inference_dirpath = os.path.join(EXPERIMENT_DIRPATH, "cov_mcmc_results", "matrix_inference")
    wt_matrix_inference = np.load(os.path.join(matrix_inference_dirpath, "wt_posterior_array.npy"))
    rd_matrix_inference = np.load(os.path.join(matrix_inference_dirpath, "rd_posterior_array.npy"))

    wt_ss_params = np.load(os.path.join(matrix_inference_dirpath, "wt_posterior.npy"))
    rd_ss_params = np.load(os.path.join(matrix_inference_dirpath, "rd_posterior.npy"))

    wt_inf_likelihood = np.load(os.path.join(matrix_inference_dirpath, "wt_likelihood.npy"))
    rd_inf_likelihood = np.load(os.path.join(matrix_inference_dirpath, "rd_likelihood.npy"))
    return (
        rd_inf_likelihood,
        rd_matrix_inference,
        rd_ss_params,
        wt_inf_likelihood,
        wt_matrix_inference,
        wt_ss_params,
    )


@app.cell
def _(np):
    print("Setting up grid for inference...", flush=True)

    CC_SAMPLE_COUNT = 10
    SR_SAMPLE_COUNT = 10
    ADV_SAMPLE_COUNT = 10

    cell_counts = np.linspace(50, 300, CC_SAMPLE_COUNT)
    sample_rates = np.linspace(0.5, 15, SR_SAMPLE_COUNT)
    advection_rates = np.linspace(0, 3, ADV_SAMPLE_COUNT)
    return advection_rates, cell_counts


@app.cell
def _(
    plt,
    rd_inf_likelihood,
    rd_matrix_inference,
    wt_inf_likelihood,
    wt_matrix_inference,
):
    plt.scatter(wt_matrix_inference[3, 5, 5, :], wt_inf_likelihood, s=1)
    plt.scatter(rd_matrix_inference[3, 5, 5, :], rd_inf_likelihood, s=1)
    return


@app.cell
def _(
    CONTROL_PALETTE,
    RD_PALETTE,
    plt,
    rd_matrix_inference,
    wt_matrix_inference,
):
    plt.hist(wt_matrix_inference[4, 9, 9, :].flatten(), color=CONTROL_PALETTE, histtype="step", bins=100);
    plt.hist(rd_matrix_inference[4, 9, 9, :].flatten(), color=RD_PALETTE, histtype="step", bins=100);
    plt.show()
    return


@app.cell
def _(cell_counts, get_basal_confluency, np, rd_ss_params, wt_ss_params):
    wt_confluency_matrix = []
    rd_confluency_matrix = []
    for cell_count in cell_counts:
        wt_confluency = get_basal_confluency(wt_ss_params[:, 6] * 50 + 30, cell_count)
        rd_confluency = get_basal_confluency(rd_ss_params[:, 6] * 50 + 30, cell_count)
        wt_confluency_matrix.append(wt_confluency)
        rd_confluency_matrix.append(rd_confluency)

    wt_confluency_matrix = np.stack(wt_confluency_matrix, axis=0)
    rd_confluency_matrix = np.stack(rd_confluency_matrix, axis=0)
    return


@app.cell
def _(
    CONTROL_PALETTE,
    RD_PALETTE,
    cell_counts,
    np,
    plt,
    rd_matrix_inference,
    wt_matrix_inference,
):
    def quick_plot_coupling():
        fig, ax = plt.subplots()
        alpha_values = np.linspace(0.1, 1.0, 10)
        for i in range(10):
            ax.plot(cell_counts, np.mean(wt_matrix_inference[:, 9, i, :], axis=-1), alpha=alpha_values[i], c=CONTROL_PALETTE)
            ax.plot(cell_counts, np.mean(rd_matrix_inference[:, 9, i, :], axis=-1), alpha=alpha_values[i], c=RD_PALETTE)
        plt.show()

    quick_plot_coupling()
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
