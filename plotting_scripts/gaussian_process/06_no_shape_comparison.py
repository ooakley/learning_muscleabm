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
    import matplotlib.font_manager as fm
    from matplotlib.ticker import AutoLocator, MaxNLocator

    from datetime import datetime
    return arviz_stats, datetime, json, np, os, pd, plt, scipy, subprocess


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
    return (
        CONTROL_PALETTE,
        FULL_WIDTH,
        METADATA_DICTIONARY,
        OUT_DIRPATH,
        RD_PALETTE,
        TEXT_WIDTH,
    )


@app.cell
def _(np, os):
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

    PARTICLE_DIRPATH = "model_experiments/2026-09-23-collisions_only"
    pa_ctl_likelihoods = np.load(os.path.join(PARTICLE_DIRPATH, "disc_cov_mcmc_results", "wt_mcmc_likelihoods.npy"))
    pa_rd_likelihoods = np.load(os.path.join(PARTICLE_DIRPATH, "disc_cov_mcmc_results", "rd_mcmc_likelihoods.npy"))

    SS_DIRPATH = "model_experiments/2026-09-16-collisions_shape"
    ss_ctl_likelihoods = np.load(os.path.join(SS_DIRPATH, "disc_cov_mcmc_results", "wt_mcmc_likelihoods.npy"))
    ss_rd_likelihoods = np.load(os.path.join(SS_DIRPATH, "disc_cov_mcmc_results", "rd_mcmc_likelihoods.npy"))
    return (
        METRICS_LABELS,
        METRICS_TO_PLOT,
        PARTICLE_DIRPATH,
        SS_DIRPATH,
        pa_ctl_likelihoods,
        pa_rd_likelihoods,
        ss_ctl_likelihoods,
        ss_rd_likelihoods,
    )


@app.cell
def _(PARTICLE_DIRPATH, arviz_stats, np, os):
    ctl_mcmc_chain = np.load(os.path.join(PARTICLE_DIRPATH, "disc_cov_mcmc_results", "wt_mcmc_chain.npy"))
    rd_mcmc_chain = np.load(os.path.join(PARTICLE_DIRPATH, "disc_cov_mcmc_results", "rd_mcmc_chain.npy"))
    ctl_rhat = arviz_stats.rhat(ctl_mcmc_chain[:, :, 0, :], chain_axis=1, draw_axis=0)
    rd_rhat = arviz_stats.rhat(rd_mcmc_chain[:, :, 0, :], chain_axis=1, draw_axis=0)
    ctl_ess = arviz_stats.ess(ctl_mcmc_chain[:, :, 0, :], chain_axis=1, draw_axis=0)
    rd_ess = arviz_stats.ess(rd_mcmc_chain[:, :, 0, :], chain_axis=1, draw_axis=0)
    return ctl_ess, ctl_rhat, rd_ess, rd_rhat


@app.cell
def _(PARTICLE_DIRPATH, ctl_ess, ctl_rhat, json, np, os, pd, rd_ess, rd_rhat):
    with open(os.path.join(PARTICLE_DIRPATH, "config.json")) as json_file:
        config_dict = json.load(json_file)

    parameter_list = [parameter_range[0] for parameter_range in config_dict["gridsearch_parameters"]]
    parameter_list.remove("numberOfCells")

    column_names = [
        "CTL $\\hat{R}$",
        "CTL ESS",
        "RD $\\hat{R}$",
        "RD ESS"
    ]

    dataframe_array = np.stack([ctl_rhat, ctl_ess, rd_rhat, rd_ess], axis=1)
    diagnostics_dataframe = pd.DataFrame(dataframe_array, columns=column_names, index=parameter_list)

    print(diagnostics_dataframe.to_latex(index=parameter_list, float_format="%.2f"))
    return


@app.cell
def _(METRICS_LABELS, METRICS_TO_PLOT, PARTICLE_DIRPATH, json, np, os, pd):
    def collate_cross_validation_metrics():
        dataframe = []
        for metric_index, metric_name in enumerate(METRICS_TO_PLOT):
            # Load CV data:
            json_filepath = os.path.join(
                PARTICLE_DIRPATH, "gaussian_process_models", f"{metric_name}", "cv_metrics.json"
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

    pa_cv_dataframe = collate_cross_validation_metrics()
    return (pa_cv_dataframe,)


@app.cell
def _(pa_cv_dataframe):
    print(pa_cv_dataframe.to_latex(index=False))
    return


@app.cell
def _(PARTICLE_DIRPATH, np, os):
    pa_order_parameter = np.load(os.path.join(PARTICLE_DIRPATH, "summary_data", "order_parameters.npy"))
    pa_order_parameter = np.mean(pa_order_parameter, axis=1)

    pa_speed = np.load(os.path.join(PARTICLE_DIRPATH, "summary_data", "speeds.npy"))
    pa_speed = np.mean(pa_speed, axis=1)
    return pa_order_parameter, pa_speed


@app.cell
def _(np, pa_order_parameter, pa_speed, plt):
    plt.scatter(pa_speed, pa_order_parameter)
    plt.show()

    flock_mask = np.logical_and(pa_speed > 0.2, pa_order_parameter > 0.26)
    print(np.argwhere(flock_mask))
    return


@app.cell
def _(
    CONTROL_PALETTE,
    FULL_WIDTH,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    RD_PALETTE,
    datetime,
    os,
    pa_ctl_likelihoods,
    pa_rd_likelihoods,
    plt,
    ss_ctl_likelihoods,
    ss_rd_likelihoods,
):
    def plot_likelihood_comparison():
        fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.25))

        # Plot control comparison:
        bins = 40
        ax.hist(
            pa_ctl_likelihoods[1024::64, :, 0].flatten(),
            bins=bins, histtype="step", color=CONTROL_PALETTE, alpha=0.5,
            density=True, label="Control - Particle"
        )
        ax.hist(
            ss_ctl_likelihoods[1024::64, :, 0].flatten(),
            bins=bins, histtype="step", color=CONTROL_PALETTE,
            density=True, label="Control - Stick-slip"
        )

        # Plot RD comparison:
        ax.hist(
            pa_rd_likelihoods[1024::64, :, 0].flatten(),
            bins=bins, histtype="step", color=RD_PALETTE, alpha=0.5,
            density=True, label="RD - Particle"
        )
        ax.hist(
            ss_rd_likelihoods[1024::64, :, 0].flatten(),
            bins=bins, histtype="step", color=RD_PALETTE,
            density=True, label="RD - Stick-slip"
        )
        ax.legend(loc="upper left")

        ax.set_xlabel("$log(\\mathscr{L}(\\theta|X))$")
        ax.set_ylabel("Density")

        edging = 0.15
        fig.subplots_adjust(edging, edging, 1 - edging, 1 - edging, wspace=0.025, hspace=0.025)

        # Show in cell:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, f"no_shape_comparison.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_likelihood_comparison()
    return


@app.cell
def _(PARTICLE_DIRPATH, SS_DIRPATH, np, os):
    # Plot KDEs of speed against coherency:
    pa_speeds = np.load(os.path.join(PARTICLE_DIRPATH, "summary_data", "speeds.npy"))
    pa_speeds = np.mean(pa_speeds, axis=1)
    ss_speeds = np.load(os.path.join(SS_DIRPATH, "summary_data", "speeds.npy"))
    ss_speeds = np.mean(ss_speeds, axis=1)

    pa_coherency = np.load(os.path.join(PARTICLE_DIRPATH, "summary_data", "coherency.npy"))
    pa_coherency = np.mean(pa_coherency, axis=1)
    ss_coherency = np.load(os.path.join(SS_DIRPATH, "summary_data", "coherency.npy"))
    ss_coherency = np.mean(ss_coherency, axis=1)

    pa_mr = np.load(os.path.join(PARTICLE_DIRPATH, "summary_data", "meander_ratios.npy"))
    pa_mr = np.mean(pa_mr, axis=1)
    ss_mr = np.load(os.path.join(SS_DIRPATH, "summary_data", "meander_ratios.npy"))
    ss_mr = np.mean(ss_mr, axis=1)
    return pa_coherency, pa_mr, pa_speeds, ss_coherency, ss_mr, ss_speeds


@app.cell
def _(pa_coherency, pa_mr, plt, ss_coherency, ss_mr):
    plt.scatter(pa_mr, pa_coherency, s=1, alpha=0.1, edgecolors="none")
    plt.scatter(ss_mr, ss_coherency, s=1, alpha=0.1, edgecolors="none")
    return


@app.cell
def _(np, scipy):
    # Binscatter of speeds against coherency:
    def get_binscatter(x, y):
        ventile_edges = np.quantile(x, np.linspace(0.15, 1.0, 21))
        ventile_centres = ventile_edges[:-1] + np.diff(ventile_edges)
        ventile_y, _, _ = scipy.stats.binned_statistic(x, y, statistic=np.nanmedian, bins=ventile_edges)
        return ventile_centres, ventile_y
    return (get_binscatter,)


@app.cell
def _(get_binscatter, pa_coherency, pa_speeds, ss_coherency, ss_speeds):
    pa_binscatter_speed, pa_binscatter_coherency = get_binscatter(pa_speeds, pa_coherency)
    ss_binscatter_speed, ss_binscatter_coherency = get_binscatter(ss_speeds, ss_coherency)
    return (
        pa_binscatter_coherency,
        pa_binscatter_speed,
        ss_binscatter_coherency,
        ss_binscatter_speed,
    )


@app.cell
def _(
    TEXT_WIDTH,
    pa_binscatter_coherency,
    pa_binscatter_speed,
    plt,
    ss_binscatter_coherency,
    ss_binscatter_speed,
):
    PA_PALETTE = "#648FFF"
    SS_PALETTE = "#FFB000"

    def plot_coherency_speed_binscatter():
        # Set up plot:
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.25))
        ax.scatter(pa_binscatter_speed, pa_binscatter_coherency, s=10, c=PA_PALETTE)
        ax.plot(pa_binscatter_speed, pa_binscatter_coherency, color=PA_PALETTE, label="Particle")
        ax.scatter(ss_binscatter_speed, ss_binscatter_coherency, s=10, c=SS_PALETTE)
        ax.plot(ss_binscatter_speed, ss_binscatter_coherency, color=SS_PALETTE, label="Stick-slip")

        ax.set_xlabel("Speed")
        ax.set_ylabel("Mean Coherency")
        ax.legend()

        # # Show in cell:
        # METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # plt.savefig(
        #     os.path.join(OUT_DIRPATH, f"coherency_speed_tradeoff.png"),
        #     dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        # )
        plt.show()

    plot_coherency_speed_binscatter()
    return PA_PALETTE, SS_PALETTE


@app.cell
def _(pa_ctl_p_mean):
    pa_ctl_p_mean.shape
    return


@app.cell
def _(np, os):
    wetlab_gp_path = "wetlab_data/gp_results"

    # Get wetlab speed data:
    ctl_speed_mean = np.load(os.path.join(wetlab_gp_path, "CTL_mean_speed_mean.npy"))
    ctl_speed_sigma = np.sqrt(np.diag(np.load(os.path.join(wetlab_gp_path, "CTL_mean_speed_sigma.npy"))))
    rd_speed_mean = np.load(os.path.join(wetlab_gp_path, "RD_mean_speed_mean.npy"))
    rd_speed_sigma = np.sqrt(np.diag(np.load(os.path.join(wetlab_gp_path, "RD_mean_speed_sigma.npy"))))

    # Get wetlab coherency fraction data:
    ctl_cf_mean = np.load(os.path.join(wetlab_gp_path, "CTL_coherency_fraction_mean.npy"))
    ctl_cf_sigma = np.sqrt(np.diag(np.load(os.path.join(wetlab_gp_path, "CTL_coherency_fraction_sigma.npy"))))
    rd_cf_mean = np.load(os.path.join(wetlab_gp_path, "RD_coherency_fraction_mean.npy"))
    rd_cf_sigma = np.sqrt(np.diag(np.load(os.path.join(wetlab_gp_path, "RD_coherency_fraction_sigma.npy"))))
    return (
        ctl_cf_mean,
        ctl_cf_sigma,
        ctl_speed_mean,
        ctl_speed_sigma,
        rd_cf_mean,
        rd_cf_sigma,
        rd_speed_mean,
        rd_speed_sigma,
    )


@app.cell
def _(pa_ctl_p_mean):
    pa_ctl_p_mean.shape
    return


@app.cell
def _(PARTICLE_DIRPATH, SS_DIRPATH, np, os):
    # Compare posterior means:
    pa_ctl_p_mean = np.load(os.path.join(PARTICLE_DIRPATH, "disc_cov_mcmc_results", "wt_posterior_mean.npy"))
    pa_rd_p_mean = np.load(os.path.join(PARTICLE_DIRPATH, "disc_cov_mcmc_results", "rd_posterior_mean.npy"))

    ss_ctl_p_mean = np.load(os.path.join(SS_DIRPATH, "disc_cov_mcmc_results", "wt_posterior_mean.npy"))
    ss_rd_p_mean = np.load(os.path.join(SS_DIRPATH, "disc_cov_mcmc_results", "rd_posterior_mean.npy"))
    return pa_ctl_p_mean, pa_rd_p_mean, ss_ctl_p_mean, ss_rd_p_mean


@app.cell
def _(
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    PA_PALETTE,
    SS_PALETTE,
    TEXT_WIDTH,
    ctl_cf_mean,
    ctl_cf_sigma,
    ctl_speed_mean,
    ctl_speed_sigma,
    datetime,
    os,
    pa_ctl_p_mean,
    pa_rd_p_mean,
    plt,
    rd_cf_mean,
    rd_cf_sigma,
    rd_speed_mean,
    rd_speed_sigma,
    ss_ctl_p_mean,
    ss_rd_p_mean,
):
    def plot_ctl_tradeoff(ax):
        # Plot scatter of fit:
        ax.scatter(
            pa_ctl_p_mean[0, :, 5], pa_ctl_p_mean[3, :, 5],
            s=2, alpha=0.5, color=PA_PALETTE, label="Particle"
        )
        ax.scatter(
            ss_ctl_p_mean[0, :, 5], ss_ctl_p_mean[3, :, 5],
            s=2, alpha=0.3, color=SS_PALETTE, label="Stick-slip"
        )

        # Plot target:
        ax.scatter(ctl_speed_mean[5], ctl_cf_mean[5], c='k', s=10, alpha=0.75, label="Control Fit Target")
        ax.errorbar(ctl_speed_mean[5], ctl_cf_mean[5], xerr=ctl_speed_sigma[5], yerr=ctl_cf_sigma[5], c='k', alpha=0.75)

        # Add labels:
        ax.text(0.98, 0.05, 'Control Posterior Predictions', ha='right', transform=ax.transAxes)
        ax.set_ylabel("Coherency")
        ax.legend()

    def plot_rd_tradeoff(ax):
        # Plot scatter of fit:
        ax.scatter(
            pa_rd_p_mean[0, :, 2], pa_rd_p_mean[3, :, 2],
            s=2, alpha=0.5, color=PA_PALETTE, label="Particle"
        )
        ax.scatter(
            ss_rd_p_mean[0, :, 2], ss_rd_p_mean[3, :, 2],
            s=2, alpha=0.3, color=SS_PALETTE, label="Stick-slip"
        )

        # Plot target:
        ax.scatter(rd_speed_mean[5], rd_cf_mean[5], c='k', s=10, alpha=0.75, label="RD Fit Target")
        ax.errorbar(rd_speed_mean[5], rd_cf_mean[5], xerr=rd_speed_sigma[5], yerr=rd_cf_sigma[5], c='k', alpha=0.75)

        # Add labels:
        ax.text(0.98, 0.05, 'RD Posterior Predictions', ha='right', transform=ax.transAxes)
        ax.set_xlabel("Speed")
        ax.set_ylabel("Coherency")
        ax.legend(loc="upper right")

    def plot_tradeoffs():
        fig, axs = plt.subplots(2, 1, figsize=(TEXT_WIDTH, 5), sharex=True, sharey=True)

        plot_ctl_tradeoff(axs[0])
        plot_rd_tradeoff(axs[1])

        edge = 0.1
        fig.subplots_adjust(edge, edge, 1 - edge, 1 - edge, wspace=0, hspace=0.1)

        # Show in cell:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, f"coherency_speed_tradeoff.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_tradeoffs()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
