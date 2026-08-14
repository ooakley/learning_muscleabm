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
    return datetime, np, os, pd, plt, scipy, subprocess


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
    PARTICLE_DIRPATH = "model_experiments/2026-06-07-collisions_only"
    pa_ctl_likelihoods = np.load(os.path.join(PARTICLE_DIRPATH, "mcmc_results", "wt_mcmc_likelihoods.npy"))
    pa_rd_likelihoods = np.load(os.path.join(PARTICLE_DIRPATH, "mcmc_results", "rd_mcmc_likelihoods.npy"))

    SS_DIRPATH = "model_experiments/2026-05-31-collisions_shape"
    ss_ctl_likelihoods = np.load(os.path.join(SS_DIRPATH, "mcmc_results", "wt_mcmc_likelihoods.npy"))
    ss_rd_likelihoods = np.load(os.path.join(SS_DIRPATH, "mcmc_results", "rd_mcmc_likelihoods.npy"))
    return (
        PARTICLE_DIRPATH,
        SS_DIRPATH,
        pa_ctl_likelihoods,
        pa_rd_likelihoods,
        ss_ctl_likelihoods,
        ss_rd_likelihoods,
    )


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
        ax.hist(
            pa_ctl_likelihoods[1024::64, :, 0].flatten(),
            bins=75, histtype="step", color=CONTROL_PALETTE, alpha=0.5,
            density=True, label="Control - Particle"
        )
        ax.hist(
            ss_ctl_likelihoods[1024::64, :, 0].flatten(),
            bins=75, histtype="step", color=CONTROL_PALETTE,
            density=True, label="Control - Stick-slip"
        )

        # Plot RD comparison:
        ax.hist(
            pa_rd_likelihoods[1024::64, :, 0].flatten(),
            bins=75, histtype="step", color=RD_PALETTE, alpha=0.5,
            density=True, label="RD - Particle"
        )
        ax.hist(
            ss_rd_likelihoods[1024::64, :, 0].flatten(),
            bins=75, histtype="step", color=RD_PALETTE,
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
def _(np):
    CAT_VAR = "C(phenotype, Treatment(reference='CTL'))[T.RD]"

    def get_fit_target(phenotype, regression_results, scaled_x):
        # Get base parameters without phenotype interaction:
        base_intercept = regression_results.params["Intercept"]
        base_intercept_se = regression_results.bse["Intercept"]
        linear_coeff = regression_results.params["scaled_particle_count"]
        linear_coeff_se = regression_results.bse["scaled_particle_count"]
        square_coeff = regression_results.params["I(scaled_particle_count ** 2)"]
        square_coeff_se = regression_results.bse["I(scaled_particle_count ** 2)"]

        # Get phenotype interactions:
        phenotype_intercept = regression_results.params[f"{CAT_VAR}"]
        phenotype_intercept_se = regression_results.bse[f"{CAT_VAR}"]
        linear_interaction = regression_results.params[f"{CAT_VAR}:scaled_particle_count"]
        linear_interaction_se = regression_results.bse[f"{CAT_VAR}:scaled_particle_count"]
        square_interaction = regression_results.params[f"{CAT_VAR}:I(scaled_particle_count ** 2)"]
        square_interaction_se = regression_results.bse[f"{CAT_VAR}:I(scaled_particle_count ** 2)"]

        # Construct quadratic:
        a = square_coeff + (phenotype * square_interaction)
        b = linear_coeff + (phenotype * linear_interaction)
        c = base_intercept + (phenotype * phenotype_intercept)

        # Do error propagation to get standard errors in coefficients (multiplication preserves percentage errors):
        sq_int_se = np.abs((phenotype * square_interaction) * (square_interaction_se / square_interaction))
        lin_int_se = np.abs((phenotype * linear_interaction) * (linear_interaction_se / linear_interaction))
        base_int_se = np.abs((phenotype * phenotype_intercept) * (phenotype_intercept_se / phenotype_intercept))

        a_se = np.sqrt(square_coeff_se**2 + sq_int_se**2)
        b_se = np.sqrt(linear_coeff_se**2 + lin_int_se**2)
        c_se = np.sqrt(base_intercept_se**2 + base_int_se**2)

        # Get regression prediction:
        y = a*(scaled_x**2) + b*scaled_x + c

        # Get regression standard error:
        quad_error = (a_se / a) * (a*(scaled_x**2))
        linear_error = (b_se / b) * (b*scaled_x)
        se = np.sqrt(quad_error**2 + linear_error**2 + c_se**2)
        return y, se
    return (get_fit_target,)


@app.cell
def _(get_fit_target, np, pd):
    from statsmodels.regression import mixed_linear_model

    WETLAB_METRICS = [
        "mean_speed",
        "mean_mr",
        "anni",
        "coherency_fraction"
    ]

    # Load wet lab data:
    site_dataframe = pd.read_csv("wetlab_data/site_dataframe.csv")
    particle_counts = np.array(site_dataframe["particle_count"])

    # Get scaled points to query:
    regression_inputs = 375
    scaled_regression_inputs = (regression_inputs - np.mean(particle_counts)) / np.std(particle_counts)

    regression_dict = {}
    for wetlab_metric in WETLAB_METRICS:
        # Retrieve results of regression, and do error propagation on parameters:
        regression_results = mixed_linear_model.MixedLMResults.load(f"wetlab_data/{wetlab_metric}.res")
        wt_fit_target, wt_fit_se = get_fit_target(-0.5, regression_results, scaled_regression_inputs)
        rd_fit_target, rd_fit_se = get_fit_target( 0.5, regression_results, scaled_regression_inputs)

        # Need to convert speed back to µm/min:
        if wetlab_metric == "mean_speed":
            wt_fit_target /= 60
            wt_fit_se /= 60
            rd_fit_target /= 60
            rd_fit_se /= 60

        # Store in dictionary:
        regression_dict[wetlab_metric] = {
            "wt_mean": wt_fit_target,
            "wt_stddev": wt_fit_se,
            "rd_mean": rd_fit_target,
            "rd_stddev": rd_fit_se
        }
    return (regression_dict,)


@app.cell
def _(PARTICLE_DIRPATH, SS_DIRPATH, np, os):
    # Compare posterior means:
    pa_ctl_p_mean = np.load(os.path.join(PARTICLE_DIRPATH, "mcmc_results", "wt_posterior_mean.npy"))
    pa_rd_p_mean = np.load(os.path.join(PARTICLE_DIRPATH, "mcmc_results", "rd_posterior_mean.npy"))

    ss_ctl_p_mean = np.load(os.path.join(SS_DIRPATH, "mcmc_results", "wt_posterior_mean.npy"))
    ss_rd_p_mean = np.load(os.path.join(SS_DIRPATH, "mcmc_results", "rd_posterior_mean.npy"))
    return pa_ctl_p_mean, pa_rd_p_mean, ss_ctl_p_mean, ss_rd_p_mean


@app.cell
def _(
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    PA_PALETTE,
    SS_PALETTE,
    TEXT_WIDTH,
    datetime,
    os,
    pa_ctl_p_mean,
    pa_rd_p_mean,
    plt,
    regression_dict,
    ss_ctl_p_mean,
    ss_rd_p_mean,
):
    def plot_ctl_tradeoff(ax):
        # Plot scatter of fit:
        ax.scatter(
            pa_ctl_p_mean[0, :, 2], pa_ctl_p_mean[3, :, 2],
            s=2, alpha=0.5, color=PA_PALETTE, label="Particle"
        )
        ax.scatter(
            ss_ctl_p_mean[0, :, 2], ss_ctl_p_mean[3, :, 2],
            s=2, alpha=0.3, color=SS_PALETTE, label="Stick-slip"
        )

        # Plot target:
        ctl_speed = regression_dict["mean_speed"]["wt_mean"]
        ctl_speed_std = regression_dict["mean_speed"]["wt_stddev"]
        ctl_coherency = regression_dict["coherency_fraction"]["wt_mean"]
        ctl_coherency_std = regression_dict["coherency_fraction"]["wt_stddev"]
        ax.scatter(ctl_speed, ctl_coherency, c='k', s=10, alpha=0.75, label="Control Fit Target")
        ax.errorbar(ctl_speed, ctl_coherency, xerr=ctl_speed_std, yerr=ctl_coherency_std, c='k', alpha=0.75)

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
        rd_speed = regression_dict["mean_speed"]["rd_mean"]
        rd_speed_std = regression_dict["mean_speed"]["rd_stddev"]
        rd_coherency = regression_dict["coherency_fraction"]["rd_mean"]
        rd_coherency_std = regression_dict["coherency_fraction"]["rd_stddev"]
        ax.scatter(rd_speed, rd_coherency, c='k', s=10, alpha=0.75, label="RD Fit Target")
        ax.errorbar(rd_speed, rd_coherency, xerr=rd_speed_std, yerr=rd_coherency_std, c='k', alpha=0.75)

        # Add labels:
        ax.text(0.98, 0.05, 'RD Posterior Predictions', ha='right', transform=ax.transAxes)
        ax.set_xlabel("Speed")
        ax.set_ylabel("Coherency")
        ax.legend()

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
