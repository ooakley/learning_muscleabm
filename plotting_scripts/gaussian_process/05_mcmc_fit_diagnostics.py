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
    return (
        MaxNLocator,
        arviz_stats,
        datetime,
        json,
        np,
        os,
        pd,
        plt,
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
    CONTRAST_PALETTE = "#DC267F"
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
        "script": "mcmc_fit_diagnostics.py",
        "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_commit_hash": commit_hash
    }
    return (
        CONTROL_PALETTE,
        FULL_HEIGHT,
        METADATA_DICTIONARY,
        OUT_DIRPATH,
        RD_PALETTE,
        TEXT_WIDTH,
    )


@app.cell
def _():
    WETLAB_METRICS = [
        "mean_speed",
        "mean_mr",
        "anni",
        "coherency_fraction"
    ]

    METRICS_LABELS = [
        "Speed ($\mu$m/min)",
        "MR",
        "ANNI",
        "Coherency"
    ]
    return METRICS_LABELS, WETLAB_METRICS


@app.cell
def _(np, os):
    EXPERIMENT_DIRPATH = "model_experiments/2026-10-02-collisions_shape"
    # mcmc_results = "wide_mcmc_results"
    # mcmc_results = "05_cov_mcmc_results"
    mcmc_results = "hm2/disc_cov_mcmc_results"

    ctl_mcmc_chain = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "wt_mcmc_chain.npy"))
    ctl_mcmc_ar = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "wt_mcmc_acceptance_rate.npy"))
    ctl_mcmc_likelihood = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "wt_mcmc_likelihoods.npy"))

    rd_mcmc_chain = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "rd_mcmc_chain.npy"))
    rd_mcmc_ar = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "rd_mcmc_acceptance_rate.npy"))
    rd_mcmc_likelihood = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "rd_mcmc_likelihoods.npy"))
    return (
        EXPERIMENT_DIRPATH,
        ctl_mcmc_ar,
        ctl_mcmc_chain,
        ctl_mcmc_likelihood,
        mcmc_results,
        rd_mcmc_chain,
    )


@app.cell
def _(ctl_mcmc_ar, np):
    np.mean(ctl_mcmc_ar[0, :])
    return


@app.cell
def _(ctl_mcmc_chain):
    ctl_mcmc_chain.shape
    return


@app.cell
def _(ctl_mcmc_likelihood):
    test_flatten = ctl_mcmc_likelihood[::32, :, 0].flatten()
    return


@app.cell
def _(ctl_mcmc_likelihood):
    ctl_mcmc_likelihood.shape
    return


@app.cell
def _(ctl_mcmc_chain, np, plt):
    chains = ctl_mcmc_chain[:, :, 0, 3]
    x_pos = np.repeat(np.arange(0, chains.shape[0]), chains.shape[1], axis=0)

    fig, ax = plt.subplots(figsize=(8.0, 2.5))
    ax.scatter(x_pos.flatten(), chains.flatten(), alpha=0.25, s=0.25, edgecolors="none", c='k')
    ax.set_xlim(0, chains.shape[0])
    ax.set_ylim(0, 1)
    plt.show()
    return


@app.cell
def _(ctl_mcmc_chain):
    ctl_mcmc_chain.shape
    return


@app.cell
def _(ctl_mcmc_chain, plt):
    # 24576
    def plot_quick_hist():
        fig, ax = plt.subplots()
        chain_length = ctl_mcmc_chain.shape[0]
        half_index = int(chain_length // 2)
        ax.hist(ctl_mcmc_chain[half_index::32, :, 0, 3].flatten(), bins=100)
        ax.set_xlim(0, 1)
        plt.show()

    plot_quick_hist()
    return


@app.cell
def _(ctl_mcmc_chain, np):
    np.linspace(0, 1, ctl_mcmc_chain.shape[1]).shape
    return


@app.cell
def _(EXPERIMENT_DIRPATH, mcmc_results, np, os):
    ctl_rung_acceptance = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "wt_rung_acceptance_rate.npy"))
    rd_rung_acceptance = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "rd_rung_acceptance_rate.npy"))
    return ctl_rung_acceptance, rd_rung_acceptance


@app.cell
def _(ctl_rung_acceptance, plt, rd_rung_acceptance):
    plt.plot(ctl_rung_acceptance)
    plt.plot(rd_rung_acceptance)
    return


@app.cell
def _(arviz_stats, ctl_mcmc_chain, rd_mcmc_chain):
    # ctl_posterior = ctl_mcmc_chain[:, :, -3, :]
    # rd_posterior = rd_mcmc_chain[:, :, 0, :]
    quarter_index = int(ctl_mcmc_chain.shape[0] / 4)
    print(quarter_index)
    ctl_rhat = arviz_stats.rhat(ctl_mcmc_chain[quarter_index:, :, 0, :], chain_axis=1, draw_axis=0)
    rd_rhat = arviz_stats.rhat(rd_mcmc_chain[quarter_index:, :, 0, :], chain_axis=1, draw_axis=0)
    return ctl_rhat, rd_rhat


@app.cell
def _(ctl_rhat):
    ctl_rhat
    return


@app.cell
def _(rd_rhat):
    rd_rhat
    return


@app.cell
def _(arviz_stats, ctl_mcmc_chain, rd_mcmc_chain):
    ctl_ess = arviz_stats.ess(ctl_mcmc_chain[:, :, 0, :], chain_axis=1, draw_axis=0)
    rd_ess = arviz_stats.ess(rd_mcmc_chain[:, :, 0, :], chain_axis=1, draw_axis=0)
    return ctl_ess, rd_ess


@app.cell
def _(ctl_ess):
    ctl_ess
    return


@app.cell
def _(rd_ess):
    rd_ess
    return


@app.cell
def _(ctl_mcmc_chain):
    ctl_mcmc_chain.shape
    return


@app.cell
def _(EXPERIMENT_DIRPATH, json, os):
    with open(os.path.join(EXPERIMENT_DIRPATH, "config.json")) as json_file:
        config_dict = json.load(json_file)

    parameter_list = [parameter_range[0] for parameter_range in config_dict["gridsearch_parameters"]]
    parameter_list.remove("numberOfCells")
    return (parameter_list,)


@app.cell
def _(ctl_ess, ctl_rhat, np, parameter_list, pd, rd_ess, rd_rhat):
    column_names = [
        "CTL $\\hat{R}$",
        "CTL ESS",
        "RD $\\hat{R}$",
        "RD ESS"
    ]
    dataframe_array = np.stack([ctl_rhat, ctl_ess, rd_rhat, rd_ess], axis=1)
    diagnostics_dataframe = pd.DataFrame(dataframe_array, columns=column_names, index=parameter_list)
    return (diagnostics_dataframe,)


@app.cell
def _(diagnostics_dataframe, parameter_list):
    print(diagnostics_dataframe.to_latex(index=parameter_list, float_format="%.2f"))
    return


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
def _(WETLAB_METRICS, get_fit_target, np, pd):
    from statsmodels.regression import mixed_linear_model

    # Load wet lab data:
    site_dataframe = pd.read_csv("wetlab_data/site_dataframe.csv")
    particle_counts = np.array(site_dataframe["particle_count"])

    # Get scaled points to query:
    regression_inputs = np.linspace(0, 400, 200)
    scaled_regression_inputs = (regression_inputs - np.mean(particle_counts)) / np.std(particle_counts)

    regression_dict = {}
    for wetlab_metric in WETLAB_METRICS:
        # Retrieve results of regression, and do error propagation on parameters:
        regression_results = mixed_linear_model.MixedLMResults.load(f"wetlab_data/{wetlab_metric}.res")
        wt_fit_target, wt_fit_se = get_fit_target(0.0, regression_results, scaled_regression_inputs)
        rd_fit_target, rd_fit_se = get_fit_target(1.0, regression_results, scaled_regression_inputs)

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
    return regression_dict, regression_inputs


@app.cell
def _(EXPERIMENT_DIRPATH, mcmc_results, np, os):
    # Load posterior predictions:
    wt_posterior_preds = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "wt_posterior_mean.npy"))
    rd_posterior_preds = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "rd_posterior_mean.npy"))
    posterior_predictions = [wt_posterior_preds, rd_posterior_preds]

    # THIN_FACTOR = 64
    # ctl_mle_idx = np.argsort(ctl_mcmc_likelihood[16384::THIN_FACTOR, :, 0].flatten())[-1024:]
    # rd_mle_idx = np.argsort(rd_mcmc_likelihood[16384::THIN_FACTOR, :, 0].flatten())[-1024:]
    # ctl_mle = ctl_mcmc_chain[::THIN_FACTOR, :, 0, :].reshape(-1, 11)[ctl_mle_idx, :]
    # rd_mle = rd_mcmc_chain[::THIN_FACTOR, :, 0, :].reshape(-1, 11)[rd_mle_idx, :]

    # posterior_predictions = [wt_posterior_preds[:, ctl_mle_idx], rd_posterior_preds[:, rd_mle_idx]]
    return (posterior_predictions,)


@app.cell
def _(posterior_predictions):
    posterior_predictions[0].shape
    return


@app.cell
def _(EXPERIMENT_DIRPATH, mcmc_results, np, os):
    wt_posterior_std = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "wt_posterior_std.npy"))
    rd_posterior_std = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "rd_posterior_std.npy"))
    return rd_posterior_std, wt_posterior_std


@app.cell
def _(EXPERIMENT_DIRPATH, mcmc_results, np, os):
    wt_posterior_sigma = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "wt_posterior_sigma.npy"))
    rd_posterior_sigma = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "rd_posterior_sigma.npy"))
    return (wt_posterior_sigma,)


@app.cell
def _(wt_posterior_sigma):
    wt_posterior_sigma.shape
    return


@app.cell
def _(np, wt_posterior_sigma):
    metric_sigma = wt_posterior_sigma[0]                                   # (samples, 11, 11)
    metric_std = np.sqrt(np.diagonal(metric_sigma, axis1=1, axis2=2))       # (samples, 11)
    wt_posterior_corr = metric_sigma / (metric_std[:, :, None] * metric_std[:, None, :])
    return (wt_posterior_corr,)


@app.cell
def _(wt_posterior_corr):
    wt_posterior_corr.shape
    return


@app.cell
def _(np, wt_posterior_corr):
    adjacent_corr = np.diagonal(wt_posterior_corr, offset=1, axis1=1, axis2=2)   # (samples, 10)
    print(adjacent_corr.mean(), adjacent_corr.min(), adjacent_corr.max())
    print(np.round(wt_posterior_corr.mean(axis=0), 3))                            # average 11x11 correlation
    return


@app.cell
def _(np, wt_posterior_corr):
    from scipy.optimize import minimize_scalar

    lag_matrix = np.abs(np.arange(11)[:, None] - np.arange(11)[None, :])

    def fit_lengthscale(corr):
        loss = lambda log_l: np.sum((corr - np.exp(-0.5 * (lag_matrix / np.exp(log_l)) ** 2)) ** 2)
        return np.exp(minimize_scalar(loss, bounds=(np.log(0.05), np.log(500)), method="bounded").x)

    fitted_lengthscale = np.array([fit_lengthscale(corr) for corr in wt_posterior_corr])
    return (fitted_lengthscale,)


@app.cell
def _(fitted_lengthscale, plt):
    plt.hist(fitted_lengthscale * 25, bins=100)
    return


@app.cell
def _(WETLAB_METRICS, gp_targets, np, rd_posterior_std, wt_posterior_std):
    for i in range(4):
        print(f"--- --- {WETLAB_METRICS[i]} --- ---")
        wt_Sigma = gp_targets[0][1][i]
        print(wt_Sigma.shape)
        rd_Sigma = gp_targets[1][1][i]
        print(f"Mean CTL std. dev. {np.mean(np.sqrt(np.diag(wt_Sigma)))}")
        print(f"Mean RD std. dev. {np.mean(np.sqrt(np.diag(rd_Sigma)))}")
        print(f"Mean CTL GP estimate std. dev. {np.mean(wt_posterior_std[i, :, :])}")
        print(f"Mean RD  GP estimate std. dev. {np.mean(rd_posterior_std[i, :, :])}")

        print(f"CTL ratio: {np.mean(wt_posterior_std[i, :, :]) / np.mean(np.sqrt(np.diag(wt_Sigma)))}")
        print(f"RD ratio: {np.mean(rd_posterior_std[i, :, :]) / np.mean(np.sqrt(np.diag(rd_Sigma)))}")
    return


@app.cell
def _(posterior_predictions):
    posterior_predictions[0].shape
    return


@app.cell
def _(WETLAB_METRICS, np, os):
    def get_gp_targets():
        wt_targets = []
        wt_covs = []
        rd_targets = []
        rd_covs = []

        gp_wetlab_dirpath = os.path.join("wetlab_data", "gp_results")

        for wetlab_metric in WETLAB_METRICS:
            wt_targets.append(np.load(os.path.join(gp_wetlab_dirpath, f"CTL_{wetlab_metric}_mean.npy")))
            wt_covs.append(np.load(os.path.join(gp_wetlab_dirpath, f"CTL_{wetlab_metric}_sigma.npy")))
            rd_targets.append(np.load(os.path.join(gp_wetlab_dirpath, f"RD_{wetlab_metric}_mean.npy")))
            rd_covs.append(np.load(os.path.join(gp_wetlab_dirpath, f"RD_{wetlab_metric}_sigma.npy")))

        return (wt_targets, wt_covs), (rd_targets, rd_covs)

    gp_targets = get_gp_targets()
    return (gp_targets,)


@app.cell
def _(
    CONTROL_PALETTE,
    FULL_HEIGHT,
    METADATA_DICTIONARY,
    METRICS_LABELS,
    MaxNLocator,
    OUT_DIRPATH,
    RD_PALETTE,
    TEXT_WIDTH,
    WETLAB_METRICS,
    datetime,
    gp_targets,
    np,
    os,
    plt,
    posterior_predictions,
):
    QUERY_COUNTS = list(np.linspace(50, 300, 11).astype(int))
    FIT_PALETTE = "#FFBF00"

    def plot_posterior_predictions():
        fig, axs = plt.subplots(4, 3, figsize=(TEXT_WIDTH, FULL_HEIGHT), sharex=True, sharey="row")

        # Plot against wetlab data:
        for phenotype_index in range(2):
            ph_label = ["wt", "rd"][phenotype_index]
            ph_palette = [CONTROL_PALETTE, RD_PALETTE][phenotype_index]
            for metric_index, metric_name in enumerate(WETLAB_METRICS):
                # Ensure we don't have too much tick clutter:
                axs[metric_index, phenotype_index].xaxis.set_major_locator(MaxNLocator(3))
                axs[metric_index, phenotype_index].xaxis.set_major_locator(MaxNLocator(3))

                # # Plot mean regression line:
                # mean_regression = regression_dict[metric_name][f"{ph_label}_mean"]
                # axs[metric_index, phenotype_index].plot(
                #     regression_inputs, mean_regression, c=ph_palette
                # )

                # # Fill in with full confidence interval:
                # stddev = regression_dict[metric_name][f"{ph_label}_stddev"]
                # upper_bound = mean_regression + (1.96 * stddev)
                # lower_bound = mean_regression - (1.96 * stddev)
                # axs[metric_index, phenotype_index].fill_between(
                #     regression_inputs, upper_bound, lower_bound, 
                #     color=ph_palette, alpha=0.25
                # )

                gp_mean = gp_targets[phenotype_index][0][metric_index]
                gp_sigma = gp_targets[phenotype_index][1][metric_index]
                gp_stddev = np.sqrt(np.diag(gp_sigma))

                axs[metric_index, phenotype_index].plot(
                    QUERY_COUNTS, gp_mean, c=ph_palette
                )

                # Fill in with full confidence interval:
                upper_bound = gp_mean + (1.96 * gp_stddev)
                lower_bound = gp_mean - (1.96 * gp_stddev)
                axs[metric_index, phenotype_index].fill_between(
                    QUERY_COUNTS, upper_bound, lower_bound, 
                    color=ph_palette, alpha=0.25
                )

                # Plot posterior distributions:
                posterior_means = []
                posterior_low = []
                posterior_high = []
                for count_index, count in enumerate(QUERY_COUNTS):
                    posterior_data = posterior_predictions[phenotype_index]
                    posterior_distribution = posterior_data[metric_index, :, count_index]
                    mean = np.mean(posterior_distribution)
                    eti_low = np.quantile(posterior_distribution, 0.05)
                    eti_high = np.quantile(posterior_distribution, 0.95)

                    posterior_means.append(mean)
                    posterior_low.append(eti_low)
                    posterior_high.append(eti_high)

                    errorbar = np.expand_dims(np.array([eti_low, eti_high]), axis=1)
                    errorbar -= mean
                    # axs[metric_index, phenotype_index].errorbar(
                    #     count, mean, yerr=np.abs(errorbar),
                    #     fmt='o', c='k', ms=3, alpha=0.5
                    # )

                # Plot posterior distributions
                axs[metric_index, phenotype_index].scatter(
                    QUERY_COUNTS, posterior_means, edgecolors="none",
                    color=FIT_PALETTE, alpha=0.5, s=15
                )
                axs[metric_index, phenotype_index].plot(
                    QUERY_COUNTS, posterior_means,
                    color=FIT_PALETTE, alpha=1.0
                )
                axs[metric_index, phenotype_index].fill_between(
                    QUERY_COUNTS, posterior_low, posterior_high, 
                    color=FIT_PALETTE, alpha=0.25
                )

                # Format axes:
                if phenotype_index == 0:
                    axs[metric_index, phenotype_index].set_ylabel(METRICS_LABELS[metric_index])

                if metric_index == 0:
                    axs[metric_index, phenotype_index].text(
                        0.95, 0.85, ["Control", "RD"][phenotype_index],
                        c=ph_palette,
                        horizontalalignment='right', verticalalignment='top',
                        transform=axs[metric_index, phenotype_index].transAxes
                    )
                    axs[metric_index, phenotype_index].text(
                        0.95, 0.95, "Fit Prediction",
                        c=FIT_PALETTE,
                        horizontalalignment='right', verticalalignment='top',
                        transform=axs[metric_index, phenotype_index].transAxes
                    )

        # Posterior comparison:
        for metric_index, metric_name in enumerate(WETLAB_METRICS):
            for phenotype_index in range(2):
                ph_palette = [CONTROL_PALETTE, RD_PALETTE][phenotype_index]
                means = []
                errorbars = []

                posterior_means = []
                posterior_low = []
                posterior_high = []
                for count_index, count in enumerate(QUERY_COUNTS):
                    posterior_data = posterior_predictions[phenotype_index]
                    posterior_distribution = posterior_data[metric_index, :, count_index]
                    mean = np.mean(posterior_distribution)
                    eti_low = np.quantile(posterior_distribution, 0.05)
                    eti_high = np.quantile(posterior_distribution, 0.95)
                    posterior_means.append(mean)
                    posterior_low.append(eti_low)
                    posterior_high.append(eti_high)
                    # errorbar = np.array([eti_low, eti_high])
                    # errorbar -= mean
                    # means.append(mean)
                    # errorbars.append(errorbar)

                # Plot posterior distributions
                axs[metric_index, 2].scatter(
                    QUERY_COUNTS, posterior_means, edgecolors="none",
                    color=ph_palette, alpha=0.5, s=15
                )
                axs[metric_index, 2].plot(
                    QUERY_COUNTS, posterior_means,
                    color=ph_palette, alpha=1.0
                )
                axs[metric_index, 2].fill_between(
                    QUERY_COUNTS, posterior_low, posterior_high, 
                    color=ph_palette, alpha=0.25
                )

                # # Plot line:
                # errorbars = np.stack(errorbars, axis=1)
                # axs[metric_index, 2].errorbar(
                #     QUERY_COUNTS, means, yerr=np.abs(errorbars),
                #     fmt='o-', c=ph_palette, ms=3, alpha=0.5
                # )

            # Label comparison:
            if metric_index == 0:
                axs[metric_index, 2].text(
                    0.96, 0.95, "Fit Control",
                    c=CONTROL_PALETTE,
                    horizontalalignment='right', verticalalignment='top',
                    transform=axs[metric_index, 2].transAxes
                )
                axs[metric_index, 2].text(
                    0.96, 0.85, "Fit RD",
                    c=RD_PALETTE,
                    horizontalalignment='right', verticalalignment='top',
                    transform=axs[metric_index, 2].transAxes
                )

        # Add global xlabel:
        fig.text(0.5, 0.015, "Cell Density (particles/site)", va='baseline', ha='center')

        # Adjust subplots:
        w_edging = 0.1
        h_edging = 0.06
        fig.subplots_adjust(w_edging, h_edging, 1 - w_edging, 1 - h_edging, wspace=0.1, hspace=0.1)

        # Save:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(os.path.join(OUT_DIRPATH, "posterior_predictions.png"), dpi=300, metadata=METADATA_DICTIONARY, transparent=True)
        plt.show()

    plot_posterior_predictions()
    return (QUERY_COUNTS,)


@app.cell
def _():
    # # Visualisations of high likelihood parameters from Sobol' search:
    # ctl_sobol_likelihoods = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "wt_sobol_likelihoods.npy"))
    # rd_sobol_likelihoods = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "rd_sobol_likelihoods.npy"))
    return


@app.cell
def _(EXPERIMENT_DIRPATH, mcmc_results, np, os):
    ctl_mcmc_likelihoods = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "wt_mcmc_likelihoods.npy"))
    ctl_posterior_ll = ctl_mcmc_likelihoods[512::8, :, 0].flatten()
    rd_mcmc_likelihoods = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "rd_mcmc_likelihoods.npy"))
    rd_posterior_ll = rd_mcmc_likelihoods[512::8, :, 0].flatten()
    return ctl_posterior_ll, rd_posterior_ll


@app.cell
def _(
    CONTROL_PALETTE,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    RD_PALETTE,
    TEXT_WIDTH,
    ctl_posterior_ll,
    datetime,
    os,
    plt,
    rd_posterior_ll,
):
    def plot_log_likelihood_distribution():
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.0))

        # Plot histograms:
        ax.hist(ctl_posterior_ll, histtype="step", color=CONTROL_PALETTE, bins=50, density=True, label="Control")
        ax.hist(rd_posterior_ll, histtype="step", color=RD_PALETTE, bins=50, density=True, label="RD")

        # Plot vlines of max likelihoods of Sobol' search:
        y_limits = ax.get_ylim()
        # ax.vlines(np.max(ctl_sobol_likelihoods), *y_limits, ls="--", color=CONTROL_PALETTE, label="CTL Max. GS likelihood")
        # ax.vlines(np.max(rd_sobol_likelihoods), *y_limits, ls="--", color=RD_PALETTE, label="RD Max. GS likelihood")
        ax.set_ylim(*y_limits)

        ax.legend()
        ax.set_xlabel("$log(\\mathscr{L}(\\theta|X))$")
        ax.set_ylabel("Density")

        fig.subplots_adjust(0.1, 0.2, 0.9, 0.8)

        # Save plot:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(os.path.join(OUT_DIRPATH, "posterior_likelihoods.png"), dpi=300, metadata=METADATA_DICTIONARY, transparent=True)
        plt.show()

    plot_log_likelihood_distribution()
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
def _(TEXT_WIDTH, ctl_sobol_likelihoods, load_image, np, plt):
    def plot_ctl_fit_collage(gridsize, image_type):
        # Get relevant indices:
        total_image_count = gridsize * gridsize
        sobol_indices = np.argsort(ctl_sobol_likelihoods)[-total_image_count:]

        count = 0
        collage_array = []
        for i in range(gridsize):
            collage_row = []
            for j in range(gridsize):
                collage_row.append(load_image(sobol_indices[count], image_type))
                count += 1
            collage_row = np.concatenate(collage_row, axis=1)
            collage_array.append(collage_row)

        collage_array = np.concatenate(collage_array, axis=0)
        print(collage_array.shape)

        fig, ax = plt.subplots(figsize=(TEXT_WIDTH, TEXT_WIDTH))
        ax.imshow(collage_array)
        ax.set_axis_off()
        plt.show()

    plot_ctl_fit_collage(5, "trajectory")
    return


@app.cell
def _(
    CONTROL_PALETTE,
    FULL_HEIGHT,
    METADATA_DICTIONARY,
    METRICS_LABELS,
    QUERY_COUNTS,
    RD_PALETTE,
    TEXT_WIDTH,
    WETLAB_METRICS,
    datetime,
    np,
    plt,
    posterior_predictions,
    regression_dict,
    regression_inputs,
):
    def plot_alt_posterior_predictions():
        fig, axs = plt.subplots(4, 2, figsize=(TEXT_WIDTH, FULL_HEIGHT), sharex=True, sharey="row")

        # Plot wetlab data:
        for metric_index, metric_name in enumerate(WETLAB_METRICS):
            # Plot mean regression line:
            axs[metric_index, 0].plot(
                regression_inputs,
                regression_dict[metric_name]["wt_mean"],
                c=CONTROL_PALETTE, label="Control"
            )
            axs[metric_index, 0].plot(
                regression_inputs,
                regression_dict[metric_name]["rd_mean"],
                c=RD_PALETTE, label="RD"
            )

            # Fill in with full confidence interval:
            wt_stddev = regression_dict[metric_name]["wt_stddev"]
            wt_upper_bound = regression_dict[metric_name]["wt_mean"] + (1.96 * wt_stddev)
            wt_lower_bound = regression_dict[metric_name]["wt_mean"] - (1.96 * wt_stddev)
            axs[metric_index, 0].fill_between(
                regression_inputs, wt_upper_bound, wt_lower_bound, 
                color=CONTROL_PALETTE, alpha=0.25
            )

            rd_stddev = regression_dict[metric_name]["rd_stddev"]
            rd_upper_bound = regression_dict[metric_name]["rd_mean"] + (1.96 * rd_stddev)
            rd_lower_bound = regression_dict[metric_name]["rd_mean"] - (1.96 * rd_stddev)
            axs[metric_index, 0].fill_between(
                regression_inputs, rd_upper_bound, rd_lower_bound, 
                color=RD_PALETTE, alpha=0.25
            )

            # Label y-axis
            axs[metric_index, 0].set_ylabel(METRICS_LABELS[metric_index])

            if metric_index == 0:
                axs[metric_index, 0].text(
                    0.95, 0.95, "Experimental Data",
                    horizontalalignment='right', verticalalignment='top',
                    transform=axs[metric_index, 0].transAxes
                )
                axs[metric_index, 0].legend(loc="lower left")

        # Plot posterior predictions:
        for metric_index, metric_name in enumerate(WETLAB_METRICS):
            for phenotype_index in range(2):
                ph_palette = [CONTROL_PALETTE, RD_PALETTE][phenotype_index]
                means = []
                errorbars = []
                for count_index, count in enumerate(QUERY_COUNTS):
                    posterior_data = posterior_predictions[phenotype_index]
                    posterior_distribution = posterior_data[metric_index, :, count_index]
                    mean = np.mean(posterior_distribution)
                    eti_low = np.quantile(posterior_distribution, 0.025)
                    eti_high = np.quantile(posterior_distribution, 0.975)
                    errorbar = np.array([eti_low, eti_high])
                    errorbar -= mean
                    means.append(mean)
                    errorbars.append(errorbar)

                # Plot line:
                errorbars = np.stack(errorbars, axis=1)
                label = ["Control", "RD"][phenotype_index]
                axs[metric_index, 1].errorbar(
                    QUERY_COUNTS, means, yerr=np.abs(errorbars),
                    fmt='o-', c=ph_palette, ms=5, alpha=0.5, label=label
                )

            # Label comparison:
            if metric_index == 0:
                axs[metric_index, 1].text(
                    0.96, 0.95, "Model Data",
                    c="k",
                    horizontalalignment='right', verticalalignment='top',
                    transform=axs[metric_index, 1].transAxes
                )
                axs[metric_index, 1].legend(loc="lower left")

        # Add global xlabel:
        fig.text(0.5, 0.015, "Cell Density (particles/site)", va='baseline', ha='center')

        # Adjust subplots:
        w_edging = 0.1
        h_edging = 0.06
        fig.subplots_adjust(w_edging, h_edging, 1 - w_edging, 1 - h_edging, wspace=0.1, hspace=0.1)

        # Save:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # plt.savefig(os.path.join(OUT_DIRPATH, "alt_posterior_predictions.png"), dpi=300, metadata=METADATA_DICTIONARY, transparent=True)
        plt.show()

    plot_alt_posterior_predictions()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
