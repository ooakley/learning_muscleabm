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
    return datetime, np, os, pd, plt, subprocess


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
        "script": "mcmc_fit_diagnostics.py",
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

    MODEL_METRICS = [
        "speeds",
        "meander_ratios",
        "ann_indices",
        "coherency"
    ]
    return MODEL_METRICS, WETLAB_METRICS


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
    regression_input = 350
    scaled_regression_input = (regression_input - np.mean(particle_counts)) / np.std(particle_counts)

    regression_dict = {}
    for wetlab_metric in WETLAB_METRICS:
        # Retrieve results of regression, and do error propagation on parameters:
        regression_results = mixed_linear_model.MixedLMResults.load(f"wetlab_data/{wetlab_metric}.res")
        wt_fit_target, wt_fit_se = get_fit_target(-0.5, regression_results, scaled_regression_input)
        rd_fit_target, rd_fit_se = get_fit_target( 0.5, regression_results, scaled_regression_input)

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
def _(MODEL_METRICS, np, os):
    CS_POSTERIOR_DIRPATH = "model_experiments/2026-06-12-cs_posterior"
    model_metric_dict = {}
    for model_metric in MODEL_METRICS:
        metric_data = np.load(os.path.join(CS_POSTERIOR_DIRPATH, "summary_data", f"{model_metric}.npy"))
        model_metric_dict[model_metric] = metric_data
        print(metric_data.min(), metric_data.max())
    return CS_POSTERIOR_DIRPATH, model_metric_dict


@app.cell
def _(
    CONTROL_PALETTE,
    RD_PALETTE,
    model_metric_dict,
    np,
    plt,
    regression_dict,
):
    def plot_posterior_sim():
        fig, axs = plt.subplots(4, 1, figsize=(2.5, 4), layout="constrained")
        for metric_index in range(4):
            # Retrieve target wet-lab data:
            target_data = list(regression_dict.values())[metric_index]
            simulation_data = list(model_metric_dict.values())[metric_index]
            simulation_data = np.mean(simulation_data, axis=1)

            # Plot CTL histogram:
            axs[metric_index].hist(
                simulation_data[:1024],
                histtype="step", color=CONTROL_PALETTE,
                bins=50, density=True, alpha=0.75
            )

            # Plot RD histogram:
            axs[metric_index].hist(
                simulation_data[1024:],
                histtype="step", color=RD_PALETTE,
                bins=50, density=True, alpha=0.75
            )

            # Plot control range:
            y_limits = axs[metric_index].get_ylim()
            axs[metric_index].vlines(
                target_data["wt_mean"], y_limits[0], y_limits[1],
                color=CONTROL_PALETTE, alpha=0.75
            )
            axs[metric_index].vlines(
                target_data["rd_mean"], y_limits[0], y_limits[1],
                color=RD_PALETTE, alpha=0.75
            )
            axs[metric_index].set_ylim(*y_limits)

        plt.show()

    plot_posterior_sim()
    return


@app.cell
def _():
    # def ll_calculation(t_mean, t_se, mu, sigma):
    #     l = np.abs(t_mean - mu)
    #     scale = 1 / np.sqrt(2 * np.pi * (t_se**2 + sigma**2))
    #     exponent = - l**2 / (2 * (t_se**2 + sigma**2))
    #     return np.log(scale) + exponent

    # def calculate_likelihoods():
    #     # Calculate likelihood per-metric:
    #     ctl_log_likelihoods = []
    #     rd_log_likelihoods = []
    #     for metric_index in range(4):
    #         # Get data:
    #         target_data = list(regression_dict.values())[metric_index]
    #         metric = list(model_metric_dict.values())[metric_index]
    #         metric_means = np.mean(metric, axis=1)
    #         metric_sem = np.std(metric, axis=1) / np.sqrt(8)

    #         # Do comparison:
    #         ctl_ll = ll_calculation(
    #             target_data["wt_mean"], target_data["wt_stddev"],
    #             metric_means, metric_sem
    #         )
    #         ctl_log_likelihoods.append(ctl_ll[:8192])
    #         rd_ll = ll_calculation(
    #             target_data["rd_mean"], target_data["rd_stddev"],
    #             metric_means, metric_sem
    #         )
    #         rd_log_likelihoods.append(rd_ll[8192:])

    #     return np.stack(ctl_log_likelihoods, axis=1), np.stack(rd_log_likelihoods, axis=1)
    return


@app.cell
def _():
    # ctl_ll, rd_ll = calculate_likelihoods()
    return


@app.cell
def _():
    # def plot_ctl_pareto_grid():
    #     fig, axs = plt.subplots(4, 4, figsize=(4, 4))
    #     for i in range(4):
    #         for j in range(4):
    #             if i == j:
    #                 axs[i, j].remove()
    #             axs[i, j].scatter(ctl_ll[:, i], ctl_ll[:, j], c=np.sum(ctl_ll, axis=1), s=1)
    #             # axs[i, j].set_axis_off()

    #     plt.show()

    # plot_ctl_pareto_grid()
    return


@app.cell
def _():
    # def plot_rd_pareto_grid():
    #     fig, axs = plt.subplots(4, 4, figsize=(4, 4))
    #     for i in range(4):
    #         for j in range(4):
    #             if i == j:
    #                 axs[i, j].remove()
    #             axs[i, j].scatter(rd_ll[:, i], rd_ll[:, j], s=1)
    #             # axs[i, j].set_axis_off()

    #     plt.show()

    # plot_rd_pareto_grid()
    return


@app.cell
def _():
    # total_ctl_likelihood = np.sum(ctl_ll, axis=1)
    # ctl_likelihood_indices = np.argsort(total_ctl_likelihood)[-256:]
    # total_rd_likelihood = np.sum(rd_ll, axis=1)
    # rd_likelihood_indices = np.argsort(total_rd_likelihood)[-256:] + 8192
    return


@app.cell
def _(np, os):
    CSM_POSTERIOR_DIRPATH = "model_experiments/2026-09-30-csm_posterior"
    matrix_op = np.load(os.path.join(CSM_POSTERIOR_DIRPATH, "summary_data", "matrix_order_parameters.npy"))
    return CSM_POSTERIOR_DIRPATH, matrix_op


@app.cell
def _(matrix_op, np, plt):
    csm_op65 = np.mean(matrix_op[:, :, -1], axis=1)
    plt.hist(csm_op65[:1024], bins=25, histtype="step")
    plt.hist(csm_op65[1024:], bins=25, histtype="step")
    return (csm_op65,)


@app.cell
def _(csm_op65, np):
    print(np.mean(csm_op65[:1024]))
    return


@app.cell
def _(csm_op65, np):
    np.quantile(csm_op65[:1024], 0.9)
    return


@app.cell
def _(csm_op65, np):
    print(np.mean(csm_op65[1024:]))
    return


@app.cell
def _(csm_op65, np):
    np.quantile(csm_op65[1024:], 0.9)
    return


@app.cell
def _(CSM_POSTERIOR_DIRPATH, np, os):
    ADJ_RD_DIRPATH = "model_experiments/2026-06-13-csm_posterior"
    adj_matrix_op = np.load(os.path.join(CSM_POSTERIOR_DIRPATH, "summary_data", "matrix_order_parameters.npy"))
    return (adj_matrix_op,)


@app.cell
def _():
    # def plot_likelihood_sort():
    #     # Get matrix data:
    #     op65 = np.mean(matrix_op[:, :, 2], axis=1)

    #     # Set up plots:
    #     fig, axs = plt.subplots(2, 1, sharey=True)

    #     # Plot control sort:
    #     ctl_likelihood_sort = np.argsort(np.sum(ctl_ll, axis=1))
    #     axs[0].plot(op65[:1024][ctl_likelihood_sort], color=CONTROL_PALETTE, lw=0.1)

    #     # Plot RD sort:
    #     rd_likelihood_sort = np.argsort(np.sum(rd_ll, axis=1))
    #     axs[1].plot(op65[1024:][rd_likelihood_sort], color=RD_PALETTE, lw=0.1)
    #     plt.show()

    # plot_likelihood_sort()
    return


@app.cell
def _(
    CONTROL_PALETTE,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    RD_PALETTE,
    TEXT_WIDTH,
    adj_matrix_op,
    datetime,
    matrix_op,
    np,
    os,
    plt,
):
    op65 = np.mean(matrix_op[:, :, 0], axis=1)
    adj_op65 = np.mean(adj_matrix_op[:, :, 2], axis=1)

    def plot_op_histogram():
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.0))
        # ax.hist(op65[:8192], bins=50, histtype="step", density=True, color=CONTROL_PALETTE)
        # ax.hist(op65[8192:], bins=50, histtype="step", density=True, color=RD_PALETTE)
        ax.hist(op65[:1024], bins=20, histtype="step", density=True, color=CONTROL_PALETTE, label="Control Fit")
        ax.hist(op65[1024:],  bins=20, histtype="step", density=True, color=RD_PALETTE, label="RD Fit")
        # ax.hist(adj_op65,  bins=25, histtype="step", density=True, color=ADJ_RD_PALETTE)

        ax.legend()
        ax.set_xlabel("Order Parameter $S_{65}$")

        edging = 0.2
        fig.subplots_adjust(edging, edging, 1 - edging, 1 - edging)

        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, f"mle_simulation_op.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_op_histogram()
    return (op65,)


@app.cell
def _():
    # def plot_cell_length_histogram():
    #     fig, ax = plt.subplots(figsize=(2.75, 1.75))
    #     cell_lengths = np.load(os.path.join(CSM_POSTERIOR_DIRPATH, "summary_data", "cell_lengths.npy"))
    #     cell_lengths = np.mean(cell_lengths, axis=1)
    #     ax.hist(cell_lengths[:8192], bins=40, histtype="step", density=True, color=CONTROL_PALETTE)
    #     ax.hist(cell_lengths[8192:], bins=40, histtype="step", density=True, color=RD_PALETTE)
    #     plt.show()

    # plot_cell_length_histogram()
    return


@app.cell
def _(
    FULL_WIDTH,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    datetime,
    np,
    op65,
    os,
    plt,
):
    import imageio.v3 as iio


    def load_image(dirpath, index, image_type):
        image_filepath = os.path.join(
            dirpath, "run_data",
            str(index // 1000), str(index), f"{image_type}.png"
        )  
        image_array = iio.imread(image_filepath)
        return image_array


    def generate_collage(dirpath, indices, grid_size, image_type):
        count = 0
        image_grid = []
        for i in range(grid_size):
            image_row = []
            for j in range(grid_size):
                image_array = load_image(dirpath, indices[count], image_type)
                image_row.append(image_array)
                count += 1
            image_grid.append(np.concatenate(image_row, axis=1))
        return np.concatenate(image_grid, axis=0)


    def plot_matrix_collage(dirpath, grid_size, image_type):
        rng = np.random.default_rng(0)

        # Set up plot:
        fig, axs = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.6))
        high_wt_organisation = np.argsort(op65[:1024])[::-1]
        wt_collage = generate_collage(dirpath, high_wt_organisation, grid_size, image_type)
        high_rd_organisation = np.argsort(op65[1024:])[::-1] + 1024
        rd_collage = generate_collage(dirpath, high_rd_organisation, grid_size, image_type)

        axs[0].imshow(wt_collage)
        axs[0].set_xlabel("Control MLE simulations")
        axs[0].spines[['left', 'right', 'bottom', 'top']].set_visible(False)
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        axs[1].imshow(rd_collage)
        axs[1].set_xlabel("RD MLE simulations")
        axs[1].spines[['left', 'right', 'bottom', 'top']].set_visible(False)
        axs[1].set_xticks([])
        axs[1].set_yticks([])

        fig.subplots_adjust(0, 0, 1, 1, wspace=0.05, hspace=0)

        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, f"fit_sim_{image_type}_{dirpath.split("/")[1]}.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()
    return generate_collage, plot_matrix_collage


@app.cell
def _(CS_POSTERIOR_DIRPATH, plot_matrix_collage):
    plot_matrix_collage(CS_POSTERIOR_DIRPATH, 5, "trajectory")
    return


@app.cell
def _(CS_POSTERIOR_DIRPATH, plot_matrix_collage):
    plot_matrix_collage(CS_POSTERIOR_DIRPATH, 5, "stadia")
    return


@app.cell
def _(CSM_POSTERIOR_DIRPATH, plot_matrix_collage):
    for _matrix_plot in ["trajectory", "stadia", "matrix_density", "matrix_heading"]:
        plot_matrix_collage(CSM_POSTERIOR_DIRPATH, 5, _matrix_plot)
    return


@app.cell
def _():
    # CSM_LONG_POSTERIOR_DIRPATH = "model_experiments/2026-06-15-csm_posterior_long"
    # for _matrix_plot in ["trajectory", "matrix_density", "matrix_heading"]:
    #     plot_matrix_collage(CSM_LONG_POSTERIOR_DIRPATH, 5, _matrix_plot)
    return


@app.cell
def _(CONTROL_PALETTE, RD_PALETTE, np, plt):
    test_dense = np.load("model_experiments/2026-06-14-csm_posterior/summary_data/matrix_order_parameters.npy")
    tl_op65 = np.mean(test_dense[:, :, 2], axis=1)
    plt.hist(tl_op65[:1024], bins=24, histtype="step", color=CONTROL_PALETTE)
    plt.hist(tl_op65[1024:], bins=24, histtype="step", color=RD_PALETTE)
    return


@app.cell
def _(
    FULL_WIDTH,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    datetime,
    generate_collage,
    np,
    op65,
    os,
    plt,
):
    def plot_ordered_matrix_collage(dirpath, grid_size, image_type):
        rng = np.random.default_rng(0)

        # Set up plot:
        fig, axs = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.6))
        high_wt_organisation = np.argsort(op65[:1024])[::-1]
        wt_collage = generate_collage(dirpath, high_wt_organisation, grid_size, image_type)
        high_rd_organisation = np.argsort(op65[1024:])[::-1] + 1024
        rd_collage = generate_collage(dirpath, high_rd_organisation, grid_size, image_type)

        axs[0].imshow(wt_collage)
        axs[0].set_xlabel("Control MLE simulations")
        axs[0].spines[['left', 'right', 'bottom', 'top']].set_visible(False)
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        axs[1].imshow(rd_collage)
        axs[1].set_xlabel("RD MLE simulations")
        axs[1].spines[['left', 'right', 'bottom', 'top']].set_visible(False)
        axs[1].set_xticks([])
        axs[1].set_yticks([])

        fig.subplots_adjust(0, 0, 1, 1, wspace=0.05, hspace=0)

        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, f"fit_sim_{image_type}_{dirpath.split("/")[1]}.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()
    return


@app.cell
def _(CSM_POSTERIOR_DIRPATH, np, os):
    csm_op = np.load(os.path.join(CSM_POSTERIOR_DIRPATH, "summary_data", "matrix_order_parameters.npy"))
    csm_op = np.mean(csm_op[:, :, 2], axis=1)
    return (csm_op,)


@app.cell
def _(csm_op, np):
    wt_ordered_indices = np.argsort(csm_op[:1024])
    rd_ordered_indices = np.argsort(csm_op[1024:]) + 1024
    return


if __name__ == "__main__":
    app.run()
