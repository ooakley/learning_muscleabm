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
        METADATA_DICTIONARY,
        OUT_DIRPATH,
        RD_PALETTE,
        TEXT_WIDTH,
    )


@app.cell
def _(np, os):
    PARAMETER_DIMENSION = 12
    EXPERIMENT_DIRPATH = "model_experiments/2026-09-16-collisions_shape"
    mcmc_results = "disc_cov_mcmc_results"

    ctl_mcmc_chain = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "wt_mcmc_chain.npy"))
    rd_mcmc_chain = np.load(os.path.join(EXPERIMENT_DIRPATH, mcmc_results, "rd_mcmc_chain.npy"))
    return (
        EXPERIMENT_DIRPATH,
        PARAMETER_DIMENSION,
        ctl_mcmc_chain,
        rd_mcmc_chain,
    )


@app.cell
def _(EXPERIMENT_DIRPATH, json, np, os):
    with open(os.path.join(EXPERIMENT_DIRPATH, "config.json")) as json_file:
        config_dict = json.load(json_file)

    parameter_list = [parameter_range[0] for parameter_range in config_dict["gridsearch_parameters"]]
    parameter_list.remove("numberOfCells")

    parameter_scaling = [parameter_range[1] for parameter_range in config_dict["gridsearch_parameters"]]
    parameter_scaling.pop(-1);
    parameter_scaling = np.stack(parameter_scaling, axis=0)
    return config_dict, parameter_scaling


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
def _(EXPERIMENT_DIRPATH, np, os):
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
                EXPERIMENT_DIRPATH, "summary_data", f"{metric_name}.npy"
            ))
            metrics_dict[metric_name] = np.nanmean(metric_array, axis=1)
        return metrics_dict

    metrics_dict = load_metrics()
    sobol_inputs = np.load(os.path.join(EXPERIMENT_DIRPATH, "sample_matrix.npy"))
    return metrics_dict, sobol_inputs


@app.cell
def _(
    PARAMETER_DIMENSION,
    ctl_mcmc_chain,
    np,
    parameter_scaling,
    rd_mcmc_chain,
    sobol_inputs,
):
    def get_decomposition(precision):
        eigvals, eigvecs = np.linalg.eigh(precision)
        ctl_order = np.argsort(-eigvals)
        eigvals = eigvals[ctl_order]
        eigvecs = eigvecs[:, ctl_order]
        return eigvals, eigvecs

    def scale_values(unit_inputs):
        parameter_range = np.squeeze(np.diff(parameter_scaling))
        return (unit_inputs * parameter_range) + parameter_scaling[:, 0]

    chain_length = ctl_mcmc_chain.shape[0]
    quarter_index = int(chain_length // 4)
    ctl_posterior = ctl_mcmc_chain[quarter_index::16, :, 0, :].reshape(-1, PARAMETER_DIMENSION)
    ctl_posterior = scale_values(ctl_posterior)
    ctl_log_posterior = np.log(ctl_posterior)

    rd_posterior = rd_mcmc_chain[quarter_index::16, :, 0, :].reshape(-1, PARAMETER_DIMENSION)
    rd_posterior = scale_values(rd_posterior)
    rd_log_posterior = np.log(rd_posterior)

    sobol_prior = scale_values(sobol_inputs[:, :-1])
    return (
        ctl_log_posterior,
        ctl_posterior,
        rd_log_posterior,
        rd_posterior,
        sobol_prior,
    )


@app.cell
def _(ctl_log_posterior):
    print(ctl_log_posterior.shape)
    return


@app.cell
def _(rd_log_posterior):
    print(rd_log_posterior.shape)
    return


@app.cell
def _(ctl_log_posterior, np, rd_log_posterior):
    n_keep = 7
    choice_generator = np.random.default_rng(1)

    def split_loadings(precision, n_factors):
        # Get loadings of upper n eigenvectors:
        eigvals, eigvecs = np.linalg.eigh(precision)
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        eigvals = np.clip(eigvals[:n_factors], 0, None)
        return eigvals[:n_factors], eigvecs[:, :n_factors]

    def sloppy_loadings(precision, n_factors):
        # Get loadings of lower n eigenvectors:
        eigvals, eigvecs = np.linalg.eigh(precision)
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        eigvals = np.clip(eigvals, 0, None)
        return eigvals[n_factors:], eigvecs[:, n_factors:]

    def bootstrap_eigenvectors(log_posterior, iterates=1000):
        eigval_list = []
        eigvec_list = []
        for iteration_index in range(iterates):
            bootstrapped_sample = choice_generator.choice(log_posterior, size=log_posterior.shape[0])
            bootstrapped_covariance = np.cov(bootstrapped_sample, rowvar=False)
            bootstrapped_precision = np.linalg.inv(bootstrapped_covariance)
            top_eigvals, top_eigvecs = split_loadings(bootstrapped_precision, n_keep)
            eigval_list.append(top_eigvals)
            eigvec_list.append(top_eigvecs)
        return np.stack(eigval_list, axis=0), np.stack(eigvec_list, axis=0)

    ctl_eigvals, ctl_eigvecs = bootstrap_eigenvectors(ctl_log_posterior, iterates=1000)
    rd_eigvals, rd_eigvecs = bootstrap_eigenvectors(rd_log_posterior, iterates=1000)
    return ctl_eigvals, ctl_eigvecs, n_keep, rd_eigvals, rd_eigvecs


@app.cell
def _(ctl_eigvecs, n_keep, np):
    def grassman_dist(A, B):
        gd = np.sum((A.T @ B) ** 2) / n_keep
        return gd

    grassman_dist(ctl_eigvecs[0], ctl_eigvecs[100])
    return


@app.cell
def _(CONTROL_PALETTE, log_prior, np, plt):
    def plot_prior(posterior, theta_hat, do_exp=False):
        # Get transformed prior and posterior:
        transformed_prior = log_prior @ theta_hat
        transformed_posterior = posterior @ theta_hat

        # Plot histogram comparison:
        fig, ax = plt.subplots()
        if not do_exp:
            ax.hist(transformed_prior.flatten(), histtype="step", bins=100, color='k', density=True)
            ax.hist(transformed_posterior.flatten(), histtype="step", bins=100, color=CONTROL_PALETTE, density=True)
        else:
            bins = np.linspace(0, 5, 100)
            ax.hist(np.exp(transformed_prior), histtype="step", bins=bins, color='k', density=True)
            ax.hist(np.exp(transformed_posterior), histtype="step", bins=bins, color=CONTROL_PALETTE, density=True)
            ax.set_xlim(0, 5)

        plt.show()
    return


@app.cell
def _(np):
    def sparse_rotate(eigenvalues, eigenvectors, epsilon_schedule, initial_rotation,
                      max_evaluations_per_stage=2000, gradient_tolerance=1e-9,
                      sufficient_decrease=1e-4, max_rejections=40):
        """
        Rotate the columns of `eigenvectors` within their own span so that each rotated axis
        involves as few parameters as possible.

        For a unit-norm rotated axis, let magnitude_i = sqrt(component_i**2 + epsilon**2) (a
        smoothed |component_i|), probability_i = magnitude_i / sum(magnitude), and
        entropy = -sum(probability * log(probability)). exp(entropy) is the effective number
        of parameters in the axis. The objective is the sum of this entropy over axes, and it
        is minimised over orthogonal rotations by gradient projection (a gradient step
        projected onto the tangent space of the orthogonal group, then returned to the group
        by polar decomposition, with backtracking line search). Each value of epsilon in
        `epsilon_schedule` is one annealing stage, warm-started from the previous one.

        Parameters
        ----------
        eigenvalues : (n_axes,) precision-matrix eigenvalues matching the columns of
            `eigenvectors`. Only used to give each rotated axis's curvature and to order the
            output; they do not affect the rotation.
        eigenvectors : (n_parameters, n_axes) orthonormal top-k eigenvectors of the precision
            matrix (the stiff block).
        epsilon_schedule : decreasing smoothing scales, in units of entry size. Entries of a
            unit vector are of order 1 / sqrt(n_parameters), so something like
            np.array([1.0, 0.33, 0.1, 0.033, 0.01]) / np.sqrt(n_parameters) is a sensible start.
        initial_rotation : (n_axes, n_axes) orthogonal starting rotation, e.g. random.
        max_evaluations_per_stage, gradient_tolerance, sufficient_decrease, max_rejections :
            optimiser controls; the defaults are fine for most uses.

        Returns
        -------
        rotated_axes : (n_parameters, n_axes) sparse orthonormal basis of the same subspace.
            Signs are fixed so each axis's largest component is positive; axes are ordered by
            decreasing curvature.
        rotation : (n_axes, n_axes) with rotated_axes == eigenvectors @ rotation.
        axis_curvature : (n_axes,) w' (precision restricted to the block) w for each axis.
        objective : summed entropy at the final epsilon (lower is sparser). Comparable between
            runs only if they share the same final epsilon.
        """
        n_parameters, n_axes = eigenvectors.shape
        if eigenvalues.shape != (n_axes,):
            raise ValueError("eigenvalues must have one entry per column of eigenvectors")
        if initial_rotation.shape != (n_axes, n_axes) or not np.allclose(
                initial_rotation.T @ initial_rotation, np.eye(n_axes), atol=1e-8):
            raise ValueError("initial_rotation must be an orthogonal (n_axes, n_axes) matrix")
        if len(epsilon_schedule) == 0:
            raise ValueError("epsilon_schedule must contain at least one value")

        rotation = initial_rotation.copy()

        for epsilon in epsilon_schedule:
            objective = np.inf
            step_size = 1.0
            n_rejections = 0
            projected_gradient = np.zeros((n_axes, n_axes))
            projected_gradient_norm_squared = 0.0
            trial_rotation = rotation

            # One loop handles both accepted steps and backtracking rejections, so the
            # objective and gradient are computed in a single place. The first pass evaluates
            # the starting rotation and is accepted unconditionally (objective starts at inf).
            for evaluation in range(max_evaluations_per_stage):
                trial_axes = eigenvectors @ trial_rotation
                smoothed_magnitude = np.sqrt(trial_axes ** 2 + epsilon ** 2)
                magnitude_total = smoothed_magnitude.sum(axis=0, keepdims=True)
                probability = smoothed_magnitude / magnitude_total
                axis_entropy = -(probability * np.log(probability)).sum(axis=0, keepdims=True)
                trial_objective = axis_entropy.sum()

                if trial_objective < objective - sufficient_decrease * step_size * projected_gradient_norm_squared:
                    rotation = trial_rotation
                    objective = trial_objective
                    n_rejections = 0
                    if evaluation > 0:
                        step_size *= 2.0

                    # d(entropy)/d(component) = -(log(probability) + entropy) * component / (magnitude * total)
                    gradient_wrt_axes = -(np.log(probability) + axis_entropy) * trial_axes / (
                        smoothed_magnitude * magnitude_total)
                    rotation_gradient = eigenvectors.T @ gradient_wrt_axes
                    symmetric_part = rotation.T @ rotation_gradient
                    symmetric_part = (symmetric_part + symmetric_part.T) / 2.0
                    projected_gradient = rotation_gradient - rotation @ symmetric_part
                    projected_gradient_norm_squared = np.sum(projected_gradient ** 2)
                    if np.sqrt(projected_gradient_norm_squared) < gradient_tolerance:
                        break
                else:
                    step_size /= 2.0
                    n_rejections += 1
                    if n_rejections > max_rejections:
                        break

                left_vectors, singular_values, right_vectors_transposed = np.linalg.svd(
                    rotation - step_size * projected_gradient)
                trial_rotation = left_vectors @ right_vectors_transposed

        # We retrieve the canonical form of the subspace by ensuring that the largest component of 
        # each axis is positive, and by ordering each axis by curvature.
        rotated_axes = eigenvectors @ rotation
        row_of_largest_component = np.argmax(np.abs(rotated_axes), axis=0)
        signs = np.sign(rotated_axes[row_of_largest_component, np.arange(n_axes)])
        rotation = rotation * signs
        axis_curvature = (rotation ** 2 * eigenvalues[:, None]).sum(axis=0)
        curvature_order = np.argsort(-axis_curvature)
        rotation = rotation[:, curvature_order]
        return eigenvectors @ rotation, rotation, axis_curvature[curvature_order], objective
    return (sparse_rotate,)


@app.cell
def _(n_keep, np):
    def generate_initial_rotation(generator, dimensions=n_keep):
        orthogonal, triangular = np.linalg.qr(generator.standard_normal((dimensions, dimensions)))
        initial_rotation = orthogonal * np.sign(np.diag(triangular))
        return initial_rotation
    return (generate_initial_rotation,)


@app.cell
def _(generate_initial_rotation, n_keep, np, sparse_rotate):
    def population_rotation(eigvals, eigvecs, n_points, dimensions=n_keep, seed=0):
        baseline_schedule = np.array([1.0, 0.33, 0.1, 0.033, 0.01, 0.001, 0.0001]) / np.sqrt(12)

        all_eigvecs = []
        curvatures = []
        entropies = []
        for i in range(n_points):
            sparse_eigvecs, rotation_matrix, curvature, final_entropy = \
                sparse_rotate(eigvals, eigvecs, baseline_schedule, generate_initial_rotation(i, dimensions))

            all_eigvecs.append(np.copy(sparse_eigvecs))
            curvatures.append(np.copy(curvature))
            entropies.append(np.copy(final_entropy))

        print(np.mean(entropies))
        print(np.std(entropies))

        return np.stack(all_eigvecs, axis=0), np.stack(curvatures, axis=0), np.stack(entropies, axis=0)
    return


@app.cell
def _(generate_initial_rotation, n_keep, np, sparse_rotate):
    def bootstrap_rotation(eigvals, eigvecs, dimensions=n_keep, seed=0):
        baseline_schedule = np.array([1.0, 0.33, 0.1, 0.033, 0.01, 0.001, 0.0001]) / np.sqrt(12)
        rotation_generator = np.random.default_rng(seed)

        all_eigvecs = []
        curvatures = []
        entropies = []
        for i in range(eigvals.shape[0]):
            sparse_eigvecs, rotation_matrix, curvature, final_entropy = \
                sparse_rotate(eigvals[i], eigvecs[i], baseline_schedule, generate_initial_rotation(rotation_generator, dimensions))

            all_eigvecs.append(np.copy(sparse_eigvecs))
            curvatures.append(np.copy(curvature))
            entropies.append(np.copy(final_entropy))

        return np.stack(all_eigvecs, axis=0), np.stack(curvatures, axis=0), np.stack(entropies, axis=0)
    return (bootstrap_rotation,)


@app.cell
def _(
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    PARAMETER_SYMBOLS,
    TEXT_WIDTH,
    cc,
    datetime,
    n_keep,
    np,
    os,
    plt,
):
    def plot_eigenvectors(eigenvectors, errors, filename):
        fig, axs = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.5))

        # Plot eigenparameter loadings:
        flatten_noise = np.copy(eigenvectors)
        flatten_noise[np.abs(flatten_noise) < 0.1] = 0
        component_out = axs[0].imshow(flatten_noise, cmap=cc.m_CET_D13, vmin=-0.8, vmax=0.8)
        theta_text = "\\hat{\\theta}"
        axs[0].set_xticks(np.arange(n_keep), [f"${theta_text}_{{{idx}}}$" for idx in np.arange(1, n_keep + 1)])
        axs[0].set_yticks(np.arange(12), [f"${symbol}$" for symbol in PARAMETER_SYMBOLS])
        axs[0].set_title(f"{filename} components")
        fig.colorbar(component_out, label="Parameter component", shrink=0.7)

        # Plot bootstrap error:
        error_out = axs[1].imshow(errors, cmap=cc.m_CET_L12, vmin=0, vmax=0.5)
        theta_text = "\\hat{\\theta}"
        axs[1].set_xticks(np.arange(n_keep), [f"${theta_text}_{{{idx}}}$" for idx in np.arange(1, n_keep + 1)])
        axs[1].set_yticks(np.arange(12), [f"${symbol}$" for symbol in PARAMETER_SYMBOLS])
        axs[1].set_title(f"Allocation noise")
        fig.colorbar(error_out, label="Allocation noise", shrink=0.7)

        # Adjust subplots:
        fig.subplots_adjust(0.1, 0.1, 0.9, 0.9, 0.45)

        # Save:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, f"sparse_subspace_{filename}.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )

        plt.show()
    return (plot_eigenvectors,)


@app.cell
def _(bootstrap_rotation, ctl_eigvals, ctl_eigvecs, n_keep):
    ctl_rotated, ctl_curvatures, ctl_entropies = bootstrap_rotation(ctl_eigvals, ctl_eigvecs, dimensions=n_keep, seed=0)
    return ctl_entropies, ctl_rotated


@app.cell
def _(ctl_entropies, ctl_rotated, np, plot_eigenvectors):
    ctl_median_idx = np.argmin(np.abs(ctl_entropies - np.mean(ctl_entropies)))
    ctl_allocation_error = np.std(np.abs(ctl_rotated) > 0.1, axis=0)
    # ctl_allocation_frequency = np.mean(np.abs(ctl_rotated) > 0.1, axis=0)

    plot_eigenvectors(ctl_rotated[ctl_median_idx], ctl_allocation_error, "Control")
    return (ctl_median_idx,)


@app.cell
def _(bootstrap_rotation, n_keep, rd_eigvals, rd_eigvecs):
    rd_rotated, rd_curvatures, rd_entropies = bootstrap_rotation(rd_eigvals, rd_eigvecs, dimensions=n_keep, seed=0)
    return rd_entropies, rd_rotated


@app.cell
def _(np, plot_eigenvectors, rd_entropies, rd_rotated):
    rd_median_idx = np.argmin(np.abs(rd_entropies - np.mean(rd_entropies)))
    rd_allocation_error = np.std(np.abs(rd_rotated) > 0.1, axis=0)
    # rd_allocation_frequency = np.mean(np.abs(rd_rotated) > 0.1, axis=0)

    plot_eigenvectors(rd_rotated[2], rd_allocation_error, "RD")
    return


@app.cell
def _(ctl_median_idx, ctl_rotated, np, sobol_prior):
    clipped_eigvecs = np.copy(ctl_rotated[ctl_median_idx])
    clipped_eigvecs[np.abs(clipped_eigvecs) < 0.1] = 0 
    sparse_prior = np.log(sobol_prior) @ clipped_eigvecs
    sparse_prior = np.exp(sparse_prior)
    return clipped_eigvecs, sparse_prior


@app.cell
def _(sparse_prior):
    sparse_prior.shape
    return


@app.cell
def _(CONTROL_PALETTE, PARAMETER_SYMBOLS, RD_PALETTE, np, plt):
    def plot_prior_posterior(transform, prior, ctl_posterior, rd_posterior, ax=None, label=False):
        if ax is None:
            fig, ax = plt.subplots()

        # Clip transform:
        # clipped_transform = np.copy(transform)
        # clipped_transform[np.abs(transform) < 0.1] = 0
        transform_prior = np.log(prior) @ transform
        transform_ctl = np.log(ctl_posterior) @ transform
        transform_rd = np.log(rd_posterior) @ transform

        transform_prior = np.exp(transform_prior)
        transform_ctl = np.exp(transform_ctl)
        transform_rd = np.exp(transform_rd)

        # Get axis limits:
        lower_quantile = 0.02
        upper_quantile = 0.98
        lower_lim = np.min([
            np.quantile(transform_prior, 0.05),
            np.quantile(transform_ctl, lower_quantile),
            np.quantile(transform_rd, lower_quantile),
        ])
        upper_lim = np.max([
            np.quantile(transform_prior, 0.95),
            np.quantile(transform_ctl, upper_quantile),
            np.quantile(transform_rd, upper_quantile),
        ])

        bins = np.linspace(lower_lim, upper_lim, 60)
        if not label:
            ax.hist(transform_prior, color="k", histtype="step", bins=bins, alpha=0.5, density=True)
            ax.hist(transform_ctl, color=CONTROL_PALETTE, histtype="step", bins=bins, density=True)
            ax.hist(transform_rd, color=RD_PALETTE, histtype="step", bins=bins, density=True)
        else:
            ax.hist(
                transform_prior, color="k", histtype="step",
                bins=bins, density=True, alpha=0.5, label="Prior Distribution"
            )
            ax.hist(
                transform_ctl, color=CONTROL_PALETTE, histtype="step",
                bins=bins, density=True, label="Control Distribution"
            )
            ax.hist(
                transform_rd, color=RD_PALETTE, histtype="step",
                bins=bins, density=True, label="RD Distribution"
            )

        upper_frac = []
        lower_frac = []
        for t_index, component in enumerate(transform):
            if np.abs(component) < 0.1:
                continue
            if component > 0:
                latex_string = PARAMETER_SYMBOLS[t_index]
                upper_frac.append(latex_string)
            else:
                latex_string = PARAMETER_SYMBOLS[t_index]
                lower_frac.append(latex_string)


        log_template = "$\\log \\left( {{{0}}} \\right) $"
        log_template = "${{{}}}$" 
        # Account for empty fractions:
        if upper_frac == []:
            upper_frac = "1"
            lower_frac = " ".join(lower_frac)
            xlabel = "$\\frac{" + upper_frac + "}{" + lower_frac + "}$"
            xlabel = log_template.format(xlabel)
        elif lower_frac == []:
            upper_frac = " \\cdot ".join(upper_frac)
            xlabel = f"${upper_frac}$"
            xlabel = log_template.format(upper_frac)
        else:
            upper_frac = " \\cdot ".join(upper_frac)
            lower_frac = " \\cdot ".join(lower_frac)
            xlabel = "\\frac{" + upper_frac + "}{" + lower_frac + "}"
            xlabel = log_template.format(xlabel)

        ax.set_xlabel(xlabel)
        ax.set_xlim(lower_lim, upper_lim)
    return (plot_prior_posterior,)


@app.cell
def _(
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    TEXT_WIDTH,
    clipped_eigvecs,
    ctl_posterior,
    datetime,
    n_keep,
    os,
    plot_prior_posterior,
    plt,
    rd_posterior,
    sobol_prior,
):
    def plot_all_prior_posterior():
        fig, axs = plt.subplots(n_keep, 1, figsize=(TEXT_WIDTH * 0.6, n_keep))

        theta_text = "\\hat{\\theta}"
        for i in range(n_keep):
            if i == 2:
                plot_prior_posterior(
                    clipped_eigvecs[:, i],
                    sobol_prior, ctl_posterior, rd_posterior, axs[i],
                    label=True
                )
            else:
                plot_prior_posterior(
                    clipped_eigvecs[:, i],
                    sobol_prior, ctl_posterior, rd_posterior, axs[i]
                )
            theta_label = f"${theta_text}_{i+1}$"
            axs[i].text(0.92, 0.925, theta_label, transform=axs[i].transAxes, va="top")

        middle_in = (TEXT_WIDTH * 0.45) / 2
        print(middle_in)
        fig.subplots_adjust(0.2, 0.075, 0.8, 0.85, 0, 0.9)
        fig.legend(
            loc='center right',
            bbox_to_anchor=(0.7, 0.91), fontsize=7,
            ncol=1, bbox_transform=fig.transFigure
        )

        # Save:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, f"eigenvalue_prior_posterior.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )

        plt.show()

    plot_all_prior_posterior()
    return


@app.cell
def _(metrics_dict):
    metrics_dict.keys()
    return


@app.cell
def _(cc, metrics_dict, np, plt, sparse_prior):
    def plot_metric_prior(index_i, index_j):
        # Get points to plot:
        x_values = sparse_prior[:, index_i]
        y_values = sparse_prior[:, index_j]
        x_lower = np.quantile(x_values, 0.02)
        y_lower = np.quantile(y_values, 0.02)

        x_upper = np.quantile(x_values, 0.98)
        y_upper = np.quantile(y_values, 0.98)

        # Set up coloring:
        color_values = metrics_dict["speeds"]
        nan_filter = ~np.isnan(color_values)
        color_sort = np.argsort(color_values[nan_filter])
        vmin = np.quantile(color_values[nan_filter], 0.05)
        vmax = np.quantile(color_values[nan_filter], 0.95)

        # Plot scatterplot:
        fig, ax = plt.subplots(figsize=(3.5, 3))
        scatter = ax.scatter(
            x_values[nan_filter][color_sort], y_values[nan_filter][color_sort],
            s=1, alpha=0.5, edgecolors="none",
            c=color_values[nan_filter][color_sort],
            vmin=vmin, vmax=vmax, cmap=cc.m_CET_L8
        )
        ax.set_xlim(x_lower, x_upper)
        ax.set_ylim(y_lower, y_upper)

        fig.colorbar(scatter)
        plt.show()

    plot_metric_prior(0, 1)
    return


@app.cell
def _(
    MaxNLocator,
    TEXT_WIDTH,
    cc,
    config_dict,
    metrics_dict,
    np,
    plt,
    scipy,
    sparse_prior,
):
    def plot_binscatter(parameter_index, metric_index, ax):
        # Format ticks:
        ax.xaxis.set_major_locator(MaxNLocator(3))

        # Get relevant data:
        parameter_array = sparse_prior[:, parameter_index]
        metric_values = list(metrics_dict.values())[metric_index]
        parameter_name = config_dict["gridsearch_parameters"][parameter_index][0]

        # Get binned statistic:
        lower_limit = np.quantile(parameter_array, 0.02)
        lower_limit = 0
        upper_limit = np.quantile(parameter_array, 0.98)
        bins = np.linspace(lower_limit, upper_limit, 51)
        bin_centers = bins[:-1] + ((bins[0] + bins[1]) / 2)
        bin_median, _, _ = scipy.stats.binned_statistic(parameter_array, metric_values, statistic=np.nanmedian, bins=bins)
        bin_mean, _, _ = scipy.stats.binned_statistic(parameter_array, metric_values, statistic=np.nanmean, bins=bins)
        bin_std, _, _ = scipy.stats.binned_statistic(parameter_array, metric_values, statistic=np.nanstd, bins=bins)
        # bin_sem = bin_std / np.sqrt(2**17)

        ecdf = scipy.stats.ecdf(metric_values[~np.isnan(metric_values)])
        std_quantile = ecdf.cdf.evaluate(bin_median + bin_std)
        bin_quantile = ecdf.cdf.evaluate(bin_median)

        stderr = np.abs(bin_quantile - std_quantile)

        # Plot data:
        vmin, vmax = np.nanquantile(metric_values, [0.15, 0.85])
        vmin, vmax = [0.15, 0.85]
        ax.errorbar(bin_centers, bin_quantile, stderr, c='k', alpha=0.1)
        ax.scatter(bin_centers, bin_quantile, s=1, c=bin_quantile, vmin=vmin, vmax=vmax, cmap=cc.m_CET_D7)

        # Format plot:
        ax.set_xlim(lower_limit, upper_limit)
        ax.set_ylim(0, 1)

        # Label:
        # ax.set_xlabel(parameter_name, fontsize=7)
        # ax.set_ylabel(METRICS_LABELS[metric_index])

    def plot_all_binscatter(metric_index):
        fig, axs = plt.subplots(3, 2, figsize=(TEXT_WIDTH, 3), sharey=True)

        count = 0
        for i in range(3):
            for j in range(2):
                plot_binscatter(count, metric_index, axs[i, j])
                count += 1

        # Adjust subplots:
        fig.subplots_adjust(0.15, 0.05, 0.85, 0.95, wspace=0.15, hspace=0.55)

        # Add global y-label:
        # fig.text(0.075, 0.5, Q_LABELS[metric_index], va='center', rotation='vertical', in_layout=True)

        # # Update metadata time:
        # METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # plt.savefig(
        #     os.path.join(OUT_DIRPATH, f"binscatter_{METRICS_TO_PLOT[metric_index]}.png"),
        #     dpi=300, metadata=METADATA_DICTIONARY, transparent=True, pad_inches=0
        # )
        plt.show()

    # for p_index in range(4):
    #     plot_all_binscatter(p_index)
    return (plot_all_binscatter,)


@app.cell
def _(plot_all_binscatter):
    plot_all_binscatter(3)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
