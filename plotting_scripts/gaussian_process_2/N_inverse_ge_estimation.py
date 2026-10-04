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
    return datetime, np, os, plt, scipy, subprocess


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
        "script": "inverse_ge_estimation.py",
        "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_commit_hash": commit_hash
    }
    return


@app.cell
def _(np, os):
    MATRIX_DIRPATH = "model_experiments/2026-06-03-matrix_shape"
    sample_hessians = np.load(os.path.join(MATRIX_DIRPATH, "gaussian_process_models", "op65", "hessian_estimate.npy"))
    sample_inputs = np.load(os.path.join(MATRIX_DIRPATH, "gaussian_process_models", "op65", "hessian_inputs.npy"))
    return (sample_inputs,)


@app.cell
def _(np):
    def regularise_transforms(log_transforms):
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

        # Iterate through rest of dataset:
        for run_index in range(unit_transforms.shape[0]):
            run_transform = unit_transforms[run_index, :, :]
            cosine_matrix = base_transform.T @ run_transform

            arranged_transform = []
            for t_index in range(run_transform.shape[1]):
                choice_index = np.nanargmax(np.abs(cosine_matrix[t_index, :]))
                choice_sign = np.sign(cosine_matrix[t_index, choice_index])
                arranged_transform.append(run_transform[:, choice_index] * choice_sign)
                cosine_matrix[:, choice_index] = np.nan


            arranged_transform = np.stack(arranged_transform, axis=0)
            assert(~np.any(np.isnan(arranged_transform)))
            regularised_transforms.append(arranged_transform)

        return np.stack(regularised_transforms, axis=0)

    loss_histories = np.load("model_experiments/2026-06-03-matrix_shape/gaussian_process_models/op65/global_eigenparameter_estimation/loss_histories.npy")
    log_transforms = np.load("model_experiments/2026-06-03-matrix_shape/gaussian_process_models/op65/global_eigenparameter_estimation/log_transforms.npy")

    regularised_transforms = regularise_transforms(log_transforms)

    # Get optimal transform:
    mean_loss = np.mean(loss_histories[:, 4096:], axis=1)
    final_transform = np.copy(regularised_transforms[np.argmin(mean_loss), :, :])
    print(final_transform.shape)
    print(np.linalg.norm(final_transform, axis=0))

    clip_mask = np.abs(final_transform) < 0.1
    final_transform[clip_mask] = 0
    # final_transform /= np.linalg.norm(final_transform, axis=0, keepdims=True)
    return (final_transform,)


@app.cell
def _(final_transform, np, sample_inputs):
    test_gee = final_transform @ np.log(sample_inputs.T)
    return (test_gee,)


@app.cell
def _(final_transform, np, test_gee):
    inverted_params = np.linalg.pinv(final_transform) @ test_gee
    return (inverted_params,)


@app.cell
def _(final_transform, np, scipy, test_gee):
    def get_constrained_inverse():
        constrained_inverse = []
        for i in range(2**15):
            constrained_params = scipy.optimize.lsq_linear(final_transform, test_gee[:, i], bounds=(-np.inf, 0), method='bvls')
            constrained_inverse.append(constrained_params.x)
        return np.stack(constrained_inverse, axis=1)

    constrained_inverse = get_constrained_inverse()
    return (constrained_inverse,)


@app.cell
def _(constrained_inverse, np, plt):
    plt.hist(np.exp(constrained_inverse[12, :]), bins=100)
    return


@app.cell
def _(constrained_params, np):
    np.exp(constrained_params.x)
    return


@app.cell
def _(np):
    np.log(2)
    return


@app.cell
def _(final_transform, np, test_gee):
    np.linalg.pinv(final_transform) @ test_gee[:, 0]
    return


@app.cell
def _(final_transform, np):
    free_transform = np.eye(14) - (np.linalg.pinv(final_transform) @ final_transform)
    return (free_transform,)


@app.cell
def _(free_transform, inverted_params, np):
    # Do Schur's complement reduction to 
    constrained_inversion = []
    for column_index in range(inverted_params.shape[1]):
        column_vector = inverted_params[:, column_index]

        # Extract necessary values of indeterminacy:
        reduce_indices = np.argwhere(column_vector > 0)

        # Find partial values of w, sample indeterminate part:
        inv_reduced = np.squeeze(np.linalg.inv(free_transform)[:, reduce_indices])
        w_fixed = inv_reduced @ -column_vector[reduce_indices]

        Fw_final = free_transform @ w_fixed
        constrained_vector = np.squeeze(column_vector) + np.squeeze(Fw_final)
        constrained_inversion.append(constrained_vector)

    constrained_inversion = np.stack(constrained_inversion, axis=1)
    return (constrained_inversion,)


@app.cell
def _(constrained_inversion):
    constrained_inversion.shape
    return


@app.cell
def _(constrained_inversion, np, plt):
    plt.hist(np.exp(constrained_inversion[6, :]), bins=100)
    return


@app.cell
def _():
    # rng = np.random.default_rng(0)
    # randomness_injection = rng.uniform(-np.log(4), 0, size=(sample_inputs.T.shape))
    # w = free_transform @ randomness_injection
    return


@app.cell
def _(inverted_params, w):
    full_inversion = inverted_params + w
    return (full_inversion,)


@app.cell
def _(full_inversion, np, plt, sample_inputs):
    plt.hist(np.exp(full_inversion[1, :]), bins=100)
    plt.hist(sample_inputs.T[0, :], bins=100)

    # plt.hist(np.exp(inverted_params[6, :]), bins=100)
    return


@app.cell
def _(np, plt):
    plt.scatter(
        np.linalg.norm()
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
