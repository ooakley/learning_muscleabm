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
    return datetime, np, os, plt, subprocess


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
        "script": "control_segmentation.py",
        "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_commit_hash": commit_hash
    }
    return


@app.cell
def _(np, os):
    EXPERIMENT_DIRPATH = "model_experiments/2026-09-19-matrix_shape"

    hessian_dirpath = os.path.join(EXPERIMENT_DIRPATH, "op65_fullrank_hessian")
    sample_hessians = np.load(os.path.join(hessian_dirpath, "hessian_estimate.npy"))
    sample_hessians = (sample_hessians + np.transpose(sample_hessians, axes=[0, 2, 1])) / 2
    sample_inputs = np.load(os.path.join(hessian_dirpath, "hessian_inputs.npy"))
    return EXPERIMENT_DIRPATH, hessian_dirpath, sample_hessians, sample_inputs


@app.cell
def _(np):
    import numba

    @numba.njit(fastmath=True)
    def sqrt_matrix(A):
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.clip(eigvals, 0, None)
        return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T

    @numba.njit(fastmath=True)
    def bures_wasserstein_distance_matrix(input_matrices):
        # Set up output:
        matrix_count = input_matrices.shape[0]
        out = np.zeros((matrix_count, matrix_count))

        # Get traces:
        traces = np.empty(matrix_count)
        for k in range(matrix_count):
            traces[k] = np.trace(input_matrices[k])

        # Parallelise loop:
        for i in numba.prange(matrix_count):
            # Get square root of matrix:
            A = input_matrices[i]
            sqrt_A = sqrt_matrix(A)
            trace_A = traces[i]
            for j in range(i + 1, matrix_count):
                # Get B matrix:
                B = input_matrices[j]

                # Calculate BW:
                inner = sqrt_A @ B @ sqrt_A
                # -- Symmetrise (if floating-point error introduces off-diagonals):
                inner = 0.5 * (inner + inner.T) 
                trace_term = np.trace(sqrt_matrix(inner))
                sq_dist = np.trace(A) + traces[j] - (2 * trace_term)
                sq_dist = max(sq_dist, 0.0)
                dist = np.sqrt(sq_dist)
                out[i, j] = dist
                out[j, i] = dist

        return out
    return (bures_wasserstein_distance_matrix,)


@app.cell
def _(EXPERIMENT_DIRPATH, hessian_dirpath, np, os):
    sample_order_estimates = np.load(os.path.join(hessian_dirpath, "hessian_outputs.npy"))

    simulation_op = np.load(os.path.join(EXPERIMENT_DIRPATH, "summary_data", "matrix_order_parameters.npy"))
    simulation_op = np.mean(simulation_op[:, :, 2], axis=1)
    mean_op_dist = np.mean(simulation_op)
    std_op_dist = np.std(simulation_op)

    def scale_op65(x):
        return (x * std_op_dist) + mean_op_dist
    return sample_order_estimates, scale_op65


@app.cell
def _(np, sample_hessians):
    def get_singular_values(hessians):
        u_array = []
        s_array = []
        for index in range(hessians.shape[0]):
            u, s, _ = np.linalg.svd(hessians[index])
            u_array.append(u.T)
            s_array.append(s)
        return np.stack(u_array, axis=0), np.stack(s_array, axis=0)

    u_vectors, singular_values = get_singular_values(sample_hessians)
    return (singular_values,)


@app.cell
def _(bures_wasserstein_distance_matrix, sample_hessians, singular_values):
    dist_matrix = bures_wasserstein_distance_matrix(sample_hessians[::32] / singular_values[::32, [0], None])
    return (dist_matrix,)


@app.cell
def _(dist_matrix, np):
    diag_distances = dist_matrix[np.triu_indices(dist_matrix.shape[0], 1)].flatten()
    return (diag_distances,)


@app.cell
def _(diag_distances, plt):
    plt.hist(diag_distances, bins=100)
    return


@app.cell
def _(diag_distances, plt):
    plt.hist(diag_distances, bins=100)
    return


@app.cell
def _(np, singular_values):
    def flatten_fim(sample_hessians):
        triu_index = np.triu_indices(14)
        flattened_fim_array = []
        for i in range(sample_hessians.shape[0]):
            flattened_fim_array.append(sample_hessians[i][triu_index] / singular_values[i, 0])
        return np.stack(flattened_fim_array, axis=0)
    return (flatten_fim,)


@app.cell
def _(flatten_fim, sample_hessians):
    flattened_fim = flatten_fim(sample_hessians)
    return (flattened_fim,)


@app.cell
def _():
    # D = 14

    # @numba.njit(fastmath=True)
    # def fill_matrix(ut):
    #     # Restore upper triangles to full matrices:
    #     ret = np.empty((D, D))
    #     idx = 0
    #     for i in range(D):
    #         for j in range(i, D):
    #             v = ut[idx]
    #             ret[i, j] = v
    #             ret[j, i] = v
    #             idx += 1
    #     return ret

    # @numba.njit(fastmath=True)
    # def sqrt_matrix(A):
    #     eigvals, eigvecs = np.linalg.eigh(A)
    #     eigvals = np.abs(eigvals)
    #     return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T

    # @numba.njit(fastmath=True)
    # def bures_wasserstein_distance(A_ut, B_ut):
    #     # Restore from upper triangular representations:
    #     A = fill_matrix(A_ut)
    #     B = fill_matrix(B_ut)

    #     # Get square root of matrix:
    #     sqrt_A = sqrt_matrix(A)

    #     # Calculate BW:
    #     inner = sqrt_A @ B @ sqrt_A

    #     # -- Symmetrise (if floating-point error introduces off-diagonals):
    #     inner = 0.5 * (inner + inner.T) 
    #     trace_term = np.trace(sqrt_matrix(inner))
    #     sq_dist = np.trace(A) + np.trace(B) - (2 * trace_term)

    #     return np.sqrt(sq_dist)
    return


@app.cell
def _(bures_wasserstein_distance, flattened_fim):
    bures_wasserstein_distance(flattened_fim[0, :], flattened_fim[10, :])
    return


@app.cell
def _(bures_wasserstein_distance, flattened_fim):
    import umap

    umap_manager = umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.0, metric=bures_wasserstein_distance, random_state=5)
    umap_embeddings = umap_manager.fit_transform(flattened_fim)
    return (umap_embeddings,)


@app.cell
def _(
    np,
    plt,
    sample_inputs,
    sample_order_estimates,
    scale_op65,
    umap_embeddings,
):
    def plot_bw_umap():
        fig, ax = plt.subplots(figsize=(5, 5))
        color_values = scale_op65(sample_order_estimates)
        color_values = sample_inputs[:, 11]
        color_sort = np.argsort(color_values)
        ax.scatter(
            umap_embeddings[color_sort, 0],
            umap_embeddings[color_sort, 1],
            s=0.5, alpha=0.5, edgecolors="none",
            c=color_values[color_sort],
            vmin=np.quantile(color_values, 0.05),
            vmax=np.quantile(color_values, 0.95)
        )
        plt.show()

    plot_bw_umap()
    return


@app.cell
def _():
    # np.save("test_bw_embeddings.npy", umap_embeddings)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
