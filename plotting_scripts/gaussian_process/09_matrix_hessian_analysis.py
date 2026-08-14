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
    return cc, datetime, json, np, os, pd, plt, scipy, subprocess


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
        ADJ_RD_PALETTE,
        CONTROL_PALETTE,
        FULL_HEIGHT,
        FULL_WIDTH,
        METADATA_DICTIONARY,
        OUT_DIRPATH,
        RD_PALETTE,
        TEXT_HEIGHT,
        TEXT_WIDTH,
    )


@app.cell
def _(np):
    EXPERIMENT_DIRPATH = "model_experiments/2026-06-03-matrix_shape"
    sample_hessians = np.load("model_experiments/2026-06-03-matrix_shape/gaussian_process_models/op65/hessian_estimate.npy")
    sample_hessians = (sample_hessians + np.transpose(sample_hessians, axes=[0, 2, 1])) / 2
    sample_inputs = np.load("model_experiments/2026-06-03-matrix_shape/gaussian_process_models/op65/hessian_inputs.npy")
    return EXPERIMENT_DIRPATH, sample_hessians, sample_inputs


@app.cell
def _(np):
    sample_order_estimates = np.load("model_experiments/2026-06-03-matrix_shape/gaussian_process_models/op65/hessian_outputs.npy")

    simulation_op = np.load("model_experiments/2026-06-03-matrix_shape/summary_data/matrix_order_parameters.npy")
    simulation_op = np.mean(simulation_op[:, :, 2], axis=1)
    mean_op_dist = np.mean(simulation_op)
    std_op_dist = np.std(simulation_op)

    def scale_op65(x):
        return (x * std_op_dist) + mean_op_dist
    return sample_order_estimates, scale_op65


@app.cell
def _(EXPERIMENT_DIRPATH, json, os):
    with open(os.path.join(EXPERIMENT_DIRPATH, "config.json")) as json_file:
        config_dict = json.load(json_file)

    parameter_list = [parameter_range[0] for parameter_range in config_dict["gridsearch_parameters"]]
    return (parameter_list,)


@app.cell
def _(np):
    def get_eigenvectors(hessians):
        eigenvalue_array = []
        eigenvector_array = []
        for index in range(hessians.shape[0]):
            # As matrices are symmetric, all eigenvalues are real:
            eigvals, eigenvectors = np.linalg.eigh(hessians[index])
            # Reorient everything so it makes sense:
            eigenvalue_array.append(eigvals[::-1])
            eigenvector_array.append(eigenvectors.T[::-1, :])
        return np.stack(eigenvalue_array, axis=0), np.stack(eigenvector_array, axis=0)
    return (get_eigenvectors,)


@app.cell
def _(get_eigenvectors, sample_hessians):
    eigenvalues, eigenvectors = get_eigenvectors(sample_hessians)
    return eigenvalues, eigenvectors


@app.cell
def _(cc, eigenvalues, np, plt, sample_hessians):
    def plot_example_hessian(gridsize):
        fig, axs = plt.subplots(gridsize, gridsize, figsize=(5, 5))

        count = 0
        for i in range(gridsize):
            for j in range(gridsize):
                normalised_matrix = sample_hessians[count] / np.abs(eigenvalues[count][0])
                axs[i, j].imshow(normalised_matrix, cmap=cc.m_CET_D13, vmin=-0.5, vmax=0.5)
                axs[i, j].set_axis_off()
                count += 1

        fig.subplots_adjust(0, 0, 1, 1, wspace=0.05, hspace=0.05)

        plt.show()

    plot_example_hessian(10)
    return


@app.cell
def _(eigenvectors, np, sample_inputs):
    intrinsic_activities = np.sum(np.log(sample_inputs) * eigenvectors[:, 0, :], axis=1)
    return


@app.cell
def _(np):
    def get_singular_values(hessians):
        u_array = []
        s_array = []
        for index in range(hessians.shape[0]):
            u, s, _ = np.linalg.svd(hessians[index])
            u_array.append(u.T)
            s_array.append(s)
        return np.stack(u_array, axis=0), np.stack(s_array, axis=0)
    return (get_singular_values,)


@app.cell
def _(get_singular_values, sample_hessians):
    u_vectors, singular_values = get_singular_values(sample_hessians)
    return singular_values, u_vectors


@app.cell
def _(
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    TEXT_WIDTH,
    cc,
    datetime,
    mpl,
    np,
    os,
    plt,
    singular_values,
):
    def plot_eigenvalue_distribution():
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.5))

        # Plot all ecdfs:
        color_values = np.linspace(0, 1, 14)
        for i in range(14):
            ax.ecdf(np.log(singular_values[:, i]), color=cc.m_CET_L8_r(color_values[i]), alpha=0.7)

        ax.set_xlabel("$\\log(\\lambda)$")
        ax.set_ylabel("ECDF")

        cr = mpl.colorizer.Colorizer(cmap=cc.m_CET_L8_r)
        cr.set_clim(1, 14)
        cbar = fig.colorbar(
            mpl.colorizer.ColorizingArtist(cr), ax=ax,
            fraction=0.1, label="Eigenvector Order"
        )
        cbar.set_ticks([1, 7, 14])
        cbar.ax.invert_yaxis()

        # Adjust layout:
        edging = 0.15
        fig.subplots_adjust(0.1, edging, 0.975, 1-edging)

        # Save figures:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "global_eigenvalue_distribution.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_eigenvalue_distribution()
    return


@app.cell
def _(np, plt, singular_values, u_vectors):
    from numpy.polynomial import polynomial as P

    def plot_average_interaction(value_index, i, j, ax):
        # Get eigencomponents:
        eigencomponents = np.copy(u_vectors[:, value_index, [i, j]])

        # Get gradient:
        c = P.polyfit(eigencomponents[:, 0], eigencomponents[:, 1], [1], full=False)
        c_angle = np.arctan(c[1])

        # Get the correlation coefficient:
        corr_coeff = np.corrcoef(eigencomponents[:, 0], eigencomponents[:, 1])[0, 1]
        print(corr_coeff)

        # Plot the scatter plots:
        color_values = np.log(singular_values[:, value_index])
        color_sort = np.argsort(color_values)[::-1]
        color_sort = color_values < np.quantile(color_values, 0.1)
        ax.scatter(eigencomponents[color_sort, 0], eigencomponents[color_sort, 1], s=0.5, alpha=0.05, c=color_values[color_sort])
        mean_components = np.mean(eigencomponents, axis=0)
        # ax.scatter(mean_components[0], mean_components[1], c='r')
        # ax.scatter(corr_coeff * np.cos(c_angle), corr_coeff * np.sin(c_angle), c='g')

        ax.set_aspect("equal")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        # plt.plot([0, corr_coeff * np.cos(c_angle)], [0, corr_coeff * np.sin(c_angle)], c='k')

    fig, ax = plt.subplots(figsize=(3, 3))
    plot_average_interaction(13, 0, 4, ax)
    plt.show()
    return


@app.cell
def _(
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    TEXT_HEIGHT,
    TEXT_WIDTH,
    cc,
    datetime,
    mpl,
    np,
    os,
    parameter_list,
    plt,
    u_vectors,
):
    def plot_primary_eigenvector_distribution():
        fig, axs = plt.subplots(5, 3, figsize=(TEXT_WIDTH, TEXT_HEIGHT * 0.9), sharex=True, sharey=True)

        count = 0
        for i in range(5):
            for j in range(3):
                if count == 14:
                    cr = mpl.colorizer.Colorizer(cmap=cc.m_CET_L8_r)
                    cr.set_clim(1, 14)
                    cbar = fig.colorbar(
                        mpl.colorizer.ColorizingArtist(cr), ax=axs[i, j],
                        fraction=1.0, label="Eigenvector Order"
                    )
                    cbar.set_ticks([1, 7, 14])
                    cbar.ax.invert_yaxis()
                    axs[i, j].set_axis_off()
                    continue

                # Plot all ecdfs:
                color_values = np.linspace(0, 1, 14)
                for component in range(14):
                    axs[i, j].ecdf(
                        np.abs(u_vectors[:, component, count]),
                        color=cc.m_CET_L8_r(color_values[component]),
                        alpha=0.7
                    )

                axs[i, j].text(0.98, 0.02, parameter_list[count], ha="right", fontsize=7)

                # Manage axes:
                axs[i, j].set_xlim(0, 1)
                count += 1

        fig.text(0.01, 0.5, 'ECDF', ha='left', va='center', rotation='vertical')

        edging = 0.075
        fig.subplots_adjust(edging, 0.025, 1 - edging, 1 - 0.025,  wspace=0.1, hspace=0.1)

        # Save figures:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "eigenvector_ecdf.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_primary_eigenvector_distribution()
    return


@app.cell
def _(np, parameter_list, pd, u_vectors):
    primary_component_dataset = np.abs(u_vectors[:, 0, :])
    significance_percentages = np.count_nonzero(np.abs(primary_component_dataset) > 0.1, axis=0) / primary_component_dataset.shape[0]
    significance_dataframe = pd.DataFrame(significance_percentages, columns=["Significance Fraction"], index=parameter_list)
    return (significance_dataframe,)


@app.cell
def _(parameter_list, significance_dataframe):
    print(significance_dataframe.to_latex(index=parameter_list, float_format="%.2f"))
    return


@app.cell
def _(eigenvalues, np, u_vectors):
    def calculate_geometric_median(key_eigenvectors, key_eigenvalues, weighting=None):
        if weighting is None:
            weighting = np.ones(key_eigenvectors.shape[0])
        proposal_median = np.mean(key_eigenvectors, axis=0)
        converged = False
        while not converged:
            # Get distances:
            cosine_similarities = key_eigenvectors @ np.expand_dims(proposal_median, axis=1)
            nematic_distances = 1 - np.abs(cosine_similarities)

            # Get weighted average of appropriately flipped vectors:
            flipped_vectors = np.sign(cosine_similarities) * key_eigenvectors
            combined_weights = (weighting) / np.squeeze(nematic_distances)
            combined_weights = np.expand_dims(combined_weights, axis=1)
            new_proposal = np.sum(flipped_vectors * combined_weights, axis=0) / np.sum(combined_weights)
            new_proposal /= np.linalg.norm(new_proposal)
            epsilon = 1 - np.dot(proposal_median, new_proposal)
            if epsilon < 1e-8:
                converged = True
            proposal_median = new_proposal

        # Get average distance from median:
        cosine_similarities = key_eigenvectors @ np.expand_dims(proposal_median, axis=1)
        nematic_distances = 1 - np.abs(cosine_similarities)

        return proposal_median, flipped_vectors, np.mean(nematic_distances)

    proposal_median, flipped_vectors, _ = calculate_geometric_median(u_vectors[:, 0, :], eigenvalues[:, 0])
    return calculate_geometric_median, flipped_vectors, proposal_median


@app.cell
def _(np, proposal_median):
    np.round(proposal_median, 2)
    return


@app.cell
def _(TEXT_WIDTH, flipped_vectors, np, plt, sample_inputs):
    import colorstamps

    def format_axes(ax):
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()

    def plot_eigendirections(inputs, eigenvectors, parameter_i, parameter_j, ax=None, s=1, c=None):
        if ax is None:
            fig, ax = plt.subplots()

        # Get 2D colormap from parameters:
        if c is None:
            c, _ = colorstamps.apply_stamp(
                inputs[:, parameter_i], inputs[:, parameter_j],
                'flat',
                vmin_0=0, vmax_0=1,
                vmin_1=0, vmax_1=1,
            )

        # Plot components:
        ax.scatter(
            eigenvectors[:, parameter_i],
            eigenvectors[:, parameter_j],
            s=s, alpha=0.5, c=c
        )

        theta = np.linspace(0, np.pi * 2, 100)
        ax.plot(1.025*np.cos(theta), 1.025*np.sin(theta), c='k', ls="--",  lw=1, alpha=0.5)
        format_axes(ax)
        # plt.show()

    def plot_linear_combinations():
        fig, axs = plt.subplots(12, 12, figsize=(TEXT_WIDTH, TEXT_WIDTH))
        for i in range(12):
            for j in range(12):
                if i == j:
                    axs[i, j].text(0, 0, "Placeholder", fontsize=5, horizontalalignment="center")
                    format_axes(axs[i, j])
                    continue
                if i < j:
                    plot_eigendirections(sample_inputs, flipped_vectors, i, j, ax=axs[i, j], s=0.01)
                if i > j:
                    axs[i, j].remove()
                    # plot_eigendirections(sample_inputs, flipped_vectors, i, j, ax=axs[i, j], s=0.01)
                    # , c=sampled_outputs

        fig.tight_layout()
        plt.show()
    return (colorstamps,)


@app.cell
def _(np):
    import umap
    import numba

    @numba.njit()
    def nematic_cosine(a, b):
        return 1 - np.abs(a @ b.T)

    @numba.njit()
    def nematic_euclidean(a, b):
        distance_array = np.zeros(2)
        distance_array[0] = np.sqrt(np.sum((a - b) ** 2))
        distance_array[1] = np.sqrt(np.sum((a + b) ** 2))
        return np.min(distance_array)

    @numba.njit()
    def nematic_similarity(a, b):
        return np.abs(a @ b.T)
    return nematic_cosine, umap


@app.cell
def _(eigenvalues, eigenvectors):
    primary_eigenvectors = eigenvectors[:, 0, :]
    primary_eigenvalues = eigenvalues[:, [0]]
    return primary_eigenvalues, primary_eigenvectors


@app.cell
def _(nematic_cosine, primary_eigenvectors, umap):
    umap_manager = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.0, metric=nematic_cosine, random_state=1)
    umap_embeddings = umap_manager.fit_transform(primary_eigenvectors)
    return umap_embeddings, umap_manager


@app.cell
def _(np):
    kpca_embeddings = np.load("model_experiments/2026-06-03-matrix_shape/gaussian_process_models/op65/kpca_embeddings.npy")
    isomap_embeddings = np.load("model_experiments/2026-06-03-matrix_shape/gaussian_process_models/op65/isomap_embeddings.npy")
    return (isomap_embeddings,)


@app.cell
def _(isomap_embeddings):
    import sklearn

    density_manager = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=0.02, leaf_size=50, rtol=0.01)
    density_manager = density_manager.fit(isomap_embeddings[::16, :])
    density = density_manager.score_samples(isomap_embeddings[:, :])
    return density, sklearn


@app.cell
def _(density, plt, primary_eigenvectors):
    test_index = 12
    plt.hist(primary_eigenvectors[density < 0.2, test_index], bins=40, density=True)
    plt.hist(primary_eigenvectors[:, test_index], bins=40, histtype="step", density=True);
    plt.show()
    return


@app.cell
def _(np, plt, primary_eigenvalues, umap_embeddings):
    def plot_umap_fi():
        fig, ax = plt.subplots()
        ax.scatter(
            umap_embeddings[:, 0], umap_embeddings[:, 1], s=1, alpha=0.2,
            c=primary_eigenvalues, vmin=np.quantile(primary_eigenvalues, 0.2), vmax=np.quantile(primary_eigenvalues, 0.95)
        )
        plt.show()

    plot_umap_fi()
    return


@app.cell
def _(
    cc,
    density,
    isomap_embeddings,
    np,
    plt,
    sample_order_estimates,
    scale_op65,
):
    def test_plot_isomap():
        color_values = np.log(np.clip(scale_op65(sample_order_estimates), 0, None) + 1e-3)
        vmin = np.quantile(color_values, 0.05)
        vmax = np.quantile(color_values, 0.95)

        density_mask = density > np.quantile(density, 0.0)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=25, azim=10, roll=0)

        ax.scatter(
            isomap_embeddings[density_mask, 0],
            isomap_embeddings[density_mask, 1],
            isomap_embeddings[density_mask, 2],
            c=color_values[density_mask], s=1, alpha=0.25, vmin=vmin, vmax=vmax,
            cmap=cc.m_CET_L8
        )
        ax.set_aspect("equal")
        plt.show()

    test_plot_isomap()
    return


@app.cell
def _(isomap_embeddings, np):
    import trimap

    np.random.seed(1)
    trimap_embeddder = trimap.TRIMAP(n_dims=3, weight_temp=0.25)
    trimap_embeddings = trimap_embeddder.fit_transform(isomap_embeddings[:, :])
    return (trimap_embeddings,)


@app.cell
def _(cc, np, os, plt, sample_order_estimates, scale_op65, trimap_embeddings):
    TOTAL_FRAME = 24 * 40

    def plot_3d_trimap(timestep):
        # Set up 3D plot:
        width = 10
        fig = plt.figure(figsize=(width, width * (9 / 16)))
        ax = fig.add_subplot(projection='3d')

        # We aim to complete the loop at 60 seconds:
        ax.view_init(elev=0, azim=(timestep/TOTAL_FRAME) * 720, roll=(timestep/TOTAL_FRAME) * 360)

        # Set up coloring:
        color_values = scale_op65(sample_order_estimates)
        color_sort = np.argsort(color_values)

        # Plot:
        ax.scatter(
            trimap_embeddings[color_sort, 0],
            trimap_embeddings[color_sort, 1],
            -trimap_embeddings[color_sort, 2],
            alpha=0.1, s=2, edgecolors="none",
            c=color_values[color_sort],
            vmin=np.quantile(color_values, 0.05),
            vmax=np.quantile(color_values, 0.95),
            cmap=cc.m_CET_L8
        )

        ax.set_axis_off()
        ax.set_aspect("equal")

        edging = 0.00
        fig.subplots_adjust(edging, edging, 1-edging, 1-edging)

    def write_video_to_file():

        if not os.path.exists("img_tmp"):
            os.mkdir("img_tmp")

        for timestep in range(TOTAL_FRAME):
            if (timestep + 1) % 24 == 0:
                print(timestep + 1)

            plot_3d_trimap(timestep)
            plt.savefig(os.path.join("img_tmp", f"frame_{timestep}.png"), dpi=150)
            plt.close()

    write_video_to_file()

    # plot_3d_trimap()
    return


@app.cell
def _():
    import subprocess
    subprocess.run("ffmpeg -y -i img_tmp/frame_%d.png -r 24 -vcodec libx264 -crf 18 trimap_video.mp4", shell=True)
    return (subprocess,)


@app.cell
def _(cc, np, parameter_eps, plt, trimap_embeddings):
    def plot_trimap():
        width = 8
        fig, ax = plt.subplots(figsize=(width, width * (9 / 16)))
        # color_vals = np.log(np.clip(scale_op65(sample_order_estimates), 0, None) + 1e-3)
        # color_vals = np.log(primary_eigenvalues)
        # color_vals = sobol_ouputs[5]
        # color_vals = sample_inputs[:, 13]
        color_vals = parameter_eps
        # color_vals = flipped_vectors[:, 5]
        # color_vals = color_vals[density > np.quantile(density, 0.05)]
        color_sort = np.argsort(color_vals)
        color_sort = np.arange(len(color_vals))
        print(color_sort)
        ax.scatter(
            trimap_embeddings[color_sort, 0],
            trimap_embeddings[color_sort, 1], s=1,
            alpha=0.5, edgecolors="none",
            c=color_vals[color_sort],
            vmin=np.quantile(color_vals, 0.05),
            vmax=np.quantile(color_vals, 0.95),
            cmap=cc.m_CET_L8
        )
        ax.set_aspect("equal")
        ax.set_axis_off()

        edging = 0
        fig.subplots_adjust(edging, edging, 1 - edging, 1 - edging)
        print(np.quantile(color_vals, 0.05))
        print(np.quantile(color_vals, 0.95))
        plt.show()

    plot_trimap()
    return


@app.cell
def _():
    # import datashader as ds
    # import datashader.transfer_functions as tf

    # tri_df = pd.DataFrame(trimap_embeddings, columns=["x", "y"])
    # ds_image = tf.shade(tf.spread(ds.Canvas().points(tri_df, "x", "y")), cmap="darkred")
    return


@app.cell
def _():
    # def plot_3d_scatterplot():
    #     # Set up 3D plot:
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = fig.add_subplot(projection='3d')
    #     ax.view_init(elev=30, azim=-110, roll=0)

    #     # Set up coloring:
    #     color_values = scale_op65(sample_order_estimates)
    #     # color_values = density
    #     color_sort = np.argsort(color_values)

    #     # Plot:
    #     ax.scatter(
    #         trimap_embeddings[color_sort, 0],
    #         trimap_embeddings[color_sort, 1],
    #         -trimap_embeddings[color_sort, 2],
    #         alpha=0.1, s=5, edgecolors="none",
    #         c=color_values[color_sort],
    #         vmin=np.quantile(color_values, 0.05),
    #         vmax=np.quantile(color_values, 0.95),
    #         cmap=cc.m_CET_L8
    #     )

    #     ax.set_aspect("equal")
    #     plt.show()

    # plot_3d_scatterplot()
    return


@app.cell
def _():
    # white_embeddings = (umap_embeddings - np.mean(umap_embeddings, axis=0)) / np.mean(np.std(umap_embeddings, axis=0))
    # print(white_embeddings.shape)
    # boundary_stats = gl.utils.boundary_statistic(kpca_embeddings[:, :2], 0.0005)
    return


@app.cell
def _(
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    TEXT_WIDTH,
    cc,
    datetime,
    np,
    os,
    plt,
    sample_order_estimates,
    scale_op65,
    umap_embeddings,
):
    def plot_op_umap():
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.8))

        # Set up coloring:
        color_values = scale_op65(sample_order_estimates)
        # color_values = np.log(np.clip(scale_op65(sample_order_estimates), 0, None) + 1e-3)
        color_sort = np.argsort(color_values)

        # Plot UMAP points:
        pos = ax.scatter(
            umap_embeddings[color_sort, 0], umap_embeddings[color_sort, 1],
            s=1.5, alpha=0.3, c=color_values[color_sort], cmap=cc.m_CET_L8,
            vmin=np.quantile(color_values, 0.05),
            vmax=np.quantile(color_values, 0.95),
            edgecolors='none'
        )

        # Configure axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP I")
        ax.set_ylabel("UMAP II")
        ax.set_aspect("equal")

        # Adjust layout:
        clip = 0.125
        fig.subplots_adjust(clip, 0, 1 - clip, 1, wspace=0, hspace=0)

        # Set up and format colorbar:
        cax = ax.inset_axes((1.03, 0, 0.025, 1))
        cbar = fig.colorbar(pos, cax=cax, label="Matrix Order Parameter")
        cbar.solids.set_alpha(1)

        # Save figures:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "op65_umap.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_op_umap()
    return


@app.cell
def _(
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    TEXT_WIDTH,
    cc,
    datetime,
    np,
    os,
    plt,
    primary_eigenvectors,
    umap_embeddings,
):
    normalised_components = np.abs(primary_eigenvectors) / np.sum(np.abs(primary_eigenvectors), axis=1, keepdims=True)
    parameter_entropy = np.sum(-normalised_components * np.log(normalised_components), axis=1)
    parameter_eps = np.exp(parameter_entropy)

    def plot_eps_umap():
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.8))

        # Set up coloring:
        color_values = parameter_eps
        color_sort = np.argsort(color_values)

        # Plot UMAP points:
        pos = ax.scatter(
            umap_embeddings[color_sort, 0], umap_embeddings[color_sort, 1],
            s=1, alpha=0.8, c=color_values[color_sort], cmap=cc.m_CET_L20,
            vmin=np.quantile(color_values, 0.04),
            vmax=np.quantile(color_values, 0.99),
            edgecolors='none'
        )

        # Configure axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP I")
        ax.set_ylabel("UMAP II")
        ax.set_aspect("equal")

        # Adjust layout:
        clip = 0.125
        fig.subplots_adjust(clip, 0, 1 - clip, 1, wspace=0, hspace=0)

        # Set up and format colorbar:
        cax = ax.inset_axes((1.03, 0, 0.025, 1))
        cbar = fig.colorbar(pos, cax=cax, label="Parameter Degeneracy $\\varepsilon$")
        cbar.solids.set_alpha(1)

        # Save figures:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "eps_umap.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_eps_umap()
    return (parameter_eps,)


@app.cell
def _(
    FULL_HEIGHT,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    TEXT_WIDTH,
    cc,
    datetime,
    flipped_vectors,
    np,
    os,
    parameter_list,
    plt,
    pos,
    umap_embeddings,
):
    def plot_umap(ax, color_index):
        color_values = flipped_vectors[:, color_index]
        color_sort = np.argsort(color_values)
        pos = ax.scatter(
            umap_embeddings[color_sort, 0], umap_embeddings[color_sort, 1],
            s=1, alpha=0.1, c=color_values[color_sort], cmap=cc.m_CET_D13,
            vmin=-0.6, vmax=0.6, edgecolors='none'
        )
        ax.set_aspect("equal")
        ax.text(0.03, 0.03, parameter_list[color_index], transform=ax.transAxes, fontsize=4)
        ax.set_xticks([])
        ax.set_yticks([])

        return pos


    def plot_all_components():
        fig, axs = plt.subplots(
            5, 3, figsize=(TEXT_WIDTH, FULL_HEIGHT),
            sharex=True, sharey=True
        )

        count = 0
        for i in range(5):
            for j in range(3):
                if count == 14:
                    # Remove spines:
                    axs[i, j].set_axis_off()
                    # Set up and format colorbar:
                    cax = axs[i, j].inset_axes((0.15, 0.1, 0.05, 0.8))
                    cbar = fig.colorbar(pos, cax=cax, label="$|\\theta|$ Component")
                    cbar.solids.set_alpha(1)
                    continue
                pos = plot_umap(axs[i, j], count)
                count += 1

        fig.text(0.5, 0.01, 'UMAP I', ha='center', va="bottom")
        fig.text(0.07, 0.5, 'UMAP II', ha='left', va='center', rotation='vertical')

        w_clip = 0.1
        h_clip = 0.04
        fig.subplots_adjust(w_clip, h_clip, 1 - w_clip, 1 - h_clip, wspace=0, hspace=0.075)

        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "hessian_eigcomponent_umap_grid.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_all_components()
    return


@app.cell
def _(get_eigenvectors, np, umap_manager):
    # Load Hessians of Sobol search:
    sobol_hessians = np.load("model_experiments/2026-06-03-matrix_shape/sobol_op65_hessian/hessian_estimate.npy")
    sobol_eigenvalues, sobol_eigenvectors = get_eigenvectors(sobol_hessians)
    sobol_embeddings = umap_manager.transform(sobol_eigenvectors[:, 0, :])
    return (sobol_embeddings,)


@app.cell
def _(EXPERIMENT_DIRPATH, np, os):
    # Load simulation values:
    sobol_matrix = np.load(os.path.join(EXPERIMENT_DIRPATH, "sample_matrix.npy"))
    sl_mask = sobol_matrix < 0.02
    sh_mask = sobol_matrix > 0.98
    sobol_mask = ~np.logical_or(np.any(sl_mask, axis=1), np.any(sh_mask, axis=1))
    masked_indices = np.arange(2 ** 17)[sobol_mask]

    SIM_METRICS = [
        "speeds",
        "meander_ratios",
        "ann_indices",
        "coherency",
        "cell_lengths",
        "order_parameters"
    ]

    SIM_LABELS = [
        "Speed",
        "MR",
        "ANNI",
        "Coherency",
        "Cell Length",
        "Flocking Order Parameter"
    ]

    # Load S65:
    sobol_op = np.load("model_experiments/2026-06-03-matrix_shape/summary_data/matrix_order_parameters.npy")
    sobol_op = np.mean(sobol_op[sobol_mask, :, 2], axis=1)

    sobol_ouputs = []
    for metric in SIM_METRICS:
        sobol_out = np.load(f"model_experiments/2026-06-03-matrix_shape/summary_data/{metric}.npy")
        sobol_out = np.mean(sobol_out[sobol_mask, :], axis=1)
        sobol_ouputs.append(sobol_out)
    return SIM_LABELS, masked_indices, sobol_op, sobol_ouputs


@app.cell
def _(
    FULL_HEIGHT,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    SIM_LABELS,
    TEXT_WIDTH,
    cc,
    datetime,
    np,
    os,
    plt,
    sobol_embeddings,
    sobol_ouputs,
):
    def plot_metric_umap():
        fig, axs = plt.subplots(3, 2, figsize=(TEXT_WIDTH, FULL_HEIGHT))

        count = 0
        for i in range(3):
            for j in range(2):
                # if count == 5:
                #     axs[i, j].set_axis_off()
                #     continue
                # Get heatmap information:
                color_values = sobol_ouputs[count]
                color_sort = np.argsort(color_values)
                nan_mask = ~np.isnan(color_values[color_sort])
                vmin = np.quantile(color_values[color_sort][nan_mask], 0.05)
                vmax = np.quantile(color_values[color_sort][nan_mask], 0.95)

                # Do scatter plot:
                scatter_out = axs[i, j].scatter(
                    sobol_embeddings[color_sort, 0][nan_mask], sobol_embeddings[color_sort, 1][nan_mask],
                    c=color_values[color_sort][nan_mask], cmap=cc.m_CET_L8,
                    vmin=vmin, vmax=vmax,
                    s=1, alpha=0.1, edgecolors='none'
                )

                # Format axes and label:
                axs[i, j].set_aspect("equal")
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                axs[i, j].text(0.02, 0.03, SIM_LABELS[count], transform=axs[i, j].transAxes)

                # Set up colorbar:
                cbar = fig.colorbar(scatter_out, ax=axs[i, j], fraction=0.1, shrink=0.88)
                cbar.solids.set_alpha(1)

                count += 1

        edging = 0.05
        fig.subplots_adjust(edging, edging, 1 - edging, 1 - edging, wspace=0.075, hspace=0.05)

        fig.text(0.5, 0.03, 'UMAP I', ha='center', va="center")
        fig.text(0.01, 0.5, 'UMAP II', ha='center', va='center', rotation='vertical')

        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "metric_umap_grid.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_metric_umap()
    return


@app.cell
def _(
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    TEXT_WIDTH,
    datetime,
    np,
    os,
    plt,
    sklearn,
    sobol_embeddings,
    sobol_op,
):
    def plot_masked_umap():
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.8))

        # Set up coloring:
        color_values = sobol_op
        op_mask = sobol_op >  0.04

        # Get density:
        kde = sklearn.neighbors.KernelDensity(bandwidth=0.5, rtol=0.1).fit(sobol_embeddings[op_mask, :])
        log_density = kde.score_samples(sobol_embeddings[op_mask, :])
        density_mask = log_density > -4

        # Get joint OP-density mask:
        valid_indices = np.argwhere(op_mask)
        valid_indices = np.delete(valid_indices, np.argwhere(~density_mask))
        joint_mask = np.zeros(len(op_mask)).astype(bool)
        joint_mask[valid_indices] = 1

        # Set up clustering:
        cluster_manager = sklearn.cluster.KMeans(n_clusters=3, random_state=0)
        cluster_manager.fit(sobol_embeddings[joint_mask, :])
        cluster_labels = cluster_manager.labels_

        # Plot background points:
        ax.scatter(sobol_embeddings[:, 0], sobol_embeddings[:, 1], c='k', alpha=0.1, s=0.5, edgecolors="none")

        # Plot UMAP points:
        labels = ["Cluster A", "Cluster B", "Cluster C"]
        for label in np.unique(cluster_labels):
            if label == -1:
                continue
            label_mask = cluster_labels == label
            ax.scatter(
                sobol_embeddings[joint_mask][label_mask, 0], sobol_embeddings[joint_mask][label_mask, 1],
                s=1.5, alpha=0.3, label=labels[label]
            )

        # Configure axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP I")
        ax.set_ylabel("UMAP II")
        ax.set_aspect("equal")

        ax.legend()

        # Adjust layout:
        clip = 0.125
        fig.subplots_adjust(clip, 0, 1 - clip, 1, wspace=0, hspace=0)

        # # Set up and format colorbar:
        # cax = ax.inset_axes((1.03, 0, 0.025, 1))
        # cbar = fig.colorbar(pos, cax=cax, label="Matrix Order Parameter")
        # cbar.solids.set_alpha(1)

        # Save figures:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "op65_umap_clustered.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

        return cluster_labels, joint_mask

    cluster_labels, joint_mask = plot_masked_umap()
    return cluster_labels, joint_mask


@app.cell
def _(
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    SIM_LABELS,
    TEXT_HEIGHT,
    TEXT_WIDTH,
    cluster_labels,
    datetime,
    joint_mask,
    np,
    os,
    plt,
    sobol_ouputs,
):
    def generate_cluster_arrays(input_values):
        cluster_list = []
        for label in np.unique(cluster_labels):
            label_mask = cluster_labels == label  
            cluster_list.append(input_values[joint_mask][label_mask])
        return cluster_list

    def plot_cluster_boxplots():
        # Show boxplots:
        fig, axs = plt.subplots(6, 1, figsize=(TEXT_WIDTH, TEXT_HEIGHT), sharex=True)

        for metric_index in range(6):
            axs[metric_index].boxplot(generate_cluster_arrays(sobol_ouputs[metric_index]))
            axs[metric_index].set_ylabel(SIM_LABELS[metric_index])

            if metric_index == 5:
                axs[metric_index].set_xticks([1, 2, 3], ["Cluster A", "Cluster B", "Cluster C"])

        fig.subplots_adjust(0.1, 0.025, 1 - 0.1, 1 - 0.025, hspace=0.05)

        # Save figures:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "boxplot_cluster_comparison.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_cluster_boxplots()
    return


@app.cell
def _(
    FULL_WIDTH,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    datetime,
    masked_indices,
    np,
    os,
    plt,
    sobol_embeddings,
    sobol_ouputs,
):
    import imageio.v3 as iio

    def load_image(index, image_type):
        npz_archive = np.load(os.path.join("model_experiments/2026-06-03-matrix_shape", f"{image_type}.npz"))
        image_array = npz_archive[str(index)]
        return image_array

    def get_simulation_indices(grid_size, image_type):
        i_min = np.quantile(sobol_embeddings[:, 0], 0)
        i_max = np.quantile(sobol_embeddings[:, 0], 1)
        i_bins = np.linspace(i_min, i_max, grid_size + 1)

        j_min = np.quantile(sobol_embeddings[:, 1], 0)
        j_max = np.quantile(sobol_embeddings[:, 1], 1)
        j_bins = np.linspace(j_min, j_max, grid_size + 1)

        fig, ax = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH))

        # Iterate through grid:
        for i in range(grid_size):
            i_mask = np.logical_and(
                sobol_embeddings[:, 0] >= i_bins[i],
                sobol_embeddings[:, 0] <  i_bins[i + 1],
            )
            i_center = (i_bins[i] + i_bins[i + 1]) / 2
            for j in range(grid_size):
                j_mask = np.logical_and(
                    sobol_embeddings[:, 1] >= j_bins[j],
                    sobol_embeddings[:, 1] <  j_bins[j + 1],
                )

                valid_set = np.logical_and(i_mask, j_mask)
                if np.count_nonzero(valid_set) < 100:
                    continue

                # Get distances from center of bin:
                j_center = (j_bins[j] + j_bins[j + 1]) / 2
                center_distances = np.sqrt(
                    np.sum((sobol_embeddings[valid_set] - np.array([i_center, j_center])) ** 2, axis=1)
                )

                # Get median speed in local areaL:
                speed_distances = np.abs(sobol_ouputs[0][valid_set] - np.median(sobol_ouputs[0][valid_set]))
                distances = np.sqrt(center_distances**2 + (1e6*speed_distances**2))
                selection_index = np.argmin(distances)

                # ax.scatter(
                #     sobol_embeddings[valid_set, 0],
                #     sobol_embeddings[valid_set, 1],
                #     s=1
                # )
                # ax.scatter(
                #     sobol_embeddings[valid_set, 0][selection_index],
                #     sobol_embeddings[valid_set, 1][selection_index],
                #     s=1
                # )

                selected_index = masked_indices[valid_set][selection_index]
                trajectory_image = load_image(selected_index, image_type)
                ax.imshow(
                    trajectory_image,
                    extent=(i_bins[i], i_bins[i + 1], j_bins[j], j_bins[j + 1])
                )

        ax.set_xlim(i_min, i_max)
        ax.set_ylim(j_min, j_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP I")
        ax.set_ylabel("UMAP II")

        edging = 0.04
        fig.subplots_adjust(edging, edging, 1 - edging, 1 - edging)

        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, f"umap_{image_type}_image_plot.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    get_simulation_indices(16, "matrix_heading")
    return (load_image,)


@app.cell
def _(get_eigenvectors, np, scale_op65, umap_manager):
    # Load Hessians of posterior parameters:
    fit_inputs = np.load("model_experiments/2026-05-31-collisions_shape/mcmc_results/full_hessians/hessian_inputs.npy")
    fit_hessians = np.load("model_experiments/2026-05-31-collisions_shape/mcmc_results/full_hessians/hessian_estimate.npy")
    full_eigenvalues, full_eigenvectors = get_eigenvectors(fit_hessians)
    full_umap_embeddings = umap_manager.transform(full_eigenvectors[:, 0, :])

    full_estimates = np.load("model_experiments/2026-05-31-collisions_shape/mcmc_results/full_hessians/hessian_outputs.npy")
    wt_full_estimates = scale_op65(full_estimates[:8192])
    rd_full_estimates = scale_op65(full_estimates[8192:])

    # Load Hessians of MLE parameters:
    mle_inputs = np.load("model_experiments/2026-05-31-collisions_shape/mcmc_results/mle_hessians/hessian_inputs.npy")
    mle_hessians = np.load("model_experiments/2026-05-31-collisions_shape/mcmc_results/mle_hessians/hessian_estimate.npy")
    mle_eigenvalues, mle_eigenvectors = get_eigenvectors(mle_hessians)
    mle_umap_embeddings = umap_manager.transform(mle_eigenvectors[:, 0, :])

    mle_estimates = np.load("model_experiments/2026-05-31-collisions_shape/mcmc_results/mle_hessians/hessian_outputs.npy")
    wt_mle_estimates = scale_op65(mle_estimates[:1024])
    rd_mle_estimates = scale_op65(mle_estimates[1024:])

    ctl_mle = mle_inputs[:1024, :]
    rd_mle = mle_inputs[1024:, :]
    return (
        ctl_mle,
        full_eigenvectors,
        mle_eigenvalues,
        mle_eigenvectors,
        mle_inputs,
        mle_umap_embeddings,
        rd_mle,
        rd_mle_estimates,
        wt_mle_estimates,
    )


@app.cell
def _(
    CONTROL_PALETTE,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    RD_PALETTE,
    TEXT_WIDTH,
    datetime,
    mle_eigenvalues,
    os,
    plt,
):
    wt_eigenvalues = mle_eigenvalues[:1024, 0]
    rd_eigenvalues = mle_eigenvalues[1024:, 0]

    def plot_mle_eigenvalue_distributions():
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.0))
        ax.hist(wt_eigenvalues, density=True, color=CONTROL_PALETTE, histtype="step", alpha=0.75, bins=20)
        ax.hist(rd_eigenvalues, density=True, color=RD_PALETTE, histtype="step", alpha=0.75, bins=20)
        ax.set_ylabel("Density")
        ax.set_xlabel("MLE $|\\lambda|$")
        ax.set_title("Eigenvalue Distribution of MLE Fits")

        fig.subplots_adjust(0.1, 0.175, 0.9, 0.85)

        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "mle_eigenvalue_hist.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_mle_eigenvalue_distributions()
    return


@app.cell
def _(
    CONTROL_PALETTE,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    RD_PALETTE,
    TEXT_WIDTH,
    datetime,
    full_eigenvectors,
    np,
    os,
    plt,
):
    mle_primary_eigenvectors = np.abs(full_eigenvectors[:, 0, :])
    mle_norm_eigenvectors = mle_primary_eigenvectors / np.sum(mle_primary_eigenvectors, axis=1, keepdims=True)
    mle_eps = np.exp(- np.sum(mle_norm_eigenvectors * np.log(mle_norm_eigenvectors), axis=1))
    wt_eps = mle_eps[:1024]
    rd_eps = mle_eps[1024:]

    def plot_mle_eps_distributions():
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.0))
        ax.hist(wt_eps, density=True, color=CONTROL_PALETTE, histtype="step", alpha=0.75, bins=25)
        ax.hist(rd_eps, density=True, color=RD_PALETTE, histtype="step", alpha=0.75, bins=25)
        ax.set_ylabel("Density")
        ax.set_xlabel("Parameter Degeneracy $\\varepsilon$")
        ax.set_title("Parameter Degeneracy Distribution of MLE Fits")
        fig.subplots_adjust(0.1, 0.175, 0.9, 0.85)

        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "mle_eps_hist.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_mle_eps_distributions()
    return


@app.cell
def _():
    # test_eigenvectors = np.abs(mle_eigenvectors[:, 0, :])
    # test_eigenvectors /= np.sum(test_eigenvectors, axis=1, keepdims=True)
    # entropies = np.sum(-test_eigenvectors * np.log(test_eigenvectors), axis=1)
    # print(np.count_nonzero(np.isnan(entropies)))
    # plt.hist(entropies[:1024], histtype="step", bins=50);
    # plt.hist(entropies[1024:], histtype="step", bins=50);
    # plt.show()
    # np.unique(test_eigenvectors, axis=0)
    return


@app.cell
def _(calculate_geometric_median, mle_eigenvectors):
    wt_median_estimate, _, _ = calculate_geometric_median(mle_eigenvectors[:1024, 0, :], None)
    wt_median_estimate *= -1
    rd_median_estimate, _, _ = calculate_geometric_median(mle_eigenvectors[1024:, 0, :], None)
    rd_median_estimate *= -1
    return rd_median_estimate, wt_median_estimate


@app.cell
def _(
    CONTROL_PALETTE,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    RD_PALETTE,
    TEXT_WIDTH,
    datetime,
    mle_inputs,
    np,
    os,
    plt,
    rd_median_estimate,
    rd_mle_estimates,
    wt_median_estimate,
    wt_mle_estimates,
):
    def plot_local_eigenparameters():
        fig, axs = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2), sharey=True)

        wt_converted = np.log(mle_inputs) @ wt_median_estimate.T
        axs[0].scatter(wt_converted[:1024], wt_mle_estimates, c=CONTROL_PALETTE, s=1, alpha=0.75)
        axs[0].set_xlabel("$log(\\hat{\\theta}_{CTL})$")
        axs[0].set_ylabel("Matrix Order Parameter")

        rd_converted = np.log(mle_inputs) @ rd_median_estimate.T
        axs[1].scatter(rd_converted[1024:], rd_mle_estimates, c=RD_PALETTE, s=1, alpha=0.75)
        axs[1].set_xlabel("$log(\\hat{\\theta}_{RD})$")

        clip = 0.1
        fig.subplots_adjust(0.1, 0.2, 0.9, 0.9, wspace=0.05, hspace=0.0)

        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "mle_eigenparameter_reg.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_local_eigenparameters()
    return


@app.cell
def _(np):
    def bold_format(number):
        if np.abs(number) < 0.1:
            return f"{number:.2f}"
        else:
            return f"\\textbf{{{number:.2f}}}"
    return (bold_format,)


@app.cell
def _(np, parameter_list, pd, rd_median_estimate, wt_median_estimate):
    median_dataframe = pd.DataFrame(
        np.vstack([wt_median_estimate, rd_median_estimate]).T,
        columns=["$\\hat{\\theta}_{CTL}$", "$\\hat{\\theta}_{RD}$"],
        index=parameter_list
    )
    return (median_dataframe,)


@app.cell
def _(bold_format, median_dataframe):
    print(median_dataframe.to_latex(float_format=bold_format))
    return


@app.cell
def _(
    CONTROL_PALETTE,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    RD_PALETTE,
    TEXT_WIDTH,
    datetime,
    mle_umap_embeddings,
    os,
    plt,
    umap_embeddings,
):
    def plot_mle_umap():
        fig, axs = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.5), sharex=True, sharey=True)

        # Plot control MLE fit:
        axs[0].scatter(
            umap_embeddings[:, 0], umap_embeddings[:, 1],
            s=0.1, alpha=0.2, c='k', edgecolors='none'
        )
        axs[0].scatter(
            mle_umap_embeddings[:1024, 0], mle_umap_embeddings[:1024, 1],
            s=1, alpha=0.4, c=CONTROL_PALETTE, edgecolors='none'
        )
        axs[0].set_aspect("equal")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].text(0.03, 0.03, "Control Fit", transform=axs[0].transAxes, fontsize=8)
        axs[0].set_ylabel("UMAP II")

        # Plot RD MLE fit:
        axs[1].scatter(
            umap_embeddings[:, 0], umap_embeddings[:, 1],
            s=0.1, alpha=0.2, c='k', edgecolors='none'
        )
        axs[1].scatter(
            mle_umap_embeddings[1024:, 0], mle_umap_embeddings[1024:, 1],
            s=1, alpha=0.4, c=RD_PALETTE, edgecolors='none'
        )
        axs[1].set_aspect("equal")
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].text(0.03, 0.03, "RD Fit", transform=axs[1].transAxes, fontsize=8)

        fig.text(0.5, 0.015, 'UMAP I', ha='center', va="bottom")

        # Adjust layout:
        fig.subplots_adjust(0.05, 0.1, 1 - 0.05, 1 - 0.1, wspace=0.0)

        # Save plot:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "mle_on_umap.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_mle_umap()
    return


@app.cell
def _(np):
    log_transforms = np.load("model_experiments/2026-06-03-matrix_shape/gaussian_process_models/op65/global_eigenparameter_estimation/log_transforms.npy")
    return (log_transforms,)


@app.cell
def _(np):
    loss_histories = np.load("model_experiments/2026-06-03-matrix_shape/gaussian_process_models/op65/global_eigenparameter_estimation/loss_histories.npy")
    log_scale_factors = np.load("model_experiments/2026-06-03-matrix_shape/gaussian_process_models/op65/global_eigenparameter_estimation/scale_factors.npy")
    log_scale_factors = np.squeeze(log_scale_factors)
    return log_scale_factors, loss_histories


@app.cell
def _(log_scale_factors):
    log_scale_factors.shape
    return


@app.cell
def _(log_scale_factors, log_transforms, loss_histories, np):
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
    return (regularised_transforms,)


@app.cell
def _(loss_histories, np, regularised_transforms):
    def bootstrap_component_errors(iterations):
        # Get mean loss:
        mean_loss = np.mean(loss_histories[:, 4096:], axis=1)

        # Set up bootstrap sampling:
        rng = np.random.default_rng(0)

        bootstrap_dataset = []
        for _ in range(iterations):
            gee_estimate_size = len(mean_loss)
            selected_indices = rng.choice(gee_estimate_size, size=gee_estimate_size)
            min_index = np.argmin(mean_loss[selected_indices])
            bootstrap_dataset.append(regularised_transforms[selected_indices, :, :][min_index])

        return np.stack(bootstrap_dataset, axis=0)

    bootstrap_transform = bootstrap_component_errors(8192)
    return (bootstrap_transform,)


@app.cell
def _(
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    TEXT_WIDTH,
    bootstrap_transform,
    cc,
    datetime,
    np,
    os,
    parameter_list,
    plt,
):
    def plot_bootstrap_components():
        fig, axs = plt.subplots(1, 3, figsize=(TEXT_WIDTH, TEXT_WIDTH))

        for component_index in range(3):
            # Plot errorbars:
            bootstrap_means = np.mean(bootstrap_transform[:, component_index, :], axis=0)
            axs[component_index].errorbar(
                bootstrap_means,
                np.arange(14) + 1,
                xerr=np.std(bootstrap_transform[:, component_index, :], axis=0) * 1.96,
                fmt='none', ecolor='k', alpha=0.5
            );

            # Plot points:
            axs[component_index].scatter(
                bootstrap_means,
                np.arange(14) + 1, s=20,
                c=bootstrap_means, cmap=cc.m_CET_D4,
                vmin=-0.5, vmax=0.5
            );

            # Plot guidelines:
            axs[component_index].vlines(0, 0, 14, color='k', ls="--", lw=1)
            axs[component_index].hlines(np.arange(14) + 1, -1, 1, color='k', ls="--", lw=0.2)

            # Format axes:
            axs[component_index].set_xlim(-1, 1)
            axs[component_index].set_ylim(0.5,  14.5)
            axs[component_index].yaxis.set_inverted(True)
            if component_index == 0:
                axs[component_index].set_yticks(np.arange(14) + 1, parameter_list)
            else:
                axs[component_index].set_yticks([])

            # Label plot:
            axs[component_index].set_xlabel(f"$\\vartheta_{component_index + 1}$")

        # Adjust subplots:
        clip = 0.06
        fig.subplots_adjust(0.25, clip, 0.75, 1 - clip, wspace=0.1, hspace=0.0)

        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "gee_bootstrap_estimate.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_bootstrap_components()
    return


@app.cell
def _(log_scale_factors, loss_histories, np, regularised_transforms):
    # Get mean loss:
    mean_loss = np.mean(loss_histories[:, 4096:], axis=1)
    final_transform = np.copy(regularised_transforms[np.argmin(mean_loss), :, :]).T
    final_sf = log_scale_factors[np.argmin(mean_loss), :]
    final_sf /= np.max(final_sf)

    print(final_sf)
    print(final_transform.shape)
    print(np.linalg.norm(final_transform, axis=0))

    clip_mask = np.abs(final_transform) < 0.1
    final_transform[clip_mask] = 0
    final_transform /= np.linalg.norm(final_transform, axis=0, keepdims=True)
    return (final_transform,)


@app.cell
def _(final_transform, parameter_list, pd):
    transform_dataframe = pd.DataFrame(
        final_transform,
        columns=["$\\vartheta_1$", "$\\vartheta_2$", "$\\vartheta_3$"],
        index=parameter_list
    )
    return (transform_dataframe,)


@app.cell
def _(bold_format, transform_dataframe):
    print(transform_dataframe.to_latex(float_format=bold_format))
    return


@app.cell
def _(
    FULL_HEIGHT,
    FULL_WIDTH,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    colorstamps,
    datetime,
    final_transform,
    np,
    os,
    plt,
    sample_inputs,
    umap_embeddings,
):
    ge_inputs = np.log(sample_inputs) @ final_transform

    i_index = 0
    j_index = 1

    def normalise(x, clip=0.1):
        x_low, x_high = np.quantile(x, [clip, 1-clip])
        norm_x = (x - x_low) / (x_high - x_low)
        return np.clip(norm_x, 0, 1)

    def plot_cs_scatter_transform(ax, i_index, j_index):
        # Get relevant quantiles:
        i_min, i_max = np.quantile(ge_inputs[:, i_index], [0.02, 0.98])
        j_min, j_max = np.quantile(ge_inputs[:, j_index], [0.02, 0.98])
        c, _ = colorstamps.apply_stamp(
            ge_inputs[:, i_index], ge_inputs[:, j_index],
            'flat',
            vmin_0=i_min, vmax_0=i_max,
            vmin_1=j_min, vmax_1=j_max,
        )

        # Plot scatterplot:
        ax.scatter(
            ge_inputs[:, i_index], ge_inputs[:, j_index],
            s=1, c=c, edgecolors='none', alpha=0.5
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"$\\vartheta_{i_index+1}$")
        ax.set_ylabel(f"$\\vartheta_{j_index+1}$")
        ax.set_aspect("equal")

    def plot_cs_eigenparameter_umap(ax, i_index, j_index):
        # Get colorstamped cmap:
        clip = 0.05
        i_min, i_max = np.quantile(ge_inputs[:, i_index], [clip, 1 - clip])
        j_min, j_max = np.quantile(ge_inputs[:, j_index], [clip, 1 - clip])

        c, _ = colorstamps.apply_stamp(
            ge_inputs[:, i_index], ge_inputs[:, j_index],
            'flat',
            vmin_0=i_min, vmax_0=i_max,
            vmin_1=j_min, vmax_1=j_max,
        )

        # Generate plot of embeddings:
        ax.scatter(
            umap_embeddings[:, 0], umap_embeddings[:, 1],
            s=1, alpha=0.2, c=c, edgecolors='none'
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP I")
        ax.set_ylabel("UMAP II")
        ax.set_aspect("equal")

    def plot_eigenparameter_umap():
        fig, axs = plt.subplots(3, 2, figsize=(FULL_WIDTH * 0.8, FULL_HEIGHT))

        for row_index in range(3):
            i, j = [
                [0, 1],
                [1, 2],
                [0, 2]
            ][row_index]
            plot_cs_eigenparameter_umap(axs[row_index, 0], i, j)
            plot_cs_scatter_transform(axs[row_index, 1], i, j)

        # Adjust subplot configuration:
        fig.subplots_adjust(0.1, 0.05, 0.9, 0.95, wspace=0.0, hspace=0.15)

        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "gee_coloured_umap.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    plot_eigenparameter_umap()
    return (ge_inputs,)


@app.cell
def _(eigenvectors, ge_inputs, np, plt):
    plt.scatter(ge_inputs[:, 2], np.abs(eigenvectors[:, 0, 5]), s=1, alpha=0.5)
    return


@app.cell
def _(EXPERIMENT_DIRPATH, final_transform, np, os):
    sobol_inputs = np.load(os.path.join(EXPERIMENT_DIRPATH, "sample_matrix.npy"))
    ge_sobol_inputs = np.log(sobol_inputs) @ final_transform

    ge_sobol_inputs[:, 0] = np.log(ge_sobol_inputs[:, 0])

    sobol_order_parameters = np.load("model_experiments/2026-06-03-matrix_shape/summary_data/matrix_order_parameters.npy")
    sobol_order_parameters = np.mean(sobol_order_parameters[:, :, 2], axis=1)

    sobol_lengths = np.load("model_experiments/2026-06-03-matrix_shape/summary_data/cell_lengths.npy")
    sobol_lengths = np.mean(sobol_lengths, axis=1)

    flocking_op = np.load("model_experiments/2026-06-03-matrix_shape/summary_data/meander_ratios.npy")
    flocking_op = np.mean(flocking_op, axis=1)

    # model_experiments/2026-06-03-matrix_shape/summary_data/speeds.npy
    return ge_sobol_inputs, sobol_lengths, sobol_order_parameters


@app.cell
def _(np, plt, sobol_order_parameters):
    ss_op = np.log(np.copy(sobol_order_parameters))
    ss_op = (ss_op - np.mean(ss_op)) / np.std(ss_op)
    plt.hist(ss_op, bins=100)
    return (ss_op,)


@app.cell
def _(ge_sobol_inputs, load_image, np, ss_op):
    def get_bins(inputs):
        # Get bounds:
        lower_bound, upper_bound = np.quantile(inputs, [0.025, 0.975])
        bin_edges = np.linspace(lower_bound, upper_bound, 15)
        bin_centers = bin_edges[:-1] + np.diff(bin_edges)
        return bin_centers, bin_edges

    def generate_collage(i, j, k):
        i_centers, i_edges = get_bins(ge_sobol_inputs[:, i])
        j_centers, j_edges = get_bins(ge_sobol_inputs[:, j])
        k_center = np.quantile(ge_sobol_inputs[:, k], 0.5)

        print(i_centers)
        print(j_centers)

        collage = []
        indices = []
        for i_index, i_center in enumerate(i_centers):
            col = []
            i_distances = ge_sobol_inputs[:, i] - i_center
            for j_index, j_center in enumerate(j_centers):
                j_distances = ge_sobol_inputs[:, j] - j_center

                # Get quantile of local order parameter:
                i_mask = np.logical_and(
                    ge_sobol_inputs[:, i] >= i_edges[i_index],
                    ge_sobol_inputs[:, i] < i_edges[i_index + 1]
                )
                j_mask = np.logical_and(
                    ge_sobol_inputs[:, j] >= j_edges[j_index],
                    ge_sobol_inputs[:, j] < j_edges[j_index + 1]
                )
                local_mask = np.logical_and(i_mask, j_mask)
                quantile_value = np.quantile(ss_op[local_mask], 0.5)
                k_distances = 1e3 * (ss_op - quantile_value) ** 2
                # k_distances = (ge_sobol_inputs[:, k] - np.quantile(ge_sobol_inputs[local_mask, k], 0.5)) ** 2

                combined_distances = np.sqrt((i_distances**2) + (j_distances**2) + k_distances)

                image_index = np.argmin(combined_distances)
                # print(image_index)
                image_array = load_image(image_index, "trajectory")
                col.append(image_array)
                indices.append(image_index)
            collage.append(np.concatenate(col[::-1], axis=0))

        return np.concatenate(collage, axis=1), i_edges, j_edges, indices
    return


@app.cell
def _():
    # collage, i_edges, j_edges, indices = generate_collage(0, 1, 2)
    return


@app.cell
def _(ctl_transformed, ge_sobol_inputs, indices, np, plt, rd_transformed):
    plt.scatter(ge_sobol_inputs[:, 0], ge_sobol_inputs[:, 1], s=1)
    plt.scatter(ge_sobol_inputs[indices, 0], ge_sobol_inputs[indices, 1], s=10)

    plt.scatter(np.log(ctl_transformed[:, 0]), ctl_transformed[:, 1], s=1)
    plt.scatter(np.log(rd_transformed[:, 0]),  rd_transformed[:, 1], s=1)
    return


@app.cell
def _(
    CONTROL_PALETTE,
    FULL_WIDTH,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    RD_PALETTE,
    ctl_transformed,
    datetime,
    i_edges,
    j_edges,
    np,
    os,
    plt,
    rd_transformed,
):
    def show_collage(collage):
        fig, ax = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH))
        ax.imshow(
            collage, aspect="auto",
            extent=(i_edges.min(), i_edges.max(), j_edges.min(), j_edges.max())
        )

        ax.scatter(np.log(ctl_transformed[:, 0]), ctl_transformed[:, 1], s=10, alpha=0.5, c=CONTROL_PALETTE, edgecolors="none")
        ax.scatter(np.log(rd_transformed[:, 0]), rd_transformed[:, 1], s=10, alpha=0.5, c=RD_PALETTE, edgecolors="none")
        ax.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
        ax.set_xlim(i_edges.min(), i_edges.max())
        ax.set_ylim(j_edges.min(), j_edges.max())
        ax.set_xlabel("$log(\\vartheta_1)$")
        ax.set_ylabel("$\\vartheta_2$")

        edging = 0.075
        fig.subplots_adjust(edging, edging, 1 - edging, 1 - edging)

        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "scatter_image_ge_plot.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    # show_collage(collage)
    return


@app.cell
def _(ctl_mle, final_transform, np, rd_mle):
    batch_shape = ctl_mle.shape[0]
    parameter_filler = np.ones(batch_shape) * 0.5

    ctl_transformed = np.log(ctl_mle) @ final_transform
    rd_transformed = np.log(rd_mle) @ final_transform

    adjusted_rd_mle = np.copy(rd_mle)
    adjusted_rd_mle[:, 0] *= 2.3
    # adjusted_rd_mle[:, 3] /= 1.1
    adj_rd_transformed = np.log(adjusted_rd_mle) @ final_transform
    return adj_rd_transformed, ctl_transformed, rd_transformed


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
    CONTROL_PALETTE,
    FULL_WIDTH,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    RD_PALETTE,
    ctl_transformed,
    datetime,
    ge_sobol_inputs,
    get_kde_grid,
    np,
    os,
    plt,
    rd_transformed,
):
    def comparative_kde_plot(ax, i, j, bin_count=50):    
        # Plot background distribution:
        bg_x, bg_y, bg_density = get_kde_grid(ge_sobol_inputs[:, i], ge_sobol_inputs[:, j], bin_count, 0.02)
        ax.contour(bg_x, bg_y, bg_density.reshape(bin_count, bin_count), 5, colors="k", alpha=0.1)

        # Plot control distribution:
        ctl_x, ctl_y, ctl_density = get_kde_grid(ctl_transformed[:, i], ctl_transformed[:, j], bin_count, 0.005)
        ax.contour(ctl_x, ctl_y, ctl_density.reshape(bin_count, bin_count), 3, colors=CONTROL_PALETTE, alpha=0.5)
        ax.scatter(
            ctl_transformed[:, i], ctl_transformed[:, j],
            c=CONTROL_PALETTE,
            s=1, edgecolors="none",
            alpha=0.2
        )

        # Plot control distribution:
        rd_x, rd_y, rd_density = get_kde_grid(rd_transformed[:, i], rd_transformed[:, j], bin_count, 0.005)
        ax.contour(rd_x, rd_y, rd_density.reshape(bin_count, bin_count), 3, colors=RD_PALETTE, alpha=0.5)
        ax.scatter(
            rd_transformed[:, i], rd_transformed[:, j],
            c=RD_PALETTE,
            s=1, edgecolors="none",
            alpha=0.2
        )

        x_data = np.concatenate([ctl_x, rd_x])
        ax.set_xlim(x_data.min(), x_data.max())

        y_data = np.concatenate([ctl_y, rd_y])
        ax.set_ylim(y_data.min(), y_data.max())

        # Label axes:
        ax.set_xlabel(f"$\\vartheta_{i + 1}$")
        ax.set_ylabel(f"$\\vartheta_{j + 1}$")

    def comparative_histogram(ax, i):
        # ctl_kde = scipy.stats.gaussian_kde(ctl_transformed[:, i])
        # ctl_input = np.linspace(ctl_transformed[:, i].min(), ctl_transformed[:, i].max(), 50)
        # ax.plot(ctl_input, ctl_kde(ctl_input), color=CONTROL_PALETTE, alpha=0.75)

        # rd_kde = scipy.stats.gaussian_kde(rd_transformed[:, i])
        # rd_input = np.linspace(rd_transformed[:, i].min(), rd_transformed[:, i].max(), 50)
        # ax.plot(rd_input, rd_kde(rd_input), color=RD_PALETTE, alpha=0.75)
        bins = 30
        ax.hist(
            ctl_transformed[:, i], bins=bins,
            color=CONTROL_PALETTE, alpha=0.75,
            histtype="step", density=True, label="Control"
        )

        ax.hist(
            rd_transformed[:, i], bins=bins,
            color=RD_PALETTE, alpha=0.75,
            histtype="step", density=True, label="RD"
        )

        ax.set_xlabel(f"$\\vartheta_{i + 1}$")
        ax.set_ylim(0, None)

        if i == 0:
            ax.legend()

    def comparative_kde_scattergrid():
        fig, axs = plt.subplots(3, 3, figsize=(FULL_WIDTH, FULL_WIDTH))
        for i in range(3):
            for j in range(3):
                if i == j:
                    comparative_histogram(axs[i, j], i)
                else:
                    comparative_kde_plot(axs[i, j], i, j, 50)

        # Adjust subplot configuration:
        clip = 0.05
        fig.subplots_adjust(clip, clip, 1 - clip, 1 - clip, wspace=0.25, hspace=0.25)

        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "kde_mle_comparative_plot.png"),
            dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        )
        plt.show()

    comparative_kde_scattergrid()
    return


@app.cell
def _(
    ADJ_RD_PALETTE,
    CONTROL_PALETTE,
    FULL_WIDTH,
    adj_rd_transformed,
    ctl_transformed,
    ge_sobol_inputs,
    get_kde_grid,
    np,
    plt,
):
    def adj_comparative_kde_plot(ax, i, j, bin_count=50):    
        # Plot background distribution:
        bg_x, bg_y, bg_density = get_kde_grid(ge_sobol_inputs[:, i], ge_sobol_inputs[:, j], bin_count, 0.02)
        ax.contour(bg_x, bg_y, bg_density.reshape(bin_count, bin_count), 5, colors="k", alpha=0.1)

        # Plot control distribution:
        ctl_x, ctl_y, ctl_density = get_kde_grid(ctl_transformed[:, i], ctl_transformed[:, j], bin_count, 0.005)
        ax.contour(ctl_x, ctl_y, ctl_density.reshape(bin_count, bin_count), 3, colors=CONTROL_PALETTE, alpha=0.5)
        ax.scatter(
            ctl_transformed[:, i], ctl_transformed[:, j],
            c=CONTROL_PALETTE,
            s=1, edgecolors="none",
            alpha=0.2
        )

        # Plot control distribution:
        rd_x, rd_y, rd_density = get_kde_grid(adj_rd_transformed[:, i], adj_rd_transformed[:, j], bin_count, 0.005)
        ax.contour(rd_x, rd_y, rd_density.reshape(bin_count, bin_count), 3, colors=ADJ_RD_PALETTE, alpha=0.75)
        ax.scatter(
            adj_rd_transformed[:, i], adj_rd_transformed[:, j],
            c=ADJ_RD_PALETTE,
            s=1, edgecolors="none",
            alpha=0.2
        )

        x_data = np.concatenate([ctl_x, rd_x])
        ax.set_xlim(x_data.min(), x_data.max())

        y_data = np.concatenate([ctl_y, rd_y])
        ax.set_ylim(y_data.min(), y_data.max())

        # Label axes:
        ax.set_xlabel(f"$\\vartheta_{i + 1}$")
        ax.set_ylabel(f"$\\vartheta_{j + 1}$")


    def adj_comparative_histogram(ax, i):
        # ctl_kde = scipy.stats.gaussian_kde(ctl_transformed[:, i])
        # ctl_input = np.linspace(ctl_transformed[:, i].min(), ctl_transformed[:, i].max(), 50)
        # ax.plot(ctl_input, ctl_kde(ctl_input), color=CONTROL_PALETTE, alpha=0.75)

        # rd_kde = scipy.stats.gaussian_kde(rd_transformed[:, i])
        # rd_input = np.linspace(rd_transformed[:, i].min(), rd_transformed[:, i].max(), 50)
        # ax.plot(rd_input, rd_kde(rd_input), color=RD_PALETTE, alpha=0.75)
        bins = 30
        ax.hist(
            ctl_transformed[:, i], bins=bins,
            color=CONTROL_PALETTE, alpha=0.75,
            histtype="step", density=True, label="Control"
        )

        ax.hist(
            adj_rd_transformed[:, i], bins=bins,
            color=ADJ_RD_PALETTE, alpha=0.75,
            histtype="step", density=True, label="Adjusted RD"
        )

        ax.set_xlabel(f"$\\vartheta_{i + 1}$")
        ax.set_ylim(0, None)

        if i == 0:
            ax.legend(loc="upper left")


    def comparative_adjusted_kde_scattergrid():
        fig, axs = plt.subplots(3, 3, figsize=(FULL_WIDTH, FULL_WIDTH))
        for i in range(3):
            for j in range(3):
                if i == j:
                    adj_comparative_histogram(axs[i, j], i)
                else:
                    adj_comparative_kde_plot(axs[i, j], i, j, 50)

        # Adjust subplot configuration:
        clip = 0.05
        fig.subplots_adjust(clip, clip, 1 - clip, 1 - clip, wspace=0.25, hspace=0.25)

        # METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # plt.savefig(
        #     os.path.join(OUT_DIRPATH, "adj_kde_mle_comparative_plot.png"),
        #     dpi=300, metadata=METADATA_DICTIONARY, transparent=True
        # )

        plt.show()

    comparative_adjusted_kde_scattergrid()
    return


@app.cell
def _(ctl_transformed, final_transform, np, rd_transformed):
    mean_t2_diff = np.mean(ctl_transformed[:, 1] - rd_transformed[:, 1])
    log_cd_increase =  mean_t2_diff / final_transform[0, 1]
    cd_increase = np.exp(log_cd_increase)
    return (cd_increase,)


@app.cell
def _(cd_increase):
    cd_increase
    return


@app.cell
def _(cc, ge_sobol_inputs, np, plt, sobol_lengths):
    def plot_scatter_transform(i, j):
        fig, ax = plt.subplots(figsize=(3, 3))
        color_values = sobol_lengths
        print(color_values)
        color_sort = np.argsort(color_values)
        print()
        ax.scatter(
            np.log(ge_sobol_inputs[color_sort, i]), ge_sobol_inputs[color_sort, j],
            s=1, alpha=0.2, c=color_values[color_sort], cmap=cc.m_CET_L20,
            vmin=np.quantile(color_values, 0.02),
            vmax=np.quantile(color_values, 0.98),
            edgecolors="none"
        )
        ax.set_xlabel(f"$\\theta_{i+1}$")
        ax.set_ylabel(f"$\\theta_{j+1}$")
        plt.show()

    plot_scatter_transform(0, 2)
    return


@app.cell
def _(final_transform, np):
    mesh_inputs = np.load("model_experiments/2026-05-31-collisions_shape/mcmc_results/matrix_inference/meshgrid_inputs.npy")
    ge_mesh_inputs = np.log(mesh_inputs) @ final_transform

    mesh_predictions = np.load("model_experiments/2026-05-31-collisions_shape/mcmc_results/matrix_inference/meshgrid_predictions.npy")
    return ge_mesh_inputs, mesh_predictions


@app.cell
def _(ge_mesh_inputs, mesh_predictions, np):
    test_max = ge_mesh_inputs[np.argmax(mesh_predictions)]
    return


@app.cell
def _(ge_mesh_inputs, mesh_predictions, np, plt):
    color_vals = np.log(mesh_predictions - np.min(mesh_predictions) + 1)
    print(np.max(color_vals))
    color_sort = np.argsort(color_vals)
    plt.scatter(np.exp(ge_mesh_inputs[color_sort, 1]), np.exp(ge_mesh_inputs[color_sort, 2]), c=color_vals[color_sort], s=1)
    # plt.scatter(test_max[0], test_max[1])
    return


@app.cell
def _(ge_sobol_inputs, plt, sobol_order_parameters):
    plt.scatter(ge_sobol_inputs[:, 2], sobol_order_parameters, s=1)
    return


@app.cell
def _(ge_sobol_inputs, np, plt, sobol_order_parameters):
    color_order = np.argsort(ge_sobol_inputs[:, 2])
    plt.scatter(ge_sobol_inputs[color_order, 1], sobol_order_parameters[color_order], c=ge_sobol_inputs[color_order, 2], s=1)
    return


@app.cell
def _(ge_mesh_inputs):
    ge_mesh_inputs.shape
    return


@app.cell
def _(ge_mesh_inputs, mesh_predictions, np, plt):
    def plot_phase_mesh(i, j, bin_count=10):
        fig, ax = plt.subplots(figsize=(3, 3))

        # Get primary quantiles:
        pq_edges = np.quantile(ge_mesh_inputs[:, i], np.linspace(0.0, 1.0, bin_count+1))

        # Iterate through primary quantiles:
        mean_values = []
        for pq_index in range(bin_count):
            pq_lower_mask = ge_mesh_inputs[:, i] >= pq_edges[pq_index]
            pq_upper_mask = ge_mesh_inputs[:, i] < pq_edges[pq_index + 1]
            pq_mask = np.logical_and(pq_lower_mask, pq_upper_mask)

            # Get ancillary quantiles:
            aq_edges = np.quantile(ge_mesh_inputs[pq_mask, j], np.linspace(0, 1, bin_count+1))
            for aq_index in range(bin_count):
                aq_lower_mask = ge_mesh_inputs[pq_mask, j] >= aq_edges[aq_index]
                aq_upper_mask = ge_mesh_inputs[pq_mask, j] < aq_edges[aq_index + 1]
                aq_mask = np.logical_and(aq_lower_mask, aq_upper_mask)

                # Get mean in conditional bin:
                bin_mean = np.mean(mesh_predictions[pq_mask][aq_mask])
                mean_values.append(bin_mean)

        return np.array(mean_values).reshape(bin_count, bin_count)

    mean_values = plot_phase_mesh(0, 1, bin_count=25)
    return (mean_values,)


@app.cell
def _(mean_values, plt):
    plt.imshow(mean_values.T, origin="lower")
    return


@app.cell
def _():
    # def plot_cubed_mesh(bin_count=10):
    #     fig, ax = plt.subplots(figsize=(3, 3))

    #     # Get i quantiles:
    #     i_edges = np.quantile(ge_mesh_inputs[:, 0], np.linspace(0.0, 1.0, bin_count+1))

    #     mean_values = []
    #     for i_index in range(bin_count):
    #         i_lower_mask = ge_mesh_inputs[:, 0] >= i_edges[i_index]
    #         i_upper_mask = ge_mesh_inputs[:, 0] < i_edges[i_index + 1]
    #         i_mask = np.logical_and(i_lower_mask, i_upper_mask)

    #         # Get j quantiles:
    #         j_edges = np.quantile(ge_mesh_inputs[i_mask, 1], np.linspace(0, 1, bin_count+1))
    #         for j_index in range(bin_count):
    #             j_lower_mask = ge_mesh_inputs[i_mask, 1] >= j_edges[j_index]
    #             j_upper_mask = ge_mesh_inputs[i_mask, 1] < j_edges[j_index + 1]
    #             j_mask = np.logical_and(j_lower_mask, j_upper_mask)

    #             # Get k quantiles:
    #             k_edges = np.quantile(ge_mesh_inputs[i_mask][j_mask, 2], np.linspace(0, 1, bin_count+1))
    #             for k_index in range(bin_count):
    #                 k_lower_mask = ge_mesh_inputs[i_mask][j_mask, 2] >= k_edges[k_index]
    #                 k_upper_mask = ge_mesh_inputs[i_mask][j_mask, 2] < k_edges[k_index + 1]
    #                 k_mask = np.logical_and(k_lower_mask, k_upper_mask)

    #                 # Get mean in conditional bin:
    #                 bin_mean = np.mean(mesh_predictions[i_mask][j_mask][k_mask])
    #                 mean_values.append(bin_mean)

    #     return np.array(mean_values).reshape(bin_count, bin_count, bin_count)
    return


@app.cell
def _():
    # cubed_mesh = plot_cubed_mesh(bin_count=15)
    return


@app.cell
def _(ge_sobol_inputs, np, sobol_order_parameters):
    def get_tiled_mean(bin_counts):
        x_limits = np.quantile(ge_sobol_inputs[:, 1], [0.00, 1.0])
        x_bins = np.linspace(*x_limits, bin_counts + 1)

        y_limits = np.quantile(ge_sobol_inputs[:, 2], [0.00, 1.0])
        y_bins = np.linspace(*y_limits, bin_counts + 1)

        grid = []
        for i in range(bin_counts):
            x_lower = x_bins[i]
            x_upper = x_bins[i + 1]
            x_mask = np.logical_and(
                ge_sobol_inputs[:, 1] >= x_lower,
                ge_sobol_inputs[:, 1] <  x_upper
            )
            for j in range(bin_counts):
                y_lower = y_bins[j]
                y_upper = y_bins[j + 1]
                y_mask = np.logical_and(
                    ge_sobol_inputs[:, 2] >= y_lower,
                    ge_sobol_inputs[:, 2] <  y_upper
                )

                mean_value = np.nanmean(sobol_order_parameters[np.logical_and(x_mask, y_mask)])
                grid.append(mean_value)

        return np.array(grid).reshape(bin_counts, bin_counts)

    tiled_mean = get_tiled_mean(25)
    return (tiled_mean,)


@app.cell
def _(plt, tiled_mean):
    plt.imshow(tiled_mean.T, origin="lower")
    return


@app.cell
def _(cc, ge_sobol_inputs, np, plt, sobol_order_parameters):
    def scatter_3d():
        fig = plt.figure(layout="constrained")
        ax = fig.add_subplot(projection='3d')

        color_values = sobol_order_parameters
        color_sort = np.argsort(color_values)[::-1]
        ax.scatter(
            ge_sobol_inputs[color_sort, 1],
            ge_sobol_inputs[color_sort, 2],
            color_values[color_sort],
            s=5, c=color_values[color_sort], cmap=cc.m_CET_L20,
            # vmin=np.quantile(color_values, 0.01),
            # vmax=np.quantile(color_values, 0.99),
            edgecolor="none"
        )

        ax.set_xlabel(f"$\\theta_{2}$")
        ax.set_ylabel(f"$\\theta_{3}$")
        ax.set_zlabel(f"$\\theta_{1}$")

        ax.view_init(35, 265, 0)
        plt.show()

    scatter_3d()
    return


@app.cell
def _(
    ctl_transformed,
    ge_inputs,
    np,
    plt,
    rd_mle_estimates,
    rd_transformed,
    wt_mle_estimates,
):
    def plot_fits_transform(index_i, index_j):
        fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)

        # Plot control fit:
        axs[0].scatter(
            ge_inputs[:, index_i], ge_inputs[:, index_j],
            s=0.1, c='k', alpha=1, edgecolors='none'
        )

        wt_color_values = wt_mle_estimates
        wt_color_sort = np.argsort(wt_color_values)
        axs[0].scatter(
            ctl_transformed[wt_color_sort, index_i], ctl_transformed[wt_color_sort, index_j],
            s=1.0, c=wt_color_values[wt_color_sort], alpha=1, edgecolors='none',
            vmin=np.quantile(wt_color_values, 0.02),
            vmax=np.quantile(wt_color_values, 0.98)
        )

        # Plot RD fit:
        axs[1].scatter(
            ge_inputs[:, index_i], ge_inputs[:, index_j],
            s=0.1, c='k', alpha=1, edgecolors='none'
        )

        rd_color_values = rd_mle_estimates
        rd_color_sort = np.argsort(rd_color_values)
        axs[1].scatter(
            rd_transformed[rd_color_sort, index_i], rd_transformed[rd_color_sort, index_j],
            s=1.0, c=rd_color_values[rd_color_sort], alpha=0.5, edgecolors='none',
            vmin=np.quantile(wt_color_values, 0.02),
            vmax=np.quantile(wt_color_values, 0.98)
        )

        plt.show()

    plot_fits_transform(1, 2)
    return


@app.cell
def _():
    # def plot_eigenparameter_umap():
    #     fig, ax = plt.subplots(figsize=(3.5, 3.5))
    #     color_values = ge_inputs[:, 0]
    #     color_sort = np.argsort(color_values)
    #     ax.scatter(
    #         umap_embeddings[color_sort, 0], umap_embeddings[color_sort, 1],
    #         s=1, alpha=0.15, c=color_values[color_sort], cmap=cc.m_CET_L20
    #     )
    #     ax.set_aspect("equal")
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_xlabel("UMAP I")
    #     ax.set_ylabel("UMAP II")
    #     # ax.scatter(wt_embeddings[:, 0], wt_embeddings[:, 1], s=1, c=CONTROL_PALETTE, alpha=0.25)
    #     # ax.scatter(rd_embeddings[:, 0], rd_embeddings[:, 1], s=1, c=RD_PALETTE, alpha=0.25)
    #     plt.show()

    # plot_eigenparameter_umap()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
