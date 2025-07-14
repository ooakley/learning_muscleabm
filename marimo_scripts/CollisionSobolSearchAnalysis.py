import marimo

__generated_with = "0.12.10"
app = marimo.App(width="columns")


@app.cell
def _():
    GRIDSEARCH_PARAMETERS = {
        "cueDiffusionRate": [0.005, 0.5],
        "cueKa": [0.25, 1.25],
        "fluctuationAmplitude": [0.01, 0.2],
        "fluctuationTimescale": [1.5, 20],
        "actinAdvectionRate": [0.01, 1],
        "maximumSteadyStateActinFlow": [0.5, 5],
        # Collisions:
        "collisionAdvectionRate": [0.01, 1],
        "collisionFlowReductionRate": [0.01, 1],
        "numberOfCells": [50, 350],
    }

    DIMENSIONALITY = 9
    TRAINING_ITERATIONS = 50
    return DIMENSIONALITY, GRIDSEARCH_PARAMETERS, TRAINING_ITERATIONS


@app.cell
def _():
    import pysr
    import torch
    import gpytorch
    import os
    import copy
    import tqdm
    import cv2

    import matplotlib.pyplot as plt
    import matplotlib.path as plt_path
    import matplotlib.patches as plt_patches

    import marimo as mo
    import numpy as np
    import pandas as pd

    from scipy.stats import qmc
    from torch.utils.data import TensorDataset, DataLoader
    return (
        DataLoader,
        TensorDataset,
        copy,
        cv2,
        gpytorch,
        mo,
        np,
        os,
        pd,
        plt,
        plt_patches,
        plt_path,
        pysr,
        qmc,
        torch,
        tqdm,
    )


@app.cell
def _(np):
    def get_quantile_data(x, y, bin_number=20):
        # Get quantiles as bin edges:
        quantiles = np.linspace(0, 1, bin_number + 1)
        bin_edges = np.quantile(x, quantiles)

        # Get binned data across x values:
        bin_count, _ = np.histogram(x, bins=bin_edges, density=False)
        y_sum, _ = np.histogram(x, bins=bin_edges, weights=y, density=False)
        mean_y = y_sum / bin_count

        return bin_edges, mean_y
    return (get_quantile_data,)


@app.cell
def _(os):
    os.getcwd()
    return


@app.cell
def _(mo):
    mo.md(r"""# 1 - Investigating deterministic collisions""")
    return


@app.cell
def _(GRIDSEARCH_PARAMETERS, np, os):
    deterministic_directory_path = "./gridsearch_data/deterministic_collisions"

    det_collisions_array = np.load(
        os.path.join(deterministic_directory_path, "collated_collisions.npy")
    )
    det_speed_array = np.load(
        os.path.join(deterministic_directory_path, "collated_magnitudes.npy")
    )
    det_dtheta_array = np.load(
        os.path.join(deterministic_directory_path, "collated_dtheta.npy")
    )
    det_op_array = np.load(
        os.path.join(deterministic_directory_path, "collated_order_parameters.npy")
    )

    abcuniform_directory_path = "./gridsearch_data/out_abcUniformOP"
    abcuniform_op_array = np.load(
        os.path.join(abcuniform_directory_path, "collated_order_parameters.npy")
    )
    resampled_parameters = np.load("./abc_parameters.npy")

    # Constructing input dataset for gaussian process:
    measured_properties = \
        np.stack(
            [det_collisions_array[:, 0],
             det_speed_array[:, 0],
             det_dtheta_array[:, 0]],
            axis=1
        )

    # Get predicted data from trajectory-only search:
    pred_speed_array = np.load(
        os.path.join(deterministic_directory_path, "pred_speeds.npy"),
    )
    abc_speed_array = np.load(
        os.path.join(deterministic_directory_path, "abc_speeds.npy"),
    )
    pred_dtheta_array = np.load(
        os.path.join(deterministic_directory_path, "pred_dthetas.npy"),
    )
    abc_dtheta_array = np.load(
        os.path.join(deterministic_directory_path, "abc_dthetas.npy"),
    )

    # Get input parameters:
    parameter_array = np.load(
        os.path.join(deterministic_directory_path, "collated_inputs.npy")
    )

    # Normalise base parameters:
    normalised_parameters = np.zeros_like(parameter_array)
    for index, parameter_range in enumerate(GRIDSEARCH_PARAMETERS.values()):
        _minimum = parameter_range[0]
        _maximum = parameter_range[1]
        normalised_parameters[:, index] = \
            (parameter_array[:, index] - _minimum) / (_maximum - _minimum)

    cell_number = parameter_array[:, -1]


    # Normalise resampled parameters:
    normalised_abc_parameters = np.zeros_like(resampled_parameters)
    for index, parameter_range in enumerate(GRIDSEARCH_PARAMETERS.values()):
        _minimum = parameter_range[0]
        _maximum = parameter_range[1]
        normalised_abc_parameters[:, index] = \
            (resampled_parameters[:, index] - _minimum) / (_maximum - _minimum)

    # Normalising motion parameters into speed and persistence:
    reduced_parameter_inputs = np.concatenate(
        (
            np.expand_dims(pred_speed_array, axis=1),
            np.expand_dims(pred_dtheta_array, axis=1),
            parameter_array[:, -3:]
        ),
        axis=1
    )

    for _i in range(5):
        # Get minimum and maximum values of parameter in question:
        _minimum = np.min(reduced_parameter_inputs[:, _i])
        _maximum = np.max(reduced_parameter_inputs[:, _i])

        # Do basic minmax norm to [0, 1]:
        reduced_parameter_inputs[:, _i] = \
            (reduced_parameter_inputs[:, _i] - _minimum) / (_maximum - _minimum)
    return (
        abc_dtheta_array,
        abc_speed_array,
        abcuniform_directory_path,
        abcuniform_op_array,
        cell_number,
        det_collisions_array,
        det_dtheta_array,
        det_op_array,
        det_speed_array,
        deterministic_directory_path,
        index,
        measured_properties,
        normalised_abc_parameters,
        normalised_parameters,
        parameter_array,
        parameter_range,
        pred_dtheta_array,
        pred_speed_array,
        reduced_parameter_inputs,
        resampled_parameters,
    )


@app.cell
def _(normalised_abc_parameters, plt):
    plt.hist(normalised_abc_parameters[:, 8], bins=50);
    plt.show()
    return


@app.cell
def _(abcuniform_op_array, plt):
    plt.hist(abcuniform_op_array[:, 0], bins=200);
    plt.show()
    return


@app.cell
def _(det_op_array, plt):
    plt.hist(det_op_array[:, 0], bins=200);
    plt.show()
    return


@app.cell
def _(abcuniform_op_array, plt):
    plt.scatter(
        abcuniform_op_array[:, 0], abcuniform_op_array[:, 1],
        s=10, alpha=0.5, edgecolor='none'
    )
    return


@app.cell
def _(det_op_array):
    det_op_array[:, 0]
    return


@app.cell
def _(np):
    def uniform_resample(output_array, parameters, resample_number=100):
        # Randomly sample from order parameter interval:
        rng = np.random.default_rng(0)

        # Iterate through uniform samples:
        resampled_parameters = []
        while len(resampled_parameters) < resample_number:
            # Randomly sample from valid OP interval - (0, 1):
            sample = rng.uniform(0, 1)

            # Get similar points in parameter space:
            summary_distance = np.abs(output_array - sample)
            locality_mask = summary_distance < 0.05
            if np.count_nonzero(locality_mask) == 0:
                continue
            subset_parameters = parameters[locality_mask]
            resampled = [rng.choice(subset_parameters[:, col]) for col in range(9)]
            resampled_parameters.append(resampled)

        # Restructure as parameter matrix of shape (RESAMPLE_NUMBER, PARAMETER_DIMS):
        resampled_parameters = np.stack(resampled_parameters, axis=0)
        return resampled_parameters
    return (uniform_resample,)


@app.cell
def _(det_op_array, parameter_array, uniform_resample):
    resampled = uniform_resample(
        det_op_array[:, 0], parameter_array,
        resample_number=10000
    )
    return (resampled,)


@app.cell
def _():
    # np.save("./abc_parameters.npy", resampled)
    return


@app.cell
def _():
    # test_op_array = det_op_array[:, 0]

    # rng = np.random.default_rng(2)
    # sample = rng.uniform(0, 1)
    # print(sample)

    # summary_distance = np.abs(test_op_array - sample)
    # locality_mask = summary_distance < 0.05

    # print(np.count_nonzero(locality_mask))

    # subset_parameters = normalised_parameters[locality_mask]
    return


@app.cell
def _():
    # _fig, _ax = plt.subplots(layout='constrained')
    # _ax.imshow(subset_parameters.T)
    # _ax.set_xticks([])
    # _ax.set_yticks([])

    # plt.show()
    return


@app.cell
def _():
    # resampled = \
    #     [rng.choice(subset_parameters[:, col], 200) for col in range(9)]

    # resampled = np.stack(resampled, axis=0)

    # _fig, _ax = plt.subplots(layout='constrained')
    # _ax.imshow(resampled)
    # _ax.set_xticks([])
    # _ax.set_yticks([])

    # plt.show()
    return


@app.cell
def _(GRIDSEARCH_PARAMETERS, np, plt, plt_patches, plt_path):
    def plot_pcp(parameter_data, out_data, filter_point=0, epsilon=0.05):
        # Useful numbers:
        parameter_number = parameter_data.shape[1]

        # Instantiate matplotlib object:
        fig, host = plt.subplots(figsize=(20, 2.5))

        # Find ranges for the parameter data:
        parameter_minimums = parameter_data.min(axis=0)
        parameter_maximums = parameter_data.max(axis=0)
        parameter_ranges = parameter_maximums - parameter_minimums
        padding_values = parameter_ranges * 0.05
        axis_ranges = (
            parameter_minimums - padding_values, parameter_maximums + padding_values
        )

        # Generate matrix where all values are scaled:
        normed_parameters = (parameter_data - parameter_minimums) / parameter_ranges
        normed_out = (out_data - out_data.min()) / (out_data.max() - out_data.min())
        normed_out = np.expand_dims(normed_out, axis=1)
        full_data = np.concatenate([normed_parameters, normed_out], axis=1)

        # Format dependent axes:
        dependent_axes = [host.twinx() for _ in range(parameter_number)]
        for index, ax in enumerate(dependent_axes):
            ax.set_ylim(axis_ranges[0][index], axis_ranges[1][index])
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", index / (parameter_number)))

        # Format host y axis:
        # host.set_ylim(0, 1)
        host.yaxis.tick_right()
        host.spines['top'].set_visible(False)
        host.spines['bottom'].set_visible(False)
        host.spines['left'].set_visible(False)

        # Format full figure x-axis:
        host.set_xlim(0, parameter_number)
        host.set_xticks(range(parameter_number + 1))
        host.set_xticklabels(
            list(GRIDSEARCH_PARAMETERS.keys()) + ["Speed"] + ["dTheta"] + ["OP"],
            fontsize=10
        )
        host.tick_params(axis='x', which='major', pad=7, rotation=0)
        host.xaxis.tick_bottom()
        # host.set_title("All Inputs", fontsize=12)

        # Plot Bezier curves between points on index:
        # for curve_index in range(parameter_data.shape[0]):

        # Get similar points in parameter space:
        sample_size = int(len(out_data) * epsilon)
        summary_distances = np.abs(out_data - filter_point)**2
        selection_indices = np.argsort(summary_distances)[:sample_size]
        for curve_index in selection_indices:
            """
            To create Bezier curves:
            For each axis, there will be a control vertex at the point itself, one at
            1/3rd towards the previous and one at one third towards the next axis.
            The first and last axis have one less control vertex x-coordinate of the
            control vertices.
            At each integer (for the axes) and two inbetween y-coordinates,
            repeat every point three times, except the first and last only twice.
            """
            control_vertices = \
                list(zip(
                    [x for x in np.linspace(
                        0, len(full_data) - 1, len(full_data) * 3 - 2, endpoint=True
                    )],
                    np.repeat(full_data[curve_index, :], 3)[1:-1]
            ))
            # for x,y in verts: host.plot(x, y, 'go')
            # to show the control points of the beziers
            codes = \
                [plt_path.Path.MOVETO] \
                + [plt_path.Path.CURVE4 for _ in range(len(control_vertices) - 1)]
            path = plt_path.Path(control_vertices, codes)
            # colors[category[curve_index] - 1]
            patch = plt_patches.PathPatch(
                path, facecolor='none', lw=1, edgecolor='k', alpha=0.05
            )
            # legend_handles[category[curve_index] - 1] = patch
            host.add_patch(patch)

            # if category_names is not None:
            #     host.legend(legend_handles, category_names,
            #                 loc='lower center', bbox_to_anchor=(0.5, -0.18),
            #                 ncol=len(category_names), fancybox=True, shadow=True)

        plt.tight_layout()
        # plt.show()

        return fig
    return (plot_pcp,)


@app.cell
def _(
    abc_dtheta_array,
    abc_speed_array,
    abcuniform_op_array,
    np,
    plot_pcp,
    plt,
    resampled,
):
    resampled_plus = np.concatenate(
        [
            resampled,
            np.expand_dims(abc_speed_array, axis=1),
            np.expand_dims(1/abc_dtheta_array, axis=1)
        ], axis=1
    )
    _ = plot_pcp(resampled_plus, abcuniform_op_array[:, 0], filter_point=0.5, epsilon=0.01)
    plt.show()
    return (resampled_plus,)


@app.cell
def _():
    return


@app.cell
def _(abcuniform_op_array, cv2, np, plot_pcp, plt, resampled_plus):
    size = 250, 2000
    fps = 15
    out = cv2.VideoWriter(
        './pcp_scan.mp4', cv2.VideoWriter_fourcc(*'avc1'),
        fps, (size[1], size[0]), True
    )

    for _index, filter_point in enumerate(np.linspace(0, 1, 50)):
        print(f"Generating plot number {_index + 1}...")
        # Plot filtered PCP:
        _fig = plot_pcp(
            resampled_plus, abcuniform_op_array[:, 0],
            filter_point=filter_point, epsilon=0.01
        )

        # Export image to array:
        _fig.canvas.draw()
        array_plot = np.array(_fig.canvas.renderer.buffer_rgba())
        plt.close(_fig)

        # Save array plot to opencv file:
        bgr_data = cv2.cvtColor(array_plot, cv2.COLOR_RGB2BGR)
        out.write(bgr_data)

    out.release()
    return array_plot, bgr_data, filter_point, fps, out, size


@app.cell
def _(GRIDSEARCH_PARAMETERS, abcuniform_op_array, cv2, np, pd, plt, resampled):
    _size = 2000, 2000
    _fps = 10
    _out = cv2.VideoWriter(
        './scatter_matrix_scan.mp4', cv2.VideoWriter_fourcc(*'avc1'),
        _fps, (_size[1], _size[0]), True
    )

    for _index, _filter_point in enumerate(np.linspace(0, 1, 50)):
        print(f"Generating plot number {_index + 1}...")
        # Get similar points in parameter space:
        sample_size = int(len(abcuniform_op_array[:, 0]) * 0.01)
        summary_distances = np.abs(abcuniform_op_array[:, 0] - _filter_point)**2
        selection_indices = np.argsort(summary_distances)[:sample_size]

        # Plot filtered scatter matrix
        _fig, _ax = plt.subplots(figsize=(20, 20))

        df = pd.DataFrame(
            resampled[selection_indices, :], columns=list(GRIDSEARCH_PARAMETERS.keys())
        )
        pd.plotting.scatter_matrix(df, alpha=1, ax=_ax, range_padding=0.2);

        # Export image to array:
        _fig.canvas.draw()
        _array_plot = np.array(_fig.canvas.renderer.buffer_rgba())
        plt.close(_fig)

        # Save array plot to opencv file:
        _bgr_data = cv2.cvtColor(_array_plot, cv2.COLOR_RGB2BGR)
        _out.write(_bgr_data)

    _out.release()
    return df, sample_size, selection_indices, summary_distances


@app.cell
def _(det_op_array, plt):
    # Investigating heteroscedasticity:
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # Plot the actual scatter plot:
    _ax.scatter(
        det_op_array[:, 0], det_op_array[:, 1],
        s=10, alpha=0.5, edgecolors='none'
    )

    # Format:
    _ax.set_xlim(0, 1);
    _ax.set_ylim(0, 0.3);
    _ax.set_xlabel("Order Parameter - Mean")
    _ax.set_ylabel("Order Parameter - Standard Deviations")
    # _fig.suptitle("Speed-Persistence with Stochastic Collisions")

    plt.show()
    return


@app.cell
def _(copy, det_op_array, np, qmc):
    def generate_ld_batch(seed):
        # Initialise random number generator:
        rng = np.random.default_rng(seed)

        # Low discrepancy output sampling:
        modifiable_op_array = copy.deepcopy(det_op_array[:, 0])

        # Create shuffled sequence of indices that we modify as they are sampled:
        index_array = np.arange(0, 2**14)
        rng.shuffle(index_array)

        # Initialise with first two samples from array:
        low_discrepancy_op = []
        low_discrepancy_op.extend(modifiable_op_array[index_array[0:2]])
        low_discrepancy_op = np.expand_dims(low_discrepancy_op, 1)

        sampled_indices = [index_array[0], index_array[1]]
        index_array = np.delete(index_array, [0, 1])

        discrepancies = []
        for _ in range(1000 - 2):
            # Calaculating initial discrepancy:
            base_discrepancy = qmc.discrepancy(low_discrepancy_op, iterative=True)
            rng.shuffle(index_array)
            discrepancies.append(base_discrepancy)

            # Iterating through candidate discrepancies:
            for _index in range(500):
                # Calculating effect of proposed data point:
                sampled_op = modifiable_op_array[index_array[_index]]
                new_discrepancy = \
                    qmc.update_discrepancy(
                        [sampled_op], low_discrepancy_op, base_discrepancy
                    )

                # If low discrepancy, add to update:
                #  or new_discrepancy < 0.01
                if new_discrepancy < base_discrepancy:
                    # Add to low disc distribution:
                    sampled_op = np.expand_dims([sampled_op], 1)
                    low_discrepancy_op = np.append(low_discrepancy_op, sampled_op, axis=0)

                    # Update index lists:
                    sampled_indices.append(index_array[_index])
                    index_array = np.delete(index_array, _index)
                    break

        return sampled_indices
    return (generate_ld_batch,)


@app.cell
def _(generate_ld_batch):
    ld_dataset = generate_ld_batch(0)
    return (ld_dataset,)


@app.cell
def _(ld_dataset):
    sampled_indices = ld_dataset
    return (sampled_indices,)


@app.cell
def _(ld_dataset, np):
    print(len(ld_dataset))
    print(len(np.unique(ld_dataset)))
    return


@app.cell
def _(det_op_array, ld_dataset, plt):
    plt.hist(det_op_array[ld_dataset, 0]);
    plt.show()
    return


@app.cell
def _(det_op_array, measured_properties, parameter_array, sampled_indices):
    low_discrepancy_parameters = parameter_array[sampled_indices, :]
    low_discrepancy_properties = measured_properties[sampled_indices, :]
    low_discrepancy_outputs = det_op_array[sampled_indices, 0]
    return (
        low_discrepancy_outputs,
        low_discrepancy_parameters,
        low_discrepancy_properties,
    )


@app.cell
def _(low_discrepancy_parameters):
    low_discrepancy_parameters.shape
    return


@app.cell
def _(low_discrepancy_outputs, low_discrepancy_parameters, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # Plot the actual scatter plot:
    _ax.scatter(
        low_discrepancy_parameters[:, 5], low_discrepancy_outputs,
        s=10, alpha=0.5, edgecolors='none'
    )

    # Format:
    # _ax.set_xlim(0, 3.15);
    # _ax.set_ylim(0, 1.0);
    _ax.set_xlabel("Average dTheta")
    _ax.set_ylabel("Order Parameter")
    # _fig.suptitle("Speed-Persistence with Stochastic Collisions")

    plt.show()
    return


@app.cell
def _(low_discrepancy_outputs, low_discrepancy_parameters, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # Plot the actual scatter plot:
    _ax.scatter(
        low_discrepancy_parameters[:, 1], low_discrepancy_outputs,
        s=10, alpha=0.5, edgecolors='none'
    )
    # _ax.scatter(
    #     pred_speed_array[det_stationary_op_mean > 0.6],
    #     det_stationary_speed_mean[det_stationary_op_mean > 0.6],
    #     s=10, alpha=0.5, edgecolors='none', c='r'
    # )

    # Format:
    # _ax.set_xlim(0, 5.5);
    # _ax.set_ylim(0, 5.5);
    _ax.set_xlabel("Predicted Speed")
    _ax.set_ylabel("Collisional Speed")
    # _fig.suptitle("Speed-Persistence with Stochastic Collisions")

    plt.show()
    return


@app.cell
def _(DIMENSIONALITY, gpytorch, np):
    # We will use the simplest form of GP model, exact inference
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            # Initialising from base class:
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

            # Defining scale parameter prior:
            mu_0 = 0.0
            sigma_0 = 15.0
            lognormal_prior = gpytorch.priors.LogNormalPrior(
                mu_0 + np.log(DIMENSIONALITY) / 2, sigma_0
            )

            # Defining mean and covariance functions:
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=DIMENSIONALITY,
                    lengthscale_prior=lognormal_prior
                )
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    class ApproximateGPModel(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points, dimensions=DIMENSIONALITY):
            # Set up distribution:
            variational_distribution = \
                gpytorch.variational.CholeskyVariationalDistribution(
                    inducing_points.size(0)
            )

            # Set up variational strategy:
            variational_strategy = \
                gpytorch.variational.VariationalStrategy(
                    self, inducing_points, variational_distribution,
                    learn_inducing_locations=True
            )

            # Inherit rest of init logic from approximate gp:
            super(ApproximateGPModel, self).__init__(variational_strategy)

            # Defining scale parameter prior:
            mu_0 = 0.5
            sigma_0 = 15.0
            lognormal_prior = gpytorch.priors.LogNormalPrior(
                mu_0 + np.log(dimensions) / 2, sigma_0
            )

            # Define mean and convariance functions:
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=dimensions,
                    lengthscale_prior=lognormal_prior
                )
            ) + gpytorch.kernels.ConstantKernel()

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    return ApproximateGPModel, ExactGPModel


@app.cell
def _(
    DataLoader,
    TensorDataset,
    abcuniform_op_array,
    measured_properties,
    normalised_abc_parameters,
    torch,
):
    # Oversampled dataset for inducing points:
    inducing_points = torch.tensor(
        normalised_abc_parameters[:500, :], dtype=torch.float32
    )

    train_parameters = torch.tensor(normalised_abc_parameters[:9000], dtype=torch.float32)
    train_props = torch.tensor(measured_properties, dtype=torch.float32)
    train_op = torch.tensor(abcuniform_op_array[:9000, 0], dtype=torch.float32)

    train_dataset = TensorDataset(train_parameters, train_op)
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)

    test_parameters = torch.tensor(normalised_abc_parameters[9000:], dtype=torch.float32)
    test_props = torch.tensor(measured_properties[1::2], dtype=torch.float32)
    test_op = torch.tensor(abcuniform_op_array[9000:, 0], dtype=torch.float32)
    return (
        inducing_points,
        test_op,
        test_parameters,
        test_props,
        train_dataset,
        train_loader,
        train_op,
        train_parameters,
        train_props,
    )


@app.cell
def _(ApproximateGPModel, DataLoader, TensorDataset, gpytorch, torch):
    def instantiate_model(inducing_points, dimensions):
        # Convert inducing points to torch:
        inducing_points = torch.tensor(
            inducing_points, dtype=torch.float32
        )

        # Set up likelihoods:
        likelihood = gpytorch.likelihoods.BetaLikelihood()
        model = ApproximateGPModel(inducing_points, dimensions=dimensions)
        return model, likelihood

    def train_model(model, likelihood, x_dataset, y_dataset, epochs=50):
        # Convert datasets to torch:
        train_x = torch.tensor(x_dataset, dtype=torch.float32)
        train_y = torch.tensor(y_dataset, dtype=torch.float32)

        # Initialise dataloaders:
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)

        # Set up training process:
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.02)

        # Set up loss:
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
        loss_history = []
        for i in range(epochs):
            # Run through entire dataset:
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()

            # Print progress:
            if (i + 1) % 25 == 0:
                print(i + 1)
                print(loss.detach())

            loss_history.append(loss.detach())

        return loss_history
    return instantiate_model, train_model


@app.cell
def _(instantiate_model, normalised_abc_parameters):
    # Initialise GP model that regresses on all nine model parameters:
    full_param_model, full_param_likelihood = instantiate_model(
        normalised_abc_parameters[:500, :], 9
    )
    return full_param_likelihood, full_param_model


@app.cell
def _(
    abcuniform_op_array,
    full_param_likelihood,
    full_param_model,
    normalised_abc_parameters,
    train_model,
):
    # Train this model:
    full_loss_history = train_model(
        full_param_model, full_param_likelihood,
        normalised_abc_parameters, abcuniform_op_array[:, 0], epochs=250
    )
    return (full_loss_history,)


@app.cell
def _(instantiate_model, reduced_parameter_inputs, sampled_indices):
    # Initialise GP model that regresses on speed value, persistence value, and collision parameters:
    reduced_param_model, reduced_param_likelihood = instantiate_model(
        reduced_parameter_inputs[sampled_indices, :], 5
    )
    return reduced_param_likelihood, reduced_param_model


@app.cell
def _(
    det_op_array,
    reduced_param_likelihood,
    reduced_param_model,
    reduced_parameter_inputs,
    train_model,
):
    # Train this model:
    reduced_loss_history = train_model(
        reduced_param_model, reduced_param_likelihood,
        reduced_parameter_inputs, det_op_array[:, 0], epochs=1000
    )
    return (reduced_loss_history,)


@app.cell
def _(full_loss_history, plt, reduced_loss_history):
    _fig, _ax = plt.subplots(figsize=(4.75, 4.25))
    _ax.plot(full_loss_history)
    _ax.plot(reduced_loss_history)
    plt.show()
    return


@app.cell
def _():
    _GRIDSEARCH_PARAMETERS = {
        "cueDiffusionRate": [0.005, 0.5],
        "cueKa": [0.25, 1.25],
        "fluctuationAmplitude": [0.01, 0.2],
        "fluctuationTimescale": [1.5, 20],
        "actinAdvectionRate": [0.01, 1],
        "maximumSteadyStateActinFlow": [0.5, 5],
        # Collisions:
        "collisionAdvectionRate": [0.01, 1],
        "collisionFlowReductionRate": [0.01, 1],
        "numberOfCells": [50, 350],
    }

    # _IGNORABLE_PARAMETERS = [
    #     1, 3, 8
    # ]
    return


@app.cell
def _(full_param_model):
    full_param_model.covar_module.kernels[0].base_kernel.lengthscale.detach().numpy()[0]
    return


@app.cell
def _(full_param_likelihood, full_param_model, test_parameters, torch):
    full_param_model.eval()
    full_param_likelihood.eval()

    with torch.no_grad():
        predictions = full_param_likelihood(full_param_model(test_parameters))
    return (predictions,)


@app.cell
def _(np, predictions):
    mean_predictions = np.mean(predictions.mean.detach().numpy(), axis=0)
    std_predictions = np.std(predictions.mean.detach().numpy(), axis=0)
    return mean_predictions, std_predictions


@app.cell
def _(np, plt):
    def plot_op_predictions(ground_truth, predictions):
        # Plotting AUC model fit:
        left_lim = -0.05
        right_lim = 1.05
        fig, ax = plt.subplots(figsize=(4.5, 4.5))

        # Plot data points:
        ax.scatter(
            ground_truth, predictions,
            s=10, edgecolors='none', alpha=0.25
        )

        # Plot line of equality:
        ax.plot(
            np.linspace(left_lim, right_lim, 2),
            np.linspace(left_lim, right_lim, 2),
            c='k', linestyle='dashed'
        )

        # Set labels:
        ax.set_xlim(left_lim, right_lim)
        ax.set_xlabel("Ground Truth Order Parameter")
        ax.set_ylim(left_lim, right_lim)
        ax.set_ylabel("Predicted Order Parameter")

        plt.show()
    return (plot_op_predictions,)


@app.cell
def _(mean_predictions, plot_op_predictions, test_op):
    plot_op_predictions(test_op.detach(), mean_predictions)
    return


@app.cell
def _(deterministic_directory_path, mean_predictions, np, os):
    np.save(
        os.path.join(deterministic_directory_path, "smoothed_op.npy"),
        mean_predictions,
    )
    return


@app.cell
def _(
    np,
    reduced_param_likelihood,
    reduced_param_model,
    reduced_parameter_inputs,
    torch,
):
    reduced_param_model.eval()
    reduced_param_likelihood.eval()

    test_red = torch.tensor(reduced_parameter_inputs, dtype=torch.float32)

    with torch.no_grad():
        red_predictions = \
            reduced_param_likelihood(reduced_param_model(test_red))

    red_mean_predictions = np.mean(red_predictions.mean.detach().numpy(), axis=0)
    return red_mean_predictions, red_predictions, test_red


@app.cell
def _(plot_op_predictions, red_mean_predictions, test_op):
    plot_op_predictions(test_op.detach(), red_mean_predictions)
    return


@app.cell
def _(pysr):
    sr_model = pysr.PySRRegressor()
    sr_model = sr_model.from_file(
        run_directory="./sr_outputs/20250701_SR_abcOP_DeterministicCollisions"
    )
    return (sr_model,)


@app.cell
def _(sr_model):
    sympy_representation = sr_model.sympy()
    sympy_representation
    return (sympy_representation,)


@app.cell
def _(sr_model):
    sr_model.equations_
    return


@app.cell
def _():
    # sympy_representation.simplify()
    return


@app.cell
def _(resampled_parameters, sr_model):
    sr_predictions = sr_model.predict(resampled_parameters)
    return (sr_predictions,)


@app.cell
def _(abcuniform_op_array, np, plt, sr_predictions):
    # Plotting AUC model fit:
    _left_lim = -0.1
    _right_lim = 1.1
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5), layout='constrained')
    _ax.scatter(
        abcuniform_op_array[:, 0], sr_predictions,
        s=10, edgecolors='none', alpha=0.5
    )
    _ax.plot(
        np.linspace(_left_lim, _right_lim, 2),
        np.linspace(_left_lim, _right_lim, 2),
        c='k', linestyle='dashed'
    )
    _ax.set_xlim(_left_lim, _right_lim)
    _ax.set_xlabel("Simulation Outputs - Order Parameter")
    _ax.set_ylim(_left_lim, _right_lim)
    _ax.set_ylabel("Symbolic Regression Predictions")
    # ax.scatter(train_x, train_y, s=5)
    return


@app.cell
def _(det_op_array, np, plt, sr_predictions):
    # Plotting AUC model fit:
    _left_lim = 0
    _right_lim = 1
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5), layout='constrained')
    _ax.scatter(
        det_op_array[:, 0], sr_predictions,
        s=10, edgecolors='none', alpha=0.5
    )
    _ax.plot(
        np.linspace(_left_lim, _right_lim, 2),
        np.linspace(_left_lim, _right_lim, 2),
        c='k', linestyle='dashed'
    )
    _ax.set_xlim(_left_lim, _right_lim)
    _ax.set_xlabel("Simulation Outputs - Order Parameter")
    _ax.set_ylim(_left_lim, _right_lim)
    _ax.set_ylabel("Symbolic Regression Predictions")
    # ax.scatter(train_x, train_y, s=5)
    return


@app.cell
def _(mean_predictions, normalised_parameters, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # Plot the actual scatter plot:
    _ax.scatter(
        normalised_parameters[:, -1], mean_predictions,
        s=10, alpha=0.5, edgecolors='none'
    )

    # Format:
    # _ax.set_xlim(0, 3.15);
    # _ax.set_ylim(0, 1.0);
    _ax.set_xlabel("Average dTheta")
    _ax.set_ylabel("Order Parameter")
    # _fig.suptitle("Speed-Persistence with Stochastic Collisions")

    plt.show()
    return


@app.cell
def _(det_op_array, normalised_parameters, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # Plot the actual scatter plot:
    _ax.scatter(
        normalised_parameters[:, -1], det_op_array[:, 0],
        s=10, alpha=0.5, edgecolors='none'
    )

    # Format:
    # _ax.set_xlim(0, 3.15);
    # _ax.set_ylim(0, 1.0);
    _ax.set_xlabel("Average dTheta")
    _ax.set_ylabel("Order Parameter")
    # _fig.suptitle("Speed-Persistence with Stochastic Collisions")

    plt.show()
    return


@app.cell
def _(model, torch):
    torch.save(model.state_dict(), './test_model_state.pth')
    return


app._unparsable_cell(
    r"""
    np.save(\"./smoothed_out.npy\", arr, allow_pickle=True, fix_imports=<no value>
    """,
    name="_"
)


@app.cell
def _(
    DataLoader,
    TensorDataset,
    low_discrepancy_parameters,
    normalised_parameters,
    np,
    torch,
    train_op,
):
    # Get rid of irrelevant parameters:
    reduced_ld_params = np.delete(low_discrepancy_parameters, [1, 3, 8], 1)
    reduced_norm_params = np.delete(normalised_parameters, [1, 3, 8], 1)

    red_inducing_points = torch.tensor(reduced_ld_params, dtype=torch.float32)

    red_train_parameters = torch.tensor(reduced_norm_params[::2, :], dtype=torch.float32)
    red_train_dataset = TensorDataset(red_train_parameters, train_op)
    red_train_loader = DataLoader(red_train_dataset, batch_size=1000, shuffle=True)

    red_test_parameters = torch.tensor(reduced_norm_params[1::4], dtype=torch.float32)

    # likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
    #     noise=fixed_noise,
    #     learn_additional_noise=True
    # )
    return (
        red_inducing_points,
        red_test_parameters,
        red_train_dataset,
        red_train_loader,
        red_train_parameters,
        reduced_ld_params,
        reduced_norm_params,
    )


@app.cell
def _(red_train_parameters, reduced_norm_params):
    print(reduced_norm_params.shape)
    print(red_train_parameters.size())
    return


@app.cell
def _(ApproximateGPModel, gpytorch, red_inducing_points):
    red_likelihood = gpytorch.likelihoods.BetaLikelihood()
    red_model = ApproximateGPModel(red_inducing_points, dimensions=6)
    return red_likelihood, red_model


@app.cell
def _(gpytorch, red_likelihood, red_model, red_train_loader, torch, train_op):
    red_model.train()
    red_likelihood.train()

    red_optimizer = torch.optim.Adam(
        [{'params': red_model.parameters()}, {'params': red_likelihood.parameters()},],
        lr=0.01
    )

    # Our loss object. We're using the VariationalELBO
    red_mll = gpytorch.mlls.VariationalELBO(
        red_likelihood, red_model,
        num_data=train_op.size(0)
    )

    for _i in range(500):
        # Within each iteration, we will go over each minibatch of data
        for _x_batch, _y_batch in red_train_loader:
            red_optimizer.zero_grad()
            _output = red_model(_x_batch)
            _loss = -red_mll(_output, _y_batch)
            _loss.backward()
            red_optimizer.step()

        if (_i + 1) % 25 == 0:
            print(_i + 1)
            print(_loss.detach())
    return red_mll, red_optimizer


@app.cell
def _(model):
    model.covar_module.kernels[0].base_kernel.lengthscale.detach().numpy()[0]
    return


@app.cell
def _(np, red_likelihood, red_model, red_test_parameters, torch):
    red_model.eval()
    red_likelihood.eval()

    with torch.no_grad():
        red_predicted_op = red_likelihood(red_model(red_test_parameters)).mean.numpy()

    red_predicted_op = np.mean(red_predicted_op, axis=0)

    print(red_predicted_op.shape)
    return (red_predicted_op,)


@app.cell
def _(red_predicted_op):
    red_predicted_op
    return


@app.cell
def _(np, plt, red_predicted_op, test_op):
    # Plotting AUC model fit:
    _left_lim = 0
    _right_lim = 1
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    _ax.scatter(
        test_op.detach(), red_predicted_op,
        s=10, edgecolors='none', alpha=0.5
    )

    # _ax.errorbar(
    #     test_op.detach(), predicted_op,
    #     yerr=det_op_array[:, 1][1::8],
    #     fmt='none'
    # )

    _ax.plot(
        np.linspace(_left_lim, _right_lim, 2),
        np.linspace(_left_lim, _right_lim, 2),
        c='k', linestyle='dashed'
    )

    _ax.set_xlim(_left_lim, _right_lim)
    _ax.set_xlabel("Simulation Outputs - Order Parameter")
    _ax.set_ylim(_left_lim, _right_lim)
    _ax.set_ylabel("Predicted Outputs - Order Parameter")
    # ax.scatter(train_x, train_y, s=5)
    return


@app.cell
def _(det_op_array, np, plt, sr_predictions):
    # Plotting AUC model fit:
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))
    _ax.scatter(det_op_array[:, 0], sr_predictions, s=2.5, alpha=0.5)

    # Plot line of equality:
    _left_lim = 0
    _right_lim = 1
    _ax.plot(
        np.linspace(_left_lim, _right_lim, 2),
        np.linspace(_left_lim, _right_lim, 2),
        c='k', linestyle='dashed'
    )

    # Determine x limits:
    _ax.set_xlim(_left_lim, _right_lim)
    _ax.set_xlabel("Simulation Outputs - Order Parameter")
    _ax.set_ylim(_left_lim, _right_lim)
    _ax.set_ylabel("SR Predicted Outputs - Order Parameter")
    # ax.scatter(train_x, train_y, s=5)
    return


@app.cell
def _(cell_number, det_stationary_col_mean, det_stationary_op_mean, m, plt):
    m
    w_fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    _op_mask = det_stationary_op_mean > 0.5

    # Plot the actual scatter plot:
    _ax.vlines(216, 0, 3.5, color='k', alpha=0.5, linestyle='dotted')
    _ax.scatter(
        cell_number, det_stationary_col_mean,
        s=10, alpha=0.25, edgecolors='none'
    )
    _ax.scatter(
        cell_number[_op_mask], det_stationary_col_mean[_op_mask],
        s=10, alpha=1, edgecolors='none', color='r'
    )

    # Format:
    _ax.set_xlim(50, 350);
    _ax.set_ylim(0, 3.5);
    _ax.set_xlabel("Cell Number")
    _ax.set_ylabel("Rate of Collisions")
    # _fig.suptitle("Speed-Persistence with Stochastic Collisions")

    plt.show()
    return (w_fig,)


@app.cell
def _(cell_number, det_stationary_op_mean, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    _op_mask = det_stationary_op_mean > 0.8

    # Plot the actual scatter plot:
    _ax.scatter(
        cell_number, det_stationary_op_mean,
        s=10, alpha=0.25, edgecolors='none'
    )


    # Format:
    _ax.set_xlim(50, 350);
    _ax.set_ylim(0, 1.0);
    _ax.set_xlabel("Cell Number")
    _ax.set_ylabel("Rate of Collisions")
    # _fig.suptitle("Speed-Persistence with Stochastic Collisions")

    plt.show()
    return


@app.cell
def _(cell_number, det_stationary_col_mean):
    collision_potential = det_stationary_col_mean / cell_number
    return (collision_potential,)


@app.cell
def _(cell_number, collision_potential, det_stationary_op_mean, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    _op_mask = det_stationary_op_mean > 0.8

    # Plot the actual scatter plot:
    _ax.scatter(
        cell_number, collision_potential,
        s=10, alpha=0.25, edgecolors='none'
    )
    _ax.scatter(
        cell_number[_op_mask], collision_potential[_op_mask],
        s=10, alpha=0.5, edgecolors='none', color='r'
    )

    # Format:
    # _ax.set_xlim(50, 350);
    # _ax.set_ylim(0, 3.5);
    _ax.set_xlabel("Cell Number")
    _ax.set_ylabel("Rate of Collisions")
    # _fig.suptitle("Speed-Persistence with Stochastic Collisions")

    plt.show()
    return


@app.cell
def _(collision_potential, det_stationary_op_mean, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # Plot the actual scatter plot:
    _ax.scatter(
        collision_potential, det_stationary_op_mean,
        s=10, alpha=0.5, edgecolors='none'
    )

    # Format:
    # _ax.set_xlim(0, 3.15);
    # _ax.set_ylim(0, 1.0);
    _ax.set_xlabel("Collision Potential")
    _ax.set_ylabel("Averaged Order Parameter")
    # _fig.suptitle("Speed-Persistence with Stochastic Collisions")

    plt.show()
    return


@app.cell
def _(det_stationary_col_mean, det_stationary_op_mean):
    collision_utility = det_stationary_op_mean / (det_stationary_col_mean + 1)
    return (collision_utility,)


@app.cell
def _(collision_utility, det_stationary_speed_mean, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # Plot the actual scatter plot:
    _ax.scatter(
        det_stationary_speed_mean, collision_utility,
        s=10, alpha=0.5, edgecolors='none'
    )

    # Format:
    _ax.set_xlim(0, 5);
    # _ax.set_ylim(0, 35.0);
    _ax.set_xlabel("Average Speed")
    _ax.set_ylabel("Average Collision Utility")
    # _fig.suptitle("Speed-Persistence with Stochastic Collisions")

    plt.show()
    return


@app.cell
def _(collision_utility, np):
    np.argwhere(collision_utility > 0.7)
    return


@app.cell
def _(det_op_array):
    det_op_array.shape
    return


@app.cell
def _(det_op_array, plt):
    _fig, _ax = plt.subplots(figsize=(5.5, 4.5))
    # _ax.plot(op_array.mean(axis=0), label='Stochastic Collisions')
    _ax.plot(det_op_array.mean(axis=0), label='Deteministic Collisions')
    _ax.set_xlim(0, 1440)
    _ax.set_ylabel("Velocity Order Parameter")

    _ax.legend()
    plt.show()
    return


@app.cell
def _(det_op_array, np, op_array):
    final_timepoints_mean = np.mean(op_array[:, 1000:], axis=1)
    order_parameters = final_timepoints_mean.reshape((4096, 10)).mean(axis=1)

    det_ft_mean = np.mean(det_op_array[:, 1000:], axis=1)
    det_order_parameters = det_ft_mean.reshape((4096, 10)).mean(axis=1)
    return (
        det_ft_mean,
        det_order_parameters,
        final_timepoints_mean,
        order_parameters,
    )


@app.cell
def _(det_order_parameters, order_parameters):
    op_diff = order_parameters - det_order_parameters
    return (op_diff,)


@app.cell
def _(op_diff, plt):
    plt.hist(op_diff, bins=100);
    plt.show()
    return


@app.cell
def _(op_diff, order_parameters, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # # Calculate bin edges:
    # _speed_edges, _mean_op = get_quantile_data(speed_array, order_parameters)

    # Plot the actual scatter plot:
    _ax.hlines(0, 0, 1, color='r', alpha=0.75)
    _ax.hlines(0.1, 0, 1, color='r', alpha=0.25)
    _ax.hlines(-0.1, 0, 1, color='r', alpha=0.25)

    _ax.scatter(order_parameters, -op_diff, s=7.5, alpha=0.5, edgecolors='none')

    # Format:
    _ax.set_xlim(0, 1);
    _ax.set_ylim(-1, 1);
    _ax.set_xlabel("Stochastic Order Parameter")
    _ax.set_ylabel("Change in Order Parameter -> Deterministic")

    plt.show()
    return


@app.cell
def _(det_order_parameters, op_diff, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # # Calculate bin edges:
    # _speed_edges, _mean_op = get_quantile_data(speed_array, order_parameters)

    # Plot the actual scatter plot:
    _ax.hlines(0, 0, 1, color='r', alpha=0.75)
    _ax.hlines(0.1, 0, 1, color='r', alpha=0.25)
    _ax.hlines(-0.1, 0, 1, color='r', alpha=0.25)

    _ax.scatter(det_order_parameters, op_diff, s=7.5, alpha=0.5, edgecolors='none')
    _ax.scatter(
        det_order_parameters[3135], op_diff[3135], c='tab:orange',
        s=30, alpha=0.5, facecolors='none'
    )

    # Format:
    _ax.set_xlim(0, 1);
    _ax.set_ylim(-1, 1);
    _ax.set_xlabel("Deterministic Order Parameter")
    _ax.set_ylabel("Change in Order Parameter -> Stochastic")

    plt.show()
    return


@app.cell
def _(det_order_parameters, np, op_diff):
    np.argwhere(np.logical_and(det_order_parameters > 0.4, op_diff > 0.1))
    return


@app.cell
def _(get_quantile_data, order_parameters, plt, speed_array, y):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # Calculate bin edges:
    _speed_edges, _mean_op = get_quantile_data(speed_array, order_parameters)

    # Plot the actual scatter plot:
    _ax.scatter(speed_array, order_parameters, s=7.5, alpha=0.5, edgecolors='none')

    # Plot binscatter:
    for _i, _y in enumerate(_mean_op):
        _ax.hlines(y, _speed_edges[_i], _speed_edges[_i + 1], color='r', alpha=0.75)
        _ax.scatter((_speed_edges[_i] + _speed_edges[_i + 1]) / 2, y, c='r', s=15)

    # Format:
    _ax.set_xlim(0, 5);
    _ax.set_ylim(0, 1);
    _ax.set_xlabel("Average Speed")
    _ax.set_ylabel("Velocity Order Parameter")

    plt.show()
    return


@app.cell
def _(dtheta_array, get_quantile_data, order_parameters, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # Calculate bin edges:
    _dtheta_edges, _mean_op = get_quantile_data(dtheta_array, order_parameters)

    # Plot the actual scatter plot:
    _ax.scatter(dtheta_array, order_parameters, s=7.5, alpha=0.5, edgecolors='none')

    # Plot binscatter:
    for _i, _y in enumerate(_mean_op):
        _ax.hlines(_y, _dtheta_edges[_i], _dtheta_edges[_i + 1], color='r', alpha=0.75)
        _ax.scatter((_dtheta_edges[_i] + _dtheta_edges[_i + 1]) / 2, _y, c='r', s=15)


    _ax.set_xlim(0, 1.35)
    _ax.set_ylim(0, 1);

    _ax.set_xlabel("Average Change in Direction")
    _ax.set_ylabel("Velocity Order Parameter")

    plt.show()
    return


@app.cell
def _(dtheta_array, plt, speed_array):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # # Calculate bin edges:
    # _speed_edges, _mean_op = get_quantile_data(speed_array, order_parameters)

    # Plot the actual scatter plot:
    _ax.scatter(speed_array, dtheta_array, s=7.5, alpha=0.5, edgecolors='none')

    # Format:
    _ax.set_xlim(0, 5.5);
    _ax.set_ylim(0, 1.5);
    _ax.set_xlabel("Average Speed")
    _ax.set_ylabel("Average Change in Direction")
    _fig.suptitle("Speed-Persistence with Stochastic Collisions")

    plt.show()
    return


@app.cell
def _(det_dtheta_array, det_speed_array, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # # Calculate bin edges:
    # _speed_edges, _mean_op = get_quantile_data(speed_array, order_parameters)

    # Plot the actual scatter plot:
    _ax.scatter(det_speed_array, det_dtheta_array, s=7.5, alpha=0.5, edgecolors='none')

    # # Plot binscatter:
    # for i, y in enumerate(_mean_op):
    #     _ax.hlines(y, _speed_edges[i], _speed_edges[i + 1], color='r', alpha=0.75)
    #     _ax.scatter((_speed_edges[i] + _speed_edges[i + 1]) / 2, y, c='r', s=15)

    # Format:
    _ax.set_xlim(0, 5.5);
    _ax.set_ylim(0, 1.5);
    _ax.set_xlabel("Average Speed")
    _ax.set_ylabel("Average Change in Direction")
    _fig.suptitle("Speed-Persistence with Deterministic Collisions")

    plt.show()
    return


@app.cell
def _(get_quantile_data, order_parameters, parameters, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # Get number of cells:
    cell_numbers = parameters[:, -1]

    # Calculate bin edges:
    _cellnum_edges, _mean_op = get_quantile_data(cell_numbers, order_parameters)

    # Plot the actual scatter plot:
    _ax.scatter(cell_numbers, order_parameters, s=15, alpha=0.5, edgecolors='none')
    _ax.scatter(
        cell_numbers[2440], order_parameters[2440],
        s=30, alpha=1, edgecolors='none', c='tab:orange'
    )

    # Plot binscatter:
    for _i, _y in enumerate(_mean_op):
        _ax.hlines(_y, _cellnum_edges[_i], _cellnum_edges[_i + 1], color='r')
        _ax.scatter((_cellnum_edges[_i] + _cellnum_edges[_i + 1]) / 2, _y, c='r', s=15)

    # Set axis labels:
    _ax.set_xlim(50, 350)
    _ax.set_ylim(0, 1);
    _ax.set_xlabel("Cell Number")
    _ax.set_ylabel("Velocity Order Parameter")

    plt.show()
    return (cell_numbers,)


@app.cell
def _(cell_numbers, get_quantile_data, order_parameters, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # Calculate bin edges:
    _cellnum_edges, _mean_op = get_quantile_data(cell_numbers, order_parameters)

    # Plot the actual scatter plot:
    _ax.scatter(cell_numbers, order_parameters, s=15, alpha=0.5, edgecolors='none')

    # Plot binscatter:
    for _i, _y in enumerate(_mean_op):
        _ax.hlines(_y, _cellnum_edges[_i], _cellnum_edges[_i + 1], color='r')
        _ax.scatter((_cellnum_edges[_i] + _cellnum_edges[_i + 1]) / 2, _y, c='r', s=15)

    # Set axis labels:
    _ax.set_xlim(50, 350)
    _ax.set_ylim(0, 1);
    _ax.set_xlabel("Cell Number")
    _ax.set_ylabel("Velocity Order Parameter")

    plt.show()
    return


@app.cell
def _(get_quantile_data, order_parameters, parameters, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # Get number of cells:
    collision_advection = parameters[:, -3]

    # Calculate bin edges:
    _coladv_edges, _mean_op = get_quantile_data(collision_advection, order_parameters)

    # Plot the actual scatter plot:
    _ax.scatter(collision_advection, order_parameters, s=7.5, alpha=0.5, edgecolors='none')

    # Plot binscatter:
    for _i, _y in enumerate(_mean_op):
        _ax.hlines(_y, _coladv_edges[_i], _coladv_edges[_i + 1], color='r')
        _ax.scatter((_coladv_edges[_i] + _coladv_edges[_i + 1]) / 2, _y, c='r', s=15)

    _ax.set_xlim(0, 1);
    _ax.set_ylim(0, 1);

    # Set axis labels:
    _ax.set_xlabel("Collision Advection Rate")
    _ax.set_ylabel("Velocity Order Parameter")

    plt.show()
    return (collision_advection,)


@app.cell
def _(get_quantile_data, order_parameters, parameters, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # Get number of cells:
    collision_reduction = parameters[:, -2]

    # Calculate bin edges:
    _colred_edges, _mean_op = get_quantile_data(collision_reduction, order_parameters)

    # Plot the actual scatter plot:
    _ax.scatter(collision_reduction, order_parameters, s=7.5, alpha=0.5, edgecolors='none')

    # Plot binscatter:
    for _i, _y in enumerate(_mean_op):
        _ax.hlines(_y, _colred_edges[_i], _colred_edges[_i + 1], color='r')
        _ax.scatter((_colred_edges[_i] + _colred_edges[_i + 1]) / 2, _y, c='r', s=15)

    # Format:
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1);
    _ax.set_xlabel("Collision Flow Reduction Rate")
    _ax.set_ylabel("Velocity Order Parameter")

    plt.show()
    return (collision_reduction,)


@app.cell
def _(cell_numbers, np, order_parameters):
    # Get indices of interesting runs:
    np.argwhere(np.logical_and(order_parameters > 0.5, cell_numbers < 150))
    return


@app.cell
def _(dtheta_array, np, order_parameters):
    # Get indices of interesting runs:
    np.argwhere(np.logical_and(order_parameters > 0.5, dtheta_array > 0.2))
    return


@app.cell
def _(cell_numbers, order_parameters, plt, speed_array):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # Plot the actual scatter plot:
    _op_mask = order_parameters > 0.25
    _ax.scatter(cell_numbers, speed_array, s=7.5, alpha=0.5, edgecolors='none')
    _ax.scatter(
        cell_numbers[_op_mask], speed_array[_op_mask],
        s=10, alpha=1, c='r', edgecolors='none', label="Flocking Cell Populations"
    )
    _ax.legend(handletextpad=0.0)

    # Format:
    _ax.set_xlim(50, 350)
    # _ax.set_ylim(0, 1.5);
    _ax.set_xlabel("Cell Number")
    _ax.set_ylabel("Average Speed")

    _ax.set_title("Stochastic Collisions")

    plt.show()
    return


@app.cell
def _(cell_numbers, det_order_parameters, det_speed_array, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # Plot the actual scatter plot:
    _op_mask = det_order_parameters > 0.25
    _ax.scatter(cell_numbers, det_speed_array, s=7.5, alpha=0.5, edgecolors='none')
    _ax.scatter(
        cell_numbers[_op_mask], det_speed_array[_op_mask],
        s=10, alpha=1, c='r', edgecolors='none', label="Flocking Cell Populations"
    )
    _ax.legend(handletextpad=0.0)

    # Format:
    _ax.set_xlim(50, 350)
    # _ax.set_ylim(0, 1.5);
    _ax.set_xlabel("Cell Number")
    _ax.set_ylabel("Average Speed")
    _ax.set_title("Deterministic Collisions")


    plt.show()
    return


@app.cell
def _(cell_numbers, dtheta_array, order_parameters, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # Plot the actual scatter plot:
    _op_mask = order_parameters > 0.25
    _ax.scatter(cell_numbers, dtheta_array, s=7.5, alpha=0.5, edgecolors='none')
    _ax.scatter(
        cell_numbers[_op_mask], dtheta_array[_op_mask],
        s=10, alpha=1, c='r', edgecolors='none', label="Flocking Cell Populations"
    )
    _ax.legend(handletextpad=0.0)

    # Format:
    _ax.set_xlim(50, 350)
    _ax.set_ylim(0, 1.5);
    _ax.set_xlabel("Cell Number")
    _ax.set_ylabel("Average Change in Direction")

    _ax.set_title("Stochastic Collisions")


    plt.show()
    return


@app.cell
def _(cell_numbers, det_dtheta_array, det_order_parameters, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # Plot the actual scatter plot:
    _op_mask = det_order_parameters > 0.25
    _ax.scatter(cell_numbers, det_dtheta_array, s=7.5, alpha=0.5, edgecolors='none')
    _ax.scatter(
        cell_numbers[_op_mask], det_dtheta_array[_op_mask],
        s=10, alpha=1, c='r', edgecolors='none', label="Flocking Cell Populations"
    )
    _ax.legend(handletextpad=0.0)

    # Format:
    _ax.set_xlim(50, 350)
    _ax.set_ylim(0, 1.5);
    _ax.set_xlabel("Cell Number")
    _ax.set_ylabel("Average Change in Direction")

    _ax.set_title("Deterministic Collisions")


    plt.show()
    return


@app.cell
def _(cell_numbers, collision_reduction, order_parameters, plt):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # Plot the actual scatter plot:
    _op_mask = order_parameters > 0.25
    _ax.scatter(cell_numbers, collision_reduction, s=7.5, alpha=0.5, edgecolors='none')
    _ax.scatter(
        cell_numbers[_op_mask], collision_reduction[_op_mask],
        s=10, alpha=1, c='r', edgecolors='none', label="Flocking Cell Populations"
    )
    _ax.legend(handletextpad=0.0)

    # Format:
    _ax.set_xlim(50, 350)
    _ax.set_ylim(0, 1);
    _ax.set_xlabel("Cell Number")
    _ax.set_ylabel("Collision Reduction Rate")

    plt.show()
    return


@app.cell
def _(collision_advection, collision_reduction, order_parameters, plt):
    collision_advection

    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # Plot the actual scatter plot:
    _op_mask = order_parameters > 0.25
    _ax.scatter(
        collision_advection, collision_reduction,
        s=7.5, alpha=0.5, edgecolors='none'
    )
    _ax.scatter(
        collision_advection[_op_mask], collision_reduction[_op_mask],
        s=10, alpha=1, c='r', edgecolors='none', label="Flocking Cell Populations"
    )
    _ax.legend(handletextpad=0.0)

    # Format:
    # _ax.set_xlim(50, 350)
    # _ax.set_ylim(0, 1);
    _ax.set_xlabel("Collision Advection Rate")
    _ax.set_ylabel("Collision Reduction Rate")

    plt.show()
    return


@app.cell
def _(collision_reduction, np, order_parameters):
    np.argwhere(np.logical_and(order_parameters > 0.25, collision_reduction > 0.6))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
