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
        "actinAdvectionRate": [0.1, 2],
        "maximumSteadyStateActinFlow": [0.5, 5],
    }

    DIMENSIONALITY = 6

    TRAINING_ITERATIONS = 250
    return DIMENSIONALITY, GRIDSEARCH_PARAMETERS, TRAINING_ITERATIONS


@app.cell
def _(GRIDSEARCH_PARAMETERS):
    print(GRIDSEARCH_PARAMETERS.keys())
    return


@app.cell
def _():
    import pysr
    import torch
    import gpytorch
    import os
    import scipy

    import matplotlib.pyplot as plt

    import marimo as mo
    import numpy as np
    import pandas as pd
    return gpytorch, mo, np, os, pd, plt, pysr, scipy, torch


@app.cell
def _(np):
    def plot_scatter(ax, x, y, xlabel, ylabel):
        ax.scatter(x, y, s=10, edgecolors='none', alpha=0.25)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def get_quantile_data(x, y, bin_number=20):
        # Get quantiles as bin edges:
        quantiles = np.linspace(0, 1, bin_number + 1)
        bin_edges = np.quantile(x, quantiles)

        # Get binned data across x values:
        bin_count, _ = np.histogram(x, bins=bin_edges, density=False)
        y_sum, _ = np.histogram(x, bins=bin_edges, weights=y, density=False)
        mean_y = y_sum / bin_count

        return bin_edges, mean_y

    def plot_binscatter(ax, x, y):
        # Calculate bin edges:
        edges, mean_y = get_quantile_data(x, y)

        # Plot binscatter:
        for i, y in enumerate(mean_y):
            ax.hlines(y,  edges[i], edges[i + 1], color='tab:orange', alpha=0.75)
            ax.scatter((edges[i] + edges[i + 1]) / 2, y, c='tab:orange', s=15)
    return get_quantile_data, plot_binscatter, plot_scatter


@app.cell
def _(GRIDSEARCH_PARAMETERS, np, os):
    data_directory = "./gridsearch_data/movement_only"
    parameters = np.load(os.path.join(data_directory, "collated_inputs.npy"))
    simplified_parameters = np.delete(parameters, [1, 3], axis=1)
    speed_array = np.load(os.path.join(data_directory, "mean_magnitudes.npy"))
    dtheta_array = np.load(os.path.join(data_directory, "mean_dtheta.npy"))

    normalised_parameters = np.zeros_like(parameters)
    for index, parameter_range in enumerate(GRIDSEARCH_PARAMETERS.values()):
        _minimum = parameter_range[0]
        _maximum = parameter_range[1]
        normalised_parameters[:, index] = \
            (parameters[:, index] - _minimum) / (_maximum - _minimum)

    print(parameters.shape)
    return (
        data_directory,
        dtheta_array,
        index,
        normalised_parameters,
        parameter_range,
        parameters,
        simplified_parameters,
        speed_array,
    )


@app.cell
def _(np):
    def parameter_estimate(output_array, parameters, mean_value, epsilon=0.05):
        # Get epsilon fraction as integer:
        sample_size = int(len(output_array) * epsilon)

        # Get similar points in parameter space:
        summary_distances = np.abs(output_array - mean_value)**2
        selection_indices = np.argsort(summary_distances)[:sample_size]
        subset_parameters = parameters[selection_indices, :]
        return subset_parameters
    return (parameter_estimate,)


@app.cell
def _(normalised_parameters, parameter_estimate, speed_array):
    posterior_distribution = parameter_estimate(speed_array, normalised_parameters, 2.5)
    return (posterior_distribution,)


@app.cell
def _(np, posterior_distribution):
    covariance_matrix = np.cov(posterior_distribution, rowvar=False)
    return (covariance_matrix,)


@app.cell
def _(posterior_distribution):
    posterior_distribution.shape
    return


@app.cell
def _(posterior_distribution):
    from copulas.multivariate import GaussianMultivariate
    from copulas.univariate import BetaUnivariate, GaussianKDE

    dist = GaussianMultivariate(distribution=BetaUnivariate)
    dist.fit(posterior_distribution)
    return BetaUnivariate, GaussianKDE, GaussianMultivariate, dist


@app.cell
def _(dist, np):
    sampled = dist.sample(1500)
    sampled = np.array(sampled)
    return (sampled,)


@app.cell
def _(plt, posterior_distribution):
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1)
    _ax.scatter(posterior_distribution[:, 4], posterior_distribution[:, 5], s=10)
    return


@app.cell
def _(plt, sampled):
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1)
    _ax.scatter(sampled[:, 4], sampled[:, 5], s=10)
    return


@app.cell
def _(parameters, plot_binscatter, plot_scatter, plt, speed_array):
    _fig, _axs = plt.subplots(3, 2, figsize=(9, 13.5), sharey=True)

    plot_scatter(
        _axs[0, 0], parameters[:, 0], speed_array,
        "cueDiffusionRate", "Average Speed (µm/min)"
    )
    plot_binscatter(_axs[0, 0], parameters[:, 0], speed_array)

    plot_scatter(
        _axs[1, 0], parameters[:, 1], speed_array,
        "cueKa", "Average Speed (µm/min)"
    )
    plot_binscatter(_axs[1, 0], parameters[:, 1], speed_array)

    plot_scatter(
        _axs[2, 0], parameters[:, 2], speed_array,
        "fluctuationAmplitude", "Average Speed (µm/min)"
    )
    plot_binscatter(_axs[2, 0], parameters[:, 2], speed_array)

    plot_scatter(
        _axs[0, 1], parameters[:, 3], speed_array,
        "fluctuationTimescale", "Average Speed (µm/min)"
    )
    plot_binscatter(_axs[0, 1], parameters[:, 3], speed_array)

    plot_scatter(
        _axs[1, 1], parameters[:, 4], speed_array,
        "actinAdvectionRate", "Average Speed (µm/min)"
    )
    plot_binscatter(_axs[1, 1], parameters[:, 4], speed_array)

    plot_scatter(
        _axs[2, 1], parameters[:, 5], speed_array,
        "maximumSteadyStateActinFlow", "Average Speed (µm/min)"
    )
    plot_binscatter(_axs[2, 1], parameters[:, 5], speed_array)

    plt.show()
    return


@app.cell
def _(dtheta_array, parameters, plot_binscatter, plot_scatter, plt):
    _fig, _axs = plt.subplots(3, 2, figsize=(9, 13.5), sharey=True)

    plot_scatter(
        _axs[0, 0], parameters[:, 0], dtheta_array,
        "cueDiffusionRate", "Average Change in Direction (rad/min)"
    )
    plot_binscatter(_axs[0, 0], parameters[:, 0], dtheta_array)

    plot_scatter(
        _axs[1, 0], parameters[:, 1], dtheta_array,
        "cueKa", "Average Change in Direction (rad/min)"
    )
    plot_binscatter(_axs[1, 0], parameters[:, 1], dtheta_array)

    plot_scatter(
        _axs[2, 0], parameters[:, 2], dtheta_array,
        "fluctuationAmplitude", "Average Change in Direction (rad/min)"
    )
    plot_binscatter(_axs[2, 0], parameters[:, 2], dtheta_array)

    plot_scatter(
        _axs[0, 1], parameters[:, 3], dtheta_array,
        "fluctuationTimescale", "Average Change in Direction (rad/min)"
    )
    plot_binscatter(_axs[0, 1], parameters[:, 3], dtheta_array)

    plot_scatter(
        _axs[1, 1], parameters[:, 4], dtheta_array,
        "actinAdvectionRate", "Average Change in Direction (rad/min)"
    )
    plot_binscatter(_axs[1, 1], parameters[:, 4], dtheta_array)

    plot_scatter(
        _axs[2, 1], parameters[:, 5], dtheta_array,
        "maximumSteadyStateActinFlow", "Average Change in Direction (rad/min)"
    )
    plot_binscatter(_axs[2, 1], parameters[:, 5], dtheta_array)

    plt.show()
    return


@app.cell
def _(dtheta_array, plt, speed_array):
    _fig, _ax = plt.subplots(figsize=(3.5, 3.5))
    _ax.scatter(speed_array, dtheta_array, s=10, edgecolors='none', alpha=0.25)

    _ax.set_xlim(0, 5.5);
    _ax.set_ylim(0, 0.65);

    _ax.set_xlabel("Average Speed (µm/min)")
    _ax.set_ylabel("Average Change in Direction (rad/min)")
    return


@app.cell
def _(np, scipy):
    def calculate_average_hypervolume(parameters, seed=1, iterations=1000, num_points=10):
        # Instantiate random number generator:
        rng = np.random.default_rng(seed)

        # Sample from parameter space:
        volumes = []
        for _ in range(iterations):
            points = rng.choice(parameters, num_points, axis=0)
            hull_object = scipy.spatial.ConvexHull(points, qhull_options='QJ')
            volumes.append(hull_object.volume)

        return np.mean(volumes), np.std(volumes) / np.sqrt(iterations)
    return (calculate_average_hypervolume,)


@app.cell
def _(calculate_average_hypervolume, normalised_parameters, np, plt):
    def plot_fractional_hypervolume(property_array, xlabel, bin_number=15):
        # Calculate and plot specification capacity of different characteristics:
        quantiles = np.linspace(0, 1, 15 + 1)
        bin_edges = np.quantile(property_array, quantiles)

        total_volume, total_std = calculate_average_hypervolume(normalised_parameters)
        total_error = total_std / total_volume

        midpoints = []
        fractional_volumes = []
        approximation_errors = []
        for _index in range(15):
            print(_index)
            # Restrict to quantile:
            lower_limit = bin_edges[_index]
            upper_limit = bin_edges[_index + 1]
            mask = np.logical_and(
                property_array > lower_limit,
                property_array < upper_limit
            )

            # Calculate average volume of points in this area:
            restricted_volume, restricted_sem = \
                calculate_average_hypervolume(normalised_parameters[mask, :])
            restricted_error = restricted_sem / restricted_volume
            fractional_volume = restricted_volume/total_volume

            # Record values:
            fractional_volumes.append(fractional_volume)
            midpoints.append((lower_limit + upper_limit) / 2)

            # Propage error by quadrature:
            approximation_errors.append(
                fractional_volume*np.sqrt(total_error**2 + restricted_error**2)
            )


        _fig, _ax = plt.subplots(figsize=(4.5, 4.5))
        _ax.scatter(midpoints, fractional_volumes);
        _ax.errorbar(
            midpoints, fractional_volumes,
            yerr=approximation_errors,
            fmt='none'
        )

        for i, y in enumerate(fractional_volumes):
            _ax.hlines(y,  bin_edges[i], bin_edges[i + 1], alpha=0.75)

        _ax.set_xlabel(xlabel)
        _ax.set_ylabel("Fractional Volume in Parameter Space")
        _ax.set_ylim(0, 0.45)

        plt.show()
    return (plot_fractional_hypervolume,)


@app.cell
def _(plot_fractional_hypervolume, speed_array):
    plot_fractional_hypervolume(speed_array, "Average Speed (µm/min)")
    return


@app.cell
def _(dtheta_array, plot_fractional_hypervolume):
    plot_fractional_hypervolume(dtheta_array, "Average Change in Direction (µm/min)")
    return


@app.cell
def _(calculate_average_hypervolume, normalised_parameters, np):
    def plot_fractional_hypervolume_heatmap(
        property_array_x, property_array_y, bin_number=15
    ):
        # Calculate and plot specification capacity of different characteristics:
        quantiles = np.linspace(0, 1, bin_number + 1)
        x_bin_edges = np.quantile(property_array_x, quantiles)
        y_bin_edges = np.quantile(property_array_y, quantiles)

        total_volume, total_std = calculate_average_hypervolume(normalised_parameters)
        total_error = total_std / total_volume

        fractional_volume_matrix = []
        for i_index in range(bin_number):
            # Generate y mask:
            y_lower_limit = y_bin_edges[i_index]
            y_upper_limit = y_bin_edges[i_index + 1]
            y_mask = np.logical_and(
                property_array_y > y_lower_limit,
                property_array_y < y_upper_limit
            )

            # Instantiate row object:
            fractional_volume_row = []

            # Loop through row:
            for j_index in range(bin_number):
                # Generate x mask:
                x_lower_limit = x_bin_edges[j_index]
                x_upper_limit = x_bin_edges[j_index + 1]
                x_mask = np.logical_and(
                    property_array_x > x_lower_limit,
                    property_array_x < x_upper_limit
                )

                # Grid point mask:
                grid_mask = np.logical_and(
                    x_mask, y_mask
                )

                if np.count_nonzero(grid_mask) == 0:
                    fractional_volume_row.append(-1)
                    continue

                # Calculate average volume of points in this area:
                restricted_volume, restricted_sem = \
                    calculate_average_hypervolume(normalised_parameters[grid_mask, :])
                restricted_error = restricted_sem / restricted_volume
                fractional_volume = restricted_volume/total_volume

                # Record values:
                fractional_volume_row.append(fractional_volume)

            fractional_volume_matrix.append(np.array(fractional_volume_row))

        return np.stack(fractional_volume_matrix, axis=0)

        # _fig, _ax = plt.subplots(figsize=(4.5, 4.5))
        # _ax.scatter(midpoints, fractional_volumes);
        # _ax.errorbar(
        #     midpoints, fractional_volumes,
        #     yerr=approximation_errors,
        #     fmt='none'
        # )

        # for i, y in enumerate(fractional_volumes):
        #     _ax.hlines(y,  bin_edges[i], bin_edges[i + 1], alpha=0.75)

        # _ax.set_xlabel(xlabel)
        # _ax.set_ylabel("Fractional Volume in Parameter Space")

        # plt.show()
    return (plot_fractional_hypervolume_heatmap,)


@app.cell
def _(dtheta_array, plot_fractional_hypervolume_heatmap, speed_array):
    matrix = plot_fractional_hypervolume_heatmap(speed_array, dtheta_array, bin_number=20)
    return (matrix,)


@app.cell
def _(matrix, plt):
    palette = plt.cm.viridis.with_extremes(over='r', under='w', bad='b')

    _fig, _ax = plt.subplots()
    pos = _ax.imshow(
        matrix, vmin=0, vmax=0.045, origin='lower',
        cmap=palette
    )
    _ax.set_xlabel("Speed Quantiles")
    _ax.set_ylabel("Average Change in Direction Quantiles")
    cbar = _fig.colorbar(pos, ax=_ax)
    cbar.set_label('Parameter Diversity')

    plt.show()
    return cbar, palette, pos


@app.cell
def _(pysr):
    sr_model = pysr.PySRRegressor()
    sr_model = sr_model.from_file(
        run_directory="./sr_outputs/20250620_144910_WrPep1"
    )
    return (sr_model,)


@app.cell
def _(sr_model):
    sympy_representation = sr_model.sympy()
    return (sympy_representation,)


@app.cell
def _(sympy_representation):
    sympy_representation
    return


@app.cell
def _(sympy_representation):
    sympy_representation.simplify()
    return


@app.cell
def _(simplified_parameters, sr_model):
    sr_predictions = sr_model.predict(simplified_parameters, -1)
    return (sr_predictions,)


@app.cell
def _(np, plt, speed_array, sr_predictions):
    # Plotting AUC model fit:
    _left_lim = 0
    _right_lim = 6.5
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5), layout='constrained')
    _ax.scatter(
        speed_array, sr_predictions,
        s=10, edgecolors='none', alpha=0.5
    )
    _ax.plot(
        np.linspace(_left_lim, _right_lim, 2),
        np.linspace(_left_lim, _right_lim, 2),
        c='k', linestyle='dashed'
    )
    _ax.set_xlim(_left_lim, _right_lim)
    _ax.set_xlabel("GP Simulation Outputs - Speed")
    _ax.set_ylim(_left_lim, _right_lim)
    _ax.set_ylabel("Predicted Outputs - Speed")
    # ax.scatter(train_x, train_y, s=5)
    return


@app.cell
def _(mo):
    mo.md(r"""#1 Running GP with cell speed data:""")
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
    return (ExactGPModel,)


@app.cell
def _(ExactGPModel, gpytorch, parameters, speed_array, torch):
    train_parameters = torch.tensor(parameters[::2], dtype=torch.float32)
    train_speed = torch.tensor(speed_array[::2], dtype=torch.float32)\

    test_parameters = torch.tensor(parameters[1::2], dtype=torch.float32)
    test_speed = torch.tensor(speed_array[1::2], dtype=torch.float32)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_parameters, train_speed, likelihood)
    return (
        likelihood,
        model,
        test_parameters,
        test_speed,
        train_parameters,
        train_speed,
    )


@app.cell
def _(
    TRAINING_ITERATIONS,
    gpytorch,
    likelihood,
    model,
    torch,
    train_parameters,
    train_speed,
):
    # Train AUC model:
    likelihood.train()
    model.train()

    # Use the adam optimizer
    model_optimiser = torch.optim.Adam(model.parameters(), lr=0.1) 
    # Model contains GaussianLikelihood parameters.

    # Using MLL as our effective cost function:
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _i in range(TRAINING_ITERATIONS):
        # Zero gradients from previous iteration
        model_optimiser.zero_grad()
        # Get output from model:
        predictions = model(train_parameters)
        # Calculate loss and backprop gradients
        loss = -mll(predictions, train_speed)
        loss.backward()
        model_optimiser.step()
        if (_i + 1) % 20 == 0:
            print("--- --- --- ---")
            print(f"Iteration {_i+1}/{TRAINING_ITERATIONS}")
            print(f"Lengthscales {
                model.covar_module.base_kernel.lengthscale.detach().numpy()[0]
            }")
            # print(f"Noise estimates: {model.likelihood.noise.item()}")
    return loss, mll, model_optimiser, predictions


@app.cell
def _(likelihood, model, test_parameters, torch):
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        predicted_speed = likelihood(model(test_parameters))
    return (predicted_speed,)


@app.cell
def _(model):
    print(model.covar_module.base_kernel.lengthscale.detach().numpy())
    return


@app.cell
def _(GRIDSEARCH_PARAMETERS, model, np):
    np.set_printoptions(linewidth=2000) 

    speed_lengthscales = model.covar_module.base_kernel.lengthscale.detach().numpy()[0]
    _sorting_indices = np.argsort(speed_lengthscales)

    print(list(np.array(list(GRIDSEARCH_PARAMETERS.keys()))[_sorting_indices]))
    print(list(speed_lengthscales[_sorting_indices]))
    return (speed_lengthscales,)


@app.cell
def _(np, plt, predicted_speed, test_speed):
    # Plotting AUC model fit:
    _left_lim = 0
    _right_lim = 6.5
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5), layout='constrained')
    _ax.scatter(
        test_speed.detach(), predicted_speed.mean,
        s=10, edgecolors='none', alpha=0.5
    )
    _ax.plot(
        np.linspace(_left_lim, _right_lim, 2),
        np.linspace(_left_lim, _right_lim, 2),
        c='k', linestyle='dashed'
    )
    _ax.set_xlim(_left_lim, _right_lim)
    _ax.set_xlabel("GP Simulation Outputs - Speed")
    _ax.set_ylim(_left_lim, _right_lim)
    _ax.set_ylabel("Predicted Outputs - Speed")
    # ax.scatter(train_x, train_y, s=5)
    return


@app.cell
def _():
    # Predicting speed and persistence from other grid searches:
    return


@app.cell
def _(np, torch):
    deterministic_directory_path = "./gridsearch_data/deterministic_collisions"

    # Get input parameters:
    # det_parameter_array = np.load(
    #     os.path.join(deterministic_directory_path, "collated_inputs.npy")
    # )[:, :-3]
    det_parameter_array = np.load(
        "./abc_parameters.npy"
    )[:, :-3]
    # ^ indexing out collision-related parameters

    det_parameter_array = torch.tensor(det_parameter_array, dtype=torch.float32)
    return det_parameter_array, deterministic_directory_path


@app.cell
def _(det_parameter_array, likelihood, model, torch):
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        det_predicted_speed = likelihood(model(det_parameter_array))
    return (det_predicted_speed,)


@app.cell
def _(det_predicted_speed):
    det_predicted_mean_speed = det_predicted_speed.mean
    return (det_predicted_mean_speed,)


@app.cell
def _(det_predicted_mean_speed):
    det_predicted_mean_speed.max()
    return


@app.cell
def _(det_predicted_mean_speed, deterministic_directory_path, np, os):
    np.save(
        os.path.join(deterministic_directory_path, "abc_speeds.npy"),
        det_predicted_mean_speed
    )
    return


@app.cell
def _(mo):
    mo.md("""## 2 Running GP with cell persistence data:""")
    return


@app.cell
def _(dtheta_array, torch):
    train_dtheta = torch.tensor(dtheta_array[::2], dtype=torch.float32)
    test_dtheta = torch.tensor(dtheta_array[1::2], dtype=torch.float32)
    return test_dtheta, train_dtheta


@app.cell
def _(ExactGPModel, gpytorch, train_dtheta, train_parameters):
    dtheta_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    dtheta_model = ExactGPModel(train_parameters, train_dtheta, dtheta_likelihood)
    return dtheta_likelihood, dtheta_model


@app.cell
def _(
    TRAINING_ITERATIONS,
    dtheta_likelihood,
    dtheta_model,
    gpytorch,
    torch,
    train_dtheta,
    train_parameters,
):
    # Train AUC model:
    dtheta_model.train()
    dtheta_likelihood.train()

    # Use the adam optimizer
    dtheta_optimiser = torch.optim.Adam(dtheta_model.parameters(), lr=0.1) 

    # Using MLL as our effective cost function:
    dtheta_mll = gpytorch.mlls.ExactMarginalLogLikelihood(dtheta_likelihood, dtheta_model)

    for _i in range(TRAINING_ITERATIONS):
        # Zero gradients from previous iteration:
        dtheta_optimiser.zero_grad()
        # Get output from model:
        dtheta_predictions = dtheta_model(train_parameters)
        # Calculate loss and backprop gradients
        dtheta_loss = -dtheta_mll(dtheta_predictions, train_dtheta)
        dtheta_loss.backward()
        dtheta_optimiser.step()
        if (_i + 1) % 20 == 0:
            print("--- --- --- ---")
            print(f"Iteration {_i+1}/{TRAINING_ITERATIONS}")
            print(f"Lengthscales: {dtheta_model.covar_module.base_kernel.lengthscale.detach().numpy()[0]}")
            # print(f"Noise estimates: {model.likelihood.noise.item()}")
    return dtheta_loss, dtheta_mll, dtheta_optimiser, dtheta_predictions


@app.cell
def _(GRIDSEARCH_PARAMETERS, dtheta_model, np):
    np.set_printoptions(linewidth=2000) 

    dtheta_lengthscales = dtheta_model.covar_module.base_kernel.lengthscale.detach().numpy()[0]
    GRIDSEARCH_PARAMETERS.keys()
    _sorting_indices = np.argsort(dtheta_lengthscales)
    print(list(np.array(list(GRIDSEARCH_PARAMETERS.keys()))[_sorting_indices]))
    print(list(dtheta_lengthscales[_sorting_indices]))
    return (dtheta_lengthscales,)


@app.cell
def _(dtheta_likelihood, dtheta_model, test_parameters, torch):
    dtheta_model.eval()
    dtheta_likelihood.eval()

    with torch.no_grad():
        predicted_dtheta = dtheta_likelihood(dtheta_model(test_parameters))
    return (predicted_dtheta,)


@app.cell
def _(np, plt, predicted_dtheta, test_dtheta):
    # Plotting AUC model fit:
    _left_lim = 0
    _right_lim = 0.7
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))
    _ax.scatter(test_dtheta.detach(), predicted_dtheta.mean, s=2.5, alpha=0.5)
    _ax.plot(
        np.linspace(_left_lim, _right_lim, 2),
        np.linspace(_left_lim, _right_lim, 2),
        c='k', linestyle='dashed'
    )
    _ax.set_xlim(_left_lim, _right_lim)
    _ax.set_xlabel("Simulation Outputs - dTheta")
    _ax.set_ylim(_left_lim, _right_lim)
    _ax.set_ylabel("Predicted Outputs - dTheta")
    # ax.scatter(train_x, train_y, s=5)
    return


@app.cell
def _():
    # Predicting persistence for other grid searches:
    return


@app.cell
def _(det_parameter_array, dtheta_likelihood, dtheta_model, torch):
    dtheta_model.eval()
    dtheta_likelihood.eval()

    with torch.no_grad():
        det_predicted_dtheta = dtheta_likelihood(dtheta_model(det_parameter_array))

    det_predicted_dtheta = det_predicted_dtheta.mean
    return (det_predicted_dtheta,)


@app.cell
def _(det_predicted_dtheta, deterministic_directory_path, np, os):
    np.save(
        os.path.join(deterministic_directory_path, "abc_dthetas.npy"),
        det_predicted_dtheta
    )
    return


@app.cell
def _(det_predicted_dtheta):
    det_predicted_dtheta.max()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
