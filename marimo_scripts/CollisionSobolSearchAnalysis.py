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

    TRAINING_ITERATIONS = 500
    return DIMENSIONALITY, GRIDSEARCH_PARAMETERS, TRAINING_ITERATIONS


@app.cell
def _():
    import pysr
    import torch
    import gpytorch
    import os

    import matplotlib.pyplot as plt

    import marimo as mo
    import numpy as np
    return gpytorch, mo, np, os, plt, pysr, torch


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
def _(np, os):
    directory_path = "./gridsearch_data/stochastic_collisions"

    parameters = np.load(os.path.join(directory_path, "collated_inputs.npy"))
    # simplified_parameters = np.delete(parameters, [1, 3], axis=1)
    speed_array = np.load(os.path.join(directory_path, "mean_magnitudes.npy"))
    dtheta_array = np.load(os.path.join(directory_path, "mean_dtheta.npy"))
    op_array = np.load(os.path.join(directory_path, "collated_order_parameters.npy"))
    return directory_path, dtheta_array, op_array, parameters, speed_array


@app.cell
def _(np, os):
    deterministic_directory_path = "./gridsearch_data/deterministic_collisions"

    det_speed_array = np.load(
        os.path.join(deterministic_directory_path, "mean_magnitudes.npy")
    )
    det_dtheta_array = np.load(
        os.path.join(deterministic_directory_path, "mean_dtheta.npy")
    )
    det_op_array = np.load(
        os.path.join(deterministic_directory_path, "collated_order_parameters.npy")
    )
    return (
        det_dtheta_array,
        det_op_array,
        det_speed_array,
        deterministic_directory_path,
    )


@app.cell
def _(det_op_array, op_array, plt):
    _fig, _ax = plt.subplots(figsize=(5.5, 4.5))
    _ax.plot(op_array.mean(axis=0), label='Stochastic Collisions')
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
def _(get_quantile_data, order_parameters, plt, speed_array):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))

    # Calculate bin edges:
    _speed_edges, _mean_op = get_quantile_data(speed_array, order_parameters)

    # Plot the actual scatter plot:
    _ax.scatter(speed_array, order_parameters, s=7.5, alpha=0.5, edgecolors='none')

    # Plot binscatter:
    for i, y in enumerate(_mean_op):
        _ax.hlines(y, _speed_edges[i], _speed_edges[i + 1], color='r', alpha=0.75)
        _ax.scatter((_speed_edges[i] + _speed_edges[i + 1]) / 2, y, c='r', s=15)

    # Format:
    _ax.set_xlim(0, 5);
    _ax.set_ylim(0, 1);
    _ax.set_xlabel("Average Speed")
    _ax.set_ylabel("Velocity Order Parameter")

    plt.show()
    return i, y


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
def _(mo):
    mo.md(r"""#1 - Running Gaussian Process Regression to assess contributory factors to Flocking""")
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
            sigma_0 = 10.0
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
def _(
    ExactGPModel,
    GRIDSEARCH_PARAMETERS,
    gpytorch,
    np,
    order_parameters,
    parameters,
    torch,
):
    # Scaling parameters to the same range:
    normalised_parameters = np.zeros_like(parameters)
    for index, parameter_range in enumerate(GRIDSEARCH_PARAMETERS.values()):
        _minimum = parameter_range[0]
        _maximum = parameter_range[1]
        normalised_parameters[:, index] = normalised_parameters[:, index] \
            - _minimum / (_maximum - _minimum)

    # Generate training datasets:
    train_parameters = torch.tensor(normalised_parameters, dtype=torch.float32)
    train_speed = torch.tensor(order_parameters, dtype=torch.float32)

    # Set up likelihood and model objects:
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_parameters, train_speed, likelihood)
    return (
        index,
        likelihood,
        model,
        normalised_parameters,
        parameter_range,
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
        if (_i + 1) % 100 == 0:
            print("--- --- --- ---")
            print(f"Iteration {_i+1}/{TRAINING_ITERATIONS}")
            print(f"Lengthscales: {model.covar_module.base_kernel.lengthscale.detach().numpy()[0]}")
            # print(f"Noise estimates: {model.likelihood.noise.item()}")
    return loss, mll, model_optimiser, predictions


@app.cell
def _(likelihood, model, normalised_parameters, order_parameters, torch):
    # Set up test parameters:
    test_parameters = torch.tensor(normalised_parameters, dtype=torch.float32)
    test_speed = torch.tensor(order_parameters, dtype=torch.float32)

    # Run inference:
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        predicted_speed = likelihood(model(test_parameters))
    return predicted_speed, test_parameters, test_speed


@app.cell
def _(np, plt, predicted_speed, test_speed):
    # Plotting AUC model fit:
    _left_lim = 0
    _right_lim = 1
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))
    _ax.scatter(test_speed.detach(), predicted_speed.mean, s=2.5, alpha=0.5)
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
def _():
    return


if __name__ == "__main__":
    app.run()
