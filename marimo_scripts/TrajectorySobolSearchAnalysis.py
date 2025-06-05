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

    TRAINING_ITERATIONS = 100
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

    import matplotlib.pyplot as plt

    import marimo as mo
    import numpy as np
    return gpytorch, mo, np, os, plt, pysr, torch


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
            parameters[:, index] - _minimum / (_maximum - _minimum)
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
def _(plt, speed_array):
    _ = plt.hist(speed_array.flatten(), bins=100)
    plt.show()
    return


@app.cell
def _(dtheta_array, plt):
    _ = plt.hist(dtheta_array.flatten(), bins=100)
    plt.show()
    return


@app.cell
def _(dtheta_array, plt, speed_array):
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.scatter(speed_array, dtheta_array, s=10, edgecolors='none', alpha=0.5)

    _ax.set_xlim(0, 5.5);
    _ax.set_ylim(0, 1);

    _ax.set_xlabel("Average Speed (µm/min)")
    _ax.set_ylabel("Average Change in Direction (rad/min)")
    return


@app.cell
def _(dtheta_array, parameters, plt):
    plt.scatter(parameters[:, 0], 1/dtheta_array, s=0.5)
    return


@app.cell
def _(parameters, plt, speed_array):
    # Plotting AUC model fit:
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.scatter(parameters[:, 0], speed_array, s=10, edgecolors='none', alpha=0.5)

    # Determine x limits:
    _ax.set_xlim([0.005, 0.5])
    _ax.set_xlabel("Simulation Parameter - cueDiffusionRate")
    _ax.set_ylim(0, 7.5)
    _ax.set_ylabel("Simulation Output - Speed")
    # ax.scatter(train_x, train_y, s=5)
    return


@app.cell
def _(parameters, plt, speed_array):
    # Plotting AUC model fit:
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.scatter(parameters[:, 5], speed_array, s=10, edgecolors='none', alpha=0.5)

    # Determine x limits:
    _left_lim = 0.5
    _right_lim = 5
    _ax.set_xlim(_left_lim, _right_lim)
    _ax.set_xlabel("Simulation Parameter - maximumSteadyStateActinFlow")
    _ax.set_ylim(0, 7.5)
    _ax.set_ylabel("Simulation Output - Speed")
    # ax.scatter(train_x, train_y, s=5)
    return


@app.cell
def _(pysr):
    timeout_minutes = 10
    sr_model = pysr.PySRRegressor(
        maxsize=15,
        maxdepth=10,
        niterations=45,
        batching=True,
        batch_size=50,
        binary_operators=["+", "*", "-"],
        unary_operators=[
            "exp",
            "log",
            "square",
            "inv",
        ],
        elementwise_loss="L1DistLoss()",
        timeout_in_seconds=timeout_minutes*60,
    )
    return sr_model, timeout_minutes


@app.cell
def _(simplified_parameters, speed_array, sr_model):
    sr_model.fit(simplified_parameters, speed_array);
    return


@app.cell
def _(sr_model):
    print(sr_model)
    return


@app.cell
def _(sr_model):
    sr_model.sympy()
    return


@app.cell
def _():
    parameter_reference = {
        "x0": "cueDiffusionRate",
        "x1": "fluctuationAmplitude",
        "x2": "actinAdvectionRate",
        "x3": "maximumSteadyStateActinFlow"
    }
    return (parameter_reference,)


@app.cell
def _(simplified_parameters, sr_model):
    sr_speed_predictions = sr_model.predict(simplified_parameters[1::2], index=None)
    return (sr_speed_predictions,)


@app.cell
def _(np, plt, speed_array, sr_speed_predictions):
    # Plotting AUC model fit:
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))
    _ax.scatter(speed_array[1::2], sr_speed_predictions, s=2.5, alpha=0.5)

    # Plot line of equality:
    _left_lim = 0
    _right_lim = 6.5
    _ax.plot(
        np.linspace(_left_lim, _right_lim, 2),
        np.linspace(_left_lim, _right_lim, 2),
        c='k', linestyle='dashed'
    )

    # Determine x limits:
    _ax.set_xlim(_left_lim, _right_lim)
    _ax.set_xlabel("Simulation Outputs - Speed")
    _ax.set_ylim(_left_lim, _right_lim)
    _ax.set_ylabel("SR Predicted Outputs - Speed")
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
def _(normalised_parameters, speed_array, torch):
    train_parameters = torch.tensor(normalised_parameters[::2], dtype=torch.float32)
    train_speed = torch.tensor(speed_array[::2], dtype=torch.float32)
    return train_parameters, train_speed


@app.cell
def _(normalised_parameters, speed_array, torch):
    test_parameters = torch.tensor(normalised_parameters[1::2], dtype=torch.float32)
    test_speed = torch.tensor(speed_array[1::2], dtype=torch.float32)
    return test_parameters, test_speed


@app.cell
def _(ExactGPModel, gpytorch, train_parameters, train_speed):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_parameters, train_speed, likelihood)
    return likelihood, model


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
            print(f"Lengthscales: {model.covar_module.base_kernel.lengthscale.detach().numpy()[0]}")
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
def _(np, plt, predicted_speed, test_speed):
    # Plotting AUC model fit:
    _left_lim = 0
    _right_lim = 6.5
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))
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
    _fig, _ax = plt.subplots()
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
    return


if __name__ == "__main__":
    app.run()
