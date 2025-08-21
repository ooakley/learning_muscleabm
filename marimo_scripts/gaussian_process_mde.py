import marimo

__generated_with = "0.12.10"
app = marimo.App(width="columns")


@app.cell
def _():
    import gpytorch
    import torch
    import json

    import numpy as np
    import matplotlib.pyplot as plt

    from scipy import stats
    from torch.utils.data import TensorDataset, DataLoader
    return DataLoader, TensorDataset, gpytorch, json, np, plt, stats, torch


@app.cell
def _():
    CONSTANT_PARAMETERS = {
        "superIterationCount": 12,
        "timestepsToRun": 2880,
        "worldSize": 2048,
        "gridSize": 64,
        "dt": 1,
        "thereIsMatrixInteraction": 1,
        "aspectRatio": 1,
        # # Collision parameters:
        # "collisionFlowReductionRate": 0,
        # "collisionAdvectionRate": 0,
        # Shape:
        "stretchFactor": 0.01,
        "slipFactor": 1
    }

    GRIDSEARCH_PARAMETERS = {
        "matrixTurnoverRate": [0, 0.2],
        "matrixAdditionRate": [0, 0.2],
        "matrixAdvectionRate": [0, 1],
        "cueDiffusionRate": [0.0001, 2.5],
        "cueKa": [0.0001, 5],
        "fluctuationAmplitude": [5e-5, 5e-3],
        "fluctuationTimescale": [1, 250],
        "maximumSteadyStateActinFlow": [0.0, 3],
        "numberOfCells": [75, 150],
        "actinAdvectionRate": [0.0, 1.5],
        "cellBodyRadius": [15, 100],
        # Collisions:
        "collisionFlowReductionRate": [0.0, 0.25],
        "collisionAdvectionRate": [0.0, 1.5],
    }
    return CONSTANT_PARAMETERS, GRIDSEARCH_PARAMETERS


@app.cell
def _(GRIDSEARCH_PARAMETERS, np):
    parameters = np.load(
        "./gridsearch_data/ss_wd/collated_inputs.npy"
    )
    distances = np.load(
        "./gridsearch_data/ss_wd/collated_distances.npy"
    )

    _nan_mask = np.any(np.isnan(distances), axis=(1, 2))

    nan_parameters = parameters[_nan_mask, :]
    parameters = parameters[~_nan_mask, :]
    distances = distances[~_nan_mask, :]

    normalised_parameters = np.zeros_like(parameters)
    for index, parameter_range in enumerate(GRIDSEARCH_PARAMETERS.values()):
        _minimum = parameter_range[0]
        _maximum = parameter_range[1]
        normalised_parameters[:, index] = \
            (parameters[:, index] - _minimum) / (_maximum - _minimum)


    # Scaling distances before combining them:
    log_distances = np.log(distances)
    # euclidean_distances = np.sqrt(np.sum(log_distances[:, :]**2, axis=2))
    euclidean_distances = np.sum(log_distances, axis=2)
    return (
        distances,
        euclidean_distances,
        index,
        log_distances,
        nan_parameters,
        normalised_parameters,
        parameter_range,
        parameters,
    )


@app.cell
def _(euclidean_distances, plt):
    plt.hist(euclidean_distances[:, 5].flatten());
    plt.show()
    return


@app.cell
def _(gpytorch, np):
    class ApproximateGPModel(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points, dimensions):
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
    return (ApproximateGPModel,)


@app.cell
def _(ApproximateGPModel, DataLoader, TensorDataset, gpytorch, torch):
    def instantiate_model(inducing_points, dimensions):
        # Convert inducing points to torch:
        inducing_points = torch.tensor(
            inducing_points, dtype=torch.float32
        )

        # Set up likelihoods:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ApproximateGPModel(inducing_points, dimensions)
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
        ], lr=0.01)

        # Set up loss:
        mll = gpytorch.mlls.VariationalELBO(
            likelihood, model, num_data=train_y.size(0)
        )
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
def _(torch):
    def run_inference(model, likelihood, inputs):
        tensor_input = torch.tensor(inputs, dtype=torch.float32)
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            predictions = likelihood(model(tensor_input))
        return predictions

    def run_grad_inference(model, likelihood, inputs):
        tensor_numpy = torch.tensor(
            inputs, dtype=torch.float32, requires_grad=True
        )
        tensor_input = torch.unsqueeze(tensor_numpy, 0)
        model.eval()
        likelihood.eval()
        predictions = likelihood(model(tensor_input)).mean
        return tensor_input, predictions
    return run_grad_inference, run_inference


@app.cell
def _(
    euclidean_distances,
    instantiate_model,
    normalised_parameters,
    train_model,
):
    models = []
    for _column in range(1):
        # Instantiate and train model:
        model, likelihood = instantiate_model(normalised_parameters[:500], 13)
        loss_history = train_model(
            model, likelihood,
            normalised_parameters, euclidean_distances[:, _column],
            epochs=250
        )
        models.append((model, likelihood))
    return likelihood, loss_history, model, models


@app.cell
def _():
    # _loss_history = train_model(
    #     *models[0],
    #     normalised_parameters, euclidean_distances[:, 0],
    #     epochs=250
    # )
    return


@app.cell
def _(GRIDSEARCH_PARAMETERS, models):
    lscale = \
        models[0][0]\
        .covar_module.kernels[0].base_kernel.lengthscale.detach().numpy()[0]

    named_lscale = dict(zip(
        list(GRIDSEARCH_PARAMETERS.keys()),
        list(lscale)
    ))

    print(named_lscale)

    # print(json.dumps(named_lscale, sort_keys=False, indent=4))
    return lscale, named_lscale


@app.cell
def _(models, normalised_parameters, run_inference):
    predictions = run_inference(*models[0], normalised_parameters[:, :])
    return (predictions,)


@app.cell
def _(euclidean_distances, np, plt, predictions):
    # Plotting model fit:
    _left_lim = -17.5
    _right_lim = 25

    _fig, _ax = plt.subplots(figsize=(4.5, 4.5), layout='constrained')
    _ax.scatter(
        euclidean_distances[:, 0], predictions.mean.detach().numpy(),
        s=10, edgecolors='none', alpha=0.5
    )

    _ax.plot(
        np.linspace(_left_lim, _right_lim, 2),
        np.linspace(_left_lim, _right_lim, 2),
        c='k', linestyle='dashed'
    )

    _ax.set_xlim(_left_lim, _right_lim)
    _ax.set_xlabel("Simulation Outputs - WD Distance")
    _ax.set_ylim(_left_lim, _right_lim)
    _ax.set_ylabel("GP Distance Prediction")
    return


@app.cell
def _(np, predictions):
    np.argmin(predictions.mean.detach().numpy())
    return


@app.cell
def _(parameters):
    parameters[12708]
    return


@app.cell
def _(CONSTANT_PARAMETERS, GRIDSEARCH_PARAMETERS, json, parameters):
    optim_params = dict(zip(
        list(GRIDSEARCH_PARAMETERS.keys()),
        parameters[12708, :]
    ))
    CONSTANT_PARAMETERS.update(optim_params)
    print(json.dumps(CONSTANT_PARAMETERS, sort_keys=False, indent=4))
    return (optim_params,)


@app.cell
def _(torch):
    def calculate_jacobian(y, x, create_graph=False):
        jac = []
        flat_y = y.reshape(-1)
        grad_y = torch.zeros_like(flat_y)
        grad_matrix = torch.eye(len(flat_y))
        for i in range(len(flat_y)):
            grad_x, = torch.autograd.grad(
                flat_y, x, grad_matrix[:, i],
                create_graph=create_graph
            )
            jac.append(grad_x.reshape(x.shape))
        return torch.stack(jac).reshape(y.shape + x.shape)

    def calculate_hessian(y, x, create_graph=False):
        return calculate_jacobian(
            calculate_jacobian(y, x, create_graph=True),
            x, create_graph=create_graph
        )
    return calculate_hessian, calculate_jacobian


@app.cell
def _(calculate_jacobian, models, normalised_parameters, run_grad_inference):
    # Get posterior:
    import copy

    posterior = copy.deepcopy(normalised_parameters[12708, :])
    print(posterior)

    for i in range(1000):
        tensor_input, mean_predictions = run_grad_inference(
            *models[0], posterior
        )
        jacobian = calculate_jacobian(
            mean_predictions, tensor_input, create_graph=True
        )

        numpy_jacobian = jacobian.detach().numpy()[0][0]
        posterior -= 0.0001*numpy_jacobian

        print(mean_predictions.detach().numpy()[0])
    return (
        copy,
        i,
        jacobian,
        mean_predictions,
        numpy_jacobian,
        posterior,
        tensor_input,
    )


@app.cell
def _(posterior):
    print(posterior)
    return


@app.cell
def _(GRIDSEARCH_PARAMETERS, posterior):
    def _():
        denormalised_prediction = []
        for index, parameter_range in enumerate(GRIDSEARCH_PARAMETERS.values()):
            _minimum = parameter_range[0]
            _maximum = parameter_range[1]
            denormalised_prediction.append(
                (posterior[index] * (_maximum - _minimum)) + _minimum
            )
        return denormalised_prediction

    denormalised_prediction = _()
    return (denormalised_prediction,)


@app.cell
def _(denormalised_prediction):
    denormalised_prediction
    return


@app.cell
def _(
    CONSTANT_PARAMETERS,
    GRIDSEARCH_PARAMETERS,
    denormalised_prediction,
    json,
):
    def _():
        optim_params = dict(zip(
            list(GRIDSEARCH_PARAMETERS.keys()),
            denormalised_prediction
        ))
        CONSTANT_PARAMETERS.update(optim_params)
        print(json.dumps(CONSTANT_PARAMETERS, sort_keys=False, indent=4))

    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
