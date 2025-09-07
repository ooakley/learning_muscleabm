import marimo

__generated_with = "0.12.10"
app = marimo.App(width="columns")


@app.cell
def _():
    import gpytorch
    import torch

    import numpy as np
    import matplotlib.pyplot as plt

    from torch.utils.data import TensorDataset, DataLoader
    return DataLoader, TensorDataset, gpytorch, np, plt, torch


@app.cell
def _():
    GRIDSEARCH_PARAMETERS = {
        "cueDiffusionRate": [
            0.0025,
            2.5
        ],
        "cueKa": [
            0.1,
            5
        ],
        "fluctuationAmplitude": [
            5e-05,
            0.005
        ],
        "fluctuationTimescale": [
            1,
            150
        ],
        "maximumSteadyStateActinFlow": [
            0.0,
            1.25
        ]
    }
    return (GRIDSEARCH_PARAMETERS,)


@app.cell
def _(GRIDSEARCH_PARAMETERS, np):
    parameters = np.load("./gridsearch_data/out_wd_search/collated_inputs.npy")
    distances = np.load("./gridsearch_data/out_wd_search/collated_distances.npy")

    nan_mask = ~np.isnan(distances[:, 0])
    parameters = parameters[nan_mask, :]

    distances = distances[nan_mask, :]

    normalised_parameters = np.zeros_like(parameters)
    for index, parameter_range in enumerate(GRIDSEARCH_PARAMETERS.values()):
        _minimum = parameter_range[0]
        _maximum = parameter_range[1]
        normalised_parameters[:, index] = \
            (parameters[:, index] - _minimum) / (_maximum - _minimum)
    return (
        distances,
        index,
        nan_mask,
        normalised_parameters,
        parameter_range,
        parameters,
    )


@app.cell
def _(parameters):
    parameters.shape
    return


@app.cell
def _(distances, np):
    print(np.min(distances, axis=0))
    return


@app.cell
def _(distances, normalised_parameters, np):
    # Get appropriate values of epsilon:
    # Posterior for WT1:
    mask = distances[:, 5] < np.quantile(distances[:, 5], 0.01)
    print(np.count_nonzero(mask))
    wt_posterior = normalised_parameters[mask, :]
    # wt_posterior = parameters[mask, :]
    return mask, wt_posterior


@app.cell
def _(np, wt_posterior):
    mde = np.mean(wt_posterior, axis=0)
    print(mde)
    return (mde,)


@app.cell
def _(distances, mask, np, wt_posterior):
    weights = np.expand_dims(1 / distances[mask, 1], axis=1)
    weighted_mde = np.sum(wt_posterior * weights, axis=0) / np.sum(weights)
    weighted_mde
    return weighted_mde, weights


@app.cell
def _(
    GRIDSEARCH_PARAMETERS,
    distances,
    mask,
    np,
    plt,
    weighted_mde,
    wt_posterior,
):
    _fig, _axs = plt.subplots(5, 5, figsize=(10, 10), layout='constrained')
    ranges = list(GRIDSEARCH_PARAMETERS.values())
    range_spacing = [(max - min)*0.1 for min, max in ranges]
    axis_limits = [
        (minmax[0] - spacer, minmax[1] + spacer)
        for minmax, spacer in zip(ranges, range_spacing)
    ]

    for _diagonal in range(5):
        _axs[_diagonal, _diagonal].hist(wt_posterior[:, _diagonal], density=True)
        _axs[_diagonal, _diagonal].set_xlim(0, 1)

    for i, j in zip(np.triu_indices(5, 1)[0], np.triu_indices(5, 1)[1]):
        # Upper diagonal:
        _axs[i, j].scatter(
            wt_posterior[:, j], wt_posterior[:, i],
            s=10, alpha=0.5, edgecolor='None', c=distances[mask, 0]
        )
        _axs[i, j].scatter(weighted_mde[j], weighted_mde[i], c='r')
        _axs[i, j].set_xlim(0, 1)
        _axs[i, j].set_ylim(0, 1)

        # Lower diagonal:
        _axs[j, i].scatter(
            wt_posterior[:, i], wt_posterior[:, j],
            s=10, alpha=0.5, edgecolor='None', c=distances[mask, 0]
        )
        _axs[j, i].scatter(weighted_mde[i], weighted_mde[j], c='r')
        _axs[j, i].set_xlim(0, 1)
        _axs[j, i].set_ylim(0, 1)

    plt.show()
    return axis_limits, i, j, range_spacing, ranges


@app.cell
def _(distances, np, parameters, plt):
    def plot_posterior_histograms():
        # Get posteriors:
        masks = [
            distances[:, i] < np.quantile(distances[:, i], 0.05) for i in range(6)
        ]
        posteriors = [parameters[mask, :] for mask in masks]

        # Loop through parameters:
        fig, axs = plt.subplots(5, 1, figsize=(3, 10), layout='constrained')
        for parameter_index in range(5):
            for index, posterior in enumerate(posteriors):
                if index < 3:
                    color = 'tab:orange'
                else:
                    color = 'tab:blue'
                axs[parameter_index].hist(
                    posterior[:, parameter_index],
                    density=True, histtype='step', color=color
                )

        plt.show()
    return (plot_posterior_histograms,)


@app.cell
def _(plot_posterior_histograms):
    plot_posterior_histograms()
    return


@app.cell
def _():
    # Run Gaussian Process regression to obtain gradients of WD w/r/t model parameters:
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

    def run_inference(model, likelihood, inputs):
        tensor_input = torch.tensor(inputs, dtype=torch.float32)
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            predictions = likelihood(model(tensor_input))
        return predictions

    def run_grad_inference(model, likelihood, inputs):
        tensor_input = torch.tensor(inputs, dtype=torch.float32, requires_grad=True)
        tensor_input = torch.unsqueeze(tensor_input, 0)
        model.eval()
        likelihood.eval()
        predictions = likelihood(model(tensor_input))
        return tensor_input, predictions.mean
    return instantiate_model, run_grad_inference, run_inference, train_model


@app.cell
def _(distances, instantiate_model, normalised_parameters, train_model):
    models = []
    for _column in range(6):
        # Instantiate and train model:
        model, likelihood = instantiate_model(normalised_parameters[:500], 5)
        loss_history = train_model(
            model, likelihood,
            normalised_parameters, distances[:, _column],
            epochs=50
        )
        models.append((model, likelihood))
    return likelihood, loss_history, model, models


@app.cell
def _(models, normalised_parameters, run_inference):
    predictions = run_inference(*models[0], normalised_parameters)
    return (predictions,)


@app.cell
def _(distances, np, plt, predictions):
    # Plotting model fit:
    _left_lim = 0
    _right_lim = 4
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5), layout='constrained')
    _ax.scatter(
        distances[:, 0], predictions.mean.detach().numpy(),
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
    return


@app.cell
def _(torch):
    def calculate_jacobian(y, x, create_graph=False):
        jac = []
        flat_y = y.reshape(-1)
        grad_y = torch.zeros_like(flat_y)
        grad_matrix = torch.eye(len(flat_y))
        for i in range(len(flat_y)):
            grad_x, = torch.autograd.grad(
                flat_y, x, grad_matrix[:, i], create_graph=create_graph
            )
            jac.append(grad_x.reshape(x.shape))
        return torch.stack(jac).reshape(y.shape + x.shape)

    def calculate_hessian(y, x, create_graph=False):
        return calculate_jacobian(
            calculate_jacobian(y, x, create_graph=True), x, create_graph=create_graph
        )
    return calculate_hessian, calculate_jacobian


@app.cell
def _(
    calculate_hessian,
    distances,
    models,
    normalised_parameters,
    np,
    run_grad_inference,
):
    # Get posteriors:
    masks = [distances[:, i] < 0.075 for i in range(6)]
    posteriors = [normalised_parameters[mask, :] for mask in masks]
    mdes = [np.mean(posterior, axis=0) for posterior in posteriors]

    hessians = []
    for _mde, model_tuple in zip(mdes, models):
        # Run inference at MDE point:
        tensor_input, mean_predictions = run_grad_inference(
            *model_tuple, _mde
        )

        # Calculate hessian matrix:
        _hessian = calculate_hessian(
            mean_predictions, tensor_input, create_graph=True
        )
        hessians.append(_hessian.detach().numpy().squeeze())
    return (
        hessians,
        masks,
        mdes,
        mean_predictions,
        model_tuple,
        posteriors,
        tensor_input,
    )


@app.cell
def _(hessians, np):
    _fig, _axs
    for _hessian in hessians:
        eig_result = np.linalg.eig(_hessian)
        eigenvector_matrix = eig_result.eigenvectors
        print(eigenvector_matrix * np.expand_dims(eig_result.eigenvalues, axis=0))
    return eig_result, eigenvector_matrix


@app.cell
def _(hessian):
    hessian_numpy = hessian.detach().numpy().squeeze()
    return (hessian_numpy,)


@app.cell
def _(hessian_numpy, plt):
    plt.imshow(hessian_numpy)
    return


@app.cell
def _(eig_result):
    eig_result
    return


@app.cell
def _():
    return


@app.cell
def _(eigenvector_matrix):
    eigenvector_matrix
    return


@app.cell
def _(eigenvector_matrix, mde, np, wt_posterior):
    # Transform posterior to eigenparameter space:
    eigenparameter_matrix = np.log(wt_posterior / mde) @ eigenvector_matrix
    return (eigenparameter_matrix,)


@app.cell
def _(distances, eigenparameter_matrix, mask, plt):
    _lim = 3
    _fig, _ax = plt.subplots(figsize=(4, 4))
    _ax.scatter(
        eigenparameter_matrix[:, 0], eigenparameter_matrix[:, 1],
        s=10, c=distances[mask, 0], alpha=0.75, edgecolor='None'
    )
    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
