import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import json

    import gpytorch
    import torch

    import numpy as np
    import matplotlib.pyplot as plt

    from torch.utils.data import TensorDataset, DataLoader
    return gpytorch, json, np, os, plt, torch


@app.cell
def _(json, np, os):
    def load_gridsearch_data(experiment_folderpath):
        # Load numpy data:
        parameter_matrix = np.load(
            os.path.join(experiment_folderpath, "sample_matrix.npy")
        )
        coherency_fractions = np.load(
            os.path.join(experiment_folderpath, "summary_data", "coherency_fractions.npy")
        )
        ann_indices = np.load(
            os.path.join(experiment_folderpath, "summary_data", "ann_indices.npy")
        )

        # Get gridsearch configuration:
        with open(os.path.join(experiment_folderpath, "config.json")) as json_filestream:
            config_dictionary  = json.load(json_filestream)
        gridsearch_parameters = config_dictionary["gridsearch_parameters"]

        return parameter_matrix, coherency_fractions, ann_indices, gridsearch_parameters
    return (load_gridsearch_data,)


@app.cell
def _(gpytorch, torch):
    class SparseAdditiveGPModel(gpytorch.models.ApproximateGP):
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
            super(SparseAdditiveGPModel, self).__init__(variational_strategy)

            # Define mean and additive covariance functions:
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = \
                gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(
                        batch_shape=torch.Size([dimensions]),
                        ard_num_dims=1
                    )
                ) \
                + gpytorch.kernels.ConstantKernel()

        def forward(self, x):
            # Calculate mean of input:
            mean_x = self.mean_module(x)

            # Calculate interaction covariance matrix:
            batched_dimensions_of_x = x.mT.unsqueeze(-1)  # Now a d x n x 1 tensor
            univariate_covars = self.covar_module(batched_dimensions_of_x)
            covar_x = gpytorch.utils.sum_interaction_terms(
                univariate_covars, max_degree=2, dim=-3
            )
            conditioned_covar_x = gpytorch.add_jitter(covar_x, jitter_val=0.001)

            return gpytorch.distributions.MultivariateNormal(mean_x, conditioned_covar_x)
    return (SparseAdditiveGPModel,)


@app.cell
def _(load_gridsearch_data):
    parameter_matrix, coherency_fractions, ann_indices, gridsearch_parameters = load_gridsearch_data(
        "model_experiments/2025-12-04-collisions_only"
    )
    return coherency_fractions, parameter_matrix


@app.cell
def _(np, plt):
    loss_history = np.load('model_experiments/2025-12-04-collisions_only/gaussian_process_models/loss_history.npy')
    plt.plot(loss_history)
    return


@app.cell
def _(SparseAdditiveGPModel, torch):
    saved_state = torch.load('model_experiments/2025-12-04-collisions_only/gaussian_process_models/model.pth')
    additive_svgp = SparseAdditiveGPModel(torch.zeros((512, 10)), 10)
    additive_svgp.load_state_dict(saved_state)
    return (additive_svgp,)


@app.cell
def _(additive_svgp, plt):
    def plot_inducing_points():
        # Get points from model:
        inducing_points = additive_svgp.variational_strategy.inducing_points.detach().numpy()

        # Plot:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(inducing_points[:, 0], inducing_points[:, 1], s=5)
        ax.set_aspect("equal")
        plt.show()

    plot_inducing_points()
    return


@app.cell
def _(additive_svgp, parameter_matrix, torch):
    mv_normal = additive_svgp(torch.from_numpy(parameter_matrix[::11, :]).to(torch.float32))
    return (mv_normal,)


@app.cell
def _(mv_normal):
    mean_estimates = mv_normal.mean.detach().numpy()
    variance_estimate = mv_normal.variance.detach().numpy()
    return (mean_estimates,)


@app.cell
def _(coherency_fractions, mean_estimates, np, plt):
    def plot_estimates():
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(np.mean(coherency_fractions[::11, :], axis=1), mean_estimates, s=1)
        ax.plot([-1, 1], [-1, 1], c="r")
        ax.set_xlim(-0.025, 0.2)
        ax.set_ylim(-0.025, 0.2)
        ax.set_aspect("equal")
        plt.show()

    plot_estimates()
    return


@app.cell
def _(torch):
    testbeta = torch.distributions.beta.Beta(1, 2)
    return (testbeta,)


@app.cell
def _(testbeta):
    testbeta.cdf(0.5)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
