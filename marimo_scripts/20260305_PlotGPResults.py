import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import json
    import copy

    import gpytorch
    import torch

    import numpy as np
    import matplotlib.pyplot as plt

    from torch.utils.data import TensorDataset, DataLoader
    from dppy.finite_dpps import FiniteDPP

    import colorcet as cc
    return DataLoader, TensorDataset, gpytorch, json, np, os, plt, torch


@app.cell
def _():
    EXPERIMENT_FOLDERPATH = "model_experiments/2026-03-02-collisions_only"
    return (EXPERIMENT_FOLDERPATH,)


@app.cell
def _(torch):
    torch.set_default_dtype(torch.float64)
    return


@app.cell(hide_code=True)
def _(DataLoader, TensorDataset, gpytorch, os, torch):
    class DeepInputTransformation(torch.nn.Module):
        def __init__(self, dimension, hidden_layer_neuron_count=150):
            # Run general initialisation of the nn.Module base class:
            super().__init__()

            # Record parameters:
            self.dimension = dimension
            self.hl_neuron_count = hidden_layer_neuron_count

            # Set up layers:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(dimension, self.hl_neuron_count),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hl_neuron_count, self.hl_neuron_count),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hl_neuron_count, 10)
            )

            # Initialise weights:
            with torch.no_grad():
                self.apply(self.initialise)

        def forward(self, x):
            return self.mlp.forward(x)

        def initialise(self, m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                diagonal_index_object = range(min(m.weight.size()))
                # m.weight[diagonal_index_object, diagonal_index_object] = 1


    class SparseGPModel(gpytorch.models.ApproximateGP):
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

            # Inherit rest of init logic from approximate GP:
            super().__init__(variational_strategy)

            # Instantiate input transform:
            self.input_transform = DeepInputTransformation(dimensions)

            # Define mean and additive covariance functions:
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=None)
            )

        def forward(self, x):
            # Warp input:
            warped_x = self.input_transform(x)

            # Calculate mean of input:
            mean_x = self.mean_module(warped_x)
            covar_x = self.covar_module(warped_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    class ModelManager:

        def __init__(self, inducing_points, learning_rate):
            # Set up model:
            inducing_points = torch.tensor(
                inducing_points, dtype=torch.float64
            )
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.model = SparseGPModel(inducing_points, inducing_points.shape[1])

            # The default noise constraint is too high, more permissive constraint of positivity:
            self.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Positive())

            # Set up optimisation:
            self.optimizer = torch.optim.Adam([
                {'params': self.model.parameters()},
                {'params': self.likelihood.parameters()},
            ], lr=learning_rate)

            self.loss_history = []

        def train_epoch(self, dataloader, num_data):
            # Ensure parameters are trainable:
            self.model.train()
            self.likelihood.train()

            # Set up loss:
            mll = gpytorch.mlls.PredictiveLogLikelihood(self.likelihood, self.model, num_data=num_data)
            loss_history = []

            # Run through entire dataset:
            for batch_index, (x_batch, y_batch) in enumerate(dataloader):
                self.optimizer.zero_grad()
                output_distribution = self.model(x_batch)
                loss = -mll(output_distribution, y_batch)
                loss.backward()

                # Step through optimisers:
                self.optimizer.step()
                if (batch_index + 1) % 64 == 0:
                    print(batch_index, loss.item())

                # Ensure inducing points don't go out of bounds:
                with torch.no_grad():
                    inducing_points = self.model.variational_strategy.inducing_points.detach()
                    self.model.variational_strategy.inducing_points[inducing_points > 1] = 1
                    self.model.variational_strategy.inducing_points[inducing_points < 0] = 0

                self.loss_history.append(loss.detach())

        def train(self, x, y, batch_size, epochs=1):
            # Convert datasets to pytorch:
            x_tensor = torch.tensor(x)
            y_tensor = torch.tensor(y)
            dataset = TensorDataset(x_tensor, y_tensor)

            for _ in range(epochs):
                print(f"---> Epoch {_ + 1}...")
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                self.train_epoch(dataloader, len(y))

        def save(self, dirpath, id):
            # Generate save folder:
            id_folderpath = os.path.join(dirpath, id)
            if not os.path.exists(id_folderpath):
                os.mkdir(id_folderpath)

            # Save model components:
            model_filepath = os.path.join(id_folderpath, "model.pth")
            torch.save(self.model, model_filepath)
            likelihood_filepath = os.path.join(id_folderpath, "likelihood.pth")
            torch.save(self.likelihood, likelihood_filepath)
            optimiser_filepath = os.path.join(id_folderpath, "optimiser.pth")
            torch.save(self.optimizer, optimiser_filepath)

        def load(self, dirpath, id):
            id_folderpath = os.path.join(dirpath, id)
            self.model = torch.load(os.path.join(id_folderpath, "model.pth"), weights_only=False)
            self.likelihood = torch.load(os.path.join(id_folderpath, "likelihood.pth"), weights_only=False)
            self.optimizer = torch.load(os.path.join(id_folderpath, "optimiser.pth"), weights_only=False)
    return (ModelManager,)


@app.cell
def _(EXPERIMENT_FOLDERPATH, json, np, os):
    # Load input data:
    parameter_matrix = np.load(
        os.path.join(EXPERIMENT_FOLDERPATH, "sample_matrix.npy")
    )

    # Load metrics
    coherency_fractions = np.load(
        os.path.join(EXPERIMENT_FOLDERPATH, "summary_data", "coherency_fractions.npy")
    )
    ann_indices = np.load(
        os.path.join(EXPERIMENT_FOLDERPATH, "summary_data", "ann_indices.npy")
    )
    magnitude_cellmeans = np.load(
        os.path.join(EXPERIMENT_FOLDERPATH, "summary_data", "magnitude_cellmeans.npy")
    )
    meander_ratios = np.load(
        os.path.join(EXPERIMENT_FOLDERPATH, "summary_data", "meander_ratios.npy")
    )

    # Get gridsearch configuration:
    with open(os.path.join(EXPERIMENT_FOLDERPATH, "config.json")) as json_filestream:
        config_dictionary  = json.load(json_filestream)
    gridsearch_parameters = config_dictionary["gridsearch_parameters"]
    return gridsearch_parameters, magnitude_cellmeans, parameter_matrix


@app.cell
def _(gridsearch_parameters):
    gridsearch_parameters.keys()
    return


@app.cell
def _(magnitude_cellmeans):
    broken_sets = magnitude_cellmeans[:, 0] > 10
    return (broken_sets,)


@app.cell
def _(broken_sets, parameter_matrix, plt):
    def plot_param_distribution():
        fig, axs = plt.subplots(10, 10, figsize=(10, 10), layout="constrained")

        for i in range(10):
            for j in range(10):
                ax = axs[i, j]
                if i == j:
                    # parameter_label = list(gridsearch_parameters.keys())[i]
                    # ax.text(
                    #     0.5, 0.5, parameter_label,
                    #     fontsize=5, horizontalalignment="center",
                    #     rotation=45, rotation_mode="anchor"
                    # )
                    # ax.set_aspect("equal")
                    # ax.set_xticks([])
                    # ax.set_yticks([])
                    # ax.set_axis_off()
                    continue

                ax.scatter(parameter_matrix[broken_sets, i], parameter_matrix[broken_sets, j], s=1, c="tab:orange")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect("equal")

        # fig.tight_layout()
        plt.show()
    return (plot_param_distribution,)


@app.cell
def _(plot_param_distribution):
    plot_param_distribution()
    return


@app.cell
def _(EXPERIMENT_FOLDERPATH, ModelManager, magnitude_cellmeans, np, os):
    # Instantiate then load emulators:
    inducing_points = np.ones((10, 10))
    gp_dirpath = os.path.join(EXPERIMENT_FOLDERPATH, "gaussian_process_models")

    model_manager = ModelManager(inducing_points, 0.003)
    model_manager.load(gp_dirpath, "magnitude_cellmeans")
    output_metric = magnitude_cellmeans
    return model_manager, output_metric


@app.cell
def _(torch):
    def emulate(manager, x):
        manager.likelihood.eval()
        manager.model.eval()
        tensor_input = torch.tensor(x)
        if len(tensor_input.shape) == 1:
            tensor_input = torch.unsqueeze(tensor_input, 0)
        with torch.no_grad():
            prediction = manager.likelihood(manager.model(tensor_input))
            prediction_mean = prediction.mean.detach().numpy()
            prediction_std = prediction.stddev.detach().numpy()
        return prediction_mean, prediction_std
    return (emulate,)


@app.cell
def _(emulate, model_manager, parameter_matrix):
    gp_out, _ = emulate(model_manager, parameter_matrix)
    return (gp_out,)


@app.cell
def _(np):
    def get_whiteners(output_metric):
        if output_metric.shape[1] == 2:
            metric_mean = np.mean(output_metric[:, 0])
            metric_std = np.std(output_metric[:, 0])
        else:
            metric_mean = np.mean(np.mean(output_metric, axis=1))
            metric_std = np.std(np.mean(output_metric, axis=1))
        return metric_mean, metric_std
    return (get_whiteners,)


@app.cell
def _(get_whiteners, gp_out, output_metric):
    metric_mean, metric_std = get_whiteners(output_metric)
    gp_predictions = (gp_out * metric_std) + metric_mean
    return (gp_predictions,)


@app.cell
def _(gp_predictions, np):
    np.max(gp_predictions)
    return


@app.cell
def _(gp_predictions, np, output_metric, plt):
    def plot_fit(simulation_data, gp_data):
        # Set up plot:
        fig, ax = plt.subplots(layout="constrained")
        lower_bound = np.min([np.min(simulation_data), np.min(gp_data)])
        upper_bound = np.max([np.max(simulation_data), np.max(gp_data)])

        # Add breathing room:
        range = upper_bound - lower_bound
        spacing = 0.025 * range
        lower_bound -= spacing
        upper_bound += spacing

        # Plot data:
        ax.scatter(simulation_data, gp_data, s=0.1)
        ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], c='r')
        ax.set_xlim([lower_bound, upper_bound])
        ax.set_ylim([lower_bound, upper_bound])
        ax.set_aspect("equal")
        plt.show()

    plot_fit(np.mean(output_metric, axis=1), gp_predictions)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
