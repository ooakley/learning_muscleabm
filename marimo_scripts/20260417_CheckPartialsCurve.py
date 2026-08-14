import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import json

    import torch
    import gpytorch

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from torch.utils.data import TensorDataset, DataLoader
    return DataLoader, TensorDataset, gpytorch, np, os, plt, torch


@app.cell
def _():
    # model_experiments/2026-05-16-collisions_shape/run_data/0/379
    # model_experiments/2026-05-16-collisions_shape/run_data/0/380
    # model_experiments/2026-05-16-collisions_shape/run_data/0/712
    return


@app.cell
def _(torch):
    # Need a higher precision for computing double derivatives:
    torch.set_default_dtype(torch.float64)
    return


@app.cell
def _(DataLoader, TensorDataset, gpytorch, os, torch):
    class DeepInputTransformation(torch.nn.Module):
        def __init__(self, dimension, hidden_layer_neuron_count=16):
            # Run general initialisation of the nn.Module base class:
            super().__init__()

            # Record parameters:
            self.dimension = dimension
            self.hl_neuron_count = hidden_layer_neuron_count

            # Set up layers:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(dimension, self.hl_neuron_count),
                torch.nn.SiLU(),
                torch.nn.Linear(self.hl_neuron_count, self.hl_neuron_count),
                torch.nn.SiLU(),
                torch.nn.Linear(self.hl_neuron_count, dimension)
            )

            # Initialise weights:
            with torch.no_grad():
                self.apply(self.initialise)

        def forward(self, x):
            return self.mlp.forward(x)

        def initialise(self, m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)


    class SparseGPModel(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points, dimensions):
            # Set up distribution:
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(0)
            )

            # Set up variational strategy:
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution,
                learn_inducing_locations=True
            )

            # Inherit rest of init logic from approximate GP:
            super().__init__(variational_strategy)

            # Instantiate input transform:
            print(f"Using dimensions: {dimensions}")
            self.input_transform = DeepInputTransformation(dimensions)

            # Define mean and additive covariance functions:
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=dimensions)
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
            inducing_points = torch.tensor(inducing_points)
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.model = SparseGPModel(inducing_points, inducing_points.shape[1])

            # The default noise constraint sets the minimum too high,
            # we need the more permissive constraint of positivity:
            self.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Positive())

            # Set up optimisation - Adam seems to work best (need to properly test this):
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

            # Run through entire dataset:
            for batch_index, (x_batch, y_batch) in enumerate(dataloader):
                self.optimizer.zero_grad()
                output_distribution = self.model(x_batch)
                loss = -mll(output_distribution, y_batch)
                loss.backward()

                # Step through optimisers:
                self.optimizer.step()
                if (batch_index + 1) % 10 == 0:
                    print(batch_index, loss.item(), flush=True)

                # Ensure inducing points don't go out of bounds (implicitly
                # imposing constraints with transforms degrades performance):
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
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                self.train_epoch(dataloader, len(y))

        def save(self, experiment_dirpath, metric_name):
            # Generate GP model folder if not present:
            model_dirpath = os.path.join(experiment_dirpath, "gaussian_process_models")
            if not os.path.exists(model_dirpath):
                os.mkdir(model_dirpath)

            # Generate folder for given metric:
            metric_folderpath = os.path.join(model_dirpath, metric_name)
            if not os.path.exists(metric_folderpath):
                os.mkdir(metric_folderpath)

            # Save model components:
            model_filepath = os.path.join(metric_folderpath, "model.pth")
            torch.save(self.model, model_filepath)
            likelihood_filepath = os.path.join(metric_folderpath, "likelihood.pth")
            torch.save(self.likelihood, likelihood_filepath)
            optimiser_filepath = os.path.join(metric_folderpath, "optimiser.pth")
            torch.save(self.optimizer, optimiser_filepath)

        def load(self, experiment_dirpath, metric_name):
            id_folderpath = os.path.join(experiment_dirpath, "gaussian_process_models", metric_name)
            self.model = torch.load(os.path.join(id_folderpath, "model.pth"), weights_only=False)
            self.likelihood = torch.load(os.path.join(id_folderpath, "likelihood.pth"), weights_only=False)
            self.optimizer = torch.load(os.path.join(id_folderpath, "optimiser.pth"), weights_only=False)
    return (ModelManager,)


@app.cell
def _(np):
    parameter_matrix = np.load("model_experiments/2026-05-12-collisions_shape/sample_matrix.npy")
    speeds = np.load("model_experiments/2026-05-12-collisions_shape/summary_data/com_speeds.npy")
    coherencies = np.load("model_experiments/2026-05-12-collisions_shape/summary_data/com_coherency.npy")
    com_op = np.load("model_experiments/2026-05-12-collisions_shape/summary_data/com_order_parameters.npy")
    return coherencies, com_op, parameter_matrix, speeds


@app.cell
def _(coherencies, np, speeds):
    mean_speeds = np.mean(speeds, axis=1)
    mean_coherencies = np.mean(coherencies, axis=1)
    return mean_coherencies, mean_speeds


@app.cell
def _(coherencies, com_op, plt):
    plt.scatter(com_op, coherencies, s=1)
    return


@app.cell
def _(mean_speeds, plt):
    plt.hist(mean_speeds, bins=100);
    plt.show()
    return


@app.cell
def _(mean_coherencies, plt):
    plt.hist(mean_coherencies, bins=100);
    plt.show()
    return


@app.cell
def _(ModelManager, np, parameter_matrix):
    # Dummy points for instantiation:
    inducing_points = np.zeros((5, parameter_matrix.shape[1]))

    coherency_model_manager = ModelManager(inducing_points, 0.003)
    coherency_model_manager.load("model_experiments/2026-05-12-collisions_shape", "speeds")
    return (coherency_model_manager,)


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
def _(coherency_model_manager, emulate, np, parameter_matrix):
    cell_count_array = np.linspace(0.05, 0.95, 100)
    coherency_responses = []
    for i in range(1000):
        parameter_background = parameter_matrix[i, :]
        parameter_background = np.repeat(np.expand_dims(parameter_background, 0), len(cell_count_array), axis=0)
        parameter_background[:, 5] = cell_count_array
        means, _ = emulate(coherency_model_manager, parameter_background)
        coherency_responses.append(means)
    return (coherency_responses,)


@app.cell
def _(coherency_responses, np, plt):
    decreasing_mask = np.all(np.diff(coherency_responses, axis=1) < 0, axis=1)
    gradient_mask = np.mean(np.diff(coherency_responses, axis=1), axis=1) < -0.035
    # selection_mask = np.logical_and(decreasing_mask, gradient_mask)
    selection_mask = gradient_mask

    for idx in range(1000):
        # if np.mean(np.diff(coherency_responses[idx])) < -0.01:
        plt.plot(coherency_responses[idx], alpha=0.25)

    plt.show()
    return (selection_mask,)


@app.cell
def _():
    # def plot_increasing_responses():
    #     increasing_mask = np.all(np.diff(coherency_responses, axis=1) > 0, axis=1)
    #     gradient_mask = np.mean(np.diff(coherency_responses, axis=1), axis=1) > 0.02
    #     selection_mask = np.logical_and(increasing_mask, gradient_mask)

    #     for idx in range(20000):
    #         if np.mean(np.diff(coherency_responses[idx])) > 0.02:
    #             plt.plot(coherency_responses[idx], alpha=0.25)

    #     plt.show()
    #     return gradient_mask

    # selection_mask = plot_increasing_responses()
    return


@app.cell
def _(parameter_matrix, selection_mask):
    selected_parameters = parameter_matrix[:20000, :][selection_mask]
    return (selected_parameters,)


@app.cell
def _(plt, selected_parameters):
    plt.scatter(selected_parameters[:, 10], selected_parameters[:, 11], s=1)
    return


@app.cell
def _(np, plt, selected_parameters):
    def plot_parameter_distribution():
        fig, axs = plt.subplots(10, 10, figsize=(10, 10), layout='constrained')
        for i in range(10):
            for j in range(10):
                if i == j:
                    counts, _, _ = axs[i, j].hist(selected_parameters[:, i], bins=10, range=(0, 1), density=True)
                    axs[i, j].hlines(1.0, 0, 1, color='r')
                    y_lim = np.max(counts) + (np.max(counts) * 0.1)
                    print(y_lim)
                    axs[i, j].set_xlim(0, 1)
                    axs[i, j].set_ylim(0, y_lim)
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
                    # axs[i, j].set_axis_off()
                else:
                    axs[i, j].scatter(selected_parameters[:, i], selected_parameters[:, j], s=1)
                    axs[i, j].set_xlim(0, 1)
                    axs[i, j].set_ylim(0, 1)
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
                    # axs[i, j].set_axis_off()
        plt.show()

    plot_parameter_distribution()
    return


@app.cell
def _(np, plt):
    def test_com(var_name):
        test_base = np.load(f"model_experiments/2026-05-12-collisions_shape/run_data/0/3/{var_name}.npy")
        test_com = np.load(f"model_experiments/2026-05-12-collisions_shape/run_data/0/3/com_{var_name}.npy")
        print(test_base)
        print(test_com)
        plt.scatter(test_com, test_base)
        plt.show()

    test_com("ann_indices")
    return


@app.cell
def _(coherency_responses):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3, whiten=True)
    pca_emebddings = pca.fit_transform(coherency_responses)
    return (pca_emebddings,)


@app.cell
def _(np, parameter_matrix, pca_emebddings, plt):
    def plot_pca():
        fig, ax = plt.subplots(figsize=(4, 4))
        limit_value = np.max(np.abs(pca_emebddings[:, :2]))
        limit_value *= 1.1
        ax.scatter(pca_emebddings[:, 0], pca_emebddings[:, 1], s=0.1, c=parameter_matrix[:20000, 7])
        ax.set_xlim(-limit_value, limit_value)
        ax.set_ylim(-limit_value, limit_value)
        plt.show()

    plot_pca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
