import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import json

    import torch
    import gpytorch

    import scipy.stats

    import colorcet as cc
    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt

    from torch.utils.data import TensorDataset, DataLoader
    return DataLoader, TensorDataset, cc, gpytorch, np, os, plt, torch


@app.cell
def _(np):
    from PIL import Image
    experiment_dirpath = "model_experiments/2026-05-20-collisions_shape"
    coherency = np.load("model_experiments/2026-05-16-collisions_shape/summary_data/coherency.npy")
    mean_coherency = np.nanmean(coherency, axis=1)
    speed = np.load("model_experiments/2026-05-16-collisions_shape/summary_data/speeds.npy")
    mean_speed = np.nanmean(speed, axis=1)
    # plot_indices = np.argsort(mean_coherency)
    # plot_index = plot_indices[:-482][-6]
    # plot_index = 380
    # print(np.count_nonzero(np.isnan(mean_coherency)))
    # print(np.nanmean(coherency, axis=1)[plot_index])

    low_speed_mask = np.logical_and(0.025 < mean_speed, mean_speed < 0.075)
    high_coherence_mask = mean_coherency > 0.375
    index_list = np.argwhere(np.logical_and(low_speed_mask, high_coherence_mask))
    index_list = [int(index[0]) for index in index_list]
    plot_index = index_list[0]
    plot_filepath = f"model_experiments/2026-05-16-collisions_shape/run_data/{int(np.floor(plot_index / 1000))}/{plot_index}/trajectory.png"
    return Image, index_list, mean_coherency, mean_speed, plot_filepath


@app.cell
def _(np):
    parameter_matrix = np.load("model_experiments/2026-05-16-collisions_shape/sample_matrix.npy")
    return (parameter_matrix,)


@app.cell
def _(index_list):
    print(index_list)
    return


@app.cell
def _(cc, mean_coherency, mean_speed, np, parameter_matrix, plt):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    cell_number_sort = np.argsort(parameter_matrix[:, 5])
    ax.scatter(
        mean_speed[cell_number_sort], mean_coherency[cell_number_sort],
        s=1, alpha=0.25, cmap=cc.m_CET_L20,
        c=parameter_matrix[cell_number_sort, 5]
    )
    # ax.set_xlim([0, 0.45])
    plt.show()
    # plt.scatter(mean_speed[plot_index], mean_coherency[plot_index], s=10, c='r')
    return


@app.cell
def _(Image, plot_filepath):
    Image.open(plot_filepath)
    return


@app.cell
def _(ann_indices, plt):
    plt.hist(ann_indices.flatten(), bins=100);
    plt.show()
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
def _(ModelManager, np, parameter_matrix):
    # Dummy points for instantiation:
    inducing_points = np.zeros((5, parameter_matrix.shape[1]))

    coherency_model_manager = ModelManager(inducing_points, 0.003)
    coherency_model_manager.load("model_experiments/2026-05-16-collisions_shape", "meander_ratios")
    return (coherency_model_manager,)


@app.cell
def _(np):
    from scipy.stats import qmc

    def generate_sobol_sequence(dimension, exponent, seed):
        sobol_sampler = qmc.Sobol(d=dimension, scramble=True, rng=0)
        sample_matrix = sobol_sampler.random_base2(m=exponent)
        return sample_matrix

    # Generate base parameter matrices:
    parameters_hyperspace = generate_sobol_sequence(24, 17, 0)
    parameters_A = parameters_hyperspace[:, :12]
    parameters_B = parameters_hyperspace[:, 12:]

    # Generating the combined parameter matrix:
    parameter_matrices = []
    for parameter_index in range(12):
        parameters_ABi = np.copy(parameters_B)
        parameters_ABi[:, parameter_index] = parameters_A[:, parameter_index]
        parameter_matrices.append(parameters_ABi)
    return parameter_matrices, parameters_A, parameters_B


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
def _(
    coherency_model_manager,
    emulate,
    parameter_matrices,
    parameters_A,
    parameters_B,
):
    f_A, _ = emulate(coherency_model_manager, parameters_A)
    f_B, _ = emulate(coherency_model_manager, parameters_B)

    f_Ai = []
    for i_parameters in parameter_matrices:
        f_Ai.append(emulate(coherency_model_manager, i_parameters)[0])
    return f_A, f_Ai, f_B


@app.cell
def _(f_A, f_Ai, f_B, np):
    def get_sobol_indices(dimension):
        Si_list = []
        STi_list = []
        for index in range(dimension):
            f_squared = np.mean(f_A) ** 2
            S_i = (np.dot(f_A, f_Ai[index]) - f_squared) / (np.dot(f_A, f_A) - f_squared) 
            S_Ti = 1 - ((np.dot(f_B, f_Ai[index]) - f_squared) / (np.dot(f_A, f_A) - f_squared))
            Si_list.append(S_i)
            STi_list.append(S_Ti)
        return Si_list, STi_list

    Si_list, STi_list = get_sobol_indices(12)
    return STi_list, Si_list


@app.cell
def _(Si_list):
    Si_list
    return


@app.cell
def _(STi_list):
    STi_list
    return


@app.cell
def _(Si_list, np):
    np.sum(Si_list)
    return


@app.cell
def _(STi_list, np):
    np.sum(STi_list)
    return


@app.cell
def _():
    # Run bootstrap estimates:
    return


if __name__ == "__main__":
    app.run()
