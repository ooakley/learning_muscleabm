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
    return DataLoader, TensorDataset, gpytorch, json, np, os, torch


@app.cell
def _(torch):
    torch.set_default_dtype(torch.float64)
    return


@app.cell(hide_code=True)
def _(DataLoader, TensorDataset, gpytorch, json, np, os, torch):
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
        speeds = np.load(
            os.path.join(experiment_folderpath, "summary_data", "magnitude_cellmeans.npy")
        )

        # Get gridsearch configuration:
        with open(os.path.join(experiment_folderpath, "config.json")) as json_filestream:
            config_dictionary  = json.load(json_filestream)
        gridsearch_parameters = config_dictionary["gridsearch_parameters"]

        return parameter_matrix, coherency_fractions, ann_indices, speeds, gridsearch_parameters


    class DeepInputTransformation(torch.nn.Module):
        def __init__(self, dimension, hidden_layer_neuron_count=64):
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
                gpytorch.kernels.RBFKernel(ard_num_dims=10)
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
            x_tensor = torch.tensor(x, dtype=torch.float64)
            y_tensor = torch.tensor(y, dtype=torch.float64)
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


    def run_inference(model, likelihood, inputs, batch_size=512):
        # Set up dataloading:
        tensor_input = torch.tensor(inputs, dtype=torch.float64)
        inference_dataset = TensorDataset(tensor_input)
        inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

        # Shift to eval mode:
        model.eval()
        likelihood.eval()

        # Set up outputs:
        predictions_array = []
        variance_array = []
        with torch.no_grad():
            for batch_index, inference_batch in enumerate(inference_loader):
                inference_batch = inference_batch[0]
                predictions = likelihood(model(inference_batch))
                predictions_array.append(predictions.mean.detach().numpy())
                variance_array.append(predictions.variance.detach().numpy())
                if (batch_index + 1) % 100 == 0:
                    print(batch_index + 1)

        return np.concatenate(predictions_array), np.concatenate(variance_array)

    def emulate(manager, x):
        tensor_input = torch.tensor(x, dtype=torch.float32)
        if len(tensor_input.shape) == 1:
            tensor_input = torch.unsqueeze(tensor_input, 0)
        with torch.no_grad():
            prediction = manager.likelihood(manager.model(tensor_input))
            prediction_mean = prediction.mean.detach().numpy()
            prediction_variance = prediction.variance.detach().numpy()
        return prediction_mean, prediction_variance
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
