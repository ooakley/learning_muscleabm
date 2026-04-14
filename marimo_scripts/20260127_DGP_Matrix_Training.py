import marimo

__generated_with = "0.19.7"
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
    from dppy.finite_dpps import FiniteDPP
    return (
        DataLoader,
        FiniteDPP,
        TensorDataset,
        gpytorch,
        json,
        np,
        os,
        plt,
        torch,
    )


@app.cell
def _(torch):
    torch.set_default_dtype(torch.float64)
    return


@app.cell
def _(json, np, os):
    def load_gridsearch_data(experiment_folderpath):
        # Load numpy data:
        parameter_matrix = np.load(
            os.path.join(experiment_folderpath, "sample_matrix.npy")
        )
        density_idr = np.load(
            os.path.join(experiment_folderpath, "summary_data", "density_idr.npy")
        )
        matrix_op = np.load(
            os.path.join(experiment_folderpath, "summary_data", "matrix_order_parameters.npy")
        )

        # Get gridsearch configuration:
        with open(os.path.join(experiment_folderpath, "config.json")) as json_filestream:
            config_dictionary  = json.load(json_filestream)
        gridsearch_parameters = config_dictionary["gridsearch_parameters"]

        return parameter_matrix, density_idr, matrix_op, gridsearch_parameters
    return (load_gridsearch_data,)


@app.cell
def _(load_gridsearch_data):
    parameter_matrix, density_idr, matrix_op, gridsearch_parameters = load_gridsearch_data(
        "model_experiments/2026-01-26-matrix_collisions"
    )
    return density_idr, matrix_op, parameter_matrix


@app.cell
def _(density_idr, np, plt):
    plt.hist(np.mean(density_idr, axis=1), bins=100);
    plt.show()
    return


@app.cell
def _(matrix_op, np, plt):
    plt.hist(np.mean(matrix_op[:, :, 0], axis=1), bins=100);
    plt.show()
    return


@app.cell
def _(density_idr, matrix_op, np, plt):
    plt.scatter(np.mean(matrix_op[:, :, 2], axis=1), np.mean(density_idr, axis=1), s=0.05)
    return


@app.cell
def _(DataLoader, TensorDataset, gpytorch, os, torch):
    class DeepInputTransformation(torch.nn.Module):
        def __init__(self, dimension, hidden_layer_neuron_count=32):
            # Run general initialisation of the nn.Module base class:
            super().__init__()

            # Record parameters:
            self.dimension = dimension
            self.hl_neuron_count = hidden_layer_neuron_count

            # Set up layers:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(dimension, self.hl_neuron_count),
                torch.nn.Tanh(),
                torch.nn.Linear(self.hl_neuron_count, self.hl_neuron_count),
                torch.nn.Tanh(),
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
                inducing_points
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
            id_folderpath = os.path.join(dirpath, "gaussian_process_models", id)
            self.model = torch.load(os.path.join(id_folderpath, "model.pth"), weights_only=False)
            self.likelihood = torch.load(os.path.join(id_folderpath, "likelihood.pth"), weights_only=False)
            self.optimizer = torch.load(os.path.join(id_folderpath, "optimiser.pth"), weights_only=False)
    return (ModelManager,)


@app.cell
def _(DataLoader, TensorDataset, np, plt, torch):
    def run_inference(model, likelihood, inputs, batch_size=512):
        # Set up dataloading:
        tensor_input = torch.tensor(inputs)
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


    def plot_variance(data_variance, predicted_variance):
        fig, ax = plt.subplots()
        ax.scatter(data_variance, predicted_variance, s=0.01)
        # Plot linear trend:
        upper_extent = np.max([data_variance, predicted_variance])
        lower_extent = np.min([data_variance, predicted_variance])
        ax.plot(
            [lower_extent, upper_extent],
            [lower_extent, upper_extent],
            c='r', alpha=0.5
        )
        ax.set_xlim(lower_extent, upper_extent)
        ax.set_ylim(lower_extent, upper_extent)
        ax.set_aspect("equal")
        plt.show()


    def plot_gp_mean_predictions(data_mean, predicted_mean, quantity):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(
            data_mean,
            predicted_mean,
            s=0.1
        )
        # Plot linear trend:
        upper_extent = np.max([data_mean, predicted_mean])
        lower_extent = np.min([data_mean, predicted_mean])
        ax.plot(
            [lower_extent, upper_extent],
            [lower_extent, upper_extent],
            c='r', alpha=0.5
        )
        ax.set_xlim(lower_extent, upper_extent)
        ax.set_xlabel(f"Simulation {quantity} values")
        ax.set_ylim(lower_extent, upper_extent)
        ax.set_ylabel(f"GP {quantity} values")
        ax.set_aspect("equal")
        plt.show()
    return plot_gp_mean_predictions, plot_variance, run_inference


@app.cell
def _(FiniteDPP, np, parameter_matrix):
    # Get likelihood matrix:
    print("Getting squared exponential likelihood matrix...", flush=True)
    distance_matrix = 1 - np.matmul(parameter_matrix[::7, :], parameter_matrix[::7, :].T)
    likelihood_matrix = np.exp(distance_matrix ** 2)

    # Set up determinantal point process:
    print("Setting up point process...")
    DPP = FiniteDPP('likelihood', **{'L': likelihood_matrix})

    k = 512
    DPP.sample_mcmc_k_dpp(size=k, random_state=None)

    # Get inducing points:
    inducing_indices = DPP.list_of_samples[0][-1]
    inducing_points = parameter_matrix[::7, :][inducing_indices, :]
    return (inducing_points,)


@app.cell
def _(ModelManager, inducing_points, matrix_op, np, parameter_matrix):
    # Get information necessary to transform GP inputs/outputs:
    OP17_DIST_MEAN = np.mean(np.mean(matrix_op[:, :, 1], axis=1))
    OP17_DIST_STD = np.std(np.mean(matrix_op[:, :, 1], axis=1))
    whitened_op17 = (np.mean(matrix_op[:, :, 1], axis=1) - OP17_DIST_MEAN) / OP17_DIST_STD
    op17_model_manager = ModelManager(inducing_points, 0.003)
    op17_model_manager.train(parameter_matrix, whitened_op17, 512, epochs=15)
    return op17_model_manager, whitened_op17


@app.cell
def _(op17_model_manager):
    op17_model_manager.save("model_experiments/2026-01-26-matrix_collisions/gaussian_process_models", "op17")
    # op17_model_manager.load("model_experiments/2026-01-26-matrix_collisions", "op17")
    return


@app.cell
def _(op17_model_manager, parameter_matrix, run_inference):
    op17_predictions, op17_variance = run_inference(op17_model_manager.model, op17_model_manager.likelihood, parameter_matrix)
    return (op17_predictions,)


@app.cell
def _(op17_predictions, plot_gp_mean_predictions, whitened_op17):
    plot_gp_mean_predictions(whitened_op17, op17_predictions, "OP17")
    return


@app.cell
def _(ModelManager, inducing_points, matrix_op, np):
    # Get information necessary to transform GP inputs/outputs:
    OP3_DIST_MEAN = np.mean(np.mean(matrix_op[:, :, 0], axis=1))
    OP3_DIST_STD = np.std(np.mean(matrix_op[:, :, 0], axis=1))
    whitened_op3 = (np.mean(matrix_op[:, :, 0], axis=1) - OP3_DIST_MEAN) / OP3_DIST_STD
    op3_model_manager = ModelManager(inducing_points, 0.003)
    # op3_model_manager.train(parameter_matrix, whitened_op3, 512, epochs=15)
    # cf_model_manager.save("model_experiments/2025-12-04-collisions_only/gaussian_process_models", "coherency_fraction")
    # op3_model_manager.load("model_experiments/2025-12-04-collisions_only/gaussian_process_models", "coherency_fraction")
    return OP3_DIST_MEAN, OP3_DIST_STD, op3_model_manager, whitened_op3


@app.cell
def _(OP3_DIST_MEAN):
    print(OP3_DIST_MEAN)
    return


@app.cell
def _(OP3_DIST_STD):
    print(OP3_DIST_STD)
    return


@app.cell
def _():
    # op3_model_manager.save("model_experiments/2026-01-26-matrix_collisions/gaussian_process_models", "op3")
    return


@app.cell
def _(op3_model_manager, parameter_matrix, run_inference):
    op3_predictions, op3_variance = run_inference(op3_model_manager.model, op3_model_manager.likelihood, parameter_matrix)
    return op3_predictions, op3_variance


@app.cell
def _(op3_predictions, plot_gp_mean_predictions, whitened_op3):
    plot_gp_mean_predictions(whitened_op3, op3_predictions)
    return


@app.cell
def _(OP3_DIST_STD, matrix_op, np, op3_variance, plot_variance):
    plot_variance(np.std(matrix_op[:, :, 0], axis=1) / np.sqrt(16), np.sqrt(op3_variance) * OP3_DIST_STD)
    return


@app.cell
def _(np, op3_model_manager, run_inference):
    # Get effect of matrix advection rate:
    test_parameters = np.ones((100, 11)) * 0.1
    test_parameters[:, 10] = np.linspace(0, 1, 100)
    op3_function, _ = run_inference(op3_model_manager.model, op3_model_manager.likelihood, test_parameters)
    return (op3_function,)


@app.cell
def _(op3_function, plt):
    plt.plot(op3_function)
    return


@app.cell
def _(ModelManager, density_idr, inducing_points, np, parameter_matrix):
    # Get information necessary to transform GP inputs/outputs:
    IDR_DIST_MEAN = np.mean(np.mean(density_idr, axis=1))
    IDR_DIST_STD = np.std(np.mean(density_idr, axis=1))
    whitened_idr = (np.mean(density_idr, axis=1) - IDR_DIST_MEAN) / IDR_DIST_STD
    idr_model_manager = ModelManager(inducing_points, 0.003)
    idr_model_manager.train(parameter_matrix, whitened_idr, 512, epochs=15)
    return idr_model_manager, whitened_idr


@app.cell
def _():
    # idr_model_manager.save("model_experiments/2026-01-26-matrix_collisions/gaussian_process_models", "density_idr")
    # idr_model_manager.load("model_experiments/2026-01-26-matrix_collisions/gaussian_process_models", "density_idr")
    return


@app.cell
def _(idr_model_manager, parameter_matrix, run_inference):
    idr_predictions, idr_variance = run_inference(idr_model_manager.model, idr_model_manager.likelihood, parameter_matrix)
    return (idr_predictions,)


@app.cell
def _(idr_predictions, plot_gp_mean_predictions, whitened_idr):
    plot_gp_mean_predictions(whitened_idr, idr_predictions)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
