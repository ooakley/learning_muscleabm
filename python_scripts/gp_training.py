import argparse
import os

import gpytorch
import torch

import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from dppy.finite_dpps import FiniteDPP

# Need a higher precision for computing double derivatives:
torch.set_default_dtype(torch.float64)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a deep kernel GP regressor on a given model metric')
    parser.add_argument('--experiment_dirpath', type=str)
    parser.add_argument('--metric_name', type=str)
    args = parser.parse_args()
    return args


def load_gridsearch_data(experiment_dirpath, metric_name):
    # Load numpy data:
    parameter_matrix = np.load(
        os.path.join(experiment_dirpath, "sample_matrix.npy")
    )
    output_metric = np.load(
        os.path.join(experiment_dirpath, "summary_data", f"{metric_name}.npy")
    )

    # # Remove failed simulations:
    # nan_mask = np.any(np.isnan(output_metric), axis=1)
    # parameter_matrix = parameter_matrix[~nan_mask, :]
    # output_metric = output_metric[~nan_mask, :]
    return parameter_matrix, output_metric


def sample_inducing_points(num_points, parameter_matrix):
    # Get likelihood matrix:
    print("Getting squared exponential likelihood matrix...", flush=True)
    distance_matrix = 1 - np.matmul(parameter_matrix[::7, :], parameter_matrix[::7, :].T).astype(np.float32)
    likelihood_matrix = np.exp(distance_matrix ** 2)

    # Set up determinantal point process:
    print("Setting up point process...")
    DPP = FiniteDPP('likelihood', **{'L': likelihood_matrix})
    DPP.sample_mcmc_k_dpp(size=num_points, random_state=None)

    # Get inducing points:
    inducing_indices = DPP.list_of_samples[0][-1]
    inducing_points = parameter_matrix[::7, :][inducing_indices, :]
    return inducing_points


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


def main():
    # Parse arguments:
    args = parse_arguments()
    print(f"Using gridsearch {args.experiment_dirpath}...")
    print(f"Training on {args.metric_name}...")

    # Load gridsearch data:
    print("Loading data...")
    parameter_matrix, output_metric = load_gridsearch_data(
        args.experiment_dirpath, args.metric_name
    )

    # Whitening metric data:
    # --- Determine data format:
    if args.metric_name == "matrix_order_parameters":
        print(output_metric.shape, flush=True)
        parameter_mean = np.mean(output_metric[:, :, 2], axis=1)
        metric_mean = np.mean(parameter_mean)
        metric_std = np.std(parameter_mean)
        whitened_metric = (parameter_mean - metric_mean) / metric_std
    elif output_metric.shape[1] == 2:
        metric_mean = np.mean(output_metric[:, 0])
        metric_std = np.std(output_metric[:, 0])
        whitened_metric = (output_metric[:, 0] - metric_mean) / metric_std
    else:
        metric_mean = np.mean(np.mean(output_metric, axis=1))
        metric_std = np.std(np.mean(output_metric, axis=1))
        whitened_metric = (np.mean(output_metric, axis=1) - metric_mean) / metric_std

    # Get inducing points:
    print("Sampling inducing points...")
    inducing_points = sample_inducing_points(512, parameter_matrix)

    # Set up the model and the model's associated training apparatus:
    print("Setting up model...")
    model_manager = ModelManager(inducing_points, 0.003)

    # Train the model:
    print("Training model...")
    model_manager.train(parameter_matrix, whitened_metric, 512, epochs=20)
    print("Saving model...")
    model_manager.save(args.experiment_dirpath, args.metric_name)

    # # Shuffle datapoints prior to cross-validation (as otherwise there
    # # are weird autocorrelations from the Sobol' sampling):
    # rng = np.random.default_rng(0)
    # dataset_size = inducing_points.shape[0]
    # permuted_indices = rng.permutation(dataset_size)
    # permuted_parameters = parameter_matrix[permuted_indices, :]
    # permuted_target = np.mean(coherency_fractions, axis=1)[permuted_indices]

    # # Perform 8-fold cross validation:
    # mae_list = []
    # for k_index in range(K_FOLD_COUNT):
    #     # Set up the model and the model's associated training apparatus:
    #     model_manager = ModelManager(inducing_points, 0.003)

    #     # Get test indices:
    #     test_indices = np.arange(k_index, dataset_size, K_FOLD_COUNT)
    #     test_mask = np.zeros(dataset_size)
    #     test_mask[test_indices] = 1

    #     # Get training indices:
    #     train_mask = np.ones(dataset_size)
    #     train_mask[test_indices] = 0

    #     # Set up dataset:
    #     train_parameters = permuted_parameters[train_mask, :]
    #     train_target = permuted_target[train_mask]
    #     test_parameters = permuted_parameters[test_mask, :]
    #     test_target = permuted_target[test_mask]

    #     # Train and save model:
    #     model_manager.train(train_parameters, train_target, 512, epochs=25)
    #     # Test model:
    #     predictions, variance = run_inference(model_manager.model, model_manager.likelihood, test_parameters, 512)
    #     mean_absolute_error = np.abs(predictions - test_target) / len(predictions)
    #     mae_list.append(mean_absolute_error)

    #     # Save model:


if __name__ == "__main__":
    main()
