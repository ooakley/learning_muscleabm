import os
import json

import torch
import gpytorch

import scipy.stats

import colorcet as cc
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import qmc
from torch.utils.data import TensorDataset, DataLoader
from statsmodels.regression import mixed_linear_model

torch.set_default_dtype(torch.float64)

MCMC_DIRPATH = "model_experiments/2026-09-16-collisions_shape"
MCMC_PARAM_DIMS = 12
MATRIX_DIRPATH = "model_experiments/2026-09-25-matrix_shape"
MATRIX_PARAM_DIMS = 15

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


class ScalingManager:

    def __init__(self, matrix_parameters, order_parameters, model_manager):
        # Get relevant scaling information:
        mean_metric = np.nanmean(order_parameters[:, :, 2], axis=1)
        self.metric_mean = np.mean(mean_metric)
        self.metric_std = np.std(mean_metric)

        # Set up parameter dictionary:
        gridsearch_dict = {metric_name: metric_range for metric_name, metric_range in matrix_parameters}

        # Get matrix advection rate scaling:    
        self.ar_min = gridsearch_dict["matrixCoupling"][0]
        self.ar_max = gridsearch_dict["matrixCoupling"][1]
        self.ar_range = self.ar_max - self.ar_min

        # Get matrix sample rate scaling:
        self.sr_min = gridsearch_dict["matrixSampleRate"][0]
        self.sr_max = gridsearch_dict["matrixSampleRate"][1]
        self.sr_range = self.sr_max - self.sr_min

        # Get cell number scaling:
        self.cell_no_min = gridsearch_dict["numberOfCells"][0]
        self.cell_no_max = gridsearch_dict["numberOfCells"][1]
        self.cell_no_range = self.cell_no_max - self.cell_no_min

        self.model_manager = model_manager

    def add_scaled_matrix_parameters(self, x, advection_rate, sample_rate, cell_number):
        # Get necessary size of array:
        sample_size = x.shape[0]

        # Scale advection rate:
        scaled_ar = (advection_rate - self.ar_min) / self.ar_range
        scaled_ar = np.ones((sample_size, 1)) * scaled_ar

        # Scale sample rate:
        scaled_sr = (sample_rate - self.sr_min) / self.sr_range
        scaled_sr = np.ones((sample_size, 1)) * scaled_sr
    
        # Scale sample rate:
        scaled_cell_no = (cell_number - self.cell_no_min) / self.cell_no_range
        scaled_cell_no = np.ones((sample_size, 1)) * scaled_cell_no
        return np.concatenate([x, scaled_ar, scaled_sr, scaled_cell_no], axis=1)

    def run_inference(self, x, advection_rate, sample_rate, cell_number):
        # Scale necessary parts of input:
        full_input = self.add_scaled_matrix_parameters(
            x, advection_rate, sample_rate, cell_number
        )

        # Run inference:
        tensor_input = torch.tensor(full_input)
        prediction = self.model_manager.likelihood(
            self.model_manager.model(tensor_input)
        )

        # Scale prediction:
        scaled_prediction = \
            (prediction.mean.detach().numpy() * self.metric_std) + self.metric_mean
        
        return scaled_prediction


def main():
    MCMC_SUBPATH = 'disc_cov_mcmc_results'

    # Establish save path:
    matrix_inference_dirpath = os.path.join(MCMC_DIRPATH, MCMC_SUBPATH, "matrix_inference")
    if not os.path.exists(matrix_inference_dirpath):
        os.mkdir(matrix_inference_dirpath)

    # Load posterior distribution:
    print("Loading posterior distributions...", flush=True)
    wt_chain = np.load(os.path.join(MCMC_DIRPATH, MCMC_SUBPATH, "wt_mcmc_chain.npy"))
    rd_chain = np.load(os.path.join(MCMC_DIRPATH, MCMC_SUBPATH, "rd_mcmc_chain.npy"))

    # Subsampled posterior:
    chain_length = wt_chain.shape[0]
    half_index = int(chain_length // 2)
    wt_posterior = wt_chain[half_index::256, :, 0, :].reshape(-1, MCMC_PARAM_DIMS)
    rd_posterior = rd_chain[half_index::256, :, 0, :].reshape(-1, MCMC_PARAM_DIMS)
    print(f"Size of posteriors: {wt_posterior.shape}")

    # Get subsampled likelihoods:
    ctl_likelihoods = np.load(os.path.join(MCMC_DIRPATH, MCMC_SUBPATH, "wt_mcmc_likelihoods.npy"))
    rd_likelihoods = np.load(os.path.join(MCMC_DIRPATH, MCMC_SUBPATH, "rd_mcmc_likelihoods.npy"))
    ctl_ss_likelihoods = ctl_likelihoods[half_index::256, :, 0].flatten()
    rd_ss_likelihoods = rd_likelihoods[half_index::256, :, 0].flatten()

    # # Get threshold:
    # index_threshold = int(np.floor(len(ctl_ss_likelihoods) * 0.05))
    # ctl_mask = np.argsort(ctl_ss_likelihoods)[index_threshold:]
    # rd_mask = np.argsort(rd_ss_likelihoods)[index_threshold:]
    # wt_posterior = wt_posterior[ctl_mask, :]
    # rd_posterior = rd_posterior[rd_mask, :]
    # print(f"Size of 95 CI posteriors: {wt_posterior.shape}")

    np.save(os.path.join(matrix_inference_dirpath, "wt_likelihood.npy"), ctl_ss_likelihoods)
    np.save(os.path.join(matrix_inference_dirpath, "rd_likelihood.npy"), rd_ss_likelihoods)
    np.save(os.path.join(matrix_inference_dirpath, "wt_posterior.npy"), wt_posterior)
    np.save(os.path.join(matrix_inference_dirpath, "rd_posterior.npy"), rd_posterior)

    # Load GP model for matrix organisation:
    dummy_inducing_points = np.ones((16, MATRIX_PARAM_DIMS))
    op65_manager = ModelManager(dummy_inducing_points, 0.003)
    op65_manager.load(MATRIX_DIRPATH, "op65")

    # # Generate Sobol sequence for (relatively) even coverage of input space (for mesh plotting):
    # sobol_sampler = qmc.Sobol(d=14, scramble=True, rng=0)
    # meshgrid_inputs = sobol_sampler.random_base2(19)
    # predictions = op65_manager.likelihood(
    #     op65_manager.model(torch.from_numpy(meshgrid_inputs))
    # ).mean.detach().numpy()
    # np.save(os.path.join(matrix_inference_dirpath, "meshgrid_inputs.npy"),  meshgrid_inputs)
    # np.save(os.path.join(matrix_inference_dirpath, "meshgrid_predictions.npy"), predictions)

    # Get relevant scaling information:
    with open(os.path.join(MATRIX_DIRPATH, "config.json")) as json_file:
        matrix_config_dictionary = json.load(json_file)
    matrix_parameters = matrix_config_dictionary["gridsearch_parameters"]
    order_parameters = np.load(os.path.join(MATRIX_DIRPATH, "summary_data", "matrix_order_parameters.npy"))
    scaling_manager = ScalingManager(matrix_parameters, order_parameters, op65_manager)

    # Set up gridsearch over advection rate, sample rate and cell number:
    print("Setting up grid for inference...", flush=True)
    MC_SAMPLE_COUNT = 10
    SR_SAMPLE_COUNT = 10
    CC_SAMPLE_COUNT = 10
    matrix_coupling_values = np.linspace(0, 15, MC_SAMPLE_COUNT)
    sample_rates = np.linspace(0.5, 15, SR_SAMPLE_COUNT)
    cell_counts = np.linspace(50, 350, CC_SAMPLE_COUNT)

    print("Running inference...", flush=True)
    wt_posterior_array = []
    rd_posterior_array = []
    for cell_count in cell_counts:
        for sample_rate in sample_rates:
            for mc_val in matrix_coupling_values:
                # Run inference on (thinned) full posterior:
                wt_full_pred = scaling_manager.run_inference(
                    wt_posterior, mc_val, sample_rate, cell_count
                )
                rd_full_pred = scaling_manager.run_inference(
                    rd_posterior, mc_val, sample_rate, cell_count
                )

                # Take average:
                wt_posterior_array.append(wt_full_pred)
                rd_posterior_array.append(rd_full_pred)

        # Print progress:
        print(f"Cell count {cell_count} completed...", flush=True)

    # Reshape and save:
    print("Reshaping and saving...", flush=True)
    wt_posterior_array = np.array(wt_posterior_array).reshape(CC_SAMPLE_COUNT, SR_SAMPLE_COUNT, MC_SAMPLE_COUNT, -1)
    rd_posterior_array = np.array(rd_posterior_array).reshape(CC_SAMPLE_COUNT, SR_SAMPLE_COUNT, MC_SAMPLE_COUNT, -1)

    matrix_inference_dirpath = os.path.join(MCMC_DIRPATH, MCMC_SUBPATH, "matrix_inference")
    if not os.path.exists(matrix_inference_dirpath):
        os.mkdir(matrix_inference_dirpath)

    np.save(os.path.join(matrix_inference_dirpath, "wt_posterior_array.npy"), wt_posterior_array)
    np.save(os.path.join(matrix_inference_dirpath, "rd_posterior_array.npy"), rd_posterior_array)


if __name__ == "__main__":
    main()
