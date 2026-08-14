import os
import time
import argparse

import torch
import gpytorch

import numpy as np

from scipy.stats import qmc
from torch.utils.data import TensorDataset, DataLoader

EXPERIMENT_DIRPATH = "model_experiments/2026-06-03-matrix_shape"
PARAMETER_DIMENSION = 14
EXPONENT = 16

# Need a higher precision for computing double derivatives:
torch.set_default_dtype(torch.float64)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--task_id", type=int, required=True)
    return parser.parse_args()


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
        # print(f"Using dimensions: {dimensions}")
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


def generate_sobol_samples():
    # Set up sampler:
    sobol_sampler = qmc.Sobol(d=PARAMETER_DIMENSION, scramble=True, rng=0)
    sampled_inputs = sobol_sampler.random_base2(EXPONENT)

    # Exclude edges of parameter space (Hessian estimation begins to break down):
    trim_factor = 0.02
    sampled_inputs *= 1 - (trim_factor * 2)
    sampled_inputs += trim_factor
    return sampled_inputs


def retrieve_posterior_samples():
    wt_chain = np.load("model_experiments/2026-05-31-collisions_shape/mcmc_results/wt_mcmc_chain.npy")
    rd_chain = np.load("model_experiments/2026-05-31-collisions_shape/mcmc_results/rd_mcmc_chain.npy")
    wt_posterior = wt_chain[16384::256, :, 0, :].reshape(-1, 11)
    rd_posterior = rd_chain[16384::256, :, 0, :].reshape(-1, 11)
    sampled_inputs = np.concatenate([wt_posterior, rd_posterior], axis=0)
    sampled_inputs = np.concatenate([sampled_inputs, np.ones((sampled_inputs.shape[0], 3)) * 0.5], axis=1)
    return sampled_inputs


def retrieve_mle_samples():
    ctl_chain = np.load("model_experiments/2026-05-31-collisions_shape/mcmc_results/wt_mcmc_chain.npy")
    rd_chain = np.load("model_experiments/2026-05-31-collisions_shape/mcmc_results/rd_mcmc_chain.npy")
    ctl_likelihoods = np.load("model_experiments/2026-05-31-collisions_shape/mcmc_results/wt_mcmc_likelihoods.npy")
    rd_likelihoods = np.load("model_experiments/2026-05-31-collisions_shape/mcmc_results/rd_mcmc_likelihoods.npy")

    THIN_FACTOR = 64
    ctl_mle_idx = np.argsort(ctl_likelihoods[::THIN_FACTOR, :, 0].flatten())[-1024:]
    rd_mle_idx = np.argsort(rd_likelihoods[::THIN_FACTOR, :, 0].flatten())[-1024:]
    ctl_mle = ctl_chain[::THIN_FACTOR, :, 0, :].reshape(-1, 11)[ctl_mle_idx, :]
    rd_mle = rd_chain[::THIN_FACTOR, :, 0, :].reshape(-1, 11)[rd_mle_idx, :]
    sampled_inputs = np.concatenate([ctl_mle, rd_mle], axis=0)
    sampled_inputs = np.concatenate([sampled_inputs, np.ones((sampled_inputs.shape[0], 3)) * 0.5], axis=1)
    return sampled_inputs


def retrieve_sobol_samples():
    sobol_matrix = np.load(os.path.join(EXPERIMENT_DIRPATH, "sample_matrix.npy"))
    low_mask = sobol_matrix < 0.02
    high_mask = sobol_matrix > 0.98
    exclude_mask = np.logical_or(np.any(low_mask, axis=1), np.any(high_mask, axis=1))
    return sobol_matrix[~exclude_mask, :]


def main():
    args = parse_arguments()
    analysis_type = "SOBOL"

    # Generate outputs folder:
    if args.task_id == 0:
        print("Generating samples...", flush=True)
    if analysis_type == "FULL_GRID":
        dir_path = os.path.join(EXPERIMENT_DIRPATH, "gaussian_process_models", args.metric)
        sampled_inputs = generate_sobol_samples()
    elif analysis_type == "MLE":
        fit_source = "model_experiments/2026-05-31-collisions_shape"
        dir_path = os.path.join(fit_source, "mcmc_results", "mle_hessians")
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        sampled_inputs = retrieve_mle_samples()
    elif analysis_type == "SOBOL":
        dir_path = os.path.join(EXPERIMENT_DIRPATH, f"sobol_{args.metric}_hessian")
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        sampled_inputs = retrieve_sobol_samples()

    # Load Gaussian Process model:
    if args.task_id == 0:
        print("Loading model...", flush=True)
    model_manager = ModelManager(np.ones((10, PARAMETER_DIMENSION)), 0.003)
    model_manager.load(EXPERIMENT_DIRPATH, args.metric)

    # Get subsample of inputs to run in this instance:
    chunk_size = sampled_inputs.shape[0] // args.world_size
    start_index = args.task_id * chunk_size
    end_index = (args.task_id + 1) * chunk_size
    if args.task_id == (args.world_size - 1):
        chunk_inputs = np.copy(sampled_inputs[start_index:, :])
        print(f"Final chunk shape: {chunk_inputs.shape}", flush=True)
    else:
        chunk_inputs = np.copy(sampled_inputs[start_index:end_index, :])
        if args.task_id == 0:
            print(f"Initial chunk shape: {chunk_inputs.shape}", flush=True)

    # Convert to torch tensor:
    tensor_inputs = torch.from_numpy(chunk_inputs)

    # Take natural log to get log curvature:
    log_inputs = torch.log(tensor_inputs)

    # Define the function for which we want to retrieve the hessian:
    def localised_cost_function(parameter_input):
        row_input = torch.unsqueeze(parameter_input, 0)
        transformed_input = torch.exp(row_input)
        prediction = model_manager.likelihood(model_manager.model(transformed_input))
        prediction_mean = prediction.mean
        phantom_set_point = prediction_mean.detach()
        localised_cost = (phantom_set_point - prediction_mean) ** 2
        return localised_cost

    # Loop over inputs to retrieve the data:
    if args.task_id == 0:
        print("Estimating hessians...", flush=True)
    hessians = []
    for i in range(log_inputs.shape[0]):
        # Print progress:
        if args.task_id == 0:
            if (i + 1) % 32 == 0 :
                print(f"Processing index: {i + 1}", flush=True)
        # Estimate hessian:
        torch_hessian = torch.autograd.functional.hessian(
            localised_cost_function, log_inputs[i, :]
        )
        hessians.append(torch_hessian.detach().numpy())
    hessians = np.stack(hessians, axis=0)
    if args.task_id == 0:
        print(f"Chunked hessian shape: {hessians.shape}")

    # Save inputs and Hessians:
    if args.task_id == 0:
        print("Saving Hessian matrices...", flush=True)
    if args.task_id == 0:
        np.save(os.path.join(dir_path, "hessian_inputs.npy"), sampled_inputs)
    np.save(os.path.join(dir_path, f"hessian_{args.task_id}.npy"), hessians)

    # Estimate outputs of sampled inputs:
    if args.task_id == 0:
        print("Running predictions of input samples...", flush=True)
        full_inputs = torch.from_numpy(sampled_inputs)
        sample_predictions = model_manager.likelihood(model_manager.model(full_inputs))
        sample_predictions = sample_predictions.mean.detach().numpy()
        np.save(os.path.join(dir_path, "hessian_outputs.npy"), sample_predictions)

    # Collate Hessians if master process:
    if args.task_id == 0:
        # Wait while other processes finish:
        filename_list = [f"hessian_{task_id}.npy" for task_id in range(args.world_size)]
        filepath_list = [os.path.join(dir_path, filename) for filename in filename_list]
        subprocesses_completing = True
        while subprocesses_completing:
            print("Checking subprocesses...", flush=True)
            completion_list = [os.path.exists(filepath) for filepath in filepath_list]
            if all(completion_list):
                subprocesses_completing = False
                continue
            time.sleep(15)

        # Collate the Hessians from all processes:
        print("Subprocesses complete!", flush=True)
        time.sleep(30)  # Ensure all files are fully written to disk:
        hessians = [np.load(filepath) for filepath in filepath_list]
        hessians_array = np.concatenate(hessians, axis=0)

        # Save full array:
        print(f"Final Hessians shape: {hessians_array.shape}")
        print("Saving full array...", flush=True)
        np.save(os.path.join(dir_path, "hessian_estimate.npy"), hessians_array)

        # Remove sharded arrays:
        [os.remove(filepath) for filepath in filepath_list]


if __name__ == "__main__":
    main()
