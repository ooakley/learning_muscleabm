import os
# import functools
import subprocess

import torch.multiprocessing as multiprocessing

import psutil
import torch
import gpytorch

import numpy as np

from scipy.stats import qmc
from torch.utils.data import TensorDataset, DataLoader
from threadpoolctl import threadpool_limits

# Torch config management:
torch.set_default_dtype(torch.float64)
print(f"Initial pytorch thread assignment: {torch.get_num_threads()}")
print(f"Initial pytorch interop thread assignment: {torch.get_num_interop_threads()}")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print(f"Current pytorch thread assignment: {torch.get_num_threads()}")
print(f"Current pytorch interop thread assignment: {torch.get_num_interop_threads()}")


NUM_WORKERS = 60
PARAMETER_DIMENSION = 11


def get_global_cpu_affinity():
    # Retrieve current CPU affinity:
    taskset_output = subprocess.check_output(f"taskset -p {os.getpid()}", shell=True).decode("utf-8")

    # Truncate newline character, split, and retrieve hex string:
    hex_string = taskset_output[:-1].split(" ")[-1]

    # Convert hexadecimal to an integer
    hexademical_integer = int(hex_string, 16)
    bitmask = format(hexademical_integer, '0256b')

    # We have to reverse the bit mask so the numpy nonzero gives the right
    # indexing:
    global CPU_IDS
    CPU_IDS = np.nonzero(np.array(list(bitmask)[::-1], dtype=int))[0]
    print("Assigned CPU IDs:", flush=True)
    print(CPU_IDS, flush=True)


def initialise_worker():
    # Set affinity with pool worker ID:
    worker_name = multiprocessing.current_process().name
    pool_worker_id = int(worker_name.split("-")[-1]) - 1
    pool_worker_id = pool_worker_id % NUM_WORKERS

    # Use psutil to set affinity:
    psutil_process_interface = psutil.Process()
    psutil_process_interface.cpu_affinity([CPU_IDS[pool_worker_id]])
    print(f"Setting worker {pool_worker_id} affinity...", flush=True)


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
            # diagonal_index_object = range(min(m.weight.size()))
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


def get_hessians(inputs, cost_function):
    # Generate numpy tensor input data - torch tensors really don't play well being placed into imap:
    global numpy_inputs
    numpy_inputs = [np.expand_dims(numpy_data, axis=0) for numpy_data in np.unstack(inputs, axis=0)]

    # Generate partial Hessian calculation so we can pass to multiprocessing pool:
    global partial_hessian_function

    def partial_hessian_function(index):
        torch_input = torch.from_numpy(numpy_inputs[index])
        torch_hessian = torch.autograd.functional.hessian(cost_function, torch_input)
        return torch_hessian.detach().numpy()

    with threadpool_limits(limits=1, user_api='blas'):
        print("Initialising workers...", flush=True)
        with multiprocessing.Pool(processes=NUM_WORKERS, initializer=initialise_worker) as pool:
            print("Calculating Hessians...", flush=True)
            hessians = []
            for index, hessian in enumerate(pool.imap(partial_hessian_function, range(len(numpy_inputs)), 1)):
                if (index + 1) % 500 == 0:
                    print(index + 1, flush=True)
                hessians.append(hessian)

    # Convert to numpy and remove batch dimensions:
    hessians = [np.squeeze(hessian) for hessian in hessians]
    return np.stack(hessians, axis=0)


def main():
    print("Managing CPU affinity...")
    get_global_cpu_affinity()

    # Load Gaussian Process model:
    print("Loading model...", flush=True)
    global cf_model_manager
    cf_model_manager = ModelManager(np.ones((10, PARAMETER_DIMENSION)), 0.003)
    cf_model_manager.load("model_experiments/2026-01-26-matrix_collisions/gaussian_process_models", "op17")

    # Generate Sobol sequence for (relatively) even coverage of input space:
    print("Generating samples...", flush=True)
    sobol_sampler = qmc.Sobol(d=PARAMETER_DIMENSION, scramble=True, rng=0)
    sampled_inputs = sobol_sampler.random_base2(14)  # 32768 samples across dimensions.

    # Exclude edges of parameter space (Hessian estimation begins to break down):
    trim_factor = 0.02
    sampled_inputs *= 1 - (trim_factor * 2)
    sampled_inputs += trim_factor

    # Take natural log to get log curvature:
    sampled_inputs = np.log(sampled_inputs)

    # Globalise for multiprocessing:
    global localised_cost_function

    # Define function to get the Hessian of:
    def localised_cost_function(parameter_input):
        transformed_input = torch.exp(parameter_input)
        prediction = cf_model_manager.likelihood(cf_model_manager.model(transformed_input))
        prediction_mean = prediction.mean
        phantom_set_point = prediction_mean.detach()
        localised_cost = (phantom_set_point - prediction_mean) ** 2
        return localised_cost

    # Double backprop for Hessians:
    hessians = get_hessians(sampled_inputs, localised_cost_function)

    # Save inputs and Hessians:
    print("Saving outputs...", flush=True)
    dir_path = "model_experiments/2026-01-26-matrix_collisions/gaussian_process_models/op17"
    np.save(os.path.join(dir_path, "log_hessian_dataset.npy"), hessians)
    np.save(os.path.join(dir_path, "hessian_inputs.npy"), np.exp(sampled_inputs))


if __name__ == "__main__":
    main()
