"""Deep kernel GP emulators of the model metrics, and the tools for saving and loading them."""
import os
import pickle
import sys
import types

import gpytorch
import torch

import numpy as np

from torch.utils.data import TensorDataset, DataLoader


class DeepInputTransformation(torch.nn.Module):
    def __init__(self, dimension, hidden_layer_count=2, hidden_layer_neuron_count=16):
        # Run general initialisation of the nn.Module base class:
        super().__init__()

        # Record parameters:
        self.dimension = dimension
        self.hl_count = hidden_layer_count
        self.hl_neuron_count = hidden_layer_neuron_count

        # Set up layers - each hidden layer is a linear map followed by an activation,
        # with a final linear map back to the input dimension:
        layers = []
        input_count = dimension
        for _ in range(self.hl_count):
            layers.append(torch.nn.Linear(input_count, self.hl_neuron_count))
            layers.append(torch.nn.SiLU())
            input_count = self.hl_neuron_count
        layers.append(torch.nn.Linear(input_count, dimension))
        self.mlp = torch.nn.Sequential(*layers)

        # Initialise weights:
        with torch.no_grad():
            self.apply(self.initialise)

    def forward(self, x):
        return self.mlp.forward(x)

    def initialise(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)


class SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, dimensions, hidden_layer_count=2, hidden_layer_neuron_count=16):
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
        self.input_transform = DeepInputTransformation(
            dimensions, hidden_layer_count, hidden_layer_neuron_count
        )

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


class _MainToEmulatorsUnpickler(pickle.Unpickler):
    """Unpickler for models saved before the model classes moved into this module.

    Models are saved whole, so the pickle records the module of their classes. Models
    trained by a script that defined the classes itself point to __main__, so these
    are looked up here instead.
    """

    def find_class(self, module_name, name):
        if module_name == "__main__" and hasattr(sys.modules[__name__], name):
            module_name = __name__
        return super().find_class(module_name, name)


# torch.load takes a pickle module, from which it uses Unpickler and load:
_main_to_emulators_pickle = types.ModuleType("main_to_emulators_pickle")
_main_to_emulators_pickle.Unpickler = _MainToEmulatorsUnpickler
_main_to_emulators_pickle.load = lambda file, **kwargs: _MainToEmulatorsUnpickler(file, **kwargs).load()


def load_torch_object(filepath):
    """Load a model, likelihood or optimiser saved whole with torch.save."""
    return torch.load(filepath, weights_only=False, pickle_module=_main_to_emulators_pickle)


class ModelManager:

    def __init__(self, inducing_points, learning_rate, hidden_layer_count=2, hidden_layer_neuron_count=16):
        # Set up model:
        inducing_points = torch.tensor(inducing_points)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = SparseGPModel(
            inducing_points, inducing_points.shape[1], hidden_layer_count, hidden_layer_neuron_count
        )

        # The default noise constraint sets the minimum too high,
        # we need the more permissive constraint of positivity:
        self.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Positive())

        # Set up optimisation - Adam seems to work best (need to properly test this):
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=learning_rate)

        # Learning rate schedule, stepped after every epoch if set:
        self.scheduler = None

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
            if (batch_index + 1) % 32 == 0:
                print(batch_index + 1, loss.item(), flush=True)

            # Ensure inducing points don't go out of bounds (implicitly
            # imposing constraints with transforms degrades performance):
            with torch.no_grad():
                inducing_points = self.model.variational_strategy.inducing_points.detach()
                self.model.variational_strategy.inducing_points[inducing_points > 1] = 1
                self.model.variational_strategy.inducing_points[inducing_points < 0] = 0

            self.loss_history.append(loss.detach().numpy())

    def train(self, x, y, batch_size, epochs=1):
        # Convert datasets to pytorch:
        x_tensor = torch.tensor(x)
        y_tensor = torch.tensor(y)
        dataset = TensorDataset(x_tensor, y_tensor)

        for epoch_index in range(epochs):
            print(f"Training epoch {epoch_index + 1}...")
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            self.train_epoch(dataloader, len(y))
            if self.scheduler is not None:
                self.scheduler.step()

    def save(self, id_folderpath, save_optimiser=True):
        # Save model components:
        os.makedirs(id_folderpath, exist_ok=True)
        model_filepath = os.path.join(id_folderpath, "model.pth")
        torch.save(self.model, model_filepath)
        likelihood_filepath = os.path.join(id_folderpath, "likelihood.pth")
        torch.save(self.likelihood, likelihood_filepath)
        # The optimiser state is about twice the size of the model, and is only
        # needed to resume training:
        if save_optimiser:
            optimiser_filepath = os.path.join(id_folderpath, "optimiser.pth")
            torch.save(self.optimizer, optimiser_filepath)

    def load(self, id_folderpath):
        self.model = load_torch_object(os.path.join(id_folderpath, "model.pth"))
        self.likelihood = load_torch_object(os.path.join(id_folderpath, "likelihood.pth"))
        optimiser_filepath = os.path.join(id_folderpath, "optimiser.pth")
        if os.path.exists(optimiser_filepath):
            self.optimizer = load_torch_object(optimiser_filepath)

    def get_hyperparameters(self):
        """Learned GP hyperparameters, all in whitened output units."""
        with torch.no_grad():
            hyperparameters = {
                "likelihood_noise": self.likelihood.noise.item(),
                "outputscale": self.model.covar_module.outputscale.item(),
                "constant_mean": self.model.mean_module.constant.item(),
                "lengthscales": self.model.covar_module.base_kernel.lengthscale.detach().numpy().flatten(),
            }
        return hyperparameters


def get_experiment_model_folderpath(experiment_dirpath, metric_name):
    """Folder of the GP model of a metric trained directly on an experiment (not a wave).

    gp_noise_training.py saves to these folders, and the Hessian, intervention and
    single-wave MCMC scripts load from them.
    """
    return os.path.join(experiment_dirpath, "gaussian_process_models", metric_name)


def run_inference(model, likelihood, inputs, batch_size=512, standard_deviations=False):
    """Whitened predictive mean, and latent and total variance, of the GP at each input.

    The total variance includes the likelihood noise. Pass likelihood=None to skip it,
    in which case None is returned in its place. With standard_deviations=True, standard
    deviations are returned in place of the variances.
    """
    # Set up dataloading (iterating over a DataLoader draws from torch's random number
    # generator, even without shuffling, which the training scripts' later folds depend on):
    tensor_input = torch.tensor(inputs)
    inference_dataset = TensorDataset(tensor_input)
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

    # Shift to eval mode:
    model.eval()
    if likelihood is not None:
        likelihood.eval()

    # Set up outputs:
    mean_array = []
    latent_spread_array = []
    total_spread_array = []
    with torch.no_grad():
        for batch_index, inference_batch in enumerate(inference_loader):
            # Get latent and full posterior noise estimates:
            latent_preds = model(inference_batch[0])
            mean_array.append(latent_preds.mean.detach().numpy())
            if likelihood is not None:
                preds = likelihood(latent_preds)
                total_spread = preds.stddev if standard_deviations else preds.variance
                total_spread_array.append(total_spread.detach().numpy())
            latent_spread = latent_preds.stddev if standard_deviations else latent_preds.variance
            latent_spread_array.append(latent_spread.detach().numpy())

            # Report progress:
            if (batch_index + 1) % 64 == 0:
                print(batch_index + 1)

    total_spread = np.concatenate(total_spread_array) if likelihood is not None else None
    return np.concatenate(mean_array), np.concatenate(latent_spread_array), total_spread
