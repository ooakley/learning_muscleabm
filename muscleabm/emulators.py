"""Deep kernel GP emulators of the model metrics, and the tools for saving and loading them."""
import copy
import os
import pickle
import sys
import types

import gpytorch
import torch

import numpy as np

from torch.nn.utils import parametrize
from torch.utils.data import TensorDataset, DataLoader

# Settings of the residual (DUE) deep kernel, used when a model is built with due=True. The
# residual branch is contractive, Lip(g) < 1, so the warp x + g(x) is invertible and
# bi-Lipschitz (Behrmann et al. 2019, https://arxiv.org/abs/1811.00995, Theorem 1 and
# Lemma 2), which stops the warp collapsing distinct inputs together (van Amersfoort et al.
# 2021, https://arxiv.org/abs/2102.11409). Each linear layer's spectral norm is capped at the
# coefficient, and the activation is 1-Lipschitz:
DUE_SPECTRAL_COEFFICIENT = 0.95
DUE_ACTIVATION = torch.nn.ELU


class SpectralCap(torch.nn.Module):
    """Parametrization capping a weight's spectral norm at a coefficient: W / max(1, sigma / c).

    This is the rule of Behrmann et al. 2019 (their Equation 2), but with the exact spectral norm
    rather than a power iteration estimate, which is cheap for layers this small, so the cap holds
    at every step.
    """

    def __init__(self, coefficient):
        super().__init__()
        self.coefficient = coefficient

    def forward(self, weight):
        sigma = torch.linalg.matrix_norm(weight, ord=2)
        return weight / torch.clamp(sigma / self.coefficient, min=1.0)


class DeepInputTransformation(torch.nn.Module):
    def __init__(
        self, dimension, hidden_layer_count=2, hidden_layer_neuron_count=16,
        activation=torch.nn.SiLU, spectral_cap=None
    ):
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
            layers.append(activation())
            input_count = self.hl_neuron_count
        layers.append(torch.nn.Linear(input_count, dimension))
        self.mlp = torch.nn.Sequential(*layers)

        # Initialise weights:
        with torch.no_grad():
            self.apply(self.initialise)

        # Cap the spectral norm of every linear layer, if set:
        if spectral_cap is not None:
            for layer in self.mlp:
                if isinstance(layer, torch.nn.Linear):
                    parametrize.register_parametrization(layer, "weight", SpectralCap(spectral_cap))

    def forward(self, x):
        return self.mlp.forward(x)

    def initialise(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)


class ResidualInputTransformation(torch.nn.Module):
    """One residual block, x + g(x), around a deep input transformation with capped spectral norms.

    With ELU (Lipschitz constant 1) and every linear layer capped at DUE_SPECTRAL_COEFFICIENT,
    Lip(g) <= DUE_SPECTRAL_COEFFICIENT ** (hidden_layer_count + 1), so distances between inputs
    are scaled by a factor between 1 - Lip(g) and 1 + Lip(g) (Behrmann et al. 2019, Lemma 2).
    """

    def __init__(self, dimension, hidden_layer_count=2, hidden_layer_neuron_count=16):
        super().__init__()
        self.branch = DeepInputTransformation(
            dimension, hidden_layer_count, hidden_layer_neuron_count,
            activation=DUE_ACTIVATION, spectral_cap=DUE_SPECTRAL_COEFFICIENT
        )

    def forward(self, x):
        return x + self.branch(x)


class SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(
        self, inducing_points, dimensions, hidden_layer_count=2, hidden_layer_neuron_count=16,
        due=False
    ):
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

        # Instantiate input transform, with a residual connection if set:
        if due:
            self.input_transform = ResidualInputTransformation(
                dimensions, hidden_layer_count, hidden_layer_neuron_count
            )
        else:
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


def _without_parametrizations(model):
    """The model, or if any of its linear layers are parametrized (spectrally capped), a copy
    of it with those layers replaced by plain linear layers.

    torch cannot save a parametrized module whole, only its state_dict. The plain layers hold the
    capped weights the model uses, so the copy gives the same predictions, and loads
    with load_torch_object like any other model. (torch's remove_parametrizations cannot be used
    on the copy, as it edits the parametrized class, which the copy shares with the model.)
    """
    if not any(parametrize.is_parametrized(module) for module in model.modules()):
        return model
    model_copy = copy.deepcopy(model)
    parametrized_layers = [
        (parent, layer_name, layer)
        for parent in model_copy.modules()
        for layer_name, layer in parent.named_children()
        if parametrize.is_parametrized(layer)
    ]
    for parent, layer_name, layer in parametrized_layers:
        if not isinstance(layer, torch.nn.Linear):
            raise TypeError(f"Cannot save parametrized {type(layer).__name__} layers, only linear layers.")
        with torch.no_grad():
            weight = layer.weight
            plain_layer = torch.nn.Linear(
                layer.in_features, layer.out_features, bias=layer.bias is not None,
                device=weight.device, dtype=weight.dtype
            )
            plain_layer.weight.copy_(weight)
            if layer.bias is not None:
                plain_layer.bias.copy_(layer.bias)
        setattr(parent, layer_name, plain_layer)
    return model_copy


def load_torch_object(filepath):
    """Load a model, likelihood or optimiser saved whole with torch.save."""
    return torch.load(filepath, weights_only=False, pickle_module=_main_to_emulators_pickle)


class ModelManager:

    def __init__(
        self, inducing_points, learning_rate, hidden_layer_count=2, hidden_layer_neuron_count=16,
        due=False
    ):
        # Set up model:
        inducing_points = torch.tensor(inducing_points)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = SparseGPModel(
            inducing_points, inducing_points.shape[1], hidden_layer_count, hidden_layer_neuron_count, due
        )

        # The default noise constraint sets the minimum too high,
        # we need the more permissive constraint of positivity:
        self.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Positive())

        # Set up optimisation - Adam seems to work best (need to properly test this):
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=learning_rate)

        # Learning rate schedule, if set: stepped after every epoch by train, and after every
        # optimiser step by train_steps:
        self.scheduler = None

        self.loss_history = []

    def _training_step(self, mll, x_batch, y_batch):
        """One optimiser step on a minibatch."""
        self.optimizer.zero_grad()
        output_distribution = self.model(x_batch)
        loss = -mll(output_distribution, y_batch)
        loss.backward()
        self.optimizer.step()

        # Ensure inducing points don't go out of bounds (implicitly
        # imposing constraints with transforms degrades performance):
        with torch.no_grad():
            inducing_points = self.model.variational_strategy.inducing_points.detach()
            self.model.variational_strategy.inducing_points[inducing_points > 1] = 1
            self.model.variational_strategy.inducing_points[inducing_points < 0] = 0

        self.loss_history.append(loss.detach().numpy())
        return loss

    def _prepare_training(self, x, y):
        # Ensure parameters are trainable, and set up the dataset and loss:
        self.model.train()
        self.likelihood.train()
        dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
        mll = gpytorch.mlls.PredictiveLogLikelihood(self.likelihood, self.model, num_data=len(y))
        return dataset, mll

    def train(self, x, y, batch_size, epochs=1):
        """Train for a number of passes through the data."""
        dataset, mll = self._prepare_training(x, y)
        for epoch_index in range(epochs):
            print(f"Training epoch {epoch_index + 1}...")
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for batch_index, (x_batch, y_batch) in enumerate(dataloader):
                loss = self._training_step(mll, x_batch, y_batch)
                if (batch_index + 1) % 32 == 0:
                    print(batch_index + 1, loss.item(), flush=True)
            if self.scheduler is not None:
                self.scheduler.step()

    def train_steps(self, x, y, batch_size, step_count):
        """Train for a number of optimiser steps, whatever the size of the data.

        Minibatches are drawn from shuffled passes through the data, reshuffled each time the
        data runs out, and training stops after step_count steps, part-way through a pass if
        need be.
        """
        dataset, mll = self._prepare_training(x, y)
        steps_per_epoch = -(-len(y) // batch_size)
        print(f"Training for {step_count} steps ({step_count / steps_per_epoch:.2f} epochs)...")
        step_index = 0
        while step_index < step_count:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for x_batch, y_batch in dataloader:
                loss = self._training_step(mll, x_batch, y_batch)
                if self.scheduler is not None:
                    self.scheduler.step()
                step_index += 1
                if step_index % 32 == 0:
                    print(step_index, loss.item(), flush=True)
                if step_index == step_count:
                    break

    def save(self, id_folderpath, save_optimiser=True):
        # Clear the linear operators gpytorch caches from the last forward pass, which it does
        # whenever the mode is set (the model is already in training mode). After a full-size
        # batch, the cache holds an object that cannot be pickled, so saving failed:
        self.model.train()

        # Save model components:
        os.makedirs(id_folderpath, exist_ok=True)
        model_filepath = os.path.join(id_folderpath, "model.pth")
        torch.save(_without_parametrizations(self.model), model_filepath)
        likelihood_filepath = os.path.join(id_folderpath, "likelihood.pth")
        torch.save(self.likelihood, likelihood_filepath)
        # The optimiser state is about twice the size of the model, and is only
        # needed to resume training:
        if save_optimiser:
            optimiser_filepath = os.path.join(id_folderpath, "optimiser.pth")
            torch.save(self.optimizer, optimiser_filepath)

    def load(self, id_folderpath):
        # Spectrally capped layers are loaded as plain linear layers (see
        # _without_parametrizations), so training a loaded model no longer caps them:
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
