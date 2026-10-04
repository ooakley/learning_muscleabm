import argparse
import os
import json
import sys

import gpytorch
import torch

import numpy as np

from scipy.stats import qmc
from torch.utils.data import TensorDataset, DataLoader

sys.stdout.reconfigure(line_buffering=True)

# Need a higher precision for computing double derivatives:
torch.set_default_dtype(torch.float64)

NUM_EPOCHS = 15
K_FOLD_COUNT = 8
INDUCING_POINT_COUNT = 512

# Learning rates: one for the GP (variational parameters, inducing locations,
# kernel hyperparameters and likelihood), and one for the deep kernel network:
LEARNING_RATE = 0.01
DEEP_KERNEL_LEARNING_RATE = 0.003

# Deep kernel architecture: number of hidden layers, and neurons per hidden layer:
NUM_LAYERS = 2
NUM_NEURONS = 32

# Batch size used for both training and inference:
BATCH_SIZE = 512

# Folder of the experiment holding the data aggregated across history matching waves:
GLOBAL_DATASET_FOLDER = "global_dataset"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a deep kernel GP regressor on a given model metric')
    parser.add_argument(
        '--experiment_dirpath', type=str,
        help='The {date}-{config} folder, containing config.json, the hm wave folders and global_dataset.'
    )
    parser.add_argument('--metric_name', type=str)
    args = parser.parse_args()
    return args


def load_wave_provenance(dataset_dirpath):
    """Which history matching wave each row of the global dataset came from.

    Returns the wave of each row, and its row within that wave's own sample matrix.
    """
    wave_indices_filepath = os.path.join(dataset_dirpath, "wave_indices.npy")
    if not os.path.exists(wave_indices_filepath):
        raise FileNotFoundError(
            f"No wave indices found in {dataset_dirpath}, run collate_hm.py to build the global dataset."
        )
    wave_indices = np.load(wave_indices_filepath)
    wave_row_indices = np.load(os.path.join(dataset_dirpath, "wave_row_indices.npy"))
    return wave_indices, wave_row_indices


def load_gridsearch_data(dataset_dirpath, metric_name):
    # Load parameter data:
    parameter_matrix = np.load(
        os.path.join(dataset_dirpath, "sample_matrix.npy")
    )

    # Load metric data:
    if metric_name == "op65":
        output_metric = np.load(
            os.path.join(dataset_dirpath, "summary_data", "matrix_order_parameters.npy")
        )
        # Select final estimate of OP scale curve:
        output_metric = output_metric[:, :, 2]
    else:
        output_metric = np.load(
            os.path.join(dataset_dirpath, "summary_data", f"{metric_name}.npy")
        )

    # Remove failed simulations:
    metric_mean = np.nanmean(output_metric, axis=1)
    n_valid = np.sum(~np.isnan(output_metric), axis=1)
    metric_sem = np.nanstd(output_metric, axis=1, ddof=1) / np.sqrt(n_valid)
    nan_mask = np.isnan(metric_sem)

    # Record which rows of the original (global) sample matrix survive, so that every
    # diagnostic can be saved against the original row indices:
    valid_indices = np.flatnonzero(~nan_mask)
    original_row_count = parameter_matrix.shape[0]

    return (
        parameter_matrix[~nan_mask, :], metric_mean[~nan_mask], metric_sem[~nan_mask],
        n_valid[~nan_mask], valid_indices, original_row_count
    )


def scatter_to_original(values, original_indices, original_row_count):
    """Place per-datapoint values at their rows in the original sample matrix.

    Rows of the original sample matrix with no value (failed simulations) are NaN.
    """
    output = np.full((original_row_count,) + values.shape[1:], np.nan)
    output[original_indices] = values
    return output


class DeepInputTransformation(torch.nn.Module):
    def __init__(self, dimension, hidden_layer_count, hidden_layer_neuron_count):
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
        self.input_transform = DeepInputTransformation(dimensions, NUM_LAYERS, NUM_NEURONS)

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

    def __init__(self, inducing_points, learning_rate, deep_kernel_learning_rate):
        # Set up model:
        inducing_points = torch.tensor(inducing_points)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = SparseGPModel(inducing_points, inducing_points.shape[1])

        # The default noise constraint sets the minimum too high,
        # we need the more permissive constraint of positivity:
        self.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Positive())

        # Split the model parameters into the deep kernel network and everything else:
        deep_kernel_parameters = list(self.model.input_transform.parameters())
        deep_kernel_parameter_ids = {id(parameter) for parameter in deep_kernel_parameters}
        gp_parameters = [
            parameter for parameter in self.model.parameters()
            if id(parameter) not in deep_kernel_parameter_ids
        ]

        # Set up optimisation - Adam seems to work best (need to properly test this).
        # The scheduler below decays each group from its own initial learning rate:
        self.optimizer = torch.optim.Adam([
            {'params': gp_parameters, 'lr': learning_rate},
            {'params': self.likelihood.parameters(), 'lr': learning_rate},
            {'params': deep_kernel_parameters, 'lr': deep_kernel_learning_rate},
        ])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=NUM_EPOCHS
        )

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
            self.scheduler.step()

    def save(self, id_folderpath, save_optimiser=True):
        # Save model components:
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
        self.model = torch.load(os.path.join(id_folderpath, "model.pth"), weights_only=False)
        self.likelihood = torch.load(os.path.join(id_folderpath, "likelihood.pth"), weights_only=False)
        optimiser_filepath = os.path.join(id_folderpath, "optimiser.pth")
        if os.path.exists(optimiser_filepath):
            self.optimizer = torch.load(optimiser_filepath, weights_only=False)

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


def run_inference(model_manager, inputs, batch_size=512):
    # Set up dataloading:
    tensor_input = torch.tensor(inputs)
    inference_dataset = TensorDataset(tensor_input)
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

    # Shift to eval mode:
    model_manager.model.eval()
    model_manager.likelihood.eval()

    # Set up outputs:
    predictions_array = []
    sigma_array = []
    latent_sigma_array = []
    with torch.no_grad():
        for batch_index, inference_batch in enumerate(inference_loader):
            # Retrieve parameter sets:
            inference_batch = inference_batch[0]

            # Get latent and full posterior noise estimates:
            latent_preds = model_manager.model(inference_batch)
            preds = model_manager.likelihood(latent_preds)
            predictions_array.append(preds.mean.detach().numpy())
            sigma_array.append(preds.stddev.detach().numpy())
            latent_sigma_array.append(latent_preds.stddev.detach().numpy())

            # Report progress:
            if (batch_index + 1) % 64 == 0:
                print(batch_index + 1)
    
    predictions_array = np.concatenate(predictions_array) 
    sigma_array = np.concatenate(sigma_array)
    latent_sigma_array = np.concatenate(latent_sigma_array)
    return predictions_array, sigma_array, latent_sigma_array


def run_warp_inference(model_manager, inputs, batch_size=512):
    """Positions of the inputs in the warped space that the GP kernel acts on."""
    tensor_input = torch.tensor(inputs)
    model_manager.model.eval()
    warped_array = []
    with torch.no_grad():
        for batch_start in range(0, tensor_input.shape[0], batch_size):
            warped_batch = model_manager.model.input_transform(tensor_input[batch_start:batch_start + batch_size])
            warped_array.append(warped_batch.detach().numpy())
    return np.concatenate(warped_array)


def cross_validation(
        parameter_matrix, output_metric, output_sem, replicate_counts, inducing_points,
        valid_indices, original_row_count, wave_indices, wave_row_indices, id_folderpath
    ):
    # Shuffle datapoints prior to cross-validation (as otherwise there
    # are weird autocorrelations from the Sobol' sampling):
    rng = np.random.default_rng(0)
    dataset_size = parameter_matrix.shape[0]

    # Get permutations:
    permuted_indices = rng.permutation(dataset_size)
    permuted_parameters = parameter_matrix[permuted_indices, :]
    permuted_target = output_metric[permuted_indices]
    permuted_sem = output_sem[permuted_indices]

    # Row of the original sample matrix that each permuted datapoint came from:
    permuted_original_indices = valid_indices[permuted_indices]

    # Fold in which each row of the original sample matrix is held out (-1 for failed rows):
    fold_index = np.full(original_row_count, -1, dtype=int)
    fold_index[permuted_original_indices] = np.arange(dataset_size) % K_FOLD_COUNT

    # Per-datapoint diagnostics, all stored against rows of the original sample matrix.
    # --- Inputs, in original units:
    valid_mask = np.zeros(original_row_count, dtype=bool)
    valid_mask[valid_indices] = True
    cv_arrays = {
        "valid_mask": valid_mask,
        "wave_index": wave_indices,
        "wave_row_index": wave_row_indices,
        "fold_index": fold_index,
        "target": scatter_to_original(output_metric, valid_indices, original_row_count),
        "sem": scatter_to_original(output_sem, valid_indices, original_row_count),
        "replicate_count": scatter_to_original(replicate_counts, valid_indices, original_row_count),
    }
    # --- Held-out predictions, in original units and in the whitened units of the
    # --- fold that held the row out (the units the cross-validation metrics use):
    held_out_keys = [
        "prediction", "total_std", "latent_std",
        "whitened_target", "whitened_sem",
        "whitened_prediction", "whitened_total_std", "whitened_latent_std"
    ]
    for key in held_out_keys:
        cv_arrays[key] = np.full(original_row_count, np.nan)
    # --- Predictions of every fold's model at every row (original units), so rows a model
    # --- trained on can be compared with the rows it held out, and models with each other:
    for key in ["all_fold_prediction", "all_fold_total_std", "all_fold_latent_std"]:
        cv_arrays[key] = np.full((K_FOLD_COUNT, original_row_count), np.nan)
    # --- Per-fold whitening transforms and learned hyperparameters (whitened units):
    for key in ["fold_whiten_mean", "fold_whiten_std", "fold_likelihood_noise", "fold_outputscale", "fold_constant_mean"]:
        cv_arrays[key] = np.full(K_FOLD_COUNT, np.nan)
    cv_arrays["fold_lengthscales"] = np.full((K_FOLD_COUNT, parameter_matrix.shape[1]), np.nan)

    # Per-fold summary metrics:
    cv_dict = {
        key: [] for key in [
            "mae", "mse", "sll", "lcr", "tcr",
            "train_mae", "train_mse", "noise_floor", "null_loss", "model_loss",
            "whiten_mean", "whiten_std", "likelihood_noise", "outputscale"
        ]
    }

    # Perform 8-fold cross validation:
    cv_models_folderpath = os.path.join(id_folderpath, "cv_models")
    for k_index in range(K_FOLD_COUNT):
        # Set up the model and the model's associated training apparatus:
        print(f"Performing validation with index: {k_index}...", flush=True)
        model_manager = ModelManager(inducing_points, LEARNING_RATE, DEEP_KERNEL_LEARNING_RATE)

        # Get test indices:
        test_indices = np.arange(k_index, dataset_size, K_FOLD_COUNT)
        test_mask = np.zeros(dataset_size, dtype=bool)
        test_mask[test_indices] = 1
        print(f"Test indices: {np.argwhere(test_mask)}")

        # Get training indices:
        train_mask = np.ones(dataset_size, dtype=bool)
        train_mask[test_indices] = 0

        # Set up dataset:
        train_parameters = permuted_parameters[train_mask, :]
        train_target = permuted_target[train_mask]
        train_sem = permuted_sem[train_mask]
    
        test_parameters = permuted_parameters[test_mask, :]
        test_target = permuted_target[test_mask]
        test_sem = permuted_sem[test_mask]
        print(f"Train shape: {train_parameters.shape}")
        print(f"Train shape: {test_parameters.shape}")

        # Calculate whitening transform:
        train_mean = np.mean(train_target)
        train_std = np.std(train_target)
        whitened_train = (train_target - train_mean) / train_std
        whitened_test = (test_target - train_mean) / train_std

        # Get whitened noise for hold-out set:
        whitened_test_noise = (test_sem / train_std) ** 2

        # Train model:
        model_manager.train(train_parameters, whitened_train, BATCH_SIZE, epochs=NUM_EPOCHS)

        # Save this fold's model, whitening transform and loss history, so that further
        # held-out diagnostics can be computed later without retraining:
        fold_folderpath = os.path.join(cv_models_folderpath, f"fold_{k_index}")
        os.makedirs(fold_folderpath, exist_ok=True)
        model_manager.save(fold_folderpath, save_optimiser=False)
        np.save(os.path.join(fold_folderpath, "whiten_mean.npy"), train_mean)
        np.save(os.path.join(fold_folderpath, "whiten_std.npy"), train_std)
        np.save(os.path.join(fold_folderpath, "loss_history.npy"), np.array(model_manager.loss_history))

        # Run predictions on every datapoint, then pick out the test and training sets:
        all_predictions, all_sigma, all_latent_sigma = \
            run_inference(model_manager, permuted_parameters, BATCH_SIZE)
        predictions = all_predictions[test_mask]
        test_sigma = all_sigma[test_mask]
        test_latent_sigma = all_latent_sigma[test_mask]

        # Get basic metrics:
        mean_absolute_error = np.mean(np.abs(whitened_test - predictions))
        mean_squared_error = np.mean((whitened_test - predictions) ** 2)
        cv_dict["mae"].append(mean_absolute_error)
        print(f"MAE: {mean_absolute_error}")
        cv_dict["mse"].append(mean_squared_error)
        print(f"MSE: {mean_squared_error}")

        # Get the same metrics on the training set, to show the train-test gap:
        train_mean_absolute_error = np.mean(np.abs(whitened_train - all_predictions[train_mask]))
        train_mean_squared_error = np.mean((whitened_train - all_predictions[train_mask]) ** 2)
        cv_dict["train_mae"].append(train_mean_absolute_error)
        print(f"Train MAE: {train_mean_absolute_error}")
        cv_dict["train_mse"].append(train_mean_squared_error)
        print(f"Train MSE: {train_mean_squared_error}")
    
        # Get standardised log loss:
        null_mean = np.mean(whitened_train)
        null_var = np.var(whitened_train)
        n_prefactor =  0.5 * np.log(2 * np.pi * null_var)
        n_exponent = ((whitened_test - null_mean) ** 2) / (2 * null_var)
        null_loss = np.mean(n_prefactor + n_exponent)
        print(f"Null LL: {null_loss}")
        model_prefactor = 0.5 * np.log(2 * np.pi * test_sigma**2)
        model_exponent = ((whitened_test - predictions)**2) / (2 * test_sigma**2)
        model_loss = np.mean(model_prefactor + model_exponent)
        print(f"Model LL: {model_loss}")
        standardised_log_loss = model_loss - null_loss
        print(f"SLL: {standardised_log_loss}")
        cv_dict["sll"].append(standardised_log_loss)
        cv_dict["null_loss"].append(null_loss)
        cv_dict["model_loss"].append(model_loss)

        # Retrieve how well the model reproduces the SEM variance:
        empirical_latent_estimate = (whitened_test - predictions) ** 2  - whitened_test_noise
        # --- Fraction with latent noise:
        lc_ratio = np.mean(empirical_latent_estimate) / np.mean(test_latent_sigma ** 2)
        cv_dict["lcr"].append(lc_ratio)
        print(f"Latent calibration ratio: {lc_ratio}")
        # --- Fraction with total noise:
        tc_ratio = np.mean(empirical_latent_estimate) / np.mean(test_sigma ** 2)
        cv_dict["tcr"].append(tc_ratio)
        print(f"Total calibration ratio: {tc_ratio}")

        # Record the noise floor, whitening transform and learned hyperparameters:
        hyperparameters = model_manager.get_hyperparameters()
        cv_dict["noise_floor"].append(np.mean(whitened_test_noise))
        cv_dict["whiten_mean"].append(train_mean)
        cv_dict["whiten_std"].append(train_std)
        cv_dict["likelihood_noise"].append(hyperparameters["likelihood_noise"])
        cv_dict["outputscale"].append(hyperparameters["outputscale"])
        cv_arrays["fold_whiten_mean"][k_index] = train_mean
        cv_arrays["fold_whiten_std"][k_index] = train_std
        cv_arrays["fold_likelihood_noise"][k_index] = hyperparameters["likelihood_noise"]
        cv_arrays["fold_outputscale"][k_index] = hyperparameters["outputscale"]
        cv_arrays["fold_constant_mean"][k_index] = hyperparameters["constant_mean"]
        cv_arrays["fold_lengthscales"][k_index, :] = hyperparameters["lengthscales"]

        # Store this model's predictions at every row, in original units:
        cv_arrays["all_fold_prediction"][k_index, permuted_original_indices] = (all_predictions * train_std) + train_mean
        cv_arrays["all_fold_total_std"][k_index, permuted_original_indices] = all_sigma * train_std
        cv_arrays["all_fold_latent_std"][k_index, permuted_original_indices] = all_latent_sigma * train_std

        # Store the held-out predictions against their original rows:
        test_original_indices = permuted_original_indices[test_mask]
        cv_arrays["prediction"][test_original_indices] = (predictions * train_std) + train_mean
        cv_arrays["total_std"][test_original_indices] = test_sigma * train_std
        cv_arrays["latent_std"][test_original_indices] = test_latent_sigma * train_std
        cv_arrays["whitened_target"][test_original_indices] = whitened_test
        cv_arrays["whitened_sem"][test_original_indices] = test_sem / train_std
        cv_arrays["whitened_prediction"][test_original_indices] = predictions
        cv_arrays["whitened_total_std"][test_original_indices] = test_sigma
        cv_arrays["whitened_latent_std"][test_original_indices] = test_latent_sigma

        # Save everything so far after each fold, so that finished folds survive
        # the job being cut short:
        cv_dict["completed_folds"] = k_index + 1
        np.savez(os.path.join(id_folderpath, "cv_diagnostics.npz"), **cv_arrays)
        with open(os.path.join(id_folderpath, "cv_metrics.json"), 'w') as output:
            json.dump(cv_dict, output, indent=4)

    return cv_dict


def emulate(manager, x):
    manager.likelihood.eval()
    manager.model.eval()
    tensor_input = torch.tensor(x)
    if len(tensor_input.shape) == 1:
        tensor_input = torch.unsqueeze(tensor_input, 0)
    with torch.no_grad():
        latent_pred = manager.model(tensor_input)
        pred = manager.likelihood(latent_pred)
        prediction_mean = pred.mean.detach().numpy()
        prediction_std = pred.stddev.detach().numpy()
        prediction_latent_std = latent_pred.stddev.detach().numpy()
    return prediction_mean, prediction_std


def get_sobol_indices(dimension, f_A, f_B, f_Ai):
    Si_list = []
    STi_list = []
    for index in range(dimension):
        f0_sq = np.mean(f_A) * np.mean(f_B)
        V = np.var(np.concatenate([f_A, f_B]))
        S_i  = (np.mean(f_A * f_Ai[index]) - f0_sq) / V
        S_Ti = 1 - (np.mean(f_B * f_Ai[index]) - f0_sq) / V
        Si_list.append(S_i)
        STi_list.append(S_Ti)
    return np.array(Si_list), np.array(STi_list)


def run_sobol_index_inference(model_manager, parameter_dimension):
    # Get necessary model evaluations for Sobol' indices:
    hyperspace_dimension = parameter_dimension * 2
    sobol_sampler = qmc.Sobol(d=hyperspace_dimension, scramble=True, rng=0)
    hyperspace_inputs = sobol_sampler.random_base2(m=17)

    # Extract base parameter matrices:
    parameters_A = hyperspace_inputs[:, :parameter_dimension]
    parameters_B = hyperspace_inputs[:, parameter_dimension:]

    # Generating the combined parameter matrices:
    parameter_matrices = []
    for parameter_index in range(parameter_dimension):
        parameters_ABi = np.copy(parameters_B)
        parameters_ABi[:, parameter_index] = parameters_A[:, parameter_index]
        parameter_matrices.append(parameters_ABi)

    # Estimate model values at these points:
    f_A, _ = emulate(model_manager, parameters_A)
    f_B, _ = emulate(model_manager, parameters_B)

    f_Ai = []
    for i_parameters in parameter_matrices:
        f_Ai.append(emulate(model_manager, i_parameters)[0])

    # Estimate indices from the evaluations:
    Si, STi = get_sobol_indices(parameter_dimension, f_A, f_B, f_Ai)
    return Si, STi


def main():
    # Set seed:
    torch.manual_seed(0)

    # Parse arguments:
    args = parse_arguments()
    print(f"Using experiment {args.experiment_dirpath}...")
    print(f"Training on {args.metric_name}...")

    # The models train on the dataset aggregated across history matching waves:
    dataset_dirpath = os.path.join(args.experiment_dirpath, GLOBAL_DATASET_FOLDER)
    wave_indices, wave_row_indices = load_wave_provenance(dataset_dirpath)
    wave_ids, wave_row_counts = np.unique(wave_indices, return_counts=True)
    for wave_id, wave_row_count in zip(wave_ids, wave_row_counts):
        print(f"Global dataset contains {wave_row_count} rows from hm{wave_id}...")

    # The models are saved to the most recent wave in the dataset, as it is that
    # wave's posterior that they are used to sample:
    latest_wave_id = int(np.max(wave_ids))
    wave_dirpath = os.path.join(args.experiment_dirpath, f"hm{latest_wave_id}")
    if not os.path.exists(wave_dirpath):
        raise FileNotFoundError(f"The global dataset contains hm{latest_wave_id}, but {wave_dirpath} does not exist.")
    print(f"Saving models to {wave_dirpath}...")

    # Load gridsearch data:
    print("Loading data...")
    parameter_matrix, output_metric, output_sem, replicate_counts, valid_indices, original_row_count = \
        load_gridsearch_data(dataset_dirpath, args.metric_name)
    print(f"Using {parameter_matrix.shape[0]} of {original_row_count} rows of the sample matrix...")
    if wave_indices.shape[0] != original_row_count:
        raise ValueError(
            f"Wave indices have {wave_indices.shape[0]} rows, but the sample matrix has "
            f"{original_row_count}, rerun collate_hm.py."
        )

    # Get inducing points - sampling the row indices, so they can be recorded:
    print("Sampling inducing points...")
    generator = np.random.default_rng(0)
    inducing_indices = generator.choice(parameter_matrix.shape[0], INDUCING_POINT_COUNT, replace=False)
    inducing_points = parameter_matrix[inducing_indices, :]
    print(f"Inducing points shape: {inducing_points.shape}...")

    # Set up save directories:
    gp_folderpath = os.path.join(
        wave_dirpath,
        f"pll_gp_models_e{NUM_EPOCHS}_ip{INDUCING_POINT_COUNT}_l{NUM_LAYERS}_n{NUM_NEURONS}"
    )
    os.makedirs(gp_folderpath, exist_ok=True)
    id_folderpath = os.path.join(gp_folderpath, args.metric_name)
    os.makedirs(id_folderpath, exist_ok=True)

    # Record the settings these models were trained with:
    training_dict = {
        "experiment_dirpath": args.experiment_dirpath,
        "dataset_dirpath": dataset_dirpath,
        "wave_ids": [int(wave_id) for wave_id in wave_ids],
        "wave_row_counts": [int(wave_row_count) for wave_row_count in wave_row_counts],
        "latest_wave_id": latest_wave_id,
        "metric_name": args.metric_name,
        "num_epochs": NUM_EPOCHS,
        "k_fold_count": K_FOLD_COUNT,
        "inducing_point_count": INDUCING_POINT_COUNT,
        "learning_rate": LEARNING_RATE,
        "deep_kernel_learning_rate": DEEP_KERNEL_LEARNING_RATE,
        "num_layers": NUM_LAYERS,
        "num_neurons": NUM_NEURONS,
        "batch_size": BATCH_SIZE,
        "original_row_count": int(original_row_count),
        "valid_row_count": int(parameter_matrix.shape[0]),
        "torch_version": torch.__version__,
        "gpytorch_version": gpytorch.__version__
    }
    with open(os.path.join(id_folderpath, "training_config.json"), 'w') as output:
        json.dump(training_dict, output, indent=4)

    # Set up the model and the model's associated training apparatus:
    print("Setting up model...")
    model_manager = ModelManager(inducing_points, LEARNING_RATE, DEEP_KERNEL_LEARNING_RATE)

    # Whiten metric distribution:
    metric_mean = np.mean(output_metric)
    metric_std = np.std(output_metric)
    whitened_metric = (output_metric - metric_mean) / metric_std
    whitened_sem = output_sem / metric_std

    # Save whitening transform:
    np.save(os.path.join(id_folderpath, "whiten_mean.npy"), metric_mean)
    np.save(os.path.join(id_folderpath, "whiten_std.npy"), metric_std)

    # Train the model:
    print("Training model...")
    model_manager.train(parameter_matrix, whitened_metric, BATCH_SIZE, epochs=NUM_EPOCHS)
    print("Saving model...")
    model_manager.save(id_folderpath)

    # Save loss history:
    loss_history = np.array(model_manager.loss_history)
    np.save(os.path.join(id_folderpath, "loss_history.npy"), loss_history)

    # Generate and save predictions:
    whitened_predictions, whitened_stddev, whitened_latent_stddev \
        = run_inference(model_manager, parameter_matrix, batch_size=BATCH_SIZE)
    predictions = (whitened_predictions * metric_std) + metric_mean
    stddev = (whitened_stddev * metric_std)
    latent_stddev = (whitened_latent_stddev * metric_std)
    np.save(
        os.path.join(id_folderpath, "parameter_predictions.npy"),
        np.stack([predictions, stddev, latent_stddev], axis=1)
    )

    # Save the full model's diagnostics against rows of the original sample matrix.
    # These predictions are in-sample, as the full model trained on every valid row:
    print("Saving full model diagnostics...")
    hyperparameters = model_manager.get_hyperparameters()
    valid_mask = np.zeros(original_row_count, dtype=bool)
    valid_mask[valid_indices] = True
    final_inducing_points = model_manager.model.variational_strategy.inducing_points.detach().numpy()
    np.savez(
        os.path.join(id_folderpath, "full_model_diagnostics.npz"),
        # --- Inputs, in original units:
        valid_mask=valid_mask,
        wave_index=wave_indices,
        wave_row_index=wave_row_indices,
        target=scatter_to_original(output_metric, valid_indices, original_row_count),
        sem=scatter_to_original(output_sem, valid_indices, original_row_count),
        replicate_count=scatter_to_original(replicate_counts, valid_indices, original_row_count),
        # --- Predictions, in original units:
        prediction=scatter_to_original(predictions, valid_indices, original_row_count),
        total_std=scatter_to_original(stddev, valid_indices, original_row_count),
        latent_std=scatter_to_original(latent_stddev, valid_indices, original_row_count),
        # --- Whitening transform, and learned hyperparameters in whitened units:
        whiten_mean=metric_mean,
        whiten_std=metric_std,
        likelihood_noise=hyperparameters["likelihood_noise"],
        outputscale=hyperparameters["outputscale"],
        constant_mean=hyperparameters["constant_mean"],
        lengthscales=hyperparameters["lengthscales"],
        # --- Inducing points: rows of the original sample matrix they were initialised
        # --- at, and their locations before and after training:
        initial_inducing_indices=valid_indices[inducing_indices],
        initial_inducing_points=inducing_points,
        inducing_points=final_inducing_points,
        # --- Positions in the warped space that the kernel acts on:
        warped_parameters=scatter_to_original(
            run_warp_inference(model_manager, parameter_matrix, BATCH_SIZE), valid_indices, original_row_count
        ),
        warped_inducing_points=run_warp_inference(model_manager, final_inducing_points, BATCH_SIZE)
    )

    # Perform cross-validation - the per-datapoint diagnostics and the per-fold
    # metrics are saved by the function itself as each fold finishes:
    cv_dict = cross_validation(
        parameter_matrix, output_metric, output_sem, replicate_counts, inducing_points,
        valid_indices, original_row_count, wave_indices, wave_row_indices, id_folderpath
    )
    for key in ["mae", "mse", "train_mse", "sll", "lcr", "tcr"]:
        print(f"Mean {key}: {np.mean(cv_dict[key])}")

    # Get Sobol' indices from GP model:
    print("Running Sobol' index inferece...", flush=True)
    Si, STi = run_sobol_index_inference(model_manager, parameter_matrix.shape[1])
    np.save(os.path.join(id_folderpath, "sobol_i.npy"), Si)
    np.save(os.path.join(id_folderpath, "sobol_Ti.npy"), STi)


if __name__ == "__main__":
    main()