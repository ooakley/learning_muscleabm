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

NUM_EPOCHS = 5
K_FOLD_COUNT = 8

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a deep kernel GP regressor on a given model metric')
    parser.add_argument('--experiment_dirpath', type=str)
    parser.add_argument('--metric_name', type=str)
    args = parser.parse_args()
    return args


def load_gridsearch_data(experiment_dirpath, metric_name):
    # Load parameter data:
    parameter_matrix = np.load(
        os.path.join(experiment_dirpath, "sample_matrix.npy")
    )

    # Load metric data:
    if metric_name == "op65":
        output_metric = np.load(
            os.path.join(experiment_dirpath, "summary_data", "matrix_order_parameters.npy")
        )
        # Select final estimate of OP scale curve:
        output_metric = output_metric[:, :, 2]
    else:
        output_metric = np.load(
            os.path.join(experiment_dirpath, "summary_data", f"{metric_name}.npy")
        )

    # Remove failed simulations:
    metric_mean = np.nanmean(output_metric, axis=1)
    n_valid = np.sum(~np.isnan(output_metric), axis=1)
    metric_sem = np.nanstd(output_metric, axis=1, ddof=1) / np.sqrt(n_valid)
    nan_mask = np.isnan(metric_sem)

    print("Fractional dead zone:")
    print(np.count_nonzero(metric_mean < 0.003) / metric_mean.shape[0])

    return parameter_matrix[~nan_mask, :], metric_mean[~nan_mask], metric_sem[~nan_mask]


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
        # Set up model, Fixed Noise requires an instantiating noise tensor as an API quirk:
        inducing_points = torch.tensor(inducing_points)
        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=torch.ones(1), learn_additional_noise=False
        )
        self.model = SparseGPModel(inducing_points, inducing_points.shape[1])

        # Set up optimisation:
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
        ], lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=NUM_EPOCHS
        )
        self.loss_history = []

    def train_epoch(self, dataloader, num_data):
        # Ensure parameters are trainable:
        self.model.train()
        self.likelihood.train()

        # Set up loss:
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=num_data)

        # Run through entire dataset:
        for batch_index, (x_batch, y_batch, noise_batch) in enumerate(dataloader):
            self.optimizer.zero_grad()
            output_distribution = self.model(x_batch)
            loss = -mll(output_distribution, y_batch, noise=noise_batch)
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

    def train(self, x, y, noise, batch_size, epochs=1):
        # Convert datasets to pytorch:
        x_tensor = torch.tensor(x)
        y_tensor = torch.tensor(y)
        noise_tensor = torch.tensor(noise)
        dataset = TensorDataset(x_tensor, y_tensor, noise_tensor)

        for epoch_index in range(epochs):
            print(f"Training epoch {epoch_index + 1}...")
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            self.train_epoch(dataloader, len(y))
            self.scheduler.step()

    def save(self, id_folderpath):
        # Save model components:
        model_filepath = os.path.join(id_folderpath, "model.pth")
        torch.save(self.model, model_filepath)
        likelihood_filepath = os.path.join(id_folderpath, "likelihood.pth")
        torch.save(self.likelihood, likelihood_filepath)
        optimiser_filepath = os.path.join(id_folderpath, "optimiser.pth")
        torch.save(self.optimizer, optimiser_filepath)

    def load(self, id_folderpath):
        self.model = torch.load(os.path.join(id_folderpath, "model.pth"), weights_only=False)
        self.likelihood = torch.load(os.path.join(id_folderpath, "likelihood.pth"), weights_only=False)
        self.optimizer = torch.load(os.path.join(id_folderpath, "optimiser.pth"), weights_only=False)


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
    stddev_array = []
    with torch.no_grad():
        for batch_index, inference_batch in enumerate(inference_loader):
            inference_batch = inference_batch[0]
            predictions = model_manager.model(inference_batch)
            predictions_array.append(predictions.mean.detach().numpy())
            stddev_array.append(predictions.stddev.detach().numpy())
            if (batch_index + 1) % 64 == 0:
                print(batch_index + 1)

    return np.concatenate(predictions_array), np.concatenate(stddev_array)


def cross_validation(parameter_matrix, output_metric, output_sem, inducing_points):
    # Shuffle datapoints prior to cross-validation (as otherwise there
    # are weird autocorrelations from the Sobol' sampling):
    rng = np.random.default_rng(0)
    dataset_size = parameter_matrix.shape[0]

    # Get permutations:
    permuted_indices = rng.permutation(dataset_size)
    permuted_parameters = parameter_matrix[permuted_indices, :]
    permuted_target = output_metric[permuted_indices]
    permuted_sem = output_sem[permuted_indices]

    # Perform 8-fold cross validation:
    mae_list = []
    mse_list = []
    sll_list = []
    cr_list = []
    for k_index in range(K_FOLD_COUNT):
        # Set up the model and the model's associated training apparatus:
        print(f"Performing validation with index: {k_index}...", flush=True)
        model_manager = ModelManager(inducing_points, 0.003)

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
        print(f"Test shape: {test_parameters.shape}")

        # Calculate whitening transform:
        train_mean = np.mean(train_target)
        train_std = np.std(train_target)
        whitened_train = (train_target - train_mean) / train_std
        whitened_test = (test_target - train_mean) / train_std

        # Whiten and clip noise estimates:
        whitened_train_noise = (train_sem / train_std) ** 2
        noise_clip = np.quantile(whitened_train_noise, 0.001)
        print(f"Train noise clip: {noise_clip}...")
        whitened_train_noise = np.clip(whitened_train_noise, noise_clip, None)
        whitened_test_noise = (test_sem / train_std) ** 2

        # Train and save model:
        model_manager.train(train_parameters, whitened_train, whitened_train_noise, 512, epochs=NUM_EPOCHS)

        # Run predictions on test set:
        predictions, stddev = run_inference(model_manager, test_parameters, 512)

        # Get basic metrics:
        mean_absolute_error = np.mean(np.abs(whitened_test - predictions))
        mean_squared_error = np.mean((whitened_test - predictions) ** 2)
        mae_list.append(mean_absolute_error)
        print(f"MAE: {mean_absolute_error}")
        mse_list.append(mean_squared_error)
        print(f"MSE: {mean_squared_error}")
    
        # Get standardised log loss:
        null_mean = np.mean(whitened_train)
        null_var = np.var(whitened_train)
        n_prefactor =  0.5 * np.log(2 * np.pi * null_var)
        n_exponent = ((whitened_test - null_mean) ** 2) / (2 * null_var)
        null_loss = np.mean(n_prefactor + n_exponent)
        print(f"Null LL: {null_loss}")
        model_prefactor = 0.5 * np.log(2 * np.pi * (stddev**2 + whitened_test_noise))
        model_exponent = ((whitened_test - predictions)**2) / (2 * (stddev**2 + whitened_test_noise))
        model_loss = np.mean(model_prefactor + model_exponent)
        print(f"Model LL: {model_loss}")
        standardised_log_loss = model_loss - null_loss
        print(f"SLL: {standardised_log_loss}")
        sll_list.append(standardised_log_loss)

        # Estimate latent calibration ratio
        latent_estimate = (whitened_test - predictions) ** 2  - whitened_test_noise
        calibration_ratio = np.mean(latent_estimate) / np.mean(stddev ** 2)
        print(f"Calibration ratio: {calibration_ratio}")
        cr_list.append(calibration_ratio)

        # Print diagnostic worst-calibrated predictions:
        point_loss = model_prefactor + model_exponent
        print("Median Point Loss | Mean Test Noise | Mean GP Latent Noise")
        print(np.median(point_loss), np.mean(whitened_test_noise), np.mean(stddev ** 2))
        worst = np.argsort(point_loss)[-20:]
        print("SLL Worst Empirical Noise | SLL Worst Latent Noise | SLL Worst Squared Error")
        print(whitened_test_noise[worst], stddev[worst] ** 2, (whitened_test - predictions)[worst] ** 2)

    return mae_list, mse_list, sll_list, cr_list


def emulate(manager, x):
    manager.likelihood.eval()
    manager.model.eval()
    tensor_input = torch.tensor(x)
    if len(tensor_input.shape) == 1:
        tensor_input = torch.unsqueeze(tensor_input, 0)
    with torch.no_grad():
        prediction = manager.model(tensor_input)
        prediction_mean = prediction.mean.detach().numpy()
        prediction_std = prediction.stddev.detach().numpy()
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
    print(f"Using gridsearch {args.experiment_dirpath}...")
    print(f"Training on {args.metric_name}...")

    # Load gridsearch data:
    print("Loading data...")
    parameter_matrix, output_metric, output_sem = load_gridsearch_data(
        args.experiment_dirpath, args.metric_name
    )

    # Get inducing points:
    print("Sampling inducing points...")
    generator = np.random.default_rng(0)
    inducing_points = generator.choice(parameter_matrix, 512, replace=False)
    print(f"Inducing points shape: {inducing_points.shape}...")

    # Set up save directories:
    gp_folderpath = os.path.join(args.experiment_dirpath, f"velbo_gp_models_{NUM_EPOCHS}")
    if not os.path.exists(gp_folderpath):
        os.mkdir(gp_folderpath)
    id_folderpath = os.path.join(gp_folderpath, args.metric_name)
    if not os.path.exists(id_folderpath):
        os.mkdir(id_folderpath)

    # Set up the model and the model's associated training apparatus:
    print("Setting up model...")
    model_manager = ModelManager(inducing_points, 0.003)

    # Whiten metric distribution:
    metric_mean = np.mean(output_metric)
    metric_std = np.std(output_metric)
    whitened_metric = (output_metric - metric_mean) / metric_std
    whitened_sem = output_sem / metric_std
    whitened_noise = whitened_sem ** 2
    noise_clip = np.quantile(whitened_noise, 0.001)
    print(f"0.05 Noise Quantile: {np.quantile(whitened_noise, 0.05)}")
    print(f"0.95 Noise Quantile: {np.quantile(whitened_noise, 0.95)}")
    print(f"Train noise clip: {noise_clip}")
    whitened_noise = np.clip(whitened_noise, noise_clip, None)

    # Save exact whitening transform:
    np.save(os.path.join(id_folderpath, "whiten_mean.npy"), metric_mean)
    np.save(os.path.join(id_folderpath, "whiten_std.npy"), metric_std)

    # Train the model:
    print("Training model...")
    model_manager.train(parameter_matrix, whitened_metric, whitened_noise, 512, epochs=NUM_EPOCHS)
    print("Saving model...")
    model_manager.save(id_folderpath)

    # Save loss history:
    loss_history = np.array(model_manager.loss_history)
    np.save(os.path.join(id_folderpath, "loss_history.npy"), loss_history)

    # Generate and save predictions:
    whitened_predictions, whitened_stddev = run_inference(model_manager, parameter_matrix, batch_size=512)
    predictions = (whitened_predictions * metric_std) + metric_mean
    stddev = (whitened_stddev * metric_std)
    np.save(os.path.join(id_folderpath, "parameter_predictions.npy"), np.stack([predictions, stddev], axis=1))

    # Perform cross-validation:
    mae_list, mse_list, sll_list, cr_list = cross_validation(
        parameter_matrix, output_metric, output_sem, inducing_points
    )

    # Save CV metrics:
    cv_dict = {
        "mae": mae_list,
        "mse": mse_list,
        "sll": sll_list,
        "cr_list": cr_list
    }

    cv_filepath = os.path.join(id_folderpath, "cv_metrics.json")
    with open(cv_filepath, 'w') as output:
        json.dump(cv_dict, output, indent=4)

    # Get Sobol' indices from GP model:
    print("Running Sobol' index inference...", flush=True)
    Si, STi = run_sobol_index_inference(model_manager, parameter_matrix.shape[1])
    np.save(os.path.join(id_folderpath, "sobol_i.npy"), Si)
    np.save(os.path.join(id_folderpath, "sobol_Ti.npy"), STi)


if __name__ == "__main__":
    main()
