"""Cross-validates a plain neural network regressor on a given model metric.

Baseline for the GP emulator: same data loading, whitening and fold splits as
gp_training.py, so the cross-validation metrics are directly comparable.
"""
import argparse
import os
import json
import sys

import torch

import numpy as np

from torch.utils.data import TensorDataset, DataLoader

sys.stdout.reconfigure(line_buffering=True)

# Match the precision used for the GP models:
torch.set_default_dtype(torch.float64)

K_FOLD_COUNT = 8
NUM_EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.003
HIDDEN_LAYER_COUNT = 4
HIDDEN_LAYER_NEURON_COUNT = 512


def parse_arguments():
    parser = argparse.ArgumentParser(description='Cross-validate a plain neural network regressor on a given model metric')
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

    return parameter_matrix[~nan_mask, :], metric_mean[~nan_mask], metric_sem[~nan_mask]


class RegressionNetwork(torch.nn.Module):
    def __init__(self, dimension, hidden_layer_count=HIDDEN_LAYER_COUNT, hidden_layer_neuron_count=HIDDEN_LAYER_NEURON_COUNT):
        # Run general initialisation of the nn.Module base class:
        super().__init__()

        # Record parameters:
        self.dimension = dimension
        self.hl_count = hidden_layer_count
        self.hl_neuron_count = hidden_layer_neuron_count

        # Set up layers:
        layers = []
        input_count = dimension
        for _ in range(self.hl_count):
            layers.append(torch.nn.Linear(input_count, self.hl_neuron_count))
            layers.append(torch.nn.SiLU())
            input_count = self.hl_neuron_count
        layers.append(torch.nn.Linear(input_count, 1))
        self.mlp = torch.nn.Sequential(*layers)

        # Initialise weights:
        with torch.no_grad():
            self.apply(self.initialise)

    def forward(self, x):
        return self.mlp.forward(x).squeeze(-1)

    def initialise(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)


class ModelManager:

    def __init__(self, dimension, learning_rate):
        # Set up model:
        self.model = RegressionNetwork(dimension)

        # Set up optimisation, decaying the learning rate to zero over training:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=NUM_EPOCHS
        )
        self.loss_function = torch.nn.MSELoss()
        self.loss_history = []

    def train_epoch(self, dataloader):
        # Ensure parameters are trainable:
        self.model.train()

        # Run through entire dataset:
        for batch_index, (x_batch, y_batch) in enumerate(dataloader):
            self.optimizer.zero_grad()
            loss = self.loss_function(self.model(x_batch), y_batch)
            loss.backward()
            self.optimizer.step()
            self.loss_history.append(loss.item())

    def train(self, x, y, batch_size, epochs=1):
        # Convert datasets to pytorch:
        x_tensor = torch.tensor(x)
        y_tensor = torch.tensor(y)
        dataset = TensorDataset(x_tensor, y_tensor)

        for epoch_index in range(epochs):
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            self.train_epoch(dataloader)
            self.scheduler.step()
            if (epoch_index + 1) % 10 == 0:
                recent_loss = np.mean(self.loss_history[-len(dataloader):])
                print(f"Epoch {epoch_index + 1}, mean training loss: {recent_loss}", flush=True)


def run_inference(model_manager, inputs, batch_size=512):
    # Set up dataloading:
    tensor_input = torch.tensor(inputs)
    inference_dataset = TensorDataset(tensor_input)
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

    # Shift to eval mode:
    model_manager.model.eval()

    # Set up outputs:
    predictions_array = []
    with torch.no_grad():
        for inference_batch in inference_loader:
            predictions = model_manager.model(inference_batch[0])
            predictions_array.append(predictions.detach().numpy())

    return np.concatenate(predictions_array)


def cross_validation(parameter_matrix, output_metric, output_sem):
    # Shuffle datapoints prior to cross-validation (as otherwise there
    # are weird autocorrelations from the Sobol' sampling). Same seed and
    # fold assignment as gp_training.py, so folds match the GP's:
    rng = np.random.default_rng(0)
    dataset_size = parameter_matrix.shape[0]

    # Get permutations:
    permuted_indices = rng.permutation(dataset_size)
    permuted_parameters = parameter_matrix[permuted_indices, :]
    permuted_target = output_metric[permuted_indices]
    permuted_sem = output_sem[permuted_indices]

    # Perform k-fold cross validation:
    mae_list = []
    mse_list = []
    noise_floor_list = []
    latent_mse_list = []
    for k_index in range(K_FOLD_COUNT):
        # Set up the model and the model's associated training apparatus:
        print(f"Performing validation with index: {k_index}...", flush=True)
        model_manager = ModelManager(parameter_matrix.shape[1], LEARNING_RATE)

        # Get test indices:
        test_indices = np.arange(k_index, dataset_size, K_FOLD_COUNT)
        test_mask = np.zeros(dataset_size, dtype=bool)
        test_mask[test_indices] = 1

        # Get training indices:
        train_mask = np.ones(dataset_size, dtype=bool)
        train_mask[test_indices] = 0

        # Set up dataset:
        train_parameters = permuted_parameters[train_mask, :]
        train_target = permuted_target[train_mask]

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

        # Get whitened noise for hold-out set:
        whitened_test_noise = (test_sem / train_std) ** 2

        # Train model:
        model_manager.train(train_parameters, whitened_train, BATCH_SIZE, epochs=NUM_EPOCHS)

        # Run predictions on test set:
        predictions = run_inference(model_manager, test_parameters, BATCH_SIZE)

        # Get basic metrics:
        mean_absolute_error = np.mean(np.abs(whitened_test - predictions))
        mean_squared_error = np.mean((whitened_test - predictions) ** 2)
        mae_list.append(mean_absolute_error)
        print(f"MAE: {mean_absolute_error}")
        mse_list.append(mean_squared_error)
        print(f"MSE: {mean_squared_error}")

        # Noise floor: the lowest MSE any model could reach on these noisy targets,
        # and the error left once that replicate noise is subtracted:
        noise_floor = np.mean(whitened_test_noise)
        noise_floor_list.append(noise_floor)
        print(f"Noise floor: {noise_floor}")
        latent_mean_squared_error = mean_squared_error - noise_floor
        latent_mse_list.append(latent_mean_squared_error)
        print(f"Noise-corrected MSE: {latent_mean_squared_error}")

    return mae_list, mse_list, noise_floor_list, latent_mse_list


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

    # Set up save directories:
    id_folderpath = os.path.join(
        args.experiment_dirpath, f"nn_baseline_models_b{BATCH_SIZE}", args.metric_name
    )
    os.makedirs(id_folderpath, exist_ok=True)

    # Perform cross-validation:
    mae_list, mse_list, noise_floor_list, latent_mse_list = cross_validation(
        parameter_matrix, output_metric, output_sem
    )

    # Summarise:
    print(f"Mean MSE: {np.mean(mse_list)}")
    print(f"Mean noise floor: {np.mean(noise_floor_list)}")
    print(f"Mean noise-corrected MSE: {np.mean(latent_mse_list)}")

    # Save CV metrics:
    cv_dict = {
        "mae": mae_list,
        "mse": mse_list,
        "noise_floor": noise_floor_list,
        "noise_corrected_mse": latent_mse_list
    }

    cv_filepath = os.path.join(id_folderpath, "cv_metrics.json")
    with open(cv_filepath, 'w') as output:
        json.dump(cv_dict, output, indent=4)


if __name__ == "__main__":
    main()