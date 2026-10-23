import argparse
import math
import os
import json
import sys
import types

import gpytorch
import torch

import numpy as np

from muscleabm.datasets import GLOBAL_DATASET_FOLDER, load_gridsearch_data
from muscleabm.emulators import (
    RESIDUAL_ACTIVATION, ModelManager, load_torch_object, run_inference
)
from muscleabm.history_matching import MODELS_DIRNAME, GPSettings
from muscleabm.sensitivity import run_sobol_index_inference

sys.stdout.reconfigure(line_buffering=True)

# Need a higher precision for computing double derivatives:
torch.set_default_dtype(torch.float64)

K_FOLD_COUNT = 8

# The GP architecture and training settings (training steps, inducing points, the deep kernel's
# layers, neurons, spectral coefficient and residual blocks, learning rates and batch size) are
# command line arguments, with defaults in muscleabm/history_matching.py. Every model
# trains for the same number of optimiser steps, however large the dataset, so that models of
# later waves are not trained for longer.
#
# Training runs in two stages, which can be run as separate jobs (--stage), so that the MCMC,
# which only needs the full model, does not wait for the cross-validation:
#     full:             the model trained on every row, its predictions and diagnostics;
#     cross_validation: the cross-validation models and metrics, and the Sobol' indices of the
#                       full model (which must already be saved).
# --stage all (the default) runs both, one after the other.
STAGES = ("all", "full", "cross_validation")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a deep kernel GP regressor on a given model metric')
    parser.add_argument(
        '--experiment_dirpath', type=str,
        help='The {date}-{config} folder, containing config.json, the hm wave folders and global_dataset.'
    )
    parser.add_argument('--metric_name', type=str)
    parser.add_argument(
        '--gp_models_dirname', type=str, default=MODELS_DIRNAME,
        help='Folder, inside the wave folder, to save the models to.'
    )
    parser.add_argument(
        '--hm_wave_id', type=int, default=None,
        help='History matching wave to train the models for, and save them to: they train on the rows '
             'of the global dataset from this wave and the waves before it. Defaults to the latest '
             'wave in the global dataset.'
    )
    parser.add_argument(
        '--stage', type=str, default="all", choices=STAGES,
        help='Train the full model (full), run the cross-validation and the Sobol\' indices of the '
             'saved full model (cross_validation), or both (all).'
    )
    GPSettings.add_arguments(parser)
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


def scatter_to_original(values, original_indices, original_row_count):
    """Place per-datapoint values at their rows in the original sample matrix.

    Rows of the original sample matrix with no value (failed simulations) are NaN.
    """
    output = np.full((original_row_count,) + values.shape[1:], np.nan)
    output[original_indices] = values
    return output


class DeepKernelModelManager(ModelManager):
    """Model manager with its own learning rate for the deep kernel network, and cosine decay."""

    def __init__(self, inducing_points, gp_settings):
        # Set up model:
        learning_rate = gp_settings.learning_rate
        deep_kernel_learning_rate = gp_settings.deep_kernel_learning_rate
        super().__init__(
            inducing_points, learning_rate,
            gp_settings.hidden_layer_count, gp_settings.hidden_layer_neuron_count,
            gp_settings.spectral_coefficient, gp_settings.residual_block_count
        )

        # Split the model parameters into the deep kernel network and everything else:
        deep_kernel_parameters = list(self.model.input_transform.parameters())
        deep_kernel_parameter_ids = {id(parameter) for parameter in deep_kernel_parameters}
        gp_parameters = [
            parameter for parameter in self.model.parameters()
            if id(parameter) not in deep_kernel_parameter_ids
        ]

        # Set up optimisation - Adam seems to work best (need to properly test this).
        # The scheduler below decays each group from its own initial learning rate, over the
        # optimiser steps of training:
        self.optimizer = torch.optim.Adam([
            {'params': gp_parameters, 'lr': learning_rate},
            {'params': self.likelihood.parameters(), 'lr': learning_rate},
            {'params': deep_kernel_parameters, 'lr': deep_kernel_learning_rate},
        ])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=gp_settings.training_steps
        )


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
        valid_indices, original_row_count, wave_indices, wave_row_indices, id_folderpath, gp_settings,
        run_record
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

    # Record what the folds were trained on:
    cv_dict.update(run_record)

    # Perform 8-fold cross validation:
    cv_models_folderpath = os.path.join(id_folderpath, "cv_models")
    for k_index in range(K_FOLD_COUNT):
        # Set up the model and the model's associated training apparatus. Each fold has its own
        # seed, so that its model does not depend on what was trained before it:
        print(f"Performing validation with index: {k_index}...", flush=True)
        torch.manual_seed(1 + k_index)
        model_manager = DeepKernelModelManager(inducing_points, gp_settings)

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
        model_manager.train_steps(train_parameters, whitened_train, gp_settings.batch_size, gp_settings.training_steps)

        # Save this fold's model, whitening transform and loss history, so that further
        # held-out diagnostics can be computed later without retraining:
        fold_folderpath = os.path.join(cv_models_folderpath, f"fold_{k_index}")
        os.makedirs(fold_folderpath, exist_ok=True)
        model_manager.save(fold_folderpath, save_optimiser=False)
        np.save(os.path.join(fold_folderpath, "whiten_mean.npy"), train_mean)
        np.save(os.path.join(fold_folderpath, "whiten_std.npy"), train_std)
        np.save(os.path.join(fold_folderpath, "loss_history.npy"), np.array(model_manager.loss_history))

        # Run predictions on every datapoint, then pick out the test and training sets:
        all_predictions, all_latent_sigma, all_sigma = run_inference(
            model_manager.model, model_manager.likelihood, permuted_parameters, gp_settings.batch_size,
            standard_deviations=True
        )
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


def prepare_training(args, gp_settings):
    """Load the training data, sample the inducing points and set up the models folder.

    Both stages call this, so they train on the same rows with the same inducing points.
    """
    # The models train on the dataset aggregated across history matching waves:
    dataset_dirpath = os.path.join(args.experiment_dirpath, GLOBAL_DATASET_FOLDER)
    wave_indices, wave_row_indices = load_wave_provenance(dataset_dirpath)

    # The models are saved to the given wave, by default the most recent in the dataset, as it
    # is that wave's posterior that they are used to sample. They train on that wave and the
    # ones before it, so a job that starts after a later wave has been collated does not train
    # on the later wave too:
    dataset_wave_ids = np.unique(wave_indices)
    hm_wave_id = int(np.max(dataset_wave_ids)) if args.hm_wave_id is None else args.hm_wave_id
    if hm_wave_id not in dataset_wave_ids:
        raise ValueError(
            f"hm{hm_wave_id} is not in the global dataset, which has waves {dataset_wave_ids.tolist()}."
        )
    wave_ids, wave_row_counts = np.unique(wave_indices[wave_indices <= hm_wave_id], return_counts=True)
    for wave_id, wave_row_count in zip(wave_ids, wave_row_counts):
        print(f"Global dataset contains {wave_row_count} rows from hm{wave_id}...")
    wave_dirpath = os.path.join(args.experiment_dirpath, f"hm{hm_wave_id}")
    if not os.path.exists(wave_dirpath):
        raise FileNotFoundError(f"The global dataset contains hm{hm_wave_id}, but {wave_dirpath} does not exist.")
    print(f"Saving models to {wave_dirpath}...")

    # Load gridsearch data:
    print("Loading data...")
    parameter_matrix, output_metric, output_sem, replicate_counts, valid_indices, original_row_count = \
        load_gridsearch_data(dataset_dirpath, args.metric_name)
    if wave_indices.shape[0] != original_row_count:
        raise ValueError(
            f"Wave indices have {wave_indices.shape[0]} rows, but the sample matrix has "
            f"{original_row_count}, rerun collate_hm.py."
        )

    # Rows of later waves are left out, like failed simulations:
    in_waves = wave_indices[valid_indices] <= hm_wave_id
    parameter_matrix, output_metric, output_sem, replicate_counts, valid_indices = (
        array[in_waves] for array in (parameter_matrix, output_metric, output_sem, replicate_counts, valid_indices)
    )
    print(f"Using {parameter_matrix.shape[0]} of {original_row_count} rows of the sample matrix...")

    # Get inducing points - sampling the row indices, so they can be recorded:
    print("Sampling inducing points...")
    generator = np.random.default_rng(0)
    inducing_indices = generator.choice(parameter_matrix.shape[0], gp_settings.inducing_point_count, replace=False)
    inducing_points = parameter_matrix[inducing_indices, :]
    print(f"Inducing points shape: {inducing_points.shape}...")

    # Set up save directories:
    gp_folderpath = os.path.join(wave_dirpath, args.gp_models_dirname)
    os.makedirs(gp_folderpath, exist_ok=True)
    id_folderpath = os.path.join(gp_folderpath, args.metric_name)
    os.makedirs(id_folderpath, exist_ok=True)

    return types.SimpleNamespace(
        dataset_dirpath=dataset_dirpath, hm_wave_id=hm_wave_id,
        wave_ids=wave_ids, wave_row_counts=wave_row_counts,
        wave_indices=wave_indices, wave_row_indices=wave_row_indices,
        parameter_matrix=parameter_matrix, output_metric=output_metric, output_sem=output_sem,
        replicate_counts=replicate_counts, valid_indices=valid_indices, original_row_count=original_row_count,
        inducing_indices=inducing_indices, inducing_points=inducing_points, id_folderpath=id_folderpath,
    )


def train_full_model(args, gp_settings, data):
    """Train the model on every row, and save it with its predictions and diagnostics."""
    # Unpack the training data:
    dataset_dirpath, latest_wave_id = data.dataset_dirpath, data.hm_wave_id
    wave_ids, wave_row_counts = data.wave_ids, data.wave_row_counts
    wave_indices, wave_row_indices = data.wave_indices, data.wave_row_indices
    parameter_matrix, output_metric, output_sem = data.parameter_matrix, data.output_metric, data.output_sem
    replicate_counts, valid_indices, original_row_count = data.replicate_counts, data.valid_indices, data.original_row_count
    inducing_indices, inducing_points, id_folderpath = data.inducing_indices, data.inducing_points, data.id_folderpath

    # The folder name does not depend on the GP settings, so say if this replaces models
    # trained with other settings:
    previous_config_filepath = os.path.join(id_folderpath, "training_config.json")
    if os.path.exists(previous_config_filepath):
        with open(previous_config_filepath) as previous_config_fstream:
            previous_settings = json.load(previous_config_fstream).get("gp_settings")
        if previous_settings != gp_settings.to_dict():
            print(
                f"Warning: replacing the models in {id_folderpath}, trained with other GP settings: "
                f"{previous_settings}"
            )

    # Record the settings these models were trained with:
    training_dict = {
        "experiment_dirpath": args.experiment_dirpath,
        "dataset_dirpath": dataset_dirpath,
        "wave_ids": [int(wave_id) for wave_id in wave_ids],
        "wave_row_counts": [int(wave_row_count) for wave_row_count in wave_row_counts],
        "latest_wave_id": latest_wave_id,
        "metric_name": args.metric_name,
        "gp_settings": gp_settings.to_dict(),
        "training_steps": gp_settings.training_steps,
        # Passes through the data that training_steps works out to, for the full model:
        "training_epochs": gp_settings.training_steps / math.ceil(parameter_matrix.shape[0] / gp_settings.batch_size),
        "k_fold_count": K_FOLD_COUNT,
        "inducing_point_count": gp_settings.inducing_point_count,
        "learning_rate": gp_settings.learning_rate,
        "deep_kernel_learning_rate": gp_settings.deep_kernel_learning_rate,
        "num_layers": gp_settings.hidden_layer_count,
        "num_neurons": gp_settings.hidden_layer_neuron_count,
        "spectral_coefficient": gp_settings.spectral_coefficient,
        "residual_block_count": gp_settings.residual_block_count,
        "residual_activation": RESIDUAL_ACTIVATION.__name__,
        # Bound on the Lipschitz constant of each residual block's branch; the warp is only
        # guaranteed to be invertible and bi-Lipschitz if it is below 1:
        "residual_lipschitz_bound": gp_settings.spectral_coefficient ** (gp_settings.hidden_layer_count + 1),
        "batch_size": gp_settings.batch_size,
        "original_row_count": int(original_row_count),
        "valid_row_count": int(parameter_matrix.shape[0]),
        "torch_version": torch.__version__,
        "gpytorch_version": gpytorch.__version__
    }
    with open(os.path.join(id_folderpath, "training_config.json"), 'w') as output:
        json.dump(training_dict, output, indent=4)

    # Set up the model and the model's associated training apparatus:
    print("Setting up model...")
    model_manager = DeepKernelModelManager(inducing_points, gp_settings)

    # Whiten metric distribution:
    metric_mean = np.mean(output_metric)
    metric_std = np.std(output_metric)
    whitened_metric = (output_metric - metric_mean) / metric_std

    # Save whitening transform:
    np.save(os.path.join(id_folderpath, "whiten_mean.npy"), metric_mean)
    np.save(os.path.join(id_folderpath, "whiten_std.npy"), metric_std)

    # Train the model:
    print("Training model...")
    model_manager.train_steps(parameter_matrix, whitened_metric, gp_settings.batch_size, gp_settings.training_steps)
    print("Saving model...")
    model_manager.save(id_folderpath)

    # Save loss history:
    loss_history = np.array(model_manager.loss_history)
    np.save(os.path.join(id_folderpath, "loss_history.npy"), loss_history)

    # Generate and save predictions:
    whitened_predictions, whitened_latent_stddev, whitened_stddev = run_inference(
        model_manager.model, model_manager.likelihood, parameter_matrix, batch_size=gp_settings.batch_size,
        standard_deviations=True
    )
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
            run_warp_inference(model_manager, parameter_matrix, gp_settings.batch_size), valid_indices, original_row_count
        ),
        warped_inducing_points=run_warp_inference(model_manager, final_inducing_points, gp_settings.batch_size)
    )


def run_cross_validation_stage(gp_settings, data):
    """Cross-validate the model, and estimate the Sobol' indices of the saved full model."""
    # The Sobol' indices come from the full model, so check it was trained on the same rows
    # with the same settings:
    training_config_filepath = os.path.join(data.id_folderpath, "training_config.json")
    model_filepath = os.path.join(data.id_folderpath, "model.pth")
    if not (os.path.exists(training_config_filepath) and os.path.exists(model_filepath)):
        raise FileNotFoundError(
            f"No full model found in {data.id_folderpath}: train it first (--stage full)."
        )
    with open(training_config_filepath) as training_config_fstream:
        training_config = json.load(training_config_fstream)
    run_record = {
        "gp_settings": gp_settings.to_dict(),
        "wave_ids": [int(wave_id) for wave_id in data.wave_ids],
    }
    for key, value in run_record.items():
        if training_config.get(key) != value:
            raise ValueError(
                f"The full model in {data.id_folderpath} was trained with {key} {training_config.get(key)}, "
                f"not {value}: train it again with these settings first (--stage full)."
            )

    # Perform cross-validation - the per-datapoint diagnostics and the per-fold
    # metrics are saved by the function itself as each fold finishes:
    cv_dict = cross_validation(
        data.parameter_matrix, data.output_metric, data.output_sem, data.replicate_counts, data.inducing_points,
        data.valid_indices, data.original_row_count, data.wave_indices, data.wave_row_indices, data.id_folderpath,
        gp_settings, run_record
    )
    for key in ["mae", "mse", "train_mse", "sll", "lcr", "tcr"]:
        print(f"Mean {key}: {np.mean(cv_dict[key])}")

    # Get Sobol' indices from the full GP model:
    print("Running Sobol' index inferece...", flush=True)
    model = load_torch_object(model_filepath)
    Si, STi = run_sobol_index_inference(model, data.parameter_matrix.shape[1])
    np.save(os.path.join(data.id_folderpath, "sobol_i.npy"), Si)
    np.save(os.path.join(data.id_folderpath, "sobol_Ti.npy"), STi)


def main():
    # Set seed:
    torch.manual_seed(0)

    # Parse arguments:
    args = parse_arguments()
    gp_settings = GPSettings.from_arguments(args)
    print(f"Using experiment {args.experiment_dirpath}...")
    print(f"Training on {args.metric_name}...")
    print(f"GP settings: {gp_settings.to_dict()}")
    lipschitz_bound = gp_settings.spectral_coefficient ** (gp_settings.hidden_layer_count + 1)
    if lipschitz_bound >= 1:
        print(
            f"Warning: the residual blocks' Lipschitz bound, spectral_coefficient ** (hidden_layer_count + 1) "
            f"= {lipschitz_bound:.3g}, is not below 1, so the warp is not guaranteed to be invertible "
            f"or bi-Lipschitz."
        )
    print(f"Running stage: {args.stage}...")

    data = prepare_training(args, gp_settings)
    if args.stage in ("all", "full"):
        train_full_model(args, gp_settings, data)
    if args.stage in ("all", "cross_validation"):
        run_cross_validation_stage(gp_settings, data)


if __name__ == "__main__":
    main()