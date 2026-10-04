import os
import time
import argparse
import functools
import json

import torch

import numpy as np

from scipy.stats import qmc

from muscleabm.emulators import ModelManager, get_experiment_model_folderpath

DESIGN_POINTS = 16
PARAMETER_DIMENSION = 15
EXPONENT = 16
ESTIMATE_METRICS = [
    "speeds",
    "meander_ratios",
    "ann_indices",
    "coherency",
    "order_parameters"
]
BASELINE_SCHEDULE = np.array([1.0, 0.33, 0.1, 0.033, 0.01, 0.001, 0.0001]) / np.sqrt(12)
N_COMPONENTS = 6

# Need a higher precision for computing double derivatives:
torch.set_default_dtype(torch.float64)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dirpath", required=True,
        help="Experiment containing config.json and the GP models, e.g. model_experiments/2026-09-25-matrix_shape."
    )
    parser.add_argument(
        "--analysis_type", choices=["BASIC", "SOBOL"], default="BASIC",
        help="BASIC: estimate at a fresh Sobol' sequence. SOBOL: estimate at the experiment's own sample matrix."
    )
    parser.add_argument("--metric", required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--task_id", type=int, required=True)
    return parser.parse_args()


def generate_sobol_samples(parameter_dimension=PARAMETER_DIMENSION):
    # Set up sampler:
    sobol_sampler = qmc.Sobol(d=parameter_dimension, scramble=True, rng=0)
    sampled_inputs = sobol_sampler.random_base2(EXPONENT)

    # Exclude edges of parameter space (Hessian estimation begins to break down):
    trim_factor = 0.01
    sampled_inputs *= 1 - (trim_factor * 2)
    sampled_inputs += trim_factor
    return sampled_inputs


def retrieve_sobol_samples(experiment_dirpath):
    sobol_matrix = np.load(os.path.join(experiment_dirpath, "sample_matrix.npy"))
    low_mask = sobol_matrix < 0.01
    high_mask = sobol_matrix > 0.99
    exclude_mask = np.logical_or(np.any(low_mask, axis=1), np.any(high_mask, axis=1))
    return sobol_matrix[~exclude_mask, :]


def unit_to_scaled(unit_inputs, parameter_scaling):
    parameter_range = torch.squeeze(torch.diff(parameter_scaling))
    return (unit_inputs * parameter_range) + parameter_scaling[:, 0]


def scaled_to_unit(scaled_inputs, parameter_scaling):
    parameter_range = torch.squeeze(torch.diff(parameter_scaling))
    return (scaled_inputs - parameter_scaling[:, 0]) / parameter_range


def main():
    args = parse_arguments()

    # Generate parameter samples:
    if args.task_id == 0:
        print("Generating samples...", flush=True)

    # Generate parameter samples:
    analysis_type = args.analysis_type
    if args.task_id == 0:
        print(f"Generating {analysis_type} samples...", flush=True)
    if analysis_type == "BASIC":
        dir_path = os.path.join(args.experiment_dirpath, f"{args.metric}_fullrank_FIM")
        sampled_inputs = generate_sobol_samples(PARAMETER_DIMENSION - 1)
    elif analysis_type == "SOBOL":
        dir_path = os.path.join(args.experiment_dirpath, f"sobol_{args.metric}_fullrank_hessian")
        sampled_inputs = retrieve_sobol_samples(args.experiment_dirpath)
        sampled_inputs = sampled_inputs[:, :-1]

    # Generate outputs folder:
    if args.task_id == 0:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    # Get parameter scaling information:
    with open(os.path.join(args.experiment_dirpath, "config.json")) as json_file:
        config_dict = json.load(json_file)

    parameter_scaling = [parameter_range[1] for parameter_range in config_dict["gridsearch_parameters"]]
    parameter_scaling.pop(-1)
    parameter_scaling = np.stack(parameter_scaling, axis=0)
    parameter_scaling = torch.from_numpy(parameter_scaling)

    # Load Gaussian Process model:
    if args.task_id == 0:
        print("Loading model...", flush=True)
    model_manager = ModelManager(np.ones((10, PARAMETER_DIMENSION)), 0.003)
    model_manager.load(get_experiment_model_folderpath(args.experiment_dirpath, args.metric))

    # Print lengthscales:
    if args.task_id == 0:
        print("--- --- --- ---")
        print("Lengthscales:")
        print(model_manager.model.covar_module.base_kernel.lengthscale)
        print("--- --- --- ---", flush=True)

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
    scaled_inputs = unit_to_scaled(tensor_inputs, parameter_scaling)
    log_inputs = torch.log(scaled_inputs)

    # Get design points:
    design_array = torch.from_numpy(np.linspace(0.05, 0.95, DESIGN_POINTS))
    # log_design_array = torch.log(torch.from_numpy(design_array))

    # Define the function for which we want to retrieve the hessian:
    def emulator_mean(design_value, parameter_input):
        unit_inputs = scaled_to_unit(torch.exp(parameter_input), parameter_scaling)
        full_input = torch.cat([unit_inputs, design_value.reshape(1)])
        row_input = torch.unsqueeze(full_input, 0)
        prediction = model_manager.likelihood(model_manager.model(row_input))
        return prediction.mean

    # Loop over inputs to retrieve the data:
    if args.task_id == 0:
        print("Estimating FIMs...", flush=True)
    fims = []
    for i in range(log_inputs.shape[0]):
        # Print progress:
        if args.task_id == 0:
            if (i + 1) % 32 == 0 :
                print(f"Processing index: {i + 1}", flush=True)

        # Estimate outer products:
        outer_products = []
        for n in range(DESIGN_POINTS):
            design_mean = functools.partial(emulator_mean, design_array[n])
            jacobian = torch.autograd.functional.jacobian(
                design_mean, log_inputs[i, :]
            )
            np_jacobian = jacobian.detach().numpy()
            outer_products.append(np.outer(np_jacobian, np_jacobian))

        # Derive joint Hessian:
        summed_outer_product = np.sum(np.stack(outer_products, axis=0), axis=0)
        fisher_information_matrix = summed_outer_product / DESIGN_POINTS
        fims.append(fisher_information_matrix)

    fims = np.stack(fims, axis=0)
    if args.task_id == 0:
        print(f"Chunked FIM shape: {fims.shape}")

    # Save inputs and fims:
    if args.task_id == 0:
        print("Saving matrices...", flush=True)
    if args.task_id == 0:
        np.save(os.path.join(dir_path, "fim_inputs.npy"), sampled_inputs)
    np.save(os.path.join(dir_path, f"fim_{args.task_id}.npy"), fims)

    # Estimate outputs of sampled inputs:
    if args.task_id == 0:
        print("Running predictions of input samples...", flush=True)
        design_values = np.ones(sampled_inputs.shape[0]) * 0.25
        design_values = np.expand_dims(design_values, axis=1)
        full_inputs = torch.from_numpy(np.concatenate([sampled_inputs, design_values], axis=1))

        # Get output predictions:
        sample_predictions = model_manager.likelihood(model_manager.model(full_inputs))
        sample_predictions = sample_predictions.mean.detach().numpy()
        np.save(os.path.join(dir_path, "fim_outputs.npy"), sample_predictions)

        # Get metric predictions:
        for metric in ESTIMATE_METRICS:
            metric_manager = ModelManager(np.ones((10, PARAMETER_DIMENSION)), 0.003)
            metric_manager.load(get_experiment_model_folderpath(args.experiment_dirpath, metric))
            sample_predictions = metric_manager.likelihood(metric_manager.model(full_inputs))
            sample_predictions = sample_predictions.mean.detach().numpy()
            np.save(os.path.join(dir_path, f"predicted_{metric}.npy"), sample_predictions)

    # Collate FIMs if master process:
    if args.task_id == 0:
        # Wait while other processes finish:
        filename_list = [f"fim_{task_id}.npy" for task_id in range(args.world_size)]
        filepath_list = [os.path.join(dir_path, filename) for filename in filename_list]
        subprocesses_completing = True
        while subprocesses_completing:
            print("Checking subprocesses...", flush=True)
            completion_list = [os.path.exists(filepath) for filepath in filepath_list]
            if all(completion_list):
                subprocesses_completing = False
                continue
            time.sleep(15)

        # Collate the FIMs from all processes:
        print("Subprocesses complete!", flush=True)
        time.sleep(30)  # Ensure all files are fully written to disk:
        fims = [np.load(filepath) for filepath in filepath_list]
        fim_array = np.concatenate(fims, axis=0)

        # Save full array:
        print(f"Final FIM shape: {fim_array.shape}")
        print("Saving full array...", flush=True)
        np.save(os.path.join(dir_path, "fim_estimate.npy"), fim_array)

        # Remove sharded arrays:
        [os.remove(filepath) for filepath in filepath_list]


if __name__ == "__main__":
    main()
