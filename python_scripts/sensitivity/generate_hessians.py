import os
import time
import argparse

import torch

import numpy as np

from scipy.stats import qmc

from muscleabm.emulators import ModelManager, get_experiment_model_folderpath

PARAMETER_DIMENSION = 15
EXPONENT = 16

# Need a higher precision for computing double derivatives:
torch.set_default_dtype(torch.float64)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dirpath", required=True,
        help="Experiment containing the GP models, e.g. model_experiments/2026-09-19-matrix_shape."
    )
    parser.add_argument(
        "--analysis_type", choices=["FULL_RANK", "MLE", "SOBOL"], default="FULL_RANK",
        help="FULL_RANK: estimate at a fresh Sobol' sequence. MLE: estimate at the highest likelihood "
             "MCMC samples. SOBOL: estimate at the experiment's own sample matrix."
    )
    parser.add_argument(
        "--mcmc_dirpath", default=None,
        help="MLE only: folder containing the WT and RD MCMC chains and likelihoods. The Hessians "
             "are saved to its mle_hessians folder."
    )
    parser.add_argument("--metric", required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--task_id", type=int, required=True)
    args = parser.parse_args()
    if args.analysis_type == "MLE" and args.mcmc_dirpath is None:
        parser.error("--analysis_type MLE requires --mcmc_dirpath.")
    return args


def generate_sobol_samples(parameter_dimension=PARAMETER_DIMENSION):
    # Set up sampler:
    sobol_sampler = qmc.Sobol(d=parameter_dimension, scramble=True, rng=0)
    sampled_inputs = sobol_sampler.random_base2(EXPONENT)

    # Exclude edges of parameter space (Hessian estimation begins to break down):
    trim_factor = 0.02
    sampled_inputs *= 1 - (trim_factor * 2)
    sampled_inputs += trim_factor
    return sampled_inputs


def retrieve_posterior_samples(mcmc_dirpath):
    wt_chain = np.load(os.path.join(mcmc_dirpath, "wt_mcmc_chain.npy"))
    rd_chain = np.load(os.path.join(mcmc_dirpath, "rd_mcmc_chain.npy"))
    wt_posterior = wt_chain[16384::256, :, 0, :].reshape(-1, 11)
    rd_posterior = rd_chain[16384::256, :, 0, :].reshape(-1, 11)
    sampled_inputs = np.concatenate([wt_posterior, rd_posterior], axis=0)
    sampled_inputs = np.concatenate([sampled_inputs, np.ones((sampled_inputs.shape[0], 3)) * 0.5], axis=1)
    return sampled_inputs


def retrieve_mle_samples(mcmc_dirpath):
    ctl_chain = np.load(os.path.join(mcmc_dirpath, "wt_mcmc_chain.npy"))
    rd_chain = np.load(os.path.join(mcmc_dirpath, "rd_mcmc_chain.npy"))
    ctl_likelihoods = np.load(os.path.join(mcmc_dirpath, "wt_mcmc_likelihoods.npy"))
    rd_likelihoods = np.load(os.path.join(mcmc_dirpath, "rd_mcmc_likelihoods.npy"))

    THIN_FACTOR = 64
    ctl_mle_idx = np.argsort(ctl_likelihoods[::THIN_FACTOR, :, 0].flatten())[-1024:]
    rd_mle_idx = np.argsort(rd_likelihoods[::THIN_FACTOR, :, 0].flatten())[-1024:]
    ctl_mle = ctl_chain[::THIN_FACTOR, :, 0, :].reshape(-1, 11)[ctl_mle_idx, :]
    rd_mle = rd_chain[::THIN_FACTOR, :, 0, :].reshape(-1, 11)[rd_mle_idx, :]
    sampled_inputs = np.concatenate([ctl_mle, rd_mle], axis=0)
    sampled_inputs = np.concatenate([sampled_inputs, np.ones((sampled_inputs.shape[0], 3)) * 0.5], axis=1)
    return sampled_inputs


def retrieve_sobol_samples(experiment_dirpath):
    sobol_matrix = np.load(os.path.join(experiment_dirpath, "sample_matrix.npy"))
    low_mask = sobol_matrix < 0.02
    high_mask = sobol_matrix > 0.98
    exclude_mask = np.logical_or(np.any(low_mask, axis=1), np.any(high_mask, axis=1))
    return sobol_matrix[~exclude_mask, :]


def main():
    args = parse_arguments()
    analysis_type = args.analysis_type

    # Generate parameter samples:
    if args.task_id == 0:
        print("Generating samples...", flush=True)
    if analysis_type == "FULL_RANK":
        dir_path = get_experiment_model_folderpath(args.experiment_dirpath, args.metric)
        sampled_inputs = generate_sobol_samples()
    elif analysis_type == "MLE":
        dir_path = os.path.join(args.mcmc_dirpath, "mle_hessians")
        sampled_inputs = retrieve_mle_samples(args.mcmc_dirpath)
    elif analysis_type == "SOBOL":
        dir_path = os.path.join(args.experiment_dirpath, f"sobol_{args.metric}_hessian")
        sampled_inputs = retrieve_sobol_samples(args.experiment_dirpath)

    # Generate outputs folder:
    if args.task_id == 0:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    # Load Gaussian Process model:
    if args.task_id == 0:
        print("Loading model...", flush=True)
    model_manager = ModelManager(np.ones((10, PARAMETER_DIMENSION)), 0.003)
    model_manager.load(get_experiment_model_folderpath(args.experiment_dirpath, args.metric))

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
