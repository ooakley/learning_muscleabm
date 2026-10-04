import os
import argparse

import json

import torch
import numba

import scipy.stats

import colorcet as cc
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from statsmodels.regression import mixed_linear_model

from muscleabm.datasets import DISCREPANCY_FRACTIONS, GLOBAL_DATASET_FOLDER, MODEL_METRICS, WETLAB_METRICS
from muscleabm.emulators import ModelManager

# Set up parallelism across metrics:
from concurrent.futures import ThreadPoolExecutor
_metric_executor = ThreadPoolExecutor(max_workers=4)

print("--- --- Allocation summary: --- ---")
torch.set_num_threads(16)
print("Threads allocated to torch: ", torch.get_num_threads(), flush=True)

PARAMETER_DIMENSION = 13
CAT_VAR = "C(phenotype, Treatment(reference='CTL'))[T.RD]"
QUERY_COUNTS = list(np.linspace(50, 300, 11).astype(int))
SCALED_QUERY_COUNTS = (np.array(QUERY_COUNTS) - 50) / (400 - 50)
GRIDSEARCH_COUNT_INDEX = 12
CHAIN_LENGTH = 32768 * 2
BATCH_SIZE = 8
LENGTHSCALE_FRACTION = 1.0

# Number of cold chain samples used for the covariance ratio summary:
RATIO_SAMPLE_COUNT = 5

# Number of curves passed to the GP in a single flat call:
EMULATION_CHUNK_SIZE = 64

# Standard deviation of the random walk proposal, for every parameter and temperature.
# With adaptive proposals this is only the starting point:
PROPOSAL_SCALE = 0.02

# Adaptive proposals. During burn-in, the step size of each temperature is tuned towards
# this acceptance rate, and the shape of its proposal is learned from its samples:
TARGET_ACCEPTANCE_RATE = 0.234
ADAPTATION_GAIN_EXPONENT = 0.6
# Fractions of the burn-in after which samples start being collected for the proposal
# covariance (so that the initial transient is left out), and after which it is used:
COVARIANCE_COLLECTION_START = 0.1
COVARIANCE_USE_START = 0.2
# Number of steps between refreshes of the proposal covariance:
COVARIANCE_UPDATE_INTERVAL = 32


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run parallel tempering MCMC against wet lab data using GP emulators')
    parser.add_argument(
        '--experiment_dirpath', type=str, required=True,
        help='The {date}-{config} folder, containing config.json, the hm wave folders and global_dataset.'
    )
    parser.add_argument(
        '--gp_models_dirpath', type=str, required=True,
        help='Folder written by the GP training script, containing one subfolder per metric.'
    )
    parser.add_argument(
        '--hm_wave_id', type=int, required=True,
        help='History matching wave whose posterior is sampled, e.g. 0 for the hm0 folder.'
    )
    parser.add_argument(
        '--phenotype', type=str, required=True, choices=["WT", "RD"],
        help='Which posterior to sample. Run once per phenotype to sample both in parallel.'
    )
    parser.add_argument(
        '--no_calibration', action='store_true',
        help='Set the latent calibration ratio of every metric to 1, so that the GP latent '
             'covariance is used as is, without reading it from cv_metrics.json.'
    )
    parser.add_argument(
        '--experimental_lengthscales', type=float, nargs='*', default=None,
        help='Replace the GP latent covariance across design counts with the outer product of its '
             'pointwise standard deviations and a fixed correlation kernel, whose lengthscale comes '
             'from the wet lab GP fits. Give one lengthscale per wet lab metric (in particle counts, '
             'the units of x in the wet lab GP fit, in the order of WETLAB_METRICS), or give no values '
             'to load them from the wet lab GP results folder.'
    )
    parser.add_argument(
        '--burn_in_fraction', type=float, default=0.25,
        help='Fraction of the chain discarded as burn-in. Proposals are only adapted during burn-in.'
    )
    parser.add_argument(
        '--adaptive_proposals', action='store_true',
        help='Adapt the step size and covariance of the proposal of each temperature during burn-in, '
             'then freeze them for the rest of the chain.'
    )
    args = parser.parse_args()
    if not 0 < args.burn_in_fraction < 1:
        parser.error("--burn_in_fraction must be between 0 and 1.")
    if args.experimental_lengthscales is not None:
        if len(args.experimental_lengthscales) not in [0, len(WETLAB_METRICS)]:
            parser.error(
                f"--experimental_lengthscales takes no values or {len(WETLAB_METRICS)} values, "
                f"one for each of {WETLAB_METRICS}."
            )
        if any(lengthscale <= 0 for lengthscale in args.experimental_lengthscales):
            parser.error("--experimental_lengthscales must be positive.")
    return args


class MetricInferenceManager:

    def __init__(
            self, gp_models_dirpath, metric_name, metric_data, dimension,
            no_calibration=False, correlation_matrix=None
        ):
        # Folder holding this metric's trained GP model and its diagnostics:
        id_folderpath = os.path.join(gp_models_dirpath, metric_name)

        # Fixed correlation between design counts. If given, only the pointwise variances
        # of the GP are used, and its own covariance between design counts is discarded:
        self.correlation_matrix = correlation_matrix

        # Internalise metric, and necessary calculations for inverse whitening:
        self.metric_data = metric_data
        self.metric_estimate = np.nanmean(metric_data, axis=1)
        self.metric_mean = np.nanmean(self.metric_estimate) 
        self.metric_std = np.nanstd(self.metric_estimate)

        # Prefer the exact whitening transform saved at training time, as the
        # training script computes it over a slightly different set of rows:
        whiten_mean_filepath = os.path.join(id_folderpath, "whiten_mean.npy")
        whiten_std_filepath = os.path.join(id_folderpath, "whiten_std.npy")
        if os.path.exists(whiten_mean_filepath) and os.path.exists(whiten_std_filepath):
            self.metric_mean = float(np.load(whiten_mean_filepath))
            self.metric_std = float(np.load(whiten_std_filepath))
        else:
            print(f"No saved whitening transform for {metric_name}, recomputing from summary data.", flush=True)

        # The latent calibration ratio multiplies the latent covariance:
        if no_calibration:
            # Use the GP latent covariance as is:
            self.fold_calibration_ratios = None
            self.calibration_ratio = 1.0
            print(f"{metric_name} latent calibration ratio: 1.0 (calibration disabled)", flush=True)
        else:
            # Load the ratio from the cross-validation metrics of the training script,
            # averaged over folds:
            cv_filepath = os.path.join(id_folderpath, "cv_metrics.json")
            with open(cv_filepath) as cv_fstream:
                cv_dict = json.load(cv_fstream)
            if "lcr" not in cv_dict or len(cv_dict["lcr"]) == 0:
                raise KeyError(f"No latent calibration ratio ('lcr') found in {cv_filepath}.")
            self.fold_calibration_ratios = np.array(cv_dict["lcr"])
            self.calibration_ratio = float(np.mean(self.fold_calibration_ratios))
            if not np.isfinite(self.calibration_ratio) or self.calibration_ratio <= 0:
                raise ValueError(
                    f"Latent calibration ratio for {metric_name} is {self.calibration_ratio}, "
                    "it must be positive to scale a covariance matrix."
                )
            print(
                f"{metric_name} latent calibration ratio: {self.calibration_ratio} "
                f"(folds: {np.round(self.fold_calibration_ratios, 3)})", flush=True
            )

        # Do necessary calculations for inverse whitening of SEM estimates:
        nan_mask = np.any(np.isnan(metric_data), axis=1)
        metric_sem = np.std(metric_data[~nan_mask], axis=1) / np.sqrt(16)
        log_sem = np.log(metric_sem)
        self.logsem_mean = np.mean(log_sem)
        self.logsem_std = np.std(log_sem)

        # Instantiate and load mean GP model:
        inducing_points = np.zeros((32, dimension))
        self.mean_model_manager = ModelManager(inducing_points, 0.003)
        self.mean_model_manager.load(id_folderpath)

    def emulate(self, input):
        self.mean_model_manager.likelihood.eval()
        self.mean_model_manager.model.eval()
        tensor_input = torch.tensor(input)
        # Ensure that we have a batch dimension:
        if len(tensor_input.shape) == 1:
            tensor_input = torch.unsqueeze(tensor_input, 0)
        with torch.no_grad():
            # prediction = self.mean_model_manager.likelihood(
            #     self.mean_model_manager.model(tensor_input)
            # )
            if len(tensor_input.shape) != 3:
                prediction = self.mean_model_manager.model(tensor_input)
                prediction_mean = prediction.mean.detach().numpy()
                prediction_cov = prediction.covariance_matrix.detach().numpy()
                return prediction_mean, prediction_cov

            # Input is a set of curves, (curves, points, dimension). Passing this directly
            # makes GPyTorch copy the inducing points into every curve's batch and warp
            # them all again, so instead pass the points as one flat batch, and pick each
            # curve's own covariance block out of the joint covariance:
            curve_count, point_count, dimension = tensor_input.shape
            mean_chunks = []
            cov_chunks = []
            for chunk_start in range(0, curve_count, EMULATION_CHUNK_SIZE):
                chunk_input = tensor_input[chunk_start:chunk_start + EMULATION_CHUNK_SIZE]
                chunk_count = chunk_input.shape[0]
                prediction = self.mean_model_manager.model(
                    chunk_input.reshape(chunk_count * point_count, dimension)
                )
                joint_cov = prediction.covariance_matrix.reshape(
                    chunk_count, point_count, chunk_count, point_count
                )
                curve_indices = torch.arange(chunk_count)
                mean_chunks.append(prediction.mean.reshape(chunk_count, point_count))
                cov_chunks.append(joint_cov[curve_indices, :, curve_indices, :])
            prediction_mean = torch.cat(mean_chunks, dim=0).detach().numpy()
            prediction_cov = torch.cat(cov_chunks, dim=0).detach().numpy()
        return prediction_mean, prediction_cov

    def unwhiten_output(self, gp_mean, gp_cov):
        scaled_mean = (gp_mean * self.metric_std) + self.metric_mean

        # Rebuild the covariance as the outer product of the pointwise standard
        # deviations, multiplied by the fixed correlation between design counts:
        if self.correlation_matrix is not None:
            if gp_cov.shape[-2:] != self.correlation_matrix.shape:
                raise ValueError(
                    f"GP covariance has shape {gp_cov.shape}, which does not match the "
                    f"{self.correlation_matrix.shape} correlation matrix."
                )
            gp_std = np.sqrt(np.diagonal(gp_cov, axis1=-2, axis2=-1))
            gp_cov = gp_std[..., :, None] * gp_std[..., None, :] * self.correlation_matrix
        # Unwhiten the latent covariance, and rescale it by the latent calibration ratio:
        scaled_cov =  gp_cov * (self.metric_std ** 2) * self.calibration_ratio
        return scaled_mean, scaled_cov

    def estimate(self, input):
        gp_mean, gp_cov = self.emulate(input)
        scaled_mean, scaled_cov = self.unwhiten_output(gp_mean, gp_cov)
        return scaled_mean, scaled_cov


def get_fit_target_cov(phenotype, regression_results, scaled_x):
    # Set up coefficient ordering:
    param_names = [
        "Intercept", "scaled_particle_count", "I(scaled_particle_count ** 2)",
        f"{CAT_VAR}", f"{CAT_VAR}:scaled_particle_count", f"{CAT_VAR}:I(scaled_particle_count ** 2)",
    ]

    # Get coefficients and coefficient covariance:
    beta = regression_results.params[param_names].to_numpy()
    cov_beta = regression_results.cov_params().loc[param_names, param_names].to_numpy()

    scaled_x = np.asarray(scaled_x)
    ones = np.ones_like(scaled_x)
    F = np.column_stack(
        [
            ones, scaled_x, scaled_x**2,
            phenotype*ones, phenotype*scaled_x, phenotype*scaled_x**2
        ]
    )

    # Retrieve cov matrix for fixed effects:
    y = F @ beta
    sigma_fe = F @ cov_beta @ F.T

    # Retrieve cov matrix for random effects":
    cov_re = regression_results.cov_re
    re_terms = list(cov_re.columns)
    basis_lookup = {
        "Group": ones,
        "scaled_particle_count": scaled_x,
        "I(scaled_particle_count ** 2)": scaled_x**2,
    }
    Z = np.column_stack([basis_lookup[term] for term in re_terms])
    sigma_re = Z @ cov_re.to_numpy() @ Z.T

    print("- - - -")
    print(np.diag(sigma_fe) / np.diag(sigma_re))
    print("- - - -")
    return y, sigma_fe + sigma_re


def get_proposal(x, rng):
    candidate_components = x + rng.normal(loc=0.0, scale=PROPOSAL_SCALE, size=x.shape)
    lower_bound_mask = candidate_components < 0
    upper_bound_mask = candidate_components > 1
    candidate_components[lower_bound_mask] = -candidate_components[lower_bound_mask]
    candidate_components[upper_bound_mask] = 2 - candidate_components[upper_bound_mask]
    return candidate_components


def reflect_into_unit_cube(x):
    """Reflect values off the walls of the unit cube until they lie inside it.

    Unlike a single reflection, this stays correct for steps longer than the cube.
    """
    folded_x = np.mod(x, 2.0)
    return np.where(folded_x > 1.0, 2.0 - folded_x, folded_x)


class ProposalAdapter:
    """Adapts the random walk proposal of each temperature during burn-in.

    The proposal of a temperature is a Gaussian step with covariance scale^2 * shape. The
    scale is nudged after every step so that the acceptance rate approaches the target. The
    shape starts as isotropic, and is replaced by the covariance of the samples seen at that
    temperature (pooled over chains) once enough of them have been collected. Everything is
    frozen at the end of burn-in, so the rest of the chain uses a fixed proposal.
    """

    def __init__(self, temperature_steps, dimension, burn_in_step_count):
        self.temperature_steps = temperature_steps
        self.dimension = dimension
        self.burn_in_step_count = burn_in_step_count
        self.collection_start_step = int(COVARIANCE_COLLECTION_START * burn_in_step_count)
        self.use_start_step = int(COVARIANCE_USE_START * burn_in_step_count)

        # Proposal of each temperature - starts as the fixed isotropic proposal:
        self.scales = np.ones(temperature_steps)
        self.shapes = np.tile((PROPOSAL_SCALE ** 2) * np.eye(dimension), (temperature_steps, 1, 1))
        self.cholesky_factors = np.linalg.cholesky(self.shapes)
        self.using_sample_covariance = False
        self.scale_step_count = 0

        # Running mean and scatter matrix of the samples at each temperature:
        self.sample_count = 0
        self.sample_mean = np.zeros((temperature_steps, dimension))
        self.sample_scatter = np.zeros((temperature_steps, dimension, dimension))

        self.scale_history = []

    def propose(self, x, rng):
        """Propose a step for every row of x, ordered chain by chain as in the sampler."""
        temperature_indices = np.tile(np.arange(self.temperature_steps), x.shape[0] // self.temperature_steps)
        standard_steps = rng.normal(size=x.shape)
        steps = np.einsum("rij,rj->ri", self.cholesky_factors[temperature_indices], standard_steps)
        steps *= self.scales[temperature_indices, None]
        return reflect_into_unit_cube(x + steps)

    def get_scale_limits(self):
        # Keep the step of every parameter below the width of the cube. Hot chains accept
        # almost every step, so their scale would otherwise grow without limit:
        return 1.0 / np.sqrt(np.max(np.diagonal(self.shapes, axis1=-2, axis2=-1), axis=-1))

    def update(self, step_index, temperature_acceptance_rates, temperature_samples):
        """Adapt the proposals after a step. Does nothing once burn-in is over.

        temperature_acceptance_rates: (temperatures,) fraction of chains that accepted this step.
        temperature_samples:          (chains, temperatures, dimension) states after this step.
        """
        if step_index >= self.burn_in_step_count:
            return

        # Nudge the scale of each temperature towards the target acceptance rate, with a
        # gain that shrinks as the adaptation settles:
        self.scale_step_count += 1
        gain = self.scale_step_count ** (-ADAPTATION_GAIN_EXPONENT)
        log_scales = np.log(self.scales) + gain * (temperature_acceptance_rates - TARGET_ACCEPTANCE_RATE)
        self.scales = np.minimum(np.exp(log_scales), self.get_scale_limits())
        self.scale_history.append(np.copy(self.scales))

        # Collect samples for the covariance of each temperature:
        if step_index >= self.collection_start_step:
            chain_count = temperature_samples.shape[0]
            batch_mean = np.mean(temperature_samples, axis=0)
            centred_samples = temperature_samples - batch_mean[None, :, :]
            batch_scatter = np.einsum("cti,ctj->tij", centred_samples, centred_samples)
            new_count = self.sample_count + chain_count
            mean_shift = batch_mean - self.sample_mean
            self.sample_scatter += batch_scatter + (self.sample_count * chain_count / new_count) \
                * np.einsum("ti,tj->tij", mean_shift, mean_shift)
            self.sample_mean += mean_shift * (chain_count / new_count)
            self.sample_count = new_count

        # Replace the shape of each proposal with the covariance of its samples, with the
        # usual scaling for a random walk in this many dimensions:
        steps_since_use_start = step_index - self.use_start_step
        is_refresh_step = steps_since_use_start >= 0 and steps_since_use_start % COVARIANCE_UPDATE_INTERVAL == 0
        is_final_step = step_index == self.burn_in_step_count - 1
        if (is_refresh_step or is_final_step) and self.sample_count > 2 * self.dimension:
            sample_covariances = self.sample_scatter / (self.sample_count - 1)
            self.shapes = (2.38 ** 2 / self.dimension) * (sample_covariances + 1e-10 * np.eye(self.dimension))
            self.cholesky_factors = np.linalg.cholesky(self.shapes)
            if not self.using_sample_covariance:
                # The scale was tuned for the isotropic shape, so start it again:
                self.using_sample_covariance = True
                self.scales = np.ones(self.temperature_steps)
                self.scale_step_count = 0
            self.scales = np.minimum(self.scales, self.get_scale_limits())

    def get_record(self):
        return {
            "proposal_scales": self.scales,
            "proposal_covariances": (self.scales ** 2)[:, None, None] * self.shapes,
            "proposal_scale_history": np.array(self.scale_history)
        }


def batch_ll_calculation(gp_mean, gp_cov, exp_mean, exp_cov):
    """
    gp_mean:  (B, n)     exp_mean: (n,)
    gp_cov:   (B, n, n)  exp_cov:  (n, n)
    returns:  (B,) log-likelihood per batch element
    """
    n = gp_mean.shape[-1]
    log_pdf_const = n * np.log(2 * np.pi)

    joint_sigma = gp_cov + exp_cov[None, :, :]    # (B, n, n), broadcasts exp_cov over batch
    residual = gp_mean - exp_mean[None, :]        # (B, n)

    L = np.linalg.cholesky(joint_sigma) # batched Cholesky, (B, n, n)
    z = np.linalg.solve(L, residual[..., None])[..., 0] # batched triangular solve, (B, n)
    mahalanobis_sq = np.sum(z**2, axis=-1) # (B,) — sum over design points, per batch element
    log_det = 2 * np.sum(np.log(np.diagonal(L, axis1=-2, axis2=-1)), axis=-1)  # (B,)

    return -0.5 * (log_pdf_const + log_det + mahalanobis_sq)   # (B,)


def estimate_log_likelihoods(x, inference_managers, target_data):
    # Here the batch count is the number of chains by the number of temperature steps:
    batch = x.shape[0]
    n_counts = len(SCALED_QUERY_COUNTS)
    input_x = np.empty((batch, n_counts, PARAMETER_DIMENSION))
    for i, count in enumerate(SCALED_QUERY_COUNTS):
        input_x[:, i, :] = np.insert(x, GRIDSEARCH_COUNT_INDEX, count, axis=1)

    # # Collate log-likelihoods across metrics:
    # log_likelihoods = []
    # for inference_manager, (exp_mean, exp_cov) in zip(inference_managers.values(), target_data.values()):
    #     gp_mean, gp_cov = inference_manager.estimate(input_x)
    #     assert gp_mean.shape == (batch, n_counts)
    #     assert gp_cov.shape == (batch, n_counts, n_counts)
    #     log_likelihood = batch_ll_calculation(gp_mean, gp_cov, exp_mean, exp_cov)
    #     log_likelihoods.append(np.copy(log_likelihood))

    # # Concatenate and summarise over different metrics:
    # log_likelihoods = np.stack(log_likelihoods, axis=1)

    # Parallelise across metrics:
    def _run_one(inference_manager, exp_mean, exp_cov):
        gp_mean, gp_cov = inference_manager.estimate(input_x)
        return batch_ll_calculation(gp_mean, gp_cov, exp_mean, exp_cov)

    futures = [
        _metric_executor.submit(_run_one, im, exp_mean, exp_cov)
        for im, (exp_mean, exp_cov) in zip(inference_managers.values(), target_data.values())
    ]
    log_likelihoods = np.stack([f.result() for f in futures], axis=1)

    return np.sum(log_likelihoods, axis=1)


def run_ensemble_mcmc(
        inference_managers, target_data, batch_size, temperature_steps, chain_length,
        burn_in_fraction=0.25, adaptive_proposals=False
    ):
    # Set up adaptation of the proposals, which only happens during burn-in:
    proposal_adapter = None
    if adaptive_proposals:
        proposal_adapter = ProposalAdapter(
            temperature_steps, PARAMETER_DIMENSION - 1, int(chain_length * burn_in_fraction)
        )

    # Generate first candidate point & likelihood:
    rng = np.random.default_rng(0)
    base_temperatures = 2 ** (np.linspace(0, np.sqrt(8), temperature_steps) ** 2)
    tiled_temperatures = np.tile(base_temperatures, batch_size)
    current_x = rng.uniform(size=(batch_size * temperature_steps, PARAMETER_DIMENSION - 1))

    # Estimate current likelihood:
    current_likelihood = estimate_log_likelihoods(current_x, inference_managers, target_data)
    current_likelihood /= tiled_temperatures

    # Iterate through chain:
    chain = [np.copy(current_x)]
    likelihoods = [np.copy(current_likelihood)]
    acceptance_rate = []
    rung_acceptance = np.zeros(temperature_steps - 1)
    rung_proposal_count = np.zeros(temperature_steps - 1)
    for i in range(chain_length - 1):
        if (i + 1) % 32 == 0:
            print(f"Running step: {i + 1}...", flush=True)

        # Sample new x:
        if proposal_adapter is None:
            proposal_x = get_proposal(current_x.reshape(-1, PARAMETER_DIMENSION - 1), rng)
        else:
            proposal_x = proposal_adapter.propose(current_x.reshape(-1, PARAMETER_DIMENSION - 1), rng)
        proposal_likelihood = estimate_log_likelihoods(proposal_x, inference_managers, target_data)
        tempered_likelihood = proposal_likelihood / tiled_temperatures

        # Calculate acceptance criterion:
        acceptance_ratio = np.exp(tempered_likelihood - current_likelihood)
        uniform_sample = rng.uniform(size=(batch_size * temperature_steps))
        acceptance_mask = uniform_sample < acceptance_ratio

        # Update population:
        current_x[acceptance_mask, :] = proposal_x[acceptance_mask, :]
        current_likelihood[acceptance_mask] = tempered_likelihood[acceptance_mask]

        # Do temperature swaps, reshape for convenience:
        current_x = current_x.reshape((batch_size, temperature_steps, PARAMETER_DIMENSION - 1))
        base_likelihood = (current_likelihood * tiled_temperatures).reshape((batch_size, temperature_steps))
        current_likelihood = current_likelihood.reshape((batch_size, temperature_steps))
        swap_indices = rng.choice(temperature_steps - 1, size=batch_size)
        for batch_index in range(batch_size):
            # print(f"Running swap {batch_index}...")
            # Get sampled ranks to swap:
            swap_index = swap_indices[batch_index]
            rung_proposal_count[swap_index] += 1

            # Calculate swap probability:
            acceptance_numerator = (
                (base_likelihood[batch_index, swap_index] / base_temperatures[swap_index + 1])
                + (base_likelihood[batch_index, swap_index + 1] / base_temperatures[swap_index])
            )
            acceptance_denominator = current_likelihood[batch_index, swap_index] + current_likelihood[batch_index, swap_index + 1]
            swap_acceptance_ratio = np.exp(acceptance_numerator - acceptance_denominator)

            # Do temperature swap:
            sampled_acceptance = rng.uniform()
            if sampled_acceptance < swap_acceptance_ratio:
                # print("Sampled acceptance:", sampled_acceptance)
                # print("Acceptance ratio:", swap_acceptance_ratio)
                current_x[batch_index, [swap_index, swap_index + 1], :] = current_x[batch_index, [swap_index + 1, swap_index], :]
                current_likelihood[batch_index, [swap_index, swap_index + 1]] = (
                    base_likelihood[batch_index, [swap_index + 1, swap_index]] / base_temperatures[[swap_index, swap_index + 1]]
                )
                rung_acceptance[swap_index] += 1

        # Adapt the proposals to the acceptance rate and samples of each temperature:
        temperature_acceptance = np.count_nonzero(acceptance_mask.reshape((batch_size, temperature_steps)), axis=0)
        if proposal_adapter is not None:
            proposal_adapter.update(i, temperature_acceptance / batch_size, current_x)

        # Reshape back to batch:
        current_x = current_x.reshape((-1, PARAMETER_DIMENSION - 1))
        current_likelihood = current_likelihood.flatten()

        # Add to chain:
        chain.append(np.copy(current_x))
        likelihoods.append(np.copy(current_likelihood))
        acceptance_rate.append(temperature_acceptance / batch_size)

    # Return chain:
    chain = np.stack(chain, axis=0)
    likelihoods = np.stack(likelihoods, axis=0) * tiled_temperatures

    # ---> (STEP, PARTICLE, TEMPERATURE, PARAMETER)
    levelled_chain = chain.reshape((chain_length, batch_size, temperature_steps, PARAMETER_DIMENSION - 1))
    levelled_likelihoods = likelihoods.reshape((chain_length, batch_size, temperature_steps))
    acceptance_rate = np.stack(acceptance_rate, axis=1)
    rung_acceptance_rate = rung_acceptance / rung_proposal_count
    adaptation_record = None if proposal_adapter is None else proposal_adapter.get_record()
    return levelled_chain, levelled_likelihoods, acceptance_rate, rung_acceptance_rate, adaptation_record


def get_posterior_predictions(x, inference_managers):
    batch = x.shape[0]
    n_counts = len(SCALED_QUERY_COUNTS)
    input_x = np.empty((batch, n_counts, PARAMETER_DIMENSION))
    for i, count in enumerate(SCALED_QUERY_COUNTS):
        input_x[:, i, :] = np.insert(x, GRIDSEARCH_COUNT_INDEX, count, axis=1)

    all_outputs = []
    all_errors = []
    all_sigma_gp = []
    for inference_manager in inference_managers.values():
        gp_mean, gp_cov = inference_manager.estimate(input_x)          # (batch, n_counts), (batch, n_counts, n_counts)
        gp_std = np.sqrt(np.diagonal(gp_cov, axis1=-2, axis2=-1))      # (batch, n_counts) pointwise std
        all_outputs.append(gp_mean)
        all_errors.append(gp_std)
        all_sigma_gp.append(gp_cov)

    return np.stack(all_outputs, axis=0), np.stack(all_errors, axis=0), np.stack(all_sigma_gp, axis=0)


def get_structural_uncertainty_kernel(scaled_x, amplitude, lengthscale):
    distance_matrix = scaled_x[:, None] - scaled_x[None, :]
    return amplitude**2 * np.exp(-0.5 * (distance_matrix / lengthscale) ** 2)


def get_experimental_lengthscales(argument_lengthscales, gp_wetlab_dirpath):
    """Lengthscale of the wet lab GP fit of each metric, in particle counts.

    The wet lab GP is fitted against the raw particle count, so its lengthscale is in
    those units, and is compared against the raw design counts.

    Uses the command line values if any were given, and otherwise loads the lengthscale
    of each phenotype's fit from the wet lab GP results folder and averages them.
    """
    if len(argument_lengthscales) > 0:
        return dict(zip(WETLAB_METRICS, argument_lengthscales))

    lengthscales = {}
    for wetlab_metric in WETLAB_METRICS:
        phenotype_lengthscales = []
        for phenotype in ["CTL", "RD"]:
            lengthscale_filepath = os.path.join(gp_wetlab_dirpath, f"{phenotype}_{wetlab_metric}_lengthscale.npy")
            if not os.path.exists(lengthscale_filepath):
                raise FileNotFoundError(
                    f"No wet lab lengthscale found at {lengthscale_filepath}. Save the lengthscale of "
                    "the wet lab GP fit there, or give the lengthscales as values to "
                    "--experimental_lengthscales."
                )
            # A fitted lengthscale is often saved with shape (1,) or (1, 1), so accept any single value:
            lengthscale_array = np.load(lengthscale_filepath)
            if lengthscale_array.size != 1 or not lengthscale_array.reshape(-1)[0] > 0:
                raise ValueError(f"{lengthscale_filepath} must contain a single positive lengthscale.")
            phenotype_lengthscales.append(float(lengthscale_array.reshape(-1)[0]))
        lengthscales[wetlab_metric] = float(np.mean(phenotype_lengthscales))
    return lengthscales


def print_covariance_ratios(label, mc_distribution, inference_managers, target_data, burn_in_fraction):
    """Print a simple estimate of the experimental variance relative to the GP latent variance.

    Uses the diagonals of the covariances actually fed to the likelihood: the experimental
    covariance (including structural uncertainty), and the calibrated GP latent covariance
    averaged over a few samples from the cold chain after burn-in.
    """
    # Take evenly spaced samples from the cold chain after burn-in:
    chain_length = mc_distribution.shape[0]
    cold_samples = mc_distribution[int(chain_length * burn_in_fraction):, :, 0, :].reshape(-1, PARAMETER_DIMENSION - 1)
    sample_indices = np.linspace(0, cold_samples.shape[0] - 1, RATIO_SAMPLE_COUNT).astype(int)
    _, _, sigma_gp = get_posterior_predictions(cold_samples[sample_indices, :], inference_managers)

    print(f"--- --- {label}: experimental vs GP latent variance ({RATIO_SAMPLE_COUNT} cold chain samples) --- ---")
    for metric_index, (wetlab_metric, (_, exp_cov)) in enumerate(target_data.items()):
        exp_variance = np.mean(np.diag(exp_cov))
        gp_variance = np.mean(np.diagonal(sigma_gp[metric_index], axis1=-2, axis2=-1))
        print(
            f"{wetlab_metric}: experimental std. dev. {np.sqrt(exp_variance):.5g}, "
            f"GP latent std. dev. {np.sqrt(gp_variance):.5g}, "
            f"variance ratio (experimental / GP): {exp_variance / gp_variance:.3g}",
            flush=True
        )


def main():
    # Parse arguments:
    args = parse_arguments()
    print(f"Using GP models from {args.gp_models_dirpath}...", flush=True)
    print(f"Sampling {args.phenotype} posterior...", flush=True)
    burn_in_step_count = int(CHAIN_LENGTH * args.burn_in_fraction)
    print(f"Discarding the first {burn_in_step_count} of {CHAIN_LENGTH} steps as burn-in...", flush=True)
    if args.adaptive_proposals:
        print("Adapting the proposals during burn-in...", flush=True)

    # Results are saved to the folder of the given history matching wave:
    wave_dirpath = os.path.join(args.experiment_dirpath, f"hm{args.hm_wave_id}")
    if not os.path.exists(wave_dirpath):
        raise FileNotFoundError(f"No wave folder found at {wave_dirpath}.")
    print(f"Sampling posterior of wave {args.hm_wave_id}, saving to {wave_dirpath}...", flush=True)
    if os.path.commonpath([os.path.abspath(wave_dirpath), os.path.abspath(args.gp_models_dirpath)]) \
            != os.path.abspath(wave_dirpath):
        print(f"Warning: the GP models are not inside {wave_dirpath}.", flush=True)

    # Load wet lab data:
    site_dataframe = pd.read_csv("wetlab_data/site_dataframe.csv")
    particle_counts = np.array(site_dataframe["particle_count"])
    site_dataframe["scaled_particle_count"] = (particle_counts - np.mean(particle_counts)) / np.std(particle_counts)

    # Get scaled points to query:
    scaled_query = (QUERY_COUNTS - np.mean(particle_counts)) / np.std(particle_counts)

    # Get the data we want to fit to from our regression results:
    gp_wetlab_dirpath = os.path.join("wetlab_data", "gp_results")
    wt_data = {}
    rd_data = {}

    # Optionally build a fixed correlation between design counts for each metric, with the
    # lengthscale of the wet lab GP fit, to replace the GP emulator's own:
    experimental_lengthscales = None
    correlation_matrices = {model_metric: None for model_metric in MODEL_METRICS}
    if args.experimental_lengthscales is not None:
        experimental_lengthscales = get_experimental_lengthscales(args.experimental_lengthscales, gp_wetlab_dirpath)
        # The lengthscales are in particle counts, so the correlation is built over the raw design counts:
        query_counts = np.array(QUERY_COUNTS, dtype=float)
        query_spacing = query_counts[1] - query_counts[0]
        print("Using outer product GP covariance with experimental lengthscales...", flush=True)
        for wetlab_metric, model_metric in zip(WETLAB_METRICS, MODEL_METRICS):
            lengthscale = experimental_lengthscales[wetlab_metric]
            correlation_matrix = get_structural_uncertainty_kernel(query_counts, 1.0, lengthscale)
            correlation_matrices[model_metric] = correlation_matrix
            # Report the lengthscale relative to the design count spacing, as a check on its units:
            print(
                f"{wetlab_metric}: lengthscale {lengthscale:.5g} particles ({lengthscale / query_spacing:.3g} design "
                f"count spacings), correlation between adjacent counts {correlation_matrix[0, 1]:.3g}", flush=True
            )
            if correlation_matrix[0, 1] < 1e-3 or correlation_matrix[0, 1] > 0.9999:
                print(f"Warning: check the units of the {wetlab_metric} lengthscale.", flush=True)
    for wetlab_metric in WETLAB_METRICS:
        print(f"-- -- -- -- {wetlab_metric} -- -- -- --", flush=True)
        # # Retrieve results of regression, and do error propagation on parameters:
        # regression_results = mixed_linear_model.MixedLMResults.load(f"wetlab_data/{wetlab_metric}.res")
        # wt_fit_target, wt_fit_cov = get_fit_target_cov(0.0, regression_results, scaled_query)
        # rd_fit_target, rd_fit_cov = get_fit_target_cov(1.0, regression_results, scaled_query)

        # # Need to convert speed back to µm/min:
        # if wetlab_metric == "mean_speed":
        #     wt_fit_target /= 60
        #     wt_fit_cov /= (60 ** 2)
        #     rd_fit_target /= 60
        #     rd_fit_cov /= (60 ** 2)

        # Get targets from quick flexible GP regression:
        wt_fit_target = np.load(os.path.join(gp_wetlab_dirpath, f"CTL_{wetlab_metric}_mean.npy"))
        wt_fit_cov = np.load(os.path.join(gp_wetlab_dirpath, f"CTL_{wetlab_metric}_sigma.npy"))
        rd_fit_target = np.load(os.path.join(gp_wetlab_dirpath, f"RD_{wetlab_metric}_mean.npy"))
        rd_fit_cov = np.load(os.path.join(gp_wetlab_dirpath, f"RD_{wetlab_metric}_sigma.npy"))

        # Generate kernel that encodes structural uncertainty in whether outputs should be quadratic:
        point_lengthscale = LENGTHSCALE_FRACTION * (scaled_query[1] - scaled_query[0])
        print(f"point_lengthscale: {point_lengthscale}")
        wt_scale = DISCREPANCY_FRACTIONS[wetlab_metric] * np.ptp(wt_fit_target)
        rd_scale = DISCREPANCY_FRACTIONS[wetlab_metric] * np.ptp(rd_fit_target)
        su_amplitude = 0.5 * (wt_scale + rd_scale)
        print(f"su_amplitude: {su_amplitude}")
        su_kernel = get_structural_uncertainty_kernel(scaled_query, su_amplitude, point_lengthscale)
        wt_fit_cov += su_kernel
        rd_fit_cov += su_kernel

        # Accumulate to dictionary:
        print("CTL MEAN: ", np.round(wt_fit_target, 5))
        print("CTL COV: ", np.round(wt_fit_cov, 5))
        print("RD MEAN: ", np.round(rd_fit_target, 5))
        print("RD COV: ", np.round(rd_fit_cov, 5))
        wt_data[wetlab_metric] = (wt_fit_target, wt_fit_cov)
        rd_data[wetlab_metric] = (rd_fit_target, rd_fit_cov)

    # The GP models of a wave are trained on the global dataset, using every
    # wave up to and including their own:
    global_dirpath = os.path.join(args.experiment_dirpath, GLOBAL_DATASET_FOLDER)
    wave_indices = np.load(os.path.join(global_dirpath, "wave_indices.npy"))
    if not np.any(wave_indices == args.hm_wave_id):
        raise ValueError(
            f"The global dataset contains no rows from hm{args.hm_wave_id}, run collate_hm.py."
        )
    wave_mask = wave_indices <= args.hm_wave_id

    # Get parameter matrix:
    parameter_matrix = np.load(os.path.join(global_dirpath, "sample_matrix.npy"))[wave_mask, :]
    reduced_parameter_matrix = np.concatenate(
        [
            parameter_matrix[:, :GRIDSEARCH_COUNT_INDEX],
            parameter_matrix[:, GRIDSEARCH_COUNT_INDEX + 1:]
        ], axis=1
    )
    print(f"Reduced parameter shape: {reduced_parameter_matrix.shape}", flush=True)

    # Load metrics and GP models:
    inference_managers = {}
    for metric_name in MODEL_METRICS:
        # Instantiate inference manager:
        metric_filepath = os.path.join(global_dirpath, "summary_data", f"{metric_name}.npy")
        model_metric = np.load(metric_filepath)[wave_mask, :]
        inference_manager = MetricInferenceManager(
            args.gp_models_dirpath, metric_name, model_metric, parameter_matrix.shape[1],
            no_calibration=args.no_calibration, correlation_matrix=correlation_matrices[metric_name]
        )
        inference_managers[metric_name] = inference_manager
    
    # Set up save dirpath"
    mcmc_dirpath = os.path.join(wave_dirpath, "disc_cov_mcmc_results")
    os.makedirs(mcmc_dirpath, exist_ok=True)

    # Select the target data and the file prefix for the requested phenotype:
    if args.phenotype == "WT":
        target_data, file_prefix, label = wt_data, "wt", "CTL"
    else:
        target_data, file_prefix, label = rd_data, "rd", "RD"

    # Record which GP models and calibration ratios these results were generated with:
    calibration_dict = {
        "experiment_dirpath": args.experiment_dirpath,
        "hm_wave_id": args.hm_wave_id,
        "gp_models_dirpath": args.gp_models_dirpath,
        "burn_in_fraction": args.burn_in_fraction,
        "adaptive_proposals": args.adaptive_proposals,
        "calibration_applied": not args.no_calibration,
        "gp_covariance": "full" if experimental_lengthscales is None else "outer_product",
        "experimental_lengthscales": experimental_lengthscales,
        "latent_calibration_ratios": {
            metric_name: inference_manager.calibration_ratio
            for metric_name, inference_manager in inference_managers.items()
        }
    }
    with open(os.path.join(mcmc_dirpath, f"{file_prefix}_gp_calibration.json"), 'w') as output:
        json.dump(calibration_dict, output, indent=4)

    # # Estimate likelihood of parameters from Sobol' search:
    # print("Estimating likelihood of gridsearch parameters...", flush=True)
    # wt_sobol_likelihoods = estimate_log_likelihoods(reduced_parameter_matrix, inference_managers, wt_data)
    # rd_sobol_likelihoods = estimate_log_likelihoods(reduced_parameter_matrix, inference_managers, rd_data)

    # # Save outputs:
    # print("Saving Sobol' likelihoods...", flush=True)
    # np.save(os.path.join(mcmc_dirpath, "wt_sobol_likelihoods.npy"), wt_sobol_likelihoods)
    # np.save(os.path.join(mcmc_dirpath, "rd_sobol_likelihoods.npy"), rd_sobol_likelihoods)

    # Estimate high likelihood parameter combination distribution with MCMC:
    print(f"Running parallel tempering MCMC for {label} data...", flush=True)
    mc_distribution, likelihoods, acceptance_rate, rung_acceptance_rate, adaptation_record = run_ensemble_mcmc(
        inference_managers, target_data,
        batch_size=BATCH_SIZE, temperature_steps=6, chain_length=CHAIN_LENGTH,
        burn_in_fraction=args.burn_in_fraction, adaptive_proposals=args.adaptive_proposals
    )

    # Save results:
    print(f"Saving {label} results...", flush=True)
    np.save(os.path.join(mcmc_dirpath, f"{file_prefix}_mcmc_chain.npy"), mc_distribution)
    np.save(os.path.join(mcmc_dirpath, f"{file_prefix}_mcmc_likelihoods.npy"), likelihoods)
    np.save(os.path.join(mcmc_dirpath, f"{file_prefix}_mcmc_acceptance_rate.npy"), acceptance_rate)
    np.save(os.path.join(mcmc_dirpath, f"{file_prefix}_rung_acceptance_rate.npy"), rung_acceptance_rate)
    if adaptation_record is not None:
        # Final proposal of each temperature, and how its scale evolved over burn-in:
        for record_name, record_array in adaptation_record.items():
            np.save(os.path.join(mcmc_dirpath, f"{file_prefix}_{record_name}.npy"), record_array)
        print(f"Final proposal scales, by temperature: {np.round(adaptation_record['proposal_scales'], 3)}", flush=True)
        print(
            "Acceptance rates after burn-in, by temperature: "
            f"{np.round(np.mean(acceptance_rate[:, burn_in_step_count:], axis=1), 3)}", flush=True
        )
    print_covariance_ratios(label, mc_distribution, inference_managers, target_data, args.burn_in_fraction)

    # Run posterior inference:
    print(f"Running posterior inference for {label} distribution...", flush=True)
    posterior_mean, posterior_std, posterior_sigma = get_posterior_predictions(
        mc_distribution[burn_in_step_count::64, :, 0, :].reshape(-1, PARAMETER_DIMENSION - 1),
        inference_managers
    )
    np.save(os.path.join(mcmc_dirpath, f"{file_prefix}_posterior_mean.npy"), posterior_mean)
    np.save(os.path.join(mcmc_dirpath, f"{file_prefix}_posterior_std.npy"), posterior_std)
    np.save(os.path.join(mcmc_dirpath, f"{file_prefix}_posterior_sigma.npy"), posterior_sigma)

if __name__ == "__main__":
    main()