import os
import argparse

import torch

import pandas as pd
import numpy as np

from statsmodels.regression import mixed_linear_model

from muscleabm.datasets import MODEL_METRICS, WETLAB_METRICS
from muscleabm.emulators import ModelManager, get_experiment_model_folderpath

torch.set_default_dtype(torch.float64)

PARAMETER_DIMENSION = 13
CAT_VAR = "C(phenotype, Treatment(reference='CTL'))[T.RD]"
QUERY_COUNTS = list(np.linspace(50, 300, 11).astype(int))
SCALED_QUERY_COUNTS = (np.array(QUERY_COUNTS) - 50) / (400 - 50)
GRIDSEARCH_COUNT_INDEX = 12
SEM_ESTIMATE = False
CHAIN_LENGTH = 32768 * 2
BATCH_SIZE = 8


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run parallel tempering MCMC against wet lab data using GP emulators')
    parser.add_argument(
        '--experiment_dirpath', type=str, required=True,
        help='Experiment folder containing sample_matrix.npy, summary_data and gaussian_process_models, '
             'e.g. model_experiments/2026-09-16-collisions_shape.'
    )
    args = parser.parse_args()
    return args


class MetricInferenceManager:

    def __init__(self, experiment_dirpath, metric_name, metric_data, dimension):
        # Internalise metric, and necessary calculations for inverse whitening:
        self.metric_data = metric_data
        self.metric_estimate = np.nanmean(metric_data, axis=1)
        self.metric_mean = np.nanmean(self.metric_estimate) 
        self.metric_std = np.nanstd(self.metric_estimate)

        # Do necessary calculations for inverse whitening of SEM estimates:
        nan_mask = np.any(np.isnan(metric_data), axis=1)
        metric_sem = np.std(metric_data[~nan_mask], axis=1) / np.sqrt(16)
        log_sem = np.log(metric_sem)
        self.logsem_mean = np.mean(log_sem)
        self.logsem_std = np.std(log_sem)

        # Instantiate and load mean GP model:
        inducing_points = np.zeros((32, dimension))
        self.mean_model_manager = ModelManager(inducing_points, 0.003)
        self.mean_model_manager.load(get_experiment_model_folderpath(experiment_dirpath, metric_name))

        # Instantiate and load SEM GP model:
        if SEM_ESTIMATE:
            self.sem_model_manager = ModelManager(inducing_points, 0.003)
            self.sem_model_manager.load(get_experiment_model_folderpath(experiment_dirpath, f"{metric_name}_noise"))

    def sem_emulate(self, input):
        # Place mean manager into eval mode:
        self.mean_model_manager.model.eval()
        self.mean_model_manager.likelihood.eval()

        # Place sem manager into eval mode:
        self.sem_model_manager.model.eval()
        self.sem_model_manager.likelihood.eval()

        # Prepare tensor input:
        tensor_input = torch.tensor(input)
        # Ensure that we have a batch dimension:
        if len(tensor_input.shape) == 1:
            tensor_input = torch.unsqueeze(tensor_input, 0)
        with torch.no_grad():
            # Get mean:
            mean_prediction = self.mean_model_manager.likelihood(
                self.mean_model_manager.model(tensor_input)
            )
            mean_prediction = mean_prediction.mean.detach().numpy()
            # Get SEM:
            sem_prediction = self.sem_model_manager.likelihood(
                self.sem_model_manager.model(tensor_input)
            )
            sem_prediction = sem_prediction.mean.detach().numpy()
        return mean_prediction, sem_prediction

    def sem_unwhiten_output(self, gp_mean, gp_sem):
        scaled_mean = (gp_mean * self.metric_std) + self.metric_mean
        scaled_logsem =  (gp_sem * self.logsem_std) + self.logsem_mean
        return scaled_mean, np.exp(scaled_logsem)

    def emulate(self, input):
        self.mean_model_manager.likelihood.eval()
        self.mean_model_manager.model.eval()
        tensor_input = torch.tensor(input)
        # Ensure that we have a batch dimension:
        if len(tensor_input.shape) == 1:
            tensor_input = torch.unsqueeze(tensor_input, 0)
        with torch.no_grad():
            prediction = self.mean_model_manager.likelihood(
                self.mean_model_manager.model(tensor_input)
            )
            prediction_mean = prediction.mean.detach().numpy()
            prediction_std = prediction.stddev.detach().numpy()
        return prediction_mean, prediction_std

    def unwhiten_output(self, gp_mean, gp_std):
        scaled_mean = (gp_mean * self.metric_std) + self.metric_mean
        scaled_std =  gp_std * self.metric_std
        return scaled_mean, scaled_std

    def estimate(self, input):
        if SEM_ESTIMATE:
            gp_mean, gp_sem = self.sem_emulate(input)
            scaled_mean, scaled_sem = self.sem_unwhiten_output(gp_mean, gp_sem)
        else:
            gp_mean, gp_sem = self.emulate(input)
            scaled_mean, scaled_sem = self.unwhiten_output(gp_mean, gp_sem)
        return scaled_mean, scaled_sem


def get_fit_target(phenotype, regression_results, scaled_x):
    # Get base parameters without phenotype interaction:
    base_intercept = regression_results.params["Intercept"]
    base_intercept_se = regression_results.bse["Intercept"]
    linear_coeff = regression_results.params["scaled_particle_count"]
    linear_coeff_se = regression_results.bse["scaled_particle_count"]
    square_coeff = regression_results.params["I(scaled_particle_count ** 2)"]
    square_coeff_se = regression_results.bse["I(scaled_particle_count ** 2)"]

    # Get phenotype interactions:
    phenotype_intercept = regression_results.params[f"{CAT_VAR}"]
    phenotype_intercept_se = regression_results.bse[f"{CAT_VAR}"]
    linear_interaction = regression_results.params[f"{CAT_VAR}:scaled_particle_count"]
    linear_interaction_se = regression_results.bse[f"{CAT_VAR}:scaled_particle_count"]
    square_interaction = regression_results.params[f"{CAT_VAR}:I(scaled_particle_count ** 2)"]
    square_interaction_se = regression_results.bse[f"{CAT_VAR}:I(scaled_particle_count ** 2)"]

    # Construct quadratic:
    a = square_coeff + (phenotype * square_interaction)
    b = linear_coeff + (phenotype * linear_interaction)
    c = base_intercept + (phenotype * phenotype_intercept)

    # Do error propagation to get standard errors in coefficients (multiplication preserves percentage errors):
    sq_int_se = np.abs((phenotype * square_interaction) * (square_interaction_se / square_interaction))
    lin_int_se = np.abs((phenotype * linear_interaction) * (linear_interaction_se / linear_interaction))
    base_int_se = np.abs((phenotype * phenotype_intercept) * (phenotype_intercept_se / phenotype_intercept))

    a_se = np.sqrt(square_coeff_se**2 + sq_int_se**2)
    b_se = np.sqrt(linear_coeff_se**2 + lin_int_se**2)
    c_se = np.sqrt(base_intercept_se**2 + base_int_se**2)

    # Get regression prediction:
    y = a*(scaled_x**2) + b*scaled_x + c

    # Get regression standard error:
    quad_error = (a_se / a) * (a*(scaled_x**2))
    linear_error = (b_se / b) * (b*scaled_x)
    se = np.sqrt(quad_error**2 + linear_error**2 + c_se**2)
    return y, se


def get_proposal(x, rng):
    candidate_components = x + rng.normal(loc=0.0, scale=0.025, size=x.shape)
    acceptance_mask = ~np.logical_or(candidate_components < 0, candidate_components > 1)
    sampled_x = np.copy(x)
    sampled_x[acceptance_mask] = candidate_components[acceptance_mask]
    return sampled_x


def ll_calculation(metric_data, mu, sigma):
    t_mean = metric_data["target"]
    t_se = metric_data["se"]
    absolute_error = np.abs(t_mean - mu)
    scale = 1 / np.sqrt(2 * np.pi * (t_se**2 + sigma**2))
    exponent = - absolute_error**2 / (2 * (t_se**2 + sigma**2))
    return np.log(scale) + exponent


def estimate_log_likelihoods(x, inference_managers, target_data):
    input_x = []
    for count in SCALED_QUERY_COUNTS:
        input_space = np.copy(x)
        input_space = np.insert(input_space, GRIDSEARCH_COUNT_INDEX, count, axis=1)
        input_x.append(input_space)
    input_x = np.concatenate(input_x, axis=0)

    log_likelihoods = []
    for inference_manager, metric_data in zip(inference_managers.values(), target_data.values()):
        estimates, stds = inference_manager.estimate(input_x)
        query_outputs = np.reshape(estimates, (len(QUERY_COUNTS), -1)).T
        query_errors = np.reshape(stds, (len(QUERY_COUNTS), -1)).T
        log_likelihood = ll_calculation(metric_data, query_outputs, query_errors)
        log_likelihoods.append(np.copy(log_likelihood))

    # Concatenate and summarise over different metrics:
    log_likelihoods = np.concatenate(log_likelihoods, axis=1)
    return np.sum(log_likelihoods, axis=1)


def run_ensemble_mcmc(inference_managers, target_data, batch_size, temperature_steps, chain_length):
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
        proposal_x = get_proposal(current_x.reshape(-1, PARAMETER_DIMENSION - 1), rng)
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

        # Reshape back to batch:
        current_x = current_x.reshape((-1, PARAMETER_DIMENSION - 1))
        current_likelihood = current_likelihood.flatten()

        # Add to chain:
        chain.append(np.copy(current_x))
        likelihoods.append(np.copy(current_likelihood))
        temperature_acceptance = np.count_nonzero(acceptance_mask.reshape((batch_size, temperature_steps)), axis=0)
        acceptance_rate.append(temperature_acceptance / batch_size)

    # Return chain:
    chain = np.stack(chain, axis=0)
    likelihoods = np.stack(likelihoods, axis=0) * tiled_temperatures

    # ---> (STEP, PARTICLE, TEMPERATURE, PARAMETER)
    levelled_chain = chain.reshape((chain_length, batch_size, temperature_steps, PARAMETER_DIMENSION - 1))
    levelled_likelihoods = likelihoods.reshape((chain_length, batch_size, temperature_steps))
    acceptance_rate = np.stack(acceptance_rate, axis=1)
    rung_acceptance_rate = rung_acceptance / rung_proposal_count
    return levelled_chain, levelled_likelihoods, acceptance_rate, rung_acceptance_rate


def get_posterior_predictions(x, inference_managers):
    # Get inputs across cell counts:
    input_x = []
    for count in SCALED_QUERY_COUNTS:
        input_space = np.copy(x)
        input_space = np.insert(input_space, GRIDSEARCH_COUNT_INDEX, count, axis=1)
        input_x.append(input_space)
    input_x = np.concatenate(input_x, axis=0)

    # Run inference across metric GP models:
    all_outputs = []
    all_errors = []
    for inference_manager in list(inference_managers.values()):
        estimates, stds = inference_manager.estimate(input_x)
        query_outputs = np.reshape(estimates, (len(QUERY_COUNTS), -1)).T
        query_errors = np.reshape(stds, (len(QUERY_COUNTS), -1)).T
        all_outputs.append(query_outputs)
        all_errors.append(query_errors)

    return np.stack(all_outputs, axis=0), np.stack(all_errors, axis=0)


def main():
    args = parse_arguments()

    # Load wet lab data:
    site_dataframe = pd.read_csv("wetlab_data/site_dataframe.csv")
    particle_counts = np.array(site_dataframe["particle_count"])
    site_dataframe["scaled_particle_count"] = (particle_counts - np.mean(particle_counts)) / np.std(particle_counts)

    # Get scaled points to query:
    scaled_query = (QUERY_COUNTS - np.mean(particle_counts)) / np.std(particle_counts)

    # Get the data we want to fit to from our regression results:
    wt_data = {}
    rd_data = {}
    for wetlab_metric in WETLAB_METRICS:
        # Retrieve results of regression, and do error propagation on parameters:
        regression_results = mixed_linear_model.MixedLMResults.load(f"wetlab_data/{wetlab_metric}.res")
        wt_fit_target, wt_fit_se = get_fit_target(-0.5, regression_results, scaled_query)
        rd_fit_target, rd_fit_se = get_fit_target( 0.5, regression_results, scaled_query)

        # Need to convert speed back to µm/min:
        if wetlab_metric == "mean_speed":
            wt_fit_target /= 60
            wt_fit_se /= 60
            rd_fit_target /= 60
            rd_fit_se /= 60

        # Accumulate to dictionary:
        print(f"-- -- -- -- {wetlab_metric} -- -- -- --", flush=True)
        print("CTL DATA: ", wt_fit_target, wt_fit_se)
        print("RD DATA: ", rd_fit_target, rd_fit_se)
        wt_data[wetlab_metric] = {"target": wt_fit_target, "se": wt_fit_se}
        rd_data[wetlab_metric] = {"target": rd_fit_target, "se": rd_fit_se}

    # Get parameter matrix:
    parameter_matrix = np.load(os.path.join(args.experiment_dirpath, "sample_matrix.npy"))
    reduced_parameter_matrix = np.concatenate(
        [
            parameter_matrix[:, :GRIDSEARCH_COUNT_INDEX],
            parameter_matrix[:, GRIDSEARCH_COUNT_INDEX + 1:]
        ], axis=1
    )

    # Load metrics and GP models:
    inference_managers = {}
    for metric_name in MODEL_METRICS:
        # Instantiate inference manager:
        metric_filepath = os.path.join(args.experiment_dirpath, "summary_data", f"{metric_name}.npy")
        model_metric = np.load(metric_filepath)
        inference_manager = MetricInferenceManager(
            args.experiment_dirpath, metric_name, model_metric, parameter_matrix.shape[1]
        )
        inference_managers[metric_name] = inference_manager

    # Estimate likelihood of parameters from Sobol' search:
    print("Estimating likelihood of gridsearch parameters...", flush=True)
    wt_sobol_likelihoods = estimate_log_likelihoods(reduced_parameter_matrix, inference_managers, wt_data)
    rd_sobol_likelihoods = estimate_log_likelihoods(reduced_parameter_matrix, inference_managers, rd_data)
    print("Saving Sobol' likelihoods...", flush=True)

    if SEM_ESTIMATE:
        mcmc_dirpath = os.path.join(args.experiment_dirpath, "sem_mcmc_results")
    else:
        mcmc_dirpath = os.path.join(args.experiment_dirpath, "wide_mcmc_results")
    if not os.path.exists(mcmc_dirpath):
        os.mkdir(mcmc_dirpath)
    np.save(os.path.join(mcmc_dirpath, "wt_sobol_likelihoods.npy"), wt_sobol_likelihoods)
    np.save(os.path.join(mcmc_dirpath, "rd_sobol_likelihoods.npy"), rd_sobol_likelihoods)

    # Estimate WT high likelihood parameter combination distribution with MCMC:
    print("Running parallel tempering MCMC for control data...")
    wt_mc_distribution, wt_likelihoods, wt_acceptance_rate, wt_rung_acceptance_rate = run_ensemble_mcmc(
        inference_managers, wt_data,
        batch_size=BATCH_SIZE, temperature_steps=6, chain_length=CHAIN_LENGTH
    )
    # Save results:
    print("Saving control results...")
    np.save(os.path.join(mcmc_dirpath, "wt_mcmc_chain.npy"), wt_mc_distribution)
    np.save(os.path.join(mcmc_dirpath, "wt_mcmc_likelihoods.npy"), wt_likelihoods)
    np.save(os.path.join(mcmc_dirpath, "wt_mcmc_acceptance_rate.npy"), wt_acceptance_rate)
    np.save(os.path.join(mcmc_dirpath, "wt_rung_acceptance_rate.npy"), wt_rung_acceptance_rate)

    # Run posterior inference:
    print("Running posterior inference for control distribution...")
    wt_posterior_mean, wt_posterior_std = get_posterior_predictions(
        wt_mc_distribution[int(CHAIN_LENGTH / 2)::64, :, 0, :].reshape(-1, PARAMETER_DIMENSION - 1),
        inference_managers
    )
    np.save(os.path.join(mcmc_dirpath, "wt_posterior_mean.npy"), wt_posterior_mean)
    np.save(os.path.join(mcmc_dirpath, "wt_posterior_std.npy"), wt_posterior_std)

    # Estimate RD high likelihood parameter combination distribution with MCMC:
    print("Running parallel tempering MCMC for RD data...")
    rd_mc_distribution, rd_likelihoods, rd_acceptance_rate, rd_rung_acceptance_rate = run_ensemble_mcmc(
        inference_managers, rd_data,
        batch_size=BATCH_SIZE, temperature_steps=6, chain_length=CHAIN_LENGTH
    )

    # Save results:
    print("Saving RD results...")
    np.save(os.path.join(mcmc_dirpath, "rd_mcmc_chain.npy"), rd_mc_distribution)
    np.save(os.path.join(mcmc_dirpath, "rd_mcmc_likelihoods.npy"), rd_likelihoods)
    np.save(os.path.join(mcmc_dirpath, "rd_mcmc_acceptance_rate.npy"), rd_acceptance_rate)
    np.save(os.path.join(mcmc_dirpath, "rd_rung_acceptance_rate.npy"), rd_rung_acceptance_rate)

    # Run posterior inference:
    print("Running posterior inference for RD distribution...")
    rd_posterior_mean, rd_posterior_std = get_posterior_predictions(
        rd_mc_distribution[int(CHAIN_LENGTH / 2)::64, :, 0, :].reshape(-1, PARAMETER_DIMENSION - 1),
        inference_managers
    )
    np.save(os.path.join(mcmc_dirpath, "rd_posterior_mean.npy"), rd_posterior_mean)
    np.save(os.path.join(mcmc_dirpath, "rd_posterior_std.npy"), rd_posterior_std)


if __name__ == "__main__":
    main()
