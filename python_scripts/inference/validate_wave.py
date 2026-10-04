"""Tests the GP emulators of one history matching wave against the simulations of the next.

The simulations of wave n + 1 are run at samples from the posterior of wave n, so they are
held-out data for the wave n emulators, in the region of parameter space that the posterior
occupies. For each metric this reports the measured emulator error there, how it compares
with the variance the GP predicted, and how it compares with the experimental uncertainty.
"""
import argparse
import os
import json
import warnings

import gpytorch
import torch

import numpy as np

from scipy.optimize import minimize_scalar

# Match the precision used to train the GP models:
torch.set_default_dtype(torch.float64)

WETLAB_METRICS = [
    "mean_speed",
    "mean_mr",
    "anni",
    "coherency_fraction"
]
MODEL_METRICS = [
    "speeds",
    "meander_ratios",
    "ann_indices",
    "coherency"
]

# Wet lab condition of each phenotype label used by the wave generation scripts:
PHENOTYPE_CONDITIONS = {"WT": "CTL", "RD": "RD"}

# Structural uncertainty added to the experimental covariance by the MCMC script, as a
# fraction of the range of each wet lab curve. Keep in line with mcmc_inference.py:
DISCREPANCY_FRACTIONS = {
    "mean_speed": 0.05,
    "mean_mr": 0.05,
    "anni": 0.05,
    "coherency_fraction": 0.05
}

BATCH_SIZE = 512

# Number of resamples of the curves used to put an uncertainty on the error correlation:
BOOTSTRAP_COUNT = 200


def parse_arguments():
    parser = argparse.ArgumentParser(description='Test the GP emulators of one wave against the simulations of the next')
    parser.add_argument(
        '--experiment_dirpath', type=str, required=True,
        help='The {date}-{config} folder, containing config.json and the hm wave folders.'
    )
    parser.add_argument(
        '--hm_wave_id', type=int, required=True,
        help='History matching wave whose emulators are tested, e.g. 0 for the hm0 folder. '
             'They are tested against the simulations of the next wave, e.g. hm1.'
    )
    parser.add_argument(
        '--gp_models_dirpath', type=str, required=True,
        help='Folder written by the GP training script, containing one subfolder per metric.'
    )
    args = parser.parse_args()
    return args


# The models are saved whole, so classes with these names need to exist here for them to
# load. Their layers and parameters come from the saved model, not from these definitions:
class DeepInputTransformation(torch.nn.Module):
    def forward(self, x):
        return self.mlp.forward(x)


class SparseGPModel(gpytorch.models.ApproximateGP):
    def forward(self, x):
        # Warp input:
        warped_x = self.input_transform(x)

        # Calculate mean of input:
        mean_x = self.mean_module(warped_x)
        covar_x = self.covar_module(warped_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def load_wave_data(wave_dirpath, metric_name):
    # Load parameter and metric data:
    parameter_matrix = np.load(os.path.join(wave_dirpath, "sample_matrix.npy"))
    output_metric = np.load(os.path.join(wave_dirpath, "summary_data", f"{metric_name}.npy"))
    if output_metric.shape[0] != parameter_matrix.shape[0]:
        raise ValueError(
            f"{metric_name} has {output_metric.shape[0]} rows, but the sample matrix has "
            f"{parameter_matrix.shape[0]}."
        )

    # Summarise replicates in the same way as the training script, marking failed simulations
    # (numpy warns about rows with no valid replicates, which are the ones being marked):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        metric_mean = np.nanmean(output_metric, axis=1)
        n_valid = np.sum(~np.isnan(output_metric), axis=1)
        metric_sem = np.nanstd(output_metric, axis=1, ddof=1) / np.sqrt(n_valid)
    valid_mask = ~np.isnan(metric_sem)
    return parameter_matrix, metric_mean, metric_sem, valid_mask


def load_row_phenotypes(wave_dirpath, row_count):
    """Phenotype whose posterior each row of the wave was drawn from, or None if not recorded."""
    # Waves with one simulation per posterior sample:
    sample_phenotypes_filepath = os.path.join(wave_dirpath, "sample_phenotypes.npy")
    if os.path.exists(sample_phenotypes_filepath):
        return np.load(sample_phenotypes_filepath)

    # Waves of design curves, with every posterior sample simulated at every design count:
    curve_phenotypes_filepath = os.path.join(wave_dirpath, "curve_phenotypes.npy")
    if os.path.exists(curve_phenotypes_filepath):
        curve_phenotypes = np.load(curve_phenotypes_filepath)
        return np.repeat(curve_phenotypes, row_count // curve_phenotypes.shape[0])
    return None


def get_design_count_number(wave_dirpath, row_count):
    """Number of design counts per curve if the wave is a wave of design curves, else None."""
    design_counts_filepath = os.path.join(wave_dirpath, "design_counts.npy")
    if not os.path.exists(design_counts_filepath):
        return None
    design_count_number = np.load(design_counts_filepath).shape[0]
    if row_count % design_count_number != 0:
        raise ValueError(f"{row_count} rows cannot be split into curves of {design_count_number} design counts.")
    return design_count_number


def run_inference(model, likelihood, inputs, batch_size=512):
    """Whitened predictive mean, and latent and total variance, of the GP at each input."""
    tensor_input = torch.tensor(inputs)
    model.eval()
    likelihood.eval()
    mean_array = []
    latent_variance_array = []
    total_variance_array = []
    with torch.no_grad():
        for batch_start in range(0, tensor_input.shape[0], batch_size):
            latent_preds = model(tensor_input[batch_start:batch_start + batch_size])
            preds = likelihood(latent_preds)
            mean_array.append(latent_preds.mean.detach().numpy())
            latent_variance_array.append(latent_preds.variance.detach().numpy())
            total_variance_array.append(preds.variance.detach().numpy())
    return np.concatenate(mean_array), np.concatenate(latent_variance_array), np.concatenate(total_variance_array)


def get_experimental_variances(wetlab_metric):
    """Mean pointwise experimental variance of each phenotype, as used by the MCMC likelihood.

    This is the wet lab GP variance plus the structural uncertainty variance, averaged over
    the design counts. Returns None if the wet lab results are not found.
    """
    gp_wetlab_dirpath = os.path.join("wetlab_data", "gp_results")
    targets = {}
    variances = {}
    for phenotype, condition in PHENOTYPE_CONDITIONS.items():
        mean_filepath = os.path.join(gp_wetlab_dirpath, f"{condition}_{wetlab_metric}_mean.npy")
        sigma_filepath = os.path.join(gp_wetlab_dirpath, f"{condition}_{wetlab_metric}_sigma.npy")
        if not (os.path.exists(mean_filepath) and os.path.exists(sigma_filepath)):
            return None
        targets[phenotype] = np.load(mean_filepath)
        variances[phenotype] = np.mean(np.diag(np.load(sigma_filepath)))

    # The structural uncertainty kernel has the same amplitude for both phenotypes:
    su_amplitude = 0.5 * DISCREPANCY_FRACTIONS[wetlab_metric] * sum(np.ptp(target) for target in targets.values())
    return {phenotype: variance + su_amplitude ** 2 for phenotype, variance in variances.items()}


def summarise_group(residual, sem, latent_variance):
    """Held-out error of the emulator over a group of rows, in the units of the metric."""
    squared_error = np.mean(residual ** 2)
    # Replicate noise is not the emulator's error, so subtract it:
    emulator_variance = squared_error - np.mean(sem ** 2)
    standardised_residual = residual / np.sqrt(latent_variance + sem ** 2)
    return {
        "row_count": int(residual.shape[0]),
        # Whether the simulator sits above (positive) or below the emulator:
        "bias": float(np.mean(residual)),
        "rmse": float(np.sqrt(squared_error)),
        "replicate_noise_std": float(np.sqrt(np.mean(sem ** 2))),
        "emulator_error_std": float(np.sqrt(max(emulator_variance, 0.0))),
        "gp_latent_std": float(np.sqrt(np.mean(latent_variance))),
        # Measured emulator variance over the variance the GP predicted (> 1: overconfident):
        "calibration_ratio": float(emulator_variance / np.mean(latent_variance)),
        # Around 0.05 for a calibrated emulator with Gaussian errors:
        "fraction_beyond_2_sigma": float(np.mean(np.abs(standardised_residual) > 2))
    }


def get_lag_correlation(curve_residuals, curve_sems):
    """Emulator error covariance between design counts, and its correlation at each lag.

    Uses the second moments of the residuals over the curves. Replicate noise is independent
    between simulations, so it only adds to the diagonal, and is removed there. The correlation
    at each lag (in design count spacings) is pooled over all pairs of counts that far apart,
    which keeps it stable when the error at a single count is small next to the noise.
    """
    design_count_number = curve_residuals.shape[1]
    error_covariance = (curve_residuals.T @ curve_residuals) / curve_residuals.shape[0]
    error_covariance -= np.diag(np.mean(curve_sems ** 2, axis=0))
    pooled_variance = np.mean(np.diag(error_covariance))
    lag_correlation = np.array([
        np.mean(np.diagonal(error_covariance, offset=lag)) / pooled_variance
        for lag in range(1, design_count_number)
    ])
    return error_covariance, lag_correlation


def fit_lag_lengthscale(lag_correlation):
    """Lengthscale (in design count spacings) of the squared exponential closest to the lag correlations."""
    lags = np.arange(1, lag_correlation.shape[0] + 1)
    # Weight each lag by the number of pairs of counts it was pooled over:
    weights = lag_correlation.shape[0] + 1 - lags
    def loss(log_lengthscale):
        return np.sum(weights * (lag_correlation - np.exp(-0.5 * (lags / np.exp(log_lengthscale)) ** 2)) ** 2)
    result = minimize_scalar(loss, bounds=(np.log(0.05), np.log(1000)), method="bounded")
    return float(np.exp(result.x))


def get_curve_error_structure(residual, sem, valid_mask, design_count_number):
    """How the emulator's error is related between the design counts of the same curve.

    Returns the error covariance, the lag correlations and the fitted lengthscale over the
    complete curves, with their uncertainty from resampling the curves.
    """
    complete_curve_mask = np.all(valid_mask.reshape(-1, design_count_number), axis=1)
    curve_residuals = residual.reshape(-1, design_count_number)[complete_curve_mask]
    curve_sems = sem.reshape(-1, design_count_number)[complete_curve_mask]
    curve_count = curve_residuals.shape[0]

    error_covariance, lag_correlation = get_lag_correlation(curve_residuals, curve_sems)
    lengthscale = fit_lag_lengthscale(lag_correlation)

    # Resample the curves to see how well the data pins these down:
    rng = np.random.default_rng(0)
    bootstrap_correlations = []
    bootstrap_lengthscales = []
    for _ in range(BOOTSTRAP_COUNT):
        resampled_indices = rng.integers(curve_count, size=curve_count)
        _, resampled_correlation = get_lag_correlation(curve_residuals[resampled_indices], curve_sems[resampled_indices])
        bootstrap_correlations.append(resampled_correlation)
        bootstrap_lengthscales.append(fit_lag_lengthscale(resampled_correlation))
    return {
        "curve_count": int(curve_count),
        "error_covariance": error_covariance,
        "lag_correlation": lag_correlation,
        "lag_correlation_std_error": np.std(bootstrap_correlations, axis=0),
        "lengthscale": lengthscale,
        "lengthscale_interval": np.percentile(bootstrap_lengthscales, [16, 84])
    }


def main():
    # Parse arguments:
    args = parse_arguments()
    next_wave_id = args.hm_wave_id + 1
    next_wave_dirpath = os.path.join(args.experiment_dirpath, f"hm{next_wave_id}")
    print(f"Testing GP models from {args.gp_models_dirpath} against the simulations of {next_wave_dirpath}...")
    if not os.path.exists(os.path.join(next_wave_dirpath, "summary_data")):
        raise FileNotFoundError(f"No summary data found in {next_wave_dirpath}, run collate_data.py for hm{next_wave_id}.")

    validation_dict = {
        "hm_wave_id": args.hm_wave_id,
        "validation_wave_id": next_wave_id,
        "gp_models_dirpath": args.gp_models_dirpath,
        "metrics": {}
    }
    for wetlab_metric, metric_name in zip(WETLAB_METRICS, MODEL_METRICS):
        print(f"-- -- -- -- {metric_name} -- -- -- --")
        id_folderpath = os.path.join(args.gp_models_dirpath, metric_name)

        # The test is only fair if the models have not been trained on the next wave:
        training_config_filepath = os.path.join(id_folderpath, "training_config.json")
        if os.path.exists(training_config_filepath):
            with open(training_config_filepath) as training_fstream:
                training_wave_ids = json.load(training_fstream).get("wave_ids", [])
            if next_wave_id in training_wave_ids:
                print(f"Warning: these models were trained on hm{next_wave_id}, so this is not held-out data.")

        # Load held-out simulations and the phenotype each was drawn for:
        parameter_matrix, metric_mean, metric_sem, valid_mask = load_wave_data(next_wave_dirpath, metric_name)
        row_phenotypes = load_row_phenotypes(next_wave_dirpath, parameter_matrix.shape[0])
        design_count_number = get_design_count_number(next_wave_dirpath, parameter_matrix.shape[0])
        print(f"Using {np.count_nonzero(valid_mask)} of {parameter_matrix.shape[0]} simulations...")

        # Load model and its whitening transform:
        model = torch.load(os.path.join(id_folderpath, "model.pth"), weights_only=False)
        likelihood = torch.load(os.path.join(id_folderpath, "likelihood.pth"), weights_only=False)
        whiten_mean = float(np.load(os.path.join(id_folderpath, "whiten_mean.npy")))
        whiten_std = float(np.load(os.path.join(id_folderpath, "whiten_std.npy")))

        # Predict at every row of the next wave, and convert to the units of the metric:
        whitened_mean, whitened_latent_variance, whitened_total_variance = \
            run_inference(model, likelihood, parameter_matrix, BATCH_SIZE)
        prediction = (whitened_mean * whiten_std) + whiten_mean
        latent_variance = whitened_latent_variance * whiten_std ** 2
        total_variance = whitened_total_variance * whiten_std ** 2
        residual = metric_mean - prediction

        # Summarise over all rows, and over the rows drawn for each phenotype:
        groups = {"all": valid_mask}
        if row_phenotypes is not None:
            for phenotype in PHENOTYPE_CONDITIONS:
                groups[phenotype] = valid_mask & (row_phenotypes == phenotype)
        experimental_variances = get_experimental_variances(wetlab_metric)
        if experimental_variances is None:
            print("No wet lab results found, skipping the comparison with experimental uncertainty.")
        metric_dict = {"groups": {}}
        for group_name, group_mask in groups.items():
            if np.count_nonzero(group_mask) == 0:
                continue
            group_dict = summarise_group(residual[group_mask], metric_sem[group_mask], latent_variance[group_mask])

            # Compare with the experimental uncertainty the MCMC likelihood uses:
            if experimental_variances is not None:
                if group_name == "all":
                    experimental_std = np.sqrt(np.mean(list(experimental_variances.values())))
                else:
                    experimental_std = np.sqrt(experimental_variances[group_name])
                group_dict["experimental_std"] = float(experimental_std)
                group_dict["emulator_error_std_over_experimental_std"] = float(group_dict["emulator_error_std"] / experimental_std)
                group_dict["gp_latent_std_over_experimental_std"] = float(group_dict["gp_latent_std"] / experimental_std)
            metric_dict["groups"][group_name] = group_dict

            print(
                f"{group_name} ({group_dict['row_count']} rows): emulator error std. dev. {group_dict['emulator_error_std']:.5g}, "
                f"GP latent std. dev. {group_dict['gp_latent_std']:.5g}, calibration ratio {group_dict['calibration_ratio']:.3g}, "
                f"bias {group_dict['bias']:.3g}, beyond 2 sigma {group_dict['fraction_beyond_2_sigma']:.3f}"
            )
            if experimental_variances is not None:
                print(
                    f"    experimental std. dev. {group_dict['experimental_std']:.5g}: emulator error is "
                    f"{group_dict['emulator_error_std_over_experimental_std']:.3g} times this "
                    f"(the GP claims {group_dict['gp_latent_std_over_experimental_std']:.3g} times)"
                )

        # Compare with the calibration ratio measured by cross-validation over the whole sweep:
        cv_filepath = os.path.join(id_folderpath, "cv_metrics.json")
        if os.path.exists(cv_filepath):
            with open(cv_filepath) as cv_fstream:
                cv_dict = json.load(cv_fstream)
            if len(cv_dict.get("lcr", [])) > 0:
                metric_dict["cross_validation_calibration_ratio"] = float(np.mean(cv_dict["lcr"]))
                print(f"Cross-validation calibration ratio, for comparison: {metric_dict['cross_validation_calibration_ratio']:.3g}")

        # For waves of design curves, measure how the emulator's error is correlated across counts:
        curve_arrays = {}
        if design_count_number is not None:
            curve_structure = get_curve_error_structure(residual, metric_sem, valid_mask, design_count_number)
            lengthscale_interval = curve_structure["lengthscale_interval"]
            metric_dict["curve_count"] = curve_structure["curve_count"]
            metric_dict["lag_error_correlation"] = [float(value) for value in curve_structure["lag_correlation"]]
            metric_dict["lag_error_correlation_std_error"] = [float(value) for value in curve_structure["lag_correlation_std_error"]]
            metric_dict["error_lengthscale_in_spacings"] = curve_structure["lengthscale"]
            metric_dict["error_lengthscale_interval_in_spacings"] = [float(value) for value in lengthscale_interval]
            print(f"Error correlation over {curve_structure['curve_count']} curves, by lag in design count spacings:")
            print(f"    correlation: {np.round(curve_structure['lag_correlation'], 2)}")
            print(f"    std. error:  {np.round(curve_structure['lag_correlation_std_error'], 2)}")
            print(
                f"Fitted error lengthscale: {curve_structure['lengthscale']:.3g} design count spacings "
                f"(68% interval {lengthscale_interval[0]:.3g} to {lengthscale_interval[1]:.3g})"
            )
            curve_arrays["error_covariance"] = curve_structure["error_covariance"]
            curve_arrays["lag_error_correlation"] = curve_structure["lag_correlation"]
            curve_arrays["lag_error_correlation_std_error"] = curve_structure["lag_correlation_std_error"]

        # Save the per-row arrays, against rows of the next wave's sample matrix:
        np.savez(
            os.path.join(id_folderpath, "wave_validation.npz"),
            valid_mask=valid_mask,
            target=metric_mean,
            sem=metric_sem,
            prediction=prediction,
            latent_std=np.sqrt(latent_variance),
            total_std=np.sqrt(total_variance),
            residual=residual,
            **({"phenotype": row_phenotypes} if row_phenotypes is not None else {}),
            **curve_arrays
        )
        validation_dict["metrics"][metric_name] = metric_dict

    # Save the summary of every metric:
    with open(os.path.join(args.gp_models_dirpath, "wave_validation.json"), 'w') as output:
        json.dump(validation_dict, output, indent=4)


if __name__ == "__main__":
    main()