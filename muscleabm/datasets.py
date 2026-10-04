"""Loading collated simulation data, and the metrics shared across scripts."""
import os

import numpy as np

# Folder of the experiment holding the data aggregated across history matching waves:
GLOBAL_DATASET_FOLDER = "global_dataset"

# Metrics measured from the simulations, and the wet lab metric each is fitted to:
MODEL_METRICS = [
    "speeds",
    "meander_ratios",
    "ann_indices",
    "coherency"
]
WETLAB_METRICS = [
    "mean_speed",
    "mean_mr",
    "anni",
    "coherency_fraction"
]

# Structural uncertainty added to the experimental covariance by the MCMC likelihood,
# as a fraction of the range of each wet lab curve:
DISCREPANCY_FRACTIONS = {
    "mean_speed": 0.05,
    "mean_mr": 0.05,
    "anni": 0.05,
    "coherency_fraction": 0.05
}


def load_gridsearch_data(dataset_dirpath, metric_name):
    """Load the parameters and summarised metric of every simulation that did not fail.

    dataset_dirpath holds sample_matrix.npy and summary_data/, e.g. a wave folder or the
    global dataset. Returns the parameters, the mean and standard error of the metric
    over replicates, the replicate count, the rows of the sample matrix that were kept,
    and the number of rows in the sample matrix.
    """
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
