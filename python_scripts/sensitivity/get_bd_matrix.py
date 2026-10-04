import os
import argparse

import numpy as np

N_COMPONENTS = 20
METRICS_TO_LOAD = [
    "speeds",
    "meander_ratios",
    "ann_indices",
    "coherency",
    "op65"
]
LOG_DICT = {
    "speeds": False,
    "meander_ratios": False,
    "ann_indices": True,
    "coherency": True,
    "op65": True
}


def load_metrics(experiment_dirpath):
    metric_arrays = []
    for metric_name in METRICS_TO_LOAD:
        if metric_name == "op65":
            # Get order parameter:
            metric_array = np.load(os.path.join(experiment_dirpath, "summary_data", "matrix_order_parameters.npy"))
            if LOG_DICT[metric_name]:
                metric_arrays.append(np.log(metric_array[:, :, 2]))
            else:
                metric_arrays.append(metric_array[:, :, 2])
        else:
            metric_array = np.load(os.path.join(
                experiment_dirpath, "summary_data", f"{metric_name}.npy"
            ))
            if LOG_DICT[metric_name]:
                metric_arrays.append(np.log(metric_array))
            else:
                metric_arrays.append(metric_array)

    return np.stack(metric_arrays, axis=2)


def bg_vectorised(metric_set, global_correlation=None):
    # Set covariance matrix to be diagonal if none provided:
    if global_correlation is None:
        global_correlation = np.eye(metric_set.shape[-1])

    # Get dimensions:
    set_count, repeat_count, metric_count = metric_set.shape
    distance_matrix = np.zeros((set_count, set_count))

    # Calculate Bhattacharyya distance, iterating over rows (only calculate upper triangle):
    for set_index in range(set_count - 1):
        # Get comparator sets, to allow us to vectorise calculation:
        indexed_set = metric_set[set_index, :, :]
        comparator_sets = metric_set[set_index + 1:, :, :]

        # Get mean and standard deviations of summary statistics at each parameter set:
        indexed_mean = np.mean(indexed_set, axis=0)
        comparator_means = np.mean(comparator_sets, axis=1)
        indexed_std = np.std(indexed_set, axis=0, ddof=1)
        comparator_stds = np.std(comparator_sets, axis=1, ddof=1)

        # Broadcast std averaging:
        joint_stds = np.sqrt(0.5 * (indexed_std**2 + comparator_stds**2))

        # Broadcast covariance averaging:
        outer_indexed_std = indexed_std[None, :, None] * indexed_std[None, None, :]
        outer_comparator_stds = comparator_stds[:, :, None] * comparator_stds[:, None, :]
        averaged_covariances = 0.5 * (outer_indexed_std + outer_comparator_stds) * global_correlation[None]
        rescaled_covariances = averaged_covariances / (joint_stds[:, :, None] * joint_stds[:, None, :])

        # Get mean differences:
        scaled_mean_differences = (indexed_mean - comparator_means) / joint_stds
        scaled_solution = np.linalg.solve(rescaled_covariances, scaled_mean_differences[..., None])[..., 0]
        # Get dot product of solution with the scaled mean difference:
        mean_terms = 0.125 * np.einsum('nk,nk->n', scaled_mean_differences, scaled_solution)

        # Decompose upper part of fraction into 2.logdet + rescaled:
        _, rescaled_logdets = np.linalg.slogdet(rescaled_covariances)
        log_det = 2 * np.sum(np.log(joint_stds), axis=1) + rescaled_logdets

        # Decompose lower part of fraction into logdet[indexed] + logdet[comparators] + logdet[global]:
        _, log_det_global_covar = np.linalg.slogdet(global_correlation)
        pair_averages = np.log(indexed_std).sum() + np.log(comparator_stds).sum(axis=1) + log_det_global_covar
        covariance_terms = 0.5 * (log_det - pair_averages)

        distance_matrix[set_index, set_index + 1:] = mean_terms + covariance_terms

    return distance_matrix + distance_matrix.T


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dirpath", required=True,
        help="Experiment containing summary_data, e.g. model_experiments/2026-09-19-matrix_shape."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load output metrics:
    metric_array = load_metrics(args.experiment_dirpath)

    # Filter dataset for nan:
    filtered_indices = np.arange(metric_array.shape[0])
    finite_mask = np.all(np.isfinite(metric_array), axis=(1, 2))
    metric_array = np.delete(metric_array, ~finite_mask, axis=0)
    filtered_indices = np.delete(filtered_indices, ~finite_mask)

    # Remove very low variance sets:
    metric_std = metric_array.std(axis=1, ddof=1)
    variance_mask = np.any(metric_std < 1e-4, axis=1)
    print(f"Variance mask: {np.count_nonzero(variance_mask)} removed...", flush=True)
    metric_array = np.delete(metric_array, variance_mask, axis=0)
    filtered_indices = np.delete(filtered_indices, variance_mask)

    # Quick testing:
    metric_array = metric_array[::4, :]

    # Get BD distance matrix:
    print("Getting distance matrix...", flush=True)
    bd_distance_matrix = bg_vectorised(metric_array, global_correlation=None)

    # Take log, to allow for some degree of meaning:
    log_distances = np.log1p(bd_distance_matrix)
    del bd_distance_matrix

    # Perform MSD [kind of]:
    set_count = log_distances.shape[0]
    centering_matrix = np.eye(set_count) - (np.ones(set_count) / set_count)
    centred_matrix = centering_matrix @ log_distances @ centering_matrix
    del log_distances

    # Get subset eigendecomposition:
    print("Performing eigendecomposition...", flush=True)
    eigenvalues, eigenvectors = np.linalg.eigh(centred_matrix)
    del centred_matrix

    # Derive coordinates:
    ordered_indices = np.argsort(-np.abs(eigenvalues))
    eigenvalues, eigenvectors = eigenvalues[ordered_indices], eigenvectors[:, ordered_indices]
    coordinates = (eigenvectors * np.sqrt(np.abs(eigenvalues)))[:, :N_COMPONENTS]

    # Make directory:
    dirpath = os.path.join(args.experiment_dirpath, "inpca")
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    # Save outputs:
    print(f"Saving indices: {filtered_indices.shape}...")
    np.save(os.path.join(dirpath, "filtered_indices.npy"), filtered_indices)
    print(f"Saving coordinates: {coordinates.shape}...")
    np.save(os.path.join(dirpath, "coordinates.npy"), coordinates)


if __name__ == "__main__":
    main()
