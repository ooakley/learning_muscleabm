"""Collates the sample matrices and summary data of every history matching wave.

Writes the concatenated dataset into the global_dataset folder of the experiment,
laid out like a single wave (sample_matrix.npy and summary_data/), so that it can
be loaded in the same way.
"""
import argparse
import os
import re
import json

import numpy as np

from muscleabm.datasets import GLOBAL_DATASET_FOLDER, MODEL_METRICS


def parse_arguments():
    parser = argparse.ArgumentParser(description='Collate summary data from across history matching waves.')
    parser.add_argument(
        '--experiment_dirpath', type=str, required=True,
        help='The {date}-{config} folder, containing config.json and the hm wave folders.'
    )
    parser.add_argument(
        '--max_failed_fraction', type=float, default=None,
        help='Stop with an error if more than this fraction of the simulations of any wave failed '
             'for any metric. By default no check is made.'
    )
    parser.add_argument(
        '--required_hm_wave_id', type=int, default=None,
        help='Stop with an error unless this wave is part of the collated dataset. Waves without '
             'summary data are otherwise skipped silently.'
    )
    args = parser.parse_args()
    if args.max_failed_fraction is not None and not 0 <= args.max_failed_fraction <= 1:
        parser.error('--max_failed_fraction must be between 0 and 1.')
    return args


def get_wave_ids(experiment_dirpath):
    # Wave folders are named hm0, hm1, ...:
    wave_ids = []
    for folder_name in os.listdir(experiment_dirpath):
        folder_match = re.fullmatch(r"hm(\d+)", folder_name)
        if folder_match and os.path.isdir(os.path.join(experiment_dirpath, folder_name)):
            wave_ids.append(int(folder_match.group(1)))
    return sorted(wave_ids)


def load_wave_data(experiment_dirpath, wave_id):
    """Load the sample matrix and summary data of one wave.

    Returns None if the wave has no summary data yet (generated, but not yet
    simulated and collated).
    """
    wave_dirpath = os.path.join(experiment_dirpath, f"hm{wave_id}")

    # Check which metrics have been collated for this wave:
    metric_filepaths = {
        metric_name: os.path.join(wave_dirpath, "summary_data", f"{metric_name}.npy")
        for metric_name in MODEL_METRICS
    }
    missing_metrics = [
        metric_name for metric_name, metric_filepath in metric_filepaths.items()
        if not os.path.exists(metric_filepath)
    ]
    if len(missing_metrics) == len(MODEL_METRICS):
        return None
    if len(missing_metrics) > 0:
        raise FileNotFoundError(f"hm{wave_id} is missing summary data for: {missing_metrics}.")

    # Load parameters and metrics, checking there is one row per simulation in each:
    sample_matrix = np.load(os.path.join(wave_dirpath, "sample_matrix.npy"))
    metric_data = {}
    for metric_name, metric_filepath in metric_filepaths.items():
        metric_matrix = np.load(metric_filepath)
        if metric_matrix.shape[0] != sample_matrix.shape[0]:
            raise ValueError(
                f"hm{wave_id}: {metric_name} has {metric_matrix.shape[0]} rows, "
                f"but the sample matrix has {sample_matrix.shape[0]}."
            )
        metric_data[metric_name] = metric_matrix
    return sample_matrix, metric_data


def main():
    # Parse arguments:
    args = parse_arguments()
    print(f"Collating waves from {args.experiment_dirpath}...", flush=True)

    # Load every wave that has summary data:
    wave_ids = get_wave_ids(args.experiment_dirpath)
    if len(wave_ids) == 0:
        raise FileNotFoundError(f"No hm wave folders found in {args.experiment_dirpath}.")
    collated_wave_ids = []
    sample_matrices = []
    metric_matrices = {metric_name: [] for metric_name in MODEL_METRICS}
    for wave_id in wave_ids:
        wave_data = load_wave_data(args.experiment_dirpath, wave_id)
        if wave_data is None:
            print(f"hm{wave_id}: no summary data found, skipping...", flush=True)
            continue
        sample_matrix, metric_data = wave_data
        print(f"hm{wave_id}: {sample_matrix.shape[0]} simulations", flush=True)

        # A simulation has failed for a metric if it has no valid replicates:
        for metric_name in MODEL_METRICS:
            failed_fraction = np.mean(np.all(np.isnan(metric_data[metric_name]), axis=1))
            if failed_fraction > 0:
                print(f"hm{wave_id}: {failed_fraction:.2%} of simulations failed for {metric_name}", flush=True)
            if args.max_failed_fraction is not None and failed_fraction > args.max_failed_fraction:
                raise ValueError(
                    f"hm{wave_id}: {failed_fraction:.2%} of simulations failed for {metric_name}, "
                    f"above the limit of {args.max_failed_fraction:.2%}."
                )

        # Waves can only be concatenated if they share parameters and replicate counts:
        if len(sample_matrices) > 0:
            if sample_matrix.shape[1] != sample_matrices[0].shape[1]:
                raise ValueError(
                    f"hm{wave_id} has {sample_matrix.shape[1]} parameters, "
                    f"but hm{collated_wave_ids[0]} has {sample_matrices[0].shape[1]}."
                )
            for metric_name in MODEL_METRICS:
                if metric_data[metric_name].shape[1:] != metric_matrices[metric_name][0].shape[1:]:
                    raise ValueError(
                        f"hm{wave_id}: {metric_name} has shape {metric_data[metric_name].shape}, which "
                        f"does not match {metric_matrices[metric_name][0].shape} in hm{collated_wave_ids[0]}."
                    )

        collated_wave_ids.append(wave_id)
        sample_matrices.append(sample_matrix)
        for metric_name in MODEL_METRICS:
            metric_matrices[metric_name].append(metric_data[metric_name])
    if len(collated_wave_ids) == 0:
        raise FileNotFoundError("None of the hm wave folders contain summary data.")
    if args.required_hm_wave_id is not None and args.required_hm_wave_id not in collated_wave_ids:
        raise FileNotFoundError(
            f"hm{args.required_hm_wave_id} has no summary data, so it is not part of the collated dataset."
        )

    # Record which wave each row came from, and its row within that wave:
    wave_indices = np.concatenate([
        np.full(sample_matrix.shape[0], wave_id)
        for wave_id, sample_matrix in zip(collated_wave_ids, sample_matrices)
    ])
    wave_row_indices = np.concatenate([
        np.arange(sample_matrix.shape[0]) for sample_matrix in sample_matrices
    ])

    # Generate relevant directories if not present:
    global_dirpath = os.path.join(args.experiment_dirpath, GLOBAL_DATASET_FOLDER)
    summary_dirpath = os.path.join(global_dirpath, "summary_data")
    os.makedirs(summary_dirpath, exist_ok=True)

    # Concatenate and save data:
    global_sample_matrix = np.concatenate(sample_matrices, axis=0)
    np.save(os.path.join(global_dirpath, "sample_matrix.npy"), global_sample_matrix)
    np.save(os.path.join(global_dirpath, "wave_indices.npy"), wave_indices)
    np.save(os.path.join(global_dirpath, "wave_row_indices.npy"), wave_row_indices)
    for metric_name in MODEL_METRICS:
        global_metric_matrix = np.concatenate(metric_matrices[metric_name], axis=0)
        np.save(os.path.join(summary_dirpath, f"{metric_name}.npy"), global_metric_matrix)
        print(f"{metric_name}: {global_metric_matrix.shape}", flush=True)

    # Record which waves the dataset was built from:
    collation_dict = {
        "wave_ids": collated_wave_ids,
        "wave_row_counts": [int(sample_matrix.shape[0]) for sample_matrix in sample_matrices],
        "total_row_count": int(global_sample_matrix.shape[0]),
        "metrics": MODEL_METRICS
    }
    with open(os.path.join(global_dirpath, "collation.json"), 'w') as output:
        json.dump(collation_dict, output, indent=4)
    print(f"Saved {global_sample_matrix.shape[0]} simulations to {global_dirpath}.", flush=True)


if __name__ == "__main__":
    main()