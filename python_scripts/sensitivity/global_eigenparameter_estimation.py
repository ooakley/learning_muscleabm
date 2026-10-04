import os
import argparse
import time

import torch

import scipy.sparse

import numpy as np

from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors

from muscleabm.sensitivity import get_eigenvectors

PARAMETER_DIMENSION = 14
RUN_COUNT = 1024


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dirpath", required=True,
        help="Experiment containing the op65_fullrank_hessian folder, e.g. model_experiments/2026-09-19-matrix_shape."
    )
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--task_id", type=int, required=True)
    return parser.parse_args()


def get_estimated_geodesic_distances(eigenvectors):
    # First get nematic similarity:
    similarity_matrix = np.abs(eigenvectors @ eigenvectors.T)
    nc_distance_matrix = 1 - similarity_matrix
    nc_distance_matrix = np.clip(nc_distance_matrix, 0, None)

    cosine_neighbours = NearestNeighbors(n_neighbors=5, metric="precomputed")
    cosine_neighbours.fit(nc_distance_matrix)
    kneighbours_graph = cosine_neighbours.kneighbors_graph(nc_distance_matrix, mode='distance')
    kneighbours_graph = kneighbours_graph.toarray()

    symmetric_neighbours = np.stack([kneighbours_graph, kneighbours_graph.T], axis=0)
    symmetric_neighbours = np.max(symmetric_neighbours, axis=0)

    csr_sn = scipy.sparse.csr_matrix(symmetric_neighbours)
    dijkstra_distance_matrix = dijkstra(csr_sn, directed=False)
    return dijkstra_distance_matrix


def log_sammon_mapping(input_data, distance_matrix, n_components, batch_size=512, seed=0):
    # Get relevant dimensions:
    sample_count = distance_matrix.shape[0]

    # Convert to torch:
    input_tensor = torch.from_numpy(input_data)

    # Get distance matrix:
    eig_distance_matrix = torch.from_numpy(distance_matrix)

    # Set up initial log transformation, with scale to allow log to actually work:
    rng = np.random.default_rng(seed)

    log_transform = rng.uniform(-1, 1, (PARAMETER_DIMENSION, n_components))
    log_transform = torch.from_numpy(log_transform)
    log_transform.requires_grad_()

    scale_factor = torch.ones((1, n_components)) * 0.1
    scale_factor.requires_grad_()

    opt = torch.optim.Adam([
        {'params': [log_transform], 'lr': 0.003},
        {'params': [scale_factor],  'lr': 0.003}
    ])

    # Get indices so we only calculate over upper triangle of matrices (we divide by 0 with the diagonal otherwise):
    upper_indices = torch.triu_indices(batch_size, batch_size, offset=1)
    axis_indices = torch.triu_indices(n_components, n_components, offset=1)
    # triu_eig = eig_distance_matrix[upper_indices[0], upper_indices[1]]

    print("Performing gradient descent for transform...", flush=True)
    loss_history = []
    for step_index in range(8192):
        # Randomly sample indices:
        # batch_indices = torch.randperm(sample_count)[:batch_size]
        batch_indices = rng.permutation(np.arange(sample_count))[:batch_size]
 
        # Do linear transformation:
        transformed_data = torch.log(input_tensor[batch_indices, :]) @ log_transform
        scaled_transformed_data = transformed_data * scale_factor
        transform_distance_matrix = torch.functional.cdist(scaled_transformed_data, scaled_transformed_data)

        # Get Sammon's error:
        batch_eig_distance_matrix = eig_distance_matrix[batch_indices, :][:, batch_indices]
        triu_eig = batch_eig_distance_matrix[upper_indices[0], upper_indices[1]]
        triu_transform = transform_distance_matrix[upper_indices[0], upper_indices[1]]
        sammon_error = torch.sum(((triu_eig - triu_transform) ** 2) / triu_eig) / torch.sum(triu_eig)

        # Get similarity between transformation vectors:
        normalised_transform = \
            log_transform / torch.linalg.norm(log_transform, dim=0, keepdims=True)
        similiarities = (normalised_transform.T @ normalised_transform)[axis_indices[0], axis_indices[1]]
        # Not currently part of the loss below:
        axis_similarity = torch.sum(torch.abs(similiarities))  # noqa: F841

        # Get loading entropies:
        # --- Get entropies by transform dimension:
        absolute_components = torch.abs(log_transform)
        norm_transform_components = absolute_components / torch.linalg.norm(absolute_components, dim=0, keepdims=True)
        # norm_transform_components = absolute_components / torch.sum(absolute_components, dim=0, keepdims=True)
        transform_entropies = torch.sum(-norm_transform_components * torch.log(norm_transform_components), dim=0)

        # --- Get entropies by parameter dimension:
        norm_param_components = absolute_components / torch.linalg.norm(absolute_components, dim=1, keepdims=True)
        # norm_param_components = absolute_components / torch.sum(absolute_components, dim=1, keepdims=True)
        param_entropies = torch.sum(-norm_param_components * torch.log(norm_param_components), dim=1)
        total_entropy = torch.sum(torch.concatenate([transform_entropies, param_entropies]))

        # Run backprop and take update step:
        opt.zero_grad()
        (sammon_error + (5e-3 * total_entropy)).backward()
        opt.step()
        loss_history.append(sammon_error.item())

        # Print progress:
        if (step_index + 1) % 1024 == 0:
            print(step_index + 1, sammon_error.item(), scale_factor.detach().numpy(), flush=True)

    return loss_history, log_transform.detach().numpy(), scale_factor.detach().numpy()


def main():
    # Parse command line arguments:
    args = parse_arguments()

    # Set up folder structure:
    managing_dirpath = os.path.join(args.experiment_dirpath, "global_eigenparameter_estimation")
    if not os.path.exists(managing_dirpath):
        os.mkdir(managing_dirpath)

    # Get Hessian dataset:
    print("Loading Hessian datasets...", flush=True)
    hessian_dirpath = os.path.join(args.experiment_dirpath, "op65_fullrank_hessian")
    hessian_inputs = np.load(os.path.join(hessian_dirpath, "hessian_inputs.npy"))
    hessians = np.load(os.path.join(hessian_dirpath, "hessian_estimate.npy"))

    # Compute eigendecompositions:
    print("Computing eigendecompositions...", flush=True)
    eigenvalues, eigenvectors = get_eigenvectors(hessians)

    # Compute geodesic distance matrix if not already present:
    geodesic_distance_filepath = os.path.join(managing_dirpath, "geodesic_distances.npy")
    if not os.path.exists(geodesic_distance_filepath):
        print("Computing geodesic distance on control manifold...", flush=True)
        geodesic_dm = get_estimated_geodesic_distances(eigenvectors[:, 0, :])
        np.save(geodesic_distance_filepath, geodesic_dm)
    else:
        print("Loading geodesic distance on control manifold...", flush=True)
        geodesic_dm = np.load(geodesic_distance_filepath)

    # Get subset of runs to perform in this shard:
    chunk_size = RUN_COUNT // args.world_size
    start_index = args.task_id * chunk_size
    end_index = (args.task_id + 1) * chunk_size

    # Iterate over random initialisations, collect approximate transformations:
    loss_histories = []
    log_transforms = []
    scale_factors = []
    for run_index in range(start_index, end_index):
        print(f"Running GEE batch {run_index}...", flush=True)
        loss_history, log_transform, scale_factor = \
            log_sammon_mapping(hessian_inputs, geodesic_dm, 3, batch_size=128, seed=run_index)
        loss_histories.append(loss_history)
        log_transforms.append(log_transform)
        scale_factors.append(scale_factor)

    # Save transforms and loss histories:
    np.save(
        os.path.join(managing_dirpath, f"loss_histories_{args.task_id}.npy"),
        np.stack(loss_histories, axis=0)
    )
    np.save(
        os.path.join(managing_dirpath, f"log_transforms_{args.task_id}.npy"),
        np.stack(log_transforms, axis=0)
    )
    np.save(
        os.path.join(managing_dirpath, f"scale_factors_{args.task_id}.npy"),
        np.stack(scale_factors, axis=0)
    )

    # Collate if primary shard:
    if args.task_id == 0:
        # Wait while other processes finish:
        tf_filename_list = [f"log_transforms_{task_id}.npy" for task_id in range(args.world_size)]
        tf_filepath_list = [os.path.join(managing_dirpath, filename) for filename in tf_filename_list]
        subprocesses_completing = True
        while subprocesses_completing:
            print("Checking subprocesses...", flush=True)
            completion_list = [os.path.exists(filepath) for filepath in tf_filepath_list]
            if all(completion_list):
                subprocesses_completing = False
                continue
            time.sleep(15)

        print("Subprocesses complete!", flush=True)
        time.sleep(30)  # Ensure all files are fully written to disk

        # Collate transform data:
        collated_log_transforms = [np.load(filepath) for filepath in tf_filepath_list]
        collated_log_transforms = np.concatenate(collated_log_transforms, axis=0)
        np.save(os.path.join(managing_dirpath, "log_transforms.npy"), collated_log_transforms)

        # Collate loss history data:
        lh_filename_list = [f"loss_histories_{task_id}.npy" for task_id in range(args.world_size)]
        lh_filepath_list = [os.path.join(managing_dirpath, filename) for filename in lh_filename_list]
        collated_loss_histories = [np.load(filepath) for filepath in lh_filepath_list]
        collated_loss_histories = np.concatenate(collated_loss_histories, axis=0)
        np.save(os.path.join(managing_dirpath, "loss_histories.npy"), collated_loss_histories)

        # Collate scale factor data:
        sf_filename_list = [f"scale_factors_{task_id}.npy" for task_id in range(args.world_size)]
        sf_filepath_list = [os.path.join(managing_dirpath, filename) for filename in sf_filename_list]
        collated_scale_factors = [np.load(filepath) for filepath in sf_filepath_list]
        collated_scale_factors = np.concatenate(collated_scale_factors, axis=0)
        np.save(os.path.join(managing_dirpath, "scale_factors.npy"), collated_scale_factors)

        # Remove sharded arrays:
        [os.remove(filepath) for filepath in tf_filepath_list]
        [os.remove(filepath) for filepath in lh_filepath_list]
        [os.remove(filepath) for filepath in sf_filepath_list]


if __name__ == "__main__":
    main()
