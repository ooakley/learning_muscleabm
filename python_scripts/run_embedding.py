import os

import numba
import sklearn

import numpy as np


def get_eigenvectors(hessians):
    eigenvalue_array = []
    eigenvector_array = []
    for index in range(hessians.shape[0]):
        # As matrices are symmetric, all eigenvalues are real:
        eigvals, eigenvectors = np.linalg.eigh(hessians[index])
        # Reorient everything so it makes sense:
        eigenvalue_array.append(eigvals[::-1])
        eigenvector_array.append(eigenvectors.T[::-1, :])
    return np.stack(eigenvalue_array, axis=0), np.stack(eigenvector_array, axis=0)


@numba.njit()
def nematic_cosine(a, b):
    return 1 - np.abs(a @ b.T)


@numba.njit()
def nematic_euclidean(a, b):
    distance_array = np.zeros(2)
    distance_array[0] = np.sqrt(np.sum((a - b) ** 2))
    distance_array[1] = np.sqrt(np.sum((a + b) ** 2))
    return np.min(distance_array)


@numba.njit()
def nematic_similarity(a, b):
    return np.abs(a @ b.T)


def main():
    # Load Hessians:
    EXPERIMENT_DIRPATH = "model_experiments/2026-06-03-matrix_shape"
    sample_hessians = np.load("model_experiments/2026-06-03-matrix_shape/gaussian_process_models/op65/hessian_estimate.npy")
    sample_inputs = np.load("model_experiments/2026-06-03-matrix_shape/gaussian_process_models/op65/hessian_inputs.npy")

    # Calculate eigendecompositions:
    eigenvalues, eigenvectors = get_eigenvectors(sample_hessians)

    # # Calculate kernel PCA:
    # print("Running kPCA...", flush=True)
    # kpca_manager = sklearn.decomposition.KernelPCA(n_components=10, kernel=nematic_similarity, n_jobs=64)
    # kpca_transformed = kpca_manager.fit_transform(eigenvectors[:, 0, :])

    # print("Saving kPCA...", flush=True)
    # save_filepath = os.path.join(EXPERIMENT_DIRPATH, "gaussian_process_models", "op65", "kpca_embeddings.npy")
    # np.save(save_filepath, kpca_transformed)

    # Calculate IsoMAP embeddings:
    print("Running ISOMAP...", flush=True)
    isomapper = sklearn.manifold.Isomap(n_components=5, metric=nematic_cosine, n_jobs=64)
    isomap_embeddings = isomapper.fit_transform(eigenvectors[:, 0, :])

    print("Saving ISOMAP...", flush=True)
    save_filepath = os.path.join(EXPERIMENT_DIRPATH, "gaussian_process_models", "op65", "isomap_embeddings.npy")
    np.save(save_filepath, isomap_embeddings)


if __name__ == "__main__":
    main()
