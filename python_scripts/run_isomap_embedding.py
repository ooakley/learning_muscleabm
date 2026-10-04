import os
import sys

import numba
import sklearn

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
EXPERIMENT_DIRPATH = "model_experiments/2026-09-25-matrix_shape"
TOP_K = 6

def get_eigenvectors(fims):
    eigenvalue_array = []
    eigenvector_array = []
    for index in range(fims.shape[0]):
        # As matrices are symmetric, all eigenvalues are real:
        eigvals, eigenvectors = np.linalg.eigh(fims[index])
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


# Define Bures-Wasserstein distances:
D = 14

@numba.njit(fastmath=True)
def fill_matrix(ut):
    # Restore upper triangles to full matrices:
    ret = np.empty((D, D))
    idx = 0
    for i in range(D):
        for j in range(i, D):
            v = ut[idx]
            ret[i, j] = v
            ret[j, i] = v
            idx += 1
    return ret


@numba.njit(fastmath=True)
def sqrt_matrix(A):
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.clip(eigvals, 0, None)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


@numba.njit(fastmath=True)
def bures_wasserstein_distance_matrix(input_matrices):
    # Set up output:
    matrix_count = input_matrices.shape[0]
    out = np.zeros((matrix_count, matrix_count))

    # Get traces:
    traces = np.empty(matrix_count)
    for k in range(matrix_count):
        traces[k] = np.trace(input_matrices[k])

    # Parallelise loop:
    for i in numba.prange(matrix_count):
        # Get square root of matrix:
        A = input_matrices[i]
        sqrt_A = sqrt_matrix(A)
        trace_A = traces[i]
        for j in range(i + 1, matrix_count):
            # Get B matrix:
            B = input_matrices[j]

            # Calculate BW:
            inner = sqrt_A @ B @ sqrt_A
            # -- Symmetrise (if floating-point error introduces off-diagonals):
            inner = 0.5 * (inner + inner.T) 
            trace_term = np.trace(sqrt_matrix(inner))
            sq_dist = np.trace(A) + traces[j] - (2 * trace_term)
            sq_dist = max(sq_dist, 0.0)
            dist = np.sqrt(sq_dist)
            out[i, j] = dist
            out[j, i] = dist

    return out


@numba.njit(fastmath=True)
def bures_wasserstein_distance(A_ut, B_ut):
    # Restore from upper triangular representations:
    A = fill_matrix(A_ut)
    B = fill_matrix(B_ut)

    # Get square root of matrix:
    sqrt_A = sqrt_matrix(A)

    # Calculate BW:
    inner = sqrt_A @ B @ sqrt_A

    # -- Symmetrise (if floating-point error introduces off-diagonals):
    inner = 0.5 * (inner + inner.T) 
    trace_term = np.trace(sqrt_matrix(inner))
    sq_dist = np.trace(A) + np.trace(B) - (2 * trace_term)

    return np.sqrt(sq_dist)


def flatten_fim(sample_fims, eigenvalues):
    triu_index = np.triu_indices(14)
    flattened_fim_array = []
    for i in range(sample_fims.shape[0]):
        flattened_fim_array.append(sample_fims[i][triu_index] / eigenvalues[i, 0])
    return np.stack(flattened_fim_array, axis=0)


@numba.njit(fastmath=True)
def grassman_distance(A, B):
    A_mat = A.reshape(14, TOP_K)
    B_mat = B.reshape(14, TOP_K)
    similarity = np.sum((A_mat.T @ B_mat) ** 2)
    return 1 - (similarity / TOP_K)


def main():
    # Load fims:
    fim_dirpath = os.path.join(EXPERIMENT_DIRPATH, "op65_fullrank_FIM")
    sample_fims = np.load(os.path.join(fim_dirpath, "fim_estimate.npy"))
    sample_inputs = np.load(os.path.join(fim_dirpath, "fim_inputs.npy"))

    embeddings_dirpath = os.path.join(EXPERIMENT_DIRPATH, "embeddings")
    if not os.path.exists(embeddings_dirpath):
        os.mkdir(embeddings_dirpath)

    # Calculate eigendecompositions:
    eigenvalues, eigenvectors = get_eigenvectors(sample_fims)

    # # Calculate kernel PCA:
    # print("Running kPCA...", flush=True)
    # kpca_manager = sklearn.decomposition.KernelPCA(n_components=10, kernel=nematic_similarity, n_jobs=64)
    # kpca_transformed = kpca_manager.fit_transform(eigenvectors[:, 0, :])

    # print("Saving kPCA...", flush=True)
    # save_filepath = os.path.join(EXPERIMENT_DIRPATH, "gaussian_process_models", "op65", "kpca_embeddings.npy")
    # np.save(save_filepath, kpca_transformed)

    # Calculate IsoMAP embeddings:
    print("Running base ISOMAP...")
    isomapper = sklearn.manifold.Isomap(n_components=8, metric=nematic_cosine, n_jobs=64)
    isomap_embeddings = isomapper.fit_transform(eigenvectors[:, 0, :])
    print("Saving base ISOMAP...")
    np.save(os.path.join(embeddings_dirpath, "isomap_embeddings.npy"), isomap_embeddings)

    # print("Flattening the FIM...")
    # fim_array = flatten_fim(sample_fims, eigenvalues)

    # # Get BW distance matrix:
    # print("Extracing BW distance matrix...", flush=True)
    # normalised_fim = sample_fims[:, :, :] / eigenvalues[:, [0], None]
    # print(normalised_fim.shape)
    # dist_matrix = bures_wasserstein_distance_matrix(normalised_fim)

    # print("Saving BW distance matrix...", flush=True)
    # np.save(os.path.join(embeddings_dirpath, "bw_distance_matrix.npy"), dist_matrix)

    # print("Running BW-ISOMAP...", flush=True)
    # isomapper = sklearn.manifold.Isomap(n_components=8, metric="precomputed", n_jobs=64)
    # isomap_embeddings = isomapper.fit_transform(dist_matrix)

    # print("Saving ISOMAP...", flush=True)
    # np.save(os.path.join(embeddings_dirpath, "bw_isomap_embeddings.npy"), isomap_embeddings)

    # # Get Grassman distance ISOMAP:
    # print("Running Grassman ISOMAP...", flush=True)
    # isomapper = sklearn.manifold.Isomap(n_components=TOP_K, metric=grassman_distance)
    # gd_inputs = eigenvectors[:, :TOP_K, :].transpose(0, 2, 1)
    # gd_inputs = gd_inputs.reshape(-1, 14 * TOP_K)
    # gd_embeddings = isomapper.fit_transform(gd_inputs)
    # print("Saving Grassman ISOMAP...", flush=True)
    # np.save(os.path.join(embeddings_dirpath, "grassman_isomap_embeddings.npy"), gd_embeddings)


if __name__ == "__main__":
    main()
