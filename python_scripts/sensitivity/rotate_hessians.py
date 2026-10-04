import os
import sys
import argparse

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
TOP_K = 7

def sparse_rotate(eigenvalues, eigenvectors, epsilon_schedule, initial_rotation,
                  max_evaluations_per_stage=2000, gradient_tolerance=1e-9,
                  sufficient_decrease=1e-4, max_rejections=40):
    """
    Rotate the columns of `eigenvectors` within their own span so that each rotated axis
    involves as few parameters as possible.

    For a unit-norm rotated axis, let magnitude_i = sqrt(component_i**2 + epsilon**2) (a
    smoothed |component_i|), probability_i = magnitude_i / sum(magnitude), and
    entropy = -sum(probability * log(probability)). exp(entropy) is the effective number
    of parameters in the axis. The objective is the sum of this entropy over axes, and it
    is minimised over orthogonal rotations by gradient projection (a gradient step
    projected onto the tangent space of the orthogonal group, then returned to the group
    by polar decomposition, with backtracking line search). Each value of epsilon in
    `epsilon_schedule` is one annealing stage, warm-started from the previous one.

    Parameters
    ----------
    eigenvalues : (n_axes,) precision-matrix eigenvalues matching the columns of
        `eigenvectors`. Only used to give each rotated axis's curvature and to order the
        output; they do not affect the rotation.
    eigenvectors : (n_parameters, n_axes) orthonormal top-k eigenvectors of the precision
        matrix (the stiff block).
    epsilon_schedule : decreasing smoothing scales, in units of entry size. Entries of a
        unit vector are of order 1 / sqrt(n_parameters), so something like
        np.array([1.0, 0.33, 0.1, 0.033, 0.01]) / np.sqrt(n_parameters) is a sensible start.
    initial_rotation : (n_axes, n_axes) orthogonal starting rotation, e.g. random.
    max_evaluations_per_stage, gradient_tolerance, sufficient_decrease, max_rejections :
        optimiser controls; the defaults are fine for most uses.

    Returns
    -------
    rotated_axes : (n_parameters, n_axes) sparse orthonormal basis of the same subspace.
        Signs are fixed so each axis's largest component is positive; axes are ordered by
        decreasing curvature.
    rotation : (n_axes, n_axes) with rotated_axes == eigenvectors @ rotation.
    axis_curvature : (n_axes,) w' (precision restricted to the block) w for each axis.
    objective : summed entropy at the final epsilon (lower is sparser). Comparable between
        runs only if they share the same final epsilon.
    """
    n_parameters, n_axes = eigenvectors.shape
    if eigenvalues.shape != (n_axes,):
        raise ValueError("eigenvalues must have one entry per column of eigenvectors")
    if initial_rotation.shape != (n_axes, n_axes) or not np.allclose(
            initial_rotation.T @ initial_rotation, np.eye(n_axes), atol=1e-8):
        raise ValueError("initial_rotation must be an orthogonal (n_axes, n_axes) matrix")
    if len(epsilon_schedule) == 0:
        raise ValueError("epsilon_schedule must contain at least one value")

    rotation = initial_rotation.copy()

    for epsilon in epsilon_schedule:
        objective = np.inf
        step_size = 1.0
        n_rejections = 0
        projected_gradient = np.zeros((n_axes, n_axes))
        projected_gradient_norm_squared = 0.0
        trial_rotation = rotation

        # One loop handles both accepted steps and backtracking rejections, so the
        # objective and gradient are computed in a single place. The first pass evaluates
        # the starting rotation and is accepted unconditionally (objective starts at inf).
        for evaluation in range(max_evaluations_per_stage):
            trial_axes = eigenvectors @ trial_rotation
            smoothed_magnitude = np.sqrt(trial_axes ** 2 + epsilon ** 2)
            magnitude_total = smoothed_magnitude.sum(axis=0, keepdims=True)
            probability = smoothed_magnitude / magnitude_total
            axis_entropy = -(probability * np.log(probability)).sum(axis=0, keepdims=True)
            trial_objective = axis_entropy.sum()

            if trial_objective < objective - sufficient_decrease * step_size * projected_gradient_norm_squared:
                rotation = trial_rotation
                objective = trial_objective
                n_rejections = 0
                if evaluation > 0:
                    step_size *= 2.0

                # d(entropy)/d(component) = -(log(probability) + entropy) * component / (magnitude * total)
                gradient_wrt_axes = -(np.log(probability) + axis_entropy) * trial_axes / (
                    smoothed_magnitude * magnitude_total)
                rotation_gradient = eigenvectors.T @ gradient_wrt_axes
                symmetric_part = rotation.T @ rotation_gradient
                symmetric_part = (symmetric_part + symmetric_part.T) / 2.0
                projected_gradient = rotation_gradient - rotation @ symmetric_part
                projected_gradient_norm_squared = np.sum(projected_gradient ** 2)
                if np.sqrt(projected_gradient_norm_squared) < gradient_tolerance:
                    break
            else:
                step_size /= 2.0
                n_rejections += 1
                if n_rejections > max_rejections:
                    break

            left_vectors, singular_values, right_vectors_transposed = np.linalg.svd(
                rotation - step_size * projected_gradient)
            trial_rotation = left_vectors @ right_vectors_transposed

    # We retrieve the canonical form of the subspace by ensuring that the largest component of 
    # each axis is positive, and by ordering each axis by curvature.
    rotated_axes = eigenvectors @ rotation
    row_of_largest_component = np.argmax(np.abs(rotated_axes), axis=0)
    signs = np.sign(rotated_axes[row_of_largest_component, np.arange(n_axes)])
    rotation = rotation * signs
    axis_curvature = (rotation ** 2 * eigenvalues[:, None]).sum(axis=0)
    curvature_order = np.argsort(-axis_curvature)
    rotation = rotation[:, curvature_order]
    return eigenvectors @ rotation, rotation, axis_curvature[curvature_order], objective


def generate_initial_rotation(generator):
    orthogonal, triangular = np.linalg.qr(generator.standard_normal((TOP_K, TOP_K)))
    initial_rotation = orthogonal * np.sign(np.diag(triangular))
    return initial_rotation


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dirpath", required=True,
        help="Experiment containing the op65_fullrank_FIM folder, e.g. model_experiments/2026-09-25-matrix_shape."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    print("Loading FIM dataset...")
    fim_dirpath = os.path.join(args.experiment_dirpath, "op65_fullrank_FIM")
    fim_dataset = np.load(os.path.join(fim_dirpath, "fim_estimate.npy"))
    print(f"Dataset size: {fim_dataset.shape}")

    print("Performing eigendecomposition...")
    fim_dataset = (fim_dataset + fim_dataset.transpose(0, 2, 1)) / 2.0
    eigvals, eigvecs = np.linalg.eigh(fim_dataset)
    top_vals = eigvals[:, -TOP_K:][:, ::-1]
    top_vecs = eigvecs[:, :, -TOP_K:][:, :, ::-1]

    # Set up prerequisites for rotation calculation:
    np_rng = np.random.default_rng(0)
    baseline_schedule = np.array([1.0, 0.33, 0.1, 0.033, 0.01, 0.001, 0.0001]) / np.sqrt(12)

    print(f"Rotating top-{TOP_K} eigenvectors for maximal sparsity...")
    axes_list, rotation_list, curvature_list, objective_list = [], [], [], []
    for batch_index in range(eigvecs.shape[0]):
        axes, rotation, curvature, objective = sparse_rotate(
            top_vals[batch_index], top_vecs[batch_index], baseline_schedule,
            generate_initial_rotation(np_rng),
        )
        axes_list.append(axes)
        rotation_list.append(rotation)
        curvature_list.append(curvature)
        objective_list.append(objective)

    print("Saving rotations...")
    save_dirpath = os.path.join(args.experiment_dirpath, "fim_rotated")
    if not os.path.exists(save_dirpath):
        os.mkdir(save_dirpath)
    np.savez(
        os.path.join(save_dirpath, f"fim_{TOP_K}_rotated.npz"),
        rotated_axes=np.stack(axes_list),        # (B, N, K)
        rotations=np.stack(rotation_list),       # (B, K, K)
        axis_curvature=np.stack(curvature_list), # (B, K)
        objective=np.array(objective_list),      # (B,)
    )


if __name__ == "__main__":
    main()
