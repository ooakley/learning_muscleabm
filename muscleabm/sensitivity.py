"""Sobol' indices and eigendecompositions of parameter sensitivity matrices."""
import torch

import numpy as np

from scipy.stats import qmc


def get_sobol_indices(dimension, f_A, f_B, f_Ai):
    Si_list = []
    STi_list = []
    for index in range(dimension):
        f0_sq = np.mean(f_A) * np.mean(f_B)
        V = np.var(np.concatenate([f_A, f_B]))
        S_i  = (np.mean(f_A * f_Ai[index]) - f0_sq) / V
        S_Ti = 1 - (np.mean(f_B * f_Ai[index]) - f0_sq) / V
        Si_list.append(S_i)
        STi_list.append(S_Ti)
    return np.array(Si_list), np.array(STi_list)


def get_gp_mean(model, x):
    """Whitened GP mean at each input."""
    model.eval()
    tensor_input = torch.tensor(x)
    if len(tensor_input.shape) == 1:
        tensor_input = torch.unsqueeze(tensor_input, 0)
    with torch.no_grad():
        prediction_mean = model(tensor_input).mean.detach().numpy()
    return prediction_mean


def run_sobol_index_inference(model, parameter_dimension):
    """First order and total Sobol' indices of the mean of a GP emulator."""
    # Get necessary model evaluations for Sobol' indices:
    hyperspace_dimension = parameter_dimension * 2
    sobol_sampler = qmc.Sobol(d=hyperspace_dimension, scramble=True, rng=0)
    hyperspace_inputs = sobol_sampler.random_base2(m=17)

    # Extract base parameter matrices:
    parameters_A = hyperspace_inputs[:, :parameter_dimension]
    parameters_B = hyperspace_inputs[:, parameter_dimension:]

    # Generating the combined parameter matrices:
    parameter_matrices = []
    for parameter_index in range(parameter_dimension):
        parameters_ABi = np.copy(parameters_B)
        parameters_ABi[:, parameter_index] = parameters_A[:, parameter_index]
        parameter_matrices.append(parameters_ABi)

    # Estimate model values at these points:
    f_A = get_gp_mean(model, parameters_A)
    f_B = get_gp_mean(model, parameters_B)

    f_Ai = []
    for i_parameters in parameter_matrices:
        f_Ai.append(get_gp_mean(model, i_parameters))

    # Estimate indices from the evaluations:
    Si, STi = get_sobol_indices(parameter_dimension, f_A, f_B, f_Ai)
    return Si, STi


def get_eigenvectors(matrices):
    """Eigenvalues and eigenvectors of a stack of symmetric matrices, largest first."""
    eigenvalue_array = []
    eigenvector_array = []
    for index in range(matrices.shape[0]):
        # As matrices are symmetric, all eigenvalues are real:
        eigvals, eigenvectors = np.linalg.eigh(matrices[index])
        # Reorient everything so it makes sense:
        eigenvalue_array.append(eigvals[::-1])
        eigenvector_array.append(eigenvectors.T[::-1, :])
    return np.stack(eigenvalue_array, axis=0), np.stack(eigenvector_array, axis=0)
