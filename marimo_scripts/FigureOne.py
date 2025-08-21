import marimo

__generated_with = "0.12.10"
app = marimo.App(width="columns")


@app.cell
def _():
    import gpytorch
    import torch
    import json

    import numpy as np
    import matplotlib.pyplot as plt

    from scipy import stats
    from torch.utils.data import TensorDataset, DataLoader
    return DataLoader, TensorDataset, gpytorch, json, np, plt, stats, torch


@app.cell
def _():
    CONSTANT_PARAMETERS = {
        "superIterationCount": 12,
        "timestepsToRun": 2880,
        "worldSize": 2048,
        "gridSize": 64,
        "dt": 1,
        "thereIsMatrixInteraction": 1,
        "aspectRatio": 1,
        # # Collision parameters:
        # "collisionFlowReductionRate": 0,
        # "collisionAdvectionRate": 0,
        # Shape:
        "stretchFactor": 0.01,
        "slipFactor": 1
    }

    GRIDSEARCH_PARAMETERS = {
        "matrixTurnoverRate": [0, 0.2],
        "matrixAdditionRate": [0, 0.2],
        "matrixAdvectionRate": [0, 1],
        "cueDiffusionRate": [0.0001, 2.5],
        "cueKa": [0.0001, 5],
        "fluctuationAmplitude": [5e-5, 5e-3],
        "fluctuationTimescale": [1, 250],
        "maximumSteadyStateActinFlow": [0.0, 3],
        "numberOfCells": [75, 150],
        "actinAdvectionRate": [0.0, 1.5],
        "cellBodyRadius": [15, 100],
        # Collisions:
        "collisionFlowReductionRate": [0.0, 0.25],
        "collisionAdvectionRate": [0.0, 1.5],
    }
    return CONSTANT_PARAMETERS, GRIDSEARCH_PARAMETERS


@app.cell
def _(GRIDSEARCH_PARAMETERS, np):
    parameters = np.load(
        "./gridsearch_data/ss_wd/collated_inputs.npy"
    )
    distances = np.load(
        "./gridsearch_data/ss_wd/collated_distances.npy"
    )

    _nan_mask = np.any(np.isnan(distances), axis=(1, 2))

    nan_parameters = parameters[_nan_mask, :]
    parameters = parameters[~_nan_mask, :]
    distances = distances[~_nan_mask, :]
    # order_parameters = order_parameters[~_nan_mask, 0]
    # speeds = speeds[~_nan_mask, 0]

    normalised_parameters = np.zeros_like(parameters)
    for index, parameter_range in enumerate(GRIDSEARCH_PARAMETERS.values()):
        _minimum = parameter_range[0]
        _maximum = parameter_range[1]
        normalised_parameters[:, index] = \
            (parameters[:, index] - _minimum) / (_maximum - _minimum)

    euclidean_distances = np.sum(distances[:, :, :]**2, axis=2)
    # euclidean_distances = np.sum(np.log(distances), axis=2)
    return (
        distances,
        euclidean_distances,
        index,
        nan_parameters,
        normalised_parameters,
        parameter_range,
        parameters,
    )


@app.cell
def _(distances, euclidean_distances, np, parameters):
    def get_cl_indices():
        WD_CUTOFF = 0.65
        parameter_indices = []
        for cell_index in range(6):
            # Get NROY for cell line:
            mask = np.all(distances[:, cell_index, :] < WD_CUTOFF, axis=1)

            # Get minimum distance in NROY:
            restricted_distances = euclidean_distances[:, cell_index][mask]
            min_distance_idx = np.argmin(restricted_distances)
            min_param_idx = \
                np.arange(parameters.shape[0])[mask][min_distance_idx]
            parameter_indices.append(min_param_idx)

        return parameter_indices

    best_guess = get_cl_indices()
    return best_guess, get_cl_indices


@app.cell
def _(best_guess):
    best_guess
    return


@app.cell
def _(
    CONSTANT_PARAMETERS,
    GRIDSEARCH_PARAMETERS,
    best_guess,
    json,
    parameters,
):
    optim_params = dict(zip(
        list(GRIDSEARCH_PARAMETERS.keys()),
        parameters[best_guess[5], :]
    ))
    CONSTANT_PARAMETERS.update(optim_params)
    print(json.dumps(CONSTANT_PARAMETERS, sort_keys=False, indent=4))
    return (optim_params,)


@app.cell
def _(best_guess, parameters):
    for _ in range(6):
        print(parameters[best_guess[_]][9])
    return


@app.cell
def _(
    GRIDSEARCH_PARAMETERS,
    distances,
    euclidean_distances,
    normalised_parameters,
    np,
):
    CELL_INDEX = 0

    # Posterior for WT1:
    # mask = euclidean_distances[:, CELL_INDEX] \
    #     < np.quantile(euclidean_distances[:, CELL_INDEX], 0.01)
    # mask = euclidean_distances[:, CELL_INDEX] <= -10

    mask = np.all(distances[:, CELL_INDEX, :] < 0.5, axis=1)
    print(np.argwhere(mask))
    print(euclidean_distances[:, CELL_INDEX][mask])
    print(np.count_nonzero(mask))
    wt_posterior = normalised_parameters[mask, :]
    # wt_posterior = parameters[mask, :]

    argmins = np.argmin(euclidean_distances, axis=0)
    mde = normalised_parameters[argmins[CELL_INDEX], :]

    print(list(zip(list(GRIDSEARCH_PARAMETERS.keys()), mde)))
    return CELL_INDEX, argmins, mask, mde, wt_posterior


@app.cell
def _(
    CELL_INDEX,
    GRIDSEARCH_PARAMETERS,
    euclidean_distances,
    mask,
    mde,
    np,
    plt,
    wt_posterior,
):
    parameter_count = len(GRIDSEARCH_PARAMETERS)
    _fig, _axs = plt.subplots(
        parameter_count, parameter_count, figsize=(20, 20),
        sharex='col',
        layout='constrained'
    )
    ranges = list(GRIDSEARCH_PARAMETERS.values())
    labels = list(GRIDSEARCH_PARAMETERS.keys())
    range_spacing = [(max - min)*0.1 for min, max in ranges]
    axis_limits = [
        (minmax[0] - spacer, minmax[1] + spacer)
        for minmax, spacer in zip(ranges, range_spacing)
    ]

    for _diagonal in range(parameter_count):
        _axs[_diagonal, _diagonal].hist(
            wt_posterior[:, _diagonal],
            range=(0,1), bins=10
        )
        _axs[_diagonal, _diagonal].set_xlim(0, 1)
        _axs[_diagonal, _diagonal].set_ylim(None)

    for i, j in zip(
        np.triu_indices(parameter_count, 1)[0],
        np.triu_indices(parameter_count, 1)[1]
    ):
        # Upper diagonal:
        #  , c=distances[mask, 1]
        _axs[i, j].scatter(
            wt_posterior[:, j], wt_posterior[:, i],
            s=10, edgecolor='None', c=euclidean_distances[mask, CELL_INDEX]
        )
        _axs[i, j].scatter(mde[j], mde[i], c='r')
        _axs[i, j].set_xlim(0, 1)
        _axs[i, j].set_ylim(0, 1)

        _axs[i, j].set_xlabel(labels[j])
        _axs[i, j].set_ylabel(labels[i])

        # Lower diagonal:
        _axs[j, i].scatter(
            wt_posterior[:, i], wt_posterior[:, j],
            s=10, edgecolor='None', c=euclidean_distances[mask, CELL_INDEX]
        )
        _axs[j, i].scatter(mde[i], mde[j], c='r')
        _axs[j, i].set_xlim(0, 1)
        _axs[j, i].set_ylim(0, 1)
        _axs[j, i].set_xlabel(labels[i])
        _axs[j, i].set_ylabel(labels[j])

    plt.show()
    return axis_limits, i, j, labels, parameter_count, range_spacing, ranges


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
