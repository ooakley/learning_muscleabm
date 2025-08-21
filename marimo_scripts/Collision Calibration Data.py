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
        "cueDiffusionRate": [
            0.0001,
            2.5
        ],
        "cueKa": [
            0.0001,
            5
        ],
        "fluctuationAmplitude": [
            5e-05,
            0.1
        ],
        "fluctuationTimescale": [
            1,
            20
        ],
        "maximumSteadyStateActinFlow": [
            0.0,
            3
        ],
        "numberOfCells": [
            75,
            250
        ],
        "actinAdvectionRate": [
            0.0,
            3
        ],
        "cellBodyRadius": [
            15,
            100
        ],
        "collisionFlowReductionRate": [
            0.0,
            1
        ],
        "collisionAdvectionRate": [
            0.0,
            1.5
        ]
    }
    return CONSTANT_PARAMETERS, GRIDSEARCH_PARAMETERS


@app.cell
def _(GRIDSEARCH_PARAMETERS, np):
    parameters = np.load(
        "./gridsearch_data/out_20250821NoMatrix/collated_inputs.npy"
    )
    distances = np.load(
        "./gridsearch_data/out_20250821NoMatrix/collated_distances.npy"
    )
    order_parameters = np.load(
        "./gridsearch_data/out_20250821NoMatrix/collated_order_parameters.npy"
    )
    speeds = np.load(
        "./gridsearch_data/out_20250821NoMatrix/collated_magnitudes.npy"
    )



    _nan_mask = np.any(np.isnan(distances), axis=(1, 2))

    nan_parameters = parameters[_nan_mask, :]
    parameters = parameters[~_nan_mask, :]
    distances = distances[~_nan_mask, :]
    order_parameters = order_parameters[~_nan_mask, 0]
    speeds = speeds[~_nan_mask, 0]

    print(parameters.shape)

    normalised_parameters = np.zeros_like(parameters)
    for index, parameter_range in enumerate(GRIDSEARCH_PARAMETERS.values()):
        _minimum = parameter_range[0]
        _maximum = parameter_range[1]
        normalised_parameters[:, index] = \
            (parameters[:, index] - _minimum) / (_maximum - _minimum)

    # euclidean_distances = np.sqrt(np.sum(distances[:, :, :]**2, axis=2))
    euclidean_distances = np.sum(np.log(distances), axis=2)
    return (
        distances,
        euclidean_distances,
        index,
        nan_parameters,
        normalised_parameters,
        order_parameters,
        parameter_range,
        parameters,
        speeds,
    )


@app.cell
def _(order_parameters, plt):
    plt.hist(order_parameters);
    plt.show()
    return


@app.cell
def _(np, order_parameters):
    np.argwhere(order_parameters > 0.5)
    return


@app.cell
def _(GRIDSEARCH_PARAMETERS, index, nan_parameters, np, plt):
    fig, axs = plt.subplots(13, 1, figsize=(4, 12))

    norm_nan_parameters = np.zeros_like(nan_parameters)
    for _index, _parameter_range in enumerate(GRIDSEARCH_PARAMETERS.values()):
        _minimum = _parameter_range[0]
        _maximum = _parameter_range[1]
        norm_nan_parameters[:, _index] = \
            (nan_parameters[:, index] - _minimum) / (_maximum - _minimum)
    return axs, fig, norm_nan_parameters


@app.cell
def _(GRIDSEARCH_PARAMETERS, norm_nan_parameters, np, plt):
    def _():
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
                norm_nan_parameters[:, _diagonal],
                range=(0,1), bins=20
            )
            _axs[_diagonal, _diagonal].set_xlim(-0.1, 1.1)
            _axs[_diagonal, _diagonal].set_ylim(None)

        for i, j in zip(
            np.triu_indices(parameter_count, 1)[0],
            np.triu_indices(parameter_count, 1)[1]
        ):
            # Upper diagonal:
            #  , c=distances[mask, 1]
            _axs[i, j].scatter(
                norm_nan_parameters[:, j], norm_nan_parameters[:, i],
                s=10, edgecolor='None'
            )
            _axs[i, j].set_xlim(-0.1, 1.1)
            _axs[i, j].set_ylim(-0.1, 1.1)

            _axs[i, j].set_xlabel(labels[j])
            _axs[i, j].set_ylabel(labels[i])

            # Lower diagonal:
            _axs[j, i].scatter(
                norm_nan_parameters[:, i], norm_nan_parameters[:, j],
                s=10, edgecolor='None'
            )
            _axs[j, i].set_xlim(-0.1, 1.1)
            _axs[j, i].set_ylim(-0.1, 1.1)
            _axs[j, i].set_xlabel(labels[i])
            _axs[j, i].set_ylabel(labels[j])
        return plt.show()


    _()
    return


@app.cell
def _(order_parameters, plt, speeds):
    plt.scatter(order_parameters, speeds)
    return


@app.cell
def _(np, order_parameters, speeds):
    _mask = np.logical_and(order_parameters > 0.95, speeds > 0)
    np.arange(len(order_parameters))[_mask]
    return


@app.cell
def _(parameters):
    parameters.shape
    return


@app.cell
def _(euclidean_distances, np):
    min_args = np.argmin(euclidean_distances, axis=0)
    return (min_args,)


@app.cell
def _(min_args):
    print(min_args)
    return


@app.cell
def _(euclidean_distances, np):
    print(np.min(euclidean_distances, axis=0))
    return


@app.cell
def _(distances, min_args):
    distances[min_args[0], 0, :]
    return


@app.cell
def _():
    # parameters = np.load(
    #     "./gridsearch_data/td_distance/collated_inputs.npy"
    # )
    # distances = np.load(
    #     "./gridsearch_data/td_distance/collated_distances.npy"
    # )

    # print(np.count_nonzero(np.isnan(distances[:, 0])))
    # print(np.count_nonzero(distances == 0))

    # distances[distances == 0] = np.nan
    # nan_mask = np.isnan(distances[:, 0])
    # nan_mask = np.logical_or(np.isnan(distances[:, 0]), np.any(distances==0, axis=1))

    # parameters = parameters[~nan_mask, :]
    # distances = distances[~nan_mask, :]

    # normalised_parameters = np.zeros_like(parameters)
    # for index, parameter_range in enumerate(GRIDSEARCH_PARAMETERS.values()):
    #     _minimum = parameter_range[0]
    #     _maximum = parameter_range[1]
    #     normalised_parameters[:, index] = \
    #         (parameters[:, index] - _minimum) / (_maximum - _minimum)
    return


@app.cell
def _(euclidean_distances, mask, plt):
    plt.hist(euclidean_distances[mask, 0]);
    plt.show()
    return


@app.cell
def _(CONSTANT_PARAMETERS, GRIDSEARCH_PARAMETERS, json, parameters):
    optim_params = dict(zip(
        list(GRIDSEARCH_PARAMETERS.keys()),
        parameters[5899, :]
    ))
    CONSTANT_PARAMETERS.update(optim_params)
    print(json.dumps(CONSTANT_PARAMETERS, sort_keys=False, indent=4))
    return (optim_params,)


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
    # mask = np.all(distances[:, CELL_INDEX, :3] < 0.8, axis=1)

    distance_quantiles = np.quantile(
        distances[:, CELL_INDEX, :2],
        0.05, axis=0
    )
    quantile_check = distances[:, CELL_INDEX, :2] < distance_quantiles 
    mask = np.all(quantile_check, axis=1)
    print(np.argwhere(mask))

    wt_posterior = normalised_parameters[mask, :]
    # wt_posterior = parameters[mask, :]

    argmins = np.argmin(euclidean_distances, axis=0)
    mde = normalised_parameters[argmins[CELL_INDEX], :]

    print(list(zip(list(GRIDSEARCH_PARAMETERS.keys()), mde)))
    return (
        CELL_INDEX,
        argmins,
        distance_quantiles,
        mask,
        mde,
        quantile_check,
        wt_posterior,
    )


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
def _(
    GRIDSEARCH_PARAMETERS,
    mask,
    mde,
    np,
    order_parameters,
    parameter_count,
    plt,
    wt_posterior,
):
    def _():
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
                wt_posterior[:, _diagonal]
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
                s=10, edgecolor='None',
                c=order_parameters[mask],
                vmin=0.1,
                vmax=0.2
            )
            _axs[i, j].scatter(mde[j], mde[i], c='r')
            _axs[i, j].set_xlim(0, 1)
            _axs[i, j].set_ylim(0, 1)

            _axs[i, j].set_xlabel(labels[j])
            _axs[i, j].set_ylabel(labels[i])

            # Lower diagonal:
            _axs[j, i].scatter(
                wt_posterior[:, i], wt_posterior[:, j],
                s=10, edgecolor='None',
                c=order_parameters[mask],
                vmin=0.1,
                vmax=0.2
            )
            _axs[j, i].scatter(mde[i], mde[j], c='r')
            _axs[j, i].set_xlim(0, 1)
            _axs[j, i].set_ylim(0, 1)
            _axs[j, i].set_xlabel(labels[i])
            _axs[j, i].set_ylabel(labels[j])
        return plt.show()

    _()
    return


@app.cell
def _(distances, np, parameters, plt):
    def plot_posterior_histograms():
        # Get posteriors:
        masks = [
            distances[:, i] < np.quantile(distances[:, i], 0.1) for i in range(6)
        ]
        posteriors = [parameters[mask, :] for mask in masks]

        # Loop through parameters:
        fig, axs = plt.subplots(10, 1, figsize=(3, 20), layout='constrained')
        for parameter_index in range(10):
            for index, posterior in enumerate(posteriors):
                if index < 3:
                    color = 'tab:orange'
                else:
                    color = 'tab:blue'
                axs[parameter_index].hist(
                    posterior[:, parameter_index],
                    density=True, histtype='step', color=color
                )

        plt.show()
    return (plot_posterior_histograms,)


@app.cell
def _(plot_posterior_histograms):
    plot_posterior_histograms()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
