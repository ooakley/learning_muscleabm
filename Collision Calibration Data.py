import marimo

__generated_with = "0.12.10"
app = marimo.App(width="columns")


@app.cell
def _():
    import gpytorch
    import torch

    import numpy as np
    import matplotlib.pyplot as plt

    from scipy import stats
    from torch.utils.data import TensorDataset, DataLoader
    return DataLoader, TensorDataset, gpytorch, np, plt, stats, torch


@app.cell
def _():
    GRIDSEARCH_PARAMETERS = {
        "cueDiffusionRate": [0.0001, 2.5],
        "cueKa": [0.0001, 5],
        "fluctuationAmplitude": [5e-5, 5e-3],
        "fluctuationTimescale": [1, 250],
        "maximumSteadyStateActinFlow": [0.0, 5],
        "numberOfCells": [75, 150],
        "actinAdvectionRate": [0, 5],
        "cellBodyRadius": [75, 125],
        # Collisions:
        "collisionFlowReductionRate": [0.0, 1],
        "collisionAdvectionRate": [0.0, 5],
    }
    return (GRIDSEARCH_PARAMETERS,)


@app.cell
def _(GRIDSEARCH_PARAMETERS, np):
    parameters = np.load(
        "./gridsearch_data/out_wd_search_det_collisions/collated_inputs.npy"
    )
    distances = np.load(
        "./gridsearch_data/out_wd_search_det_collisions/collated_distances.npy"
    )

    # nan_mask = np.logical_and(~np.isnan(distances[:, 0]), np.any(distances!=0.0, axis=1))
    nan_mask = ~np.isnan(distances[:, 0])
    parameters = parameters[nan_mask, :]

    distances = distances[nan_mask, :]

    normalised_parameters = np.zeros_like(parameters)
    for index, parameter_range in enumerate(GRIDSEARCH_PARAMETERS.values()):
        _minimum = parameter_range[0]
        _maximum = parameter_range[1]
        normalised_parameters[:, index] = \
            (parameters[:, index] - _minimum) / (_maximum - _minimum)
    return (
        distances,
        index,
        nan_mask,
        normalised_parameters,
        parameter_range,
        parameters,
    )


@app.cell
def _(distances, np):
    print(np.min(distances, axis=0))
    return


@app.cell
def _(distances, np):
    np.argmin(distances, axis=0)
    return


@app.cell
def _(GRIDSEARCH_PARAMETERS, parameters):
    print(list(zip(list(GRIDSEARCH_PARAMETERS.keys()), parameters[49, :])))
    return


@app.cell
def _(distances, plt):
    plt.hist(distances[:, 0], bins=200)
    plt.show()
    return


@app.cell
def _(GRIDSEARCH_PARAMETERS, distances, normalised_parameters, np):
    # Posterior for WT1:
    mask = distances[:, 0] < np.quantile(distances[:, 0], 0.1)
    # mask = distances[:, 5] <= 0.0

    print(np.count_nonzero(mask))
    wt_posterior = normalised_parameters[mask, :]
    # wt_posterior = parameters[mask, :]

    mde = np.mean(wt_posterior, axis=0)

    print(list(zip(list(GRIDSEARCH_PARAMETERS.keys()), mde)))
    return mask, mde, wt_posterior


@app.cell
def _(GRIDSEARCH_PARAMETERS, distances, mask, mde, np, plt, wt_posterior):
    _fig, _axs = plt.subplots(
        10, 10, figsize=(20, 20),
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

    for _diagonal in range(10):
        _axs[_diagonal, _diagonal].hist(wt_posterior[:, _diagonal], density=True)
        _axs[_diagonal, _diagonal].set_xlim(0, 1)
        _axs[_diagonal, _diagonal].set_ylim(None)

    for i, j in zip(np.triu_indices(10, 1)[0], np.triu_indices(10, 1)[1]):
        # Upper diagonal:
        _axs[i, j].scatter(
            wt_posterior[:, j], wt_posterior[:, i],
            s=10, edgecolor='None', c=distances[mask, 1]
        )
        _axs[i, j].scatter(mde[j], mde[i], c='r')
        _axs[i, j].set_xlim(0, 1)
        _axs[i, j].set_ylim(0, 1)

        _axs[i, j].set_xlabel(labels[j])
        _axs[i, j].set_ylabel(labels[i])

        # Lower diagonal:
        _axs[j, i].scatter(
            wt_posterior[:, i], wt_posterior[:, j],
            s=10, edgecolor='None', c=distances[mask, 1]
        )
        _axs[j, i].scatter(mde[i], mde[j], c='r')
        _axs[j, i].set_xlim(0, 1)
        _axs[j, i].set_ylim(0, 1)
        _axs[j, i].set_xlabel(labels[i])
        _axs[j, i].set_ylabel(labels[j])

    plt.show()
    return axis_limits, i, j, labels, range_spacing, ranges


@app.cell
def _(distances, np, parameters, plt):
    def plot_posterior_histograms():
        # Get posteriors:
        masks = [
            distances[:, i] < np.quantile(distances[:, i], 0.005) for i in range(6)
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
