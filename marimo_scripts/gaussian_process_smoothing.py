import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import json

    import gpytorch
    import torch

    import numpy as np
    import matplotlib.pyplot as plt

    from torch.utils.data import TensorDataset, DataLoader
    return DataLoader, TensorDataset, gpytorch, json, np, os, plt, torch


@app.cell
def _():
    APPROXIMATE_LENGTHSCALES = [0.2636, 1.2216, 2.3385, 2.1103, 0.6089, 2.0037, 0.6698, 2.3557, 2.6022, 2.7145]
    return


@app.cell
def _(np):
    # Math function utilities:
    def logit(x):
        return np.log(x) - np.log(1 - x)

    def logistic(x):
        return 1 / (1 + np.exp(-x))

    def mean_hypercube_distance(n_dimensions):
        # Approximant taken from:
        # https://math.stackexchange.com/questions/1976842/how-is-the-distance-of-two-random-points-in-a-unit-hypercube-distributed
        return np.sqrt((n_dimensions / 6) - (7/120))
    return logit, mean_hypercube_distance


@app.cell
def _(json, np, os):
    def load_gridsearch_data(experiment_folderpath):
        # Load numpy data:
        parameter_matrix = np.load(
            os.path.join(experiment_folderpath, "sample_matrix.npy")
        )
        coherency_fractions = np.load(
            os.path.join(experiment_folderpath, "summary_data", "coherency_fractions.npy")
        )
        ann_indices = np.load(
            os.path.join(experiment_folderpath, "summary_data", "ann_indices.npy")
        )

        # Get gridsearch configuration:
        with open(os.path.join(experiment_folderpath, "config.json")) as json_filestream:
            config_dictionary  = json.load(json_filestream)
        gridsearch_parameters = config_dictionary["gridsearch_parameters"]

        return parameter_matrix, coherency_fractions, ann_indices, gridsearch_parameters
    return (load_gridsearch_data,)


@app.cell
def _(load_gridsearch_data):
    parameter_matrix, coherency_fractions, ann_indices, gridsearch_parameters = load_gridsearch_data(
        "model_experiments/2025-12-01-collisions_only"
    )
    return coherency_fractions, gridsearch_parameters, parameter_matrix


@app.cell
def _(coherency_fractions, np):
    np.count_nonzero(np.isnan(coherency_fractions)) / 32
    return


@app.cell
def _(coherency_fractions, logit, np, plt):
    plt.hist(np.mean(logit(coherency_fractions), axis=1), bins=100);
    plt.show()
    return


@app.cell
def _(coherency_fractions, np, plt):
    plt.hist(np.mean(np.log(coherency_fractions), axis=1), bins=100);
    plt.show()
    return


@app.cell
def _(coherency_fractions, np, plt):
    plt.scatter(np.mean(coherency_fractions, axis=1), np.var(coherency_fractions, axis=1), s=0.1)
    return


@app.cell
def _(coherency_fractions, np):
    WT_1_CF_MEAN = 0.03646833938262533
    WT_1_CF_VAR = 1.289952251152215e-05

    mean_var_mask = np.logical_and(
        np.mean(coherency_fractions, axis=1) > WT_1_CF_MEAN,
        np.var(coherency_fractions, axis=1) < WT_1_CF_VAR
    )

    np.count_nonzero(mean_var_mask) / len(mean_var_mask)
    return (mean_var_mask,)


@app.cell
def _(coherency_fractions, mean_var_mask, np, plt):
    plt.hist(np.mean(coherency_fractions, axis=1)[mean_var_mask], bins=25);
    plt.show()
    return


@app.cell
def _(gpytorch, mean_hypercube_distance):
    class ApproximateGPModel(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points, dimensions):
            # Set up distribution:
            variational_distribution = \
                gpytorch.variational.CholeskyVariationalDistribution(
                    inducing_points.size(0)
            )

            # Set up variational strategy:
            variational_strategy = \
                gpytorch.variational.VariationalStrategy(
                    self, inducing_points, variational_distribution,
                    learn_inducing_locations=True
            )

            # Inherit rest of init logic from approximate gp:
            super(ApproximateGPModel, self).__init__(variational_strategy)

            # Defining scale parameter prior:
            mean_distance = mean_hypercube_distance(dimensions)
            print(f"Prior mean: {mean_distance}")
            normal_prior = gpytorch.priors.NormalPrior(
                mean_distance, 7 / 120
            )

            # Define mean and covariance functions:
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=1.5,
                    ard_num_dims=dimensions,
                    # lengthscale_prior=normal_prior
                )
            ) + gpytorch.kernels.ConstantKernel()

            # Set initial parameters to means of priors:
            self.covar_module.kernels[0].base_kernel.lengthscale = normal_prior.mean

            # For greater estimation accuracy:
            self.double()

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    return (ApproximateGPModel,)


@app.cell
def _(ApproximateGPModel, DataLoader, TensorDataset, gpytorch, torch):
    def instantiate_model(inducing_points):
        # Convert inducing points to torch:
        inducing_points = torch.tensor(
            inducing_points, dtype=torch.float64
        )

        # Set up likelihoods:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ApproximateGPModel(inducing_points, inducing_points.shape[1])
        return model, likelihood

    def train_model(model, likelihood, x_dataset, y_dataset, epochs=1):
        # Convert datasets to torch:
        train_x = torch.tensor(x_dataset, dtype=torch.float64)
        train_y = torch.tensor(y_dataset, dtype=torch.float64)

        # Initialise dataloaders:
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

        # Set up training process:
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.01)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 64
        )

        # Set up loss:
        mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=train_y.size(0))
        loss_history = []
        for i in range(epochs):
            # Run through entire dataset:
            for batch_index, (x_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (batch_index + 1) % 10 == 0:
                    print(batch_index, loss.item())

                with torch.no_grad():
                    inducing_points = model.variational_strategy.inducing_points.detach()
                    model.variational_strategy.inducing_points[inducing_points > 1] = 1
                    model.variational_strategy.inducing_points[inducing_points < 0] = 0

            # # Print progress:
            # if (i + 1) % 5 == 0:
            #     print(i + 1)
            #     print(loss.detach())

            loss_history.append(loss.detach())

        return loss_history

    def run_inference(model, likelihood, inputs):
        tensor_input = torch.tensor(inputs, dtype=torch.float64)
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            predictions = likelihood(model(tensor_input))
        return predictions

    def run_grad_inference(model, likelihood, inputs):
        tensor_input = torch.tensor(inputs, dtype=torch.float64, requires_grad=True)
        tensor_input = torch.unsqueeze(tensor_input, 0)
        model.eval()
        likelihood.eval()
        predictions = likelihood(model(tensor_input))
        return tensor_input, predictions.mean
    return instantiate_model, run_inference, train_model


@app.cell
def _(np, parameter_matrix):
    # from scipy.stats import qmc

    # # Find well distributed inducing points:
    # inducing_exponent = 11
    # print(2**inducing_exponent)
    # sobol_sampler = qmc.Sobol(d=parameter_matrix.shape[1], scramble=True, rng=0)
    # sample_matrix = sobol_sampler.random_base2(m=inducing_exponent)

    from dppy.finite_dpps import FiniteDPP

    # Get likelihood matrix:
    print("Getting squared exponential likelihood matrix...", flush=True)
    distance_matrix = 1 - np.matmul(parameter_matrix[::7, :], parameter_matrix[::7, :].T).astype(np.float32)
    likelihood_matrix = np.exp(distance_matrix ** 2)

    # Set up determinantal point process:
    print("Setting up point process...")
    DPP = FiniteDPP('likelihood', **{'L': likelihood_matrix})

    k = 512
    DPP.sample_mcmc_k_dpp(size=k, random_state=None)

    # Get inducing points:
    inducing_indices = DPP.list_of_samples[0][-1]
    inducing_points = parameter_matrix[::7, :][inducing_indices, :]
    return (inducing_points,)


@app.cell
def _(inducing_points, instantiate_model):
    # Instantiate model:
    model, likelihood = instantiate_model(inducing_points)
    return likelihood, model


@app.cell
def _(model):
    model.covar_module.kernels[0]
    return


@app.cell
def _(model):
    model.covar_module.kernels[0].base_kernel.lengthscale_prior.mean
    return


@app.cell
def _(model):
    model.covar_module.kernels[0].base_kernel.lengthscale
    return


@app.cell
def _(model):
    model.variational_strategy.inducing_points
    return


@app.cell
def _(model):
    for param_name, param in model.named_parameters():
        print(f'Parameter name: {param_name:42} value = {param.detach().numpy()}')
    return


@app.cell
def _(
    coherency_fractions,
    likelihood,
    model,
    np,
    parameter_matrix,
    train_model,
):
    # Train model:
    loss_history = train_model(
        model, likelihood,
        parameter_matrix, np.mean(coherency_fractions, axis=1),
        epochs=5
    )
    return


@app.cell
def _(model, plt):
    learnt_inducing_points = model.variational_strategy.inducing_points.detach().numpy()

    def plot_inducing_points(points):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(points[:, 0], points[:, 9], s=5)
        # ax.set_xlim(-0.05, 1.05)
        # ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        plt.show()

    plot_inducing_points(learnt_inducing_points)
    return (plot_inducing_points,)


@app.cell
def _(inducing_points, plot_inducing_points):
    plot_inducing_points(inducing_points)
    return


@app.cell
def _(likelihood, model, parameter_matrix, run_inference):
    predictions = run_inference(model, likelihood, parameter_matrix)
    return (predictions,)


@app.cell
def _(plt, predictions):
    plt.hist(predictions.mean.detach().numpy().flatten(), bins=100);
    plt.show()
    return


@app.cell
def _(likelihood, model, parameter_matrix, qmc, run_inference):
    extrapolation_exponent = 20  # 1048576 extrapolated points for phase plot estimation.
    extrapolation_sampler = qmc.Sobol(d=parameter_matrix.shape[1], scramble=True, rng=0)
    extrapolated_points = extrapolation_sampler.random_base2(m=extrapolation_exponent)
    extrapolations = run_inference(model, likelihood, extrapolated_points)
    return extrapolated_points, extrapolations


@app.cell
def _(np, plt, predictions):
    plt.hist(np.sqrt(predictions.variance.detach().numpy()), bins=100);
    plt.show()
    return


@app.cell
def _(coherency_fractions, logit, np, plt, predictions):
    def plot_gp_mean_predictions(plot_logits=True):
        fig, ax = plt.subplots(figsize=(5, 5))
        if plot_logits:
            ax.scatter(
                np.mean(logit(coherency_fractions), axis=1),
                predictions.mean.detach().numpy(),
                c=np.log(predictions.variance.detach().numpy()),
                s=0.05
            )
            ax.plot([-10, 0.15], [-10, 0.15], c='r')
        else:
            ax.scatter(
                np.mean(coherency_fractions, axis=1),
                predictions.mean.detach().numpy(),
                c=np.log(predictions.variance.detach().numpy()),
                s=0.05
            )
            ax.plot([-0.05, 0.15], [-0.05, 0.15], c='r')
            ax.set_xlim(-0.01, None)
            ax.set_ylim(-0.01, None)
            ax.set_aspect("equal")
        plt.show()

    plot_gp_mean_predictions(plot_logits=False)
    return


@app.cell
def _(coherency_fractions, np, plt, predictions):
    def plot_gp_var_predictions():
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(
            np.var(coherency_fractions, axis=1),
            predictions.variance.detach().numpy(),
            s=0.05
        )
        ax.plot([0, 5e-3], [0, 5e-3], c='r')
        ax.set_aspect("equal")
        # ax.set_xlim(-0.02, 0.15)
        # ax.set_ylim(-0.02, 0.15)
        plt.show()

    plot_gp_var_predictions()
    return


@app.cell
def _(gridsearch_parameters, np, parameter_matrix, plt):
    def get_mean_and_variance(i_values, summary_metric, bin_count=50):
        # Set up output arrays:
        mean_summary_array = []
        std_summary_array = []
        bin_boundaries = np.linspace(0, 1, bin_count + 1)

        for bin_index in range(bin_count):
            low_threshold = bin_boundaries[bin_index]
            low_mask = i_values > low_threshold
            high_threshold = bin_boundaries[bin_index + 1]
            high_mask = i_values < high_threshold
            bin_mask = low_mask & high_mask
            mean_summary_array.append(np.mean(summary_metric[bin_mask]))
            std_summary_array.append(np.std(summary_metric[bin_mask]))

        return mean_summary_array, std_summary_array

    def plot_mean_and_variance(parameter_index, summary_metric, y_label):
        mean_array, std_array = get_mean_and_variance(parameter_matrix[:, parameter_index], summary_metric)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(np.linspace(0, 1, 51)[:50] + 1/50, mean_array)
        ax.set_xlabel(list(gridsearch_parameters.keys())[parameter_index])
        ax.set_ylabel(y_label)
        plt.show()

    def plot_binscatter_metric(metric_values, metric_label):
        fig, axs = plt.subplots(2, 5, figsize=(18, 4))

        for axis_index, ax in enumerate(axs.flatten()):
            mean_array, std_array = get_mean_and_variance(parameter_matrix[:, axis_index], metric_values)
            ax.scatter(np.linspace(0, 1, 51)[:50] + 1/50, mean_array, s=1)
            ax.set_xlabel(list(gridsearch_parameters.keys())[axis_index])
            ax.set_ylabel(metric_label)

        fig.suptitle(metric_label)
        fig.tight_layout()
        plt.show()
    return (plot_binscatter_metric,)


@app.cell
def _(plot_binscatter_metric, predictions):
    plot_binscatter_metric(predictions.mean.detach().numpy(), "CF Predictions")
    return


@app.cell
def _(gridsearch_parameters, np, parameter_matrix, plt):
    def generate_phase_matrix(i_values, j_values, summary_metric, mesh_count=20):
        # Set up array:
        grid_boundaries =  np.linspace(0, 1, mesh_count + 1)
        phase_array = np.zeros((mesh_count, mesh_count))

        # Loop through indices of array:
        for grid_index_i in range(mesh_count):
            # Get row parameter mask:
            i_threshold_low = grid_boundaries[grid_index_i]
            i_low_mask = i_values >= i_threshold_low
            i_threshold_high = grid_boundaries[grid_index_i+1]
            i_high_mask = i_values < i_threshold_high
            i_mask = np.logical_and(i_low_mask, i_high_mask)

            for grid_index_j in range(mesh_count):
                # Get column parameter mask:
                j_threshold_low = grid_boundaries[grid_index_j]
                j_low_mask = j_values >= j_threshold_low
                j_threshold_high = grid_boundaries[grid_index_j+1]
                j_high_mask = j_values < j_threshold_high
                j_mask = np.logical_and(j_low_mask, j_high_mask)

                total_mask = np.logical_and(i_mask, j_mask)
                phase_array[grid_index_i, grid_index_j] = \
                    np.min(summary_metric[total_mask])

        return phase_array

    def phaseplot_matrix_plot(summary_metric, title):
        fig, axs = plt.subplots(10, 10, figsize=(15, 15), sharex=True, sharey=True)

        for i in range(10):
            for j in range(10):
                if i == j:
                    axs[i, j].set_xlabel(list(gridsearch_parameters.keys())[i], fontsize=8)
                    axs[i, j].set_aspect("equal")
                    continue
                if bool(axs[i, j].get_images()):
                    continue

                phase_matrix = generate_phase_matrix(parameter_matrix[:, i], parameter_matrix[:, j], summary_metric)

                # Set upper triangle:
                axs[i, j].imshow(phase_matrix, origin="lower")
                # axs[i, j].set_xlabel(list(gridsearch_parameters.keys())[i], fontsize=8)
                # axs[i, j].set_ylabel(list(gridsearch_parameters.keys())[j])

                # Set lower triangle:
                axs[j, i].imshow(phase_matrix.T, origin="lower")
                # axs[j, i].set_xlabel(list(gridsearch_parameters.keys())[j])
                # axs[j, i].set_ylabel(list(gridsearch_parameters.keys())[i])

        fig.suptitle(title)
        fig.tight_layout()
        plt.show()
    return generate_phase_matrix, phaseplot_matrix_plot


@app.cell
def _(phaseplot_matrix_plot, predictions):
    phaseplot_matrix_plot(predictions.mean.detach().numpy(), "Gaussian Process Predictions")
    return


@app.cell
def _(generate_phase_matrix, parameter_matrix, plt, predictions):
    predictions_array = generate_phase_matrix(parameter_matrix[:, 2], parameter_matrix[:, 3], predictions.mean.detach().numpy(), 20)
    plt.imshow(predictions_array)
    return


@app.cell
def _(extrapolated_points, extrapolations, generate_phase_matrix, plt):
    mean_extrapolations_array = generate_phase_matrix(
        extrapolated_points[:, 0], extrapolated_points[:, 1],
        extrapolations.mean.detach().numpy(), 20
    )
    plt.imshow(mean_extrapolations_array, origin='lower')
    return (mean_extrapolations_array,)


@app.cell
def _(mean_extrapolations_array, plt):
    import scipy
    plt.contour(scipy.ndimage.zoom(mean_extrapolations_array, 1), 3)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
