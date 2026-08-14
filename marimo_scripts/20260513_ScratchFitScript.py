import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import json
    import math

    import torch
    import gpytorch

    import scipy.stats

    import colorcet as cc
    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt

    from torch.utils.data import TensorDataset, DataLoader

    import imageio.v3 as iio
    return (
        DataLoader,
        TensorDataset,
        cc,
        gpytorch,
        np,
        os,
        pd,
        plt,
        scipy,
        torch,
    )


@app.cell
def _():
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = "cmr10"
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams['axes.unicode_minus'] = False
    return


@app.cell
def _(np, os):
    test_dir = "model_experiments/2026-05-31-collisions_shape"
    test_new_op = np.load(os.path.join(test_dir, "summary_data/order_parameters.npy"))
    test_new_directions = np.load(os.path.join(test_dir, "summary_data/mean_directions.npy"))
    return test_new_directions, test_new_op


@app.cell
def _(np):
    ar = np.load("model_experiments/2026-05-31-collisions_shape/mcmc_results/rd_mcmc_acceptance_rate.npy")
    return (ar,)


@app.cell
def _(ar, np):
    np.mean(ar[0, :])
    return


@app.cell
def _(plt, test_new_directions, test_new_op):
    plt.scatter(test_new_op.flatten(), test_new_directions.flatten(), s=0.1);
    plt.show()
    return


@app.cell
def _(np, os, plt, scipy):
    EXPERIMENT_DIRPATH = "model_experiments/2026-05-20-collisions_shape"
    def plot_noise_predictions(metric_name):
        # Load metric:
        model_metric = np.load(os.path.join(EXPERIMENT_DIRPATH, "summary_data", f"{metric_name}.npy"))
        nan_mask = np.any(np.isnan(model_metric), axis=1)
        metric_sem = np.std(model_metric[~nan_mask], axis=1) / np.sqrt(16)
        log_sem = np.log(metric_sem)

        # Load GP predictions:
        noise_dirpath = os.path.join(EXPERIMENT_DIRPATH, "gaussian_process_models", f"{metric_name}_noise")
        noise_predictions = np.load(os.path.join(noise_dirpath, "parameter_predictions.npy"))[:, 0]

        # Get density for predictions:
        rng = np.random.default_rng(0)
        random_indices = rng.choice(len(log_sem), 2048)
        full_dataset = np.stack([log_sem, noise_predictions], axis=1)
        estimate_dataset = full_dataset[random_indices, :]
        density = scipy.stats.gaussian_kde(estimate_dataset.T)(full_dataset.T)
        density_sort = np.argsort(density)

        # Plot predictions:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.scatter(
            log_sem[density_sort],
            noise_predictions[density_sort],
            c=density[density_sort], s=1
        )

        # Plot linear guideline:
        limits = [np.min(full_dataset), np.max(full_dataset)]
        ax.plot(limits, limits, c='r', ls='--')

        # Format axes:
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal")
        plt.show()

    plot_noise_predictions("speeds")
    return (EXPERIMENT_DIRPATH,)


@app.cell
def _(np):
    sobol_likelihoods = np.load("model_experiments/2026-05-20-collisions_shape/mcmc_results/wt_sobol_likelihoods.npy")
    return (sobol_likelihoods,)


@app.cell
def _(EXPERIMENT_DIRPATH, np, os, plt, sobol_likelihoods):
    data_dirpath = "model_experiments/2026-05-20-collisions_shape/run_data"
    def plot_images():
        # Get relevant indices:
        run_indices = [83091] + list(np.argsort(sobol_likelihoods)[-25:])
        # run_indices = np.argwhere(broken_mask)
        model_metric = np.load(os.path.join(EXPERIMENT_DIRPATH, "summary_data", "coherency.npy"))
        parameter_matrix = np.load("model_experiments/2026-05-20-collisions_shape/sample_matrix.npy")
        image_archive = np.load("model_experiments/2026-05-20-collisions_shape/trajectory.npz")

        # Get relevant images:
        matrix_images = []
        for index in run_indices:
            print(sobol_likelihoods[index])
            matrix_image = image_archive[str(index)]
            matrix_images.append(matrix_image)

        # Plot images:
        fig, axs = plt.subplots(5, 5, layout="constrained", figsize=(7, 7))
        count = 0
        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(matrix_images[count])
                axs[i, j].set_axis_off()
                axs[i, j].set_aspect("equal")
                count += 1

        # Show image:
        plt.show()


    plot_images()
    return


@app.cell
def _(np):
    parameter_matrix = np.load("model_experiments/2026-05-20-collisions_shape/sample_matrix.npy")
    base_pm = np.copy(parameter_matrix)

    model_metric = np.load("model_experiments/2026-05-20-collisions_shape/summary_data/speeds.npy")
    base_metric = np.copy(model_metric)
    model_metric_predictions = \
        np.load("model_experiments/2026-05-20-collisions_shape/gaussian_process_models/speeds/parameter_predictions.npy")

    model_metric = np.nanmean(model_metric, axis=1)
    nan_mask = np.isnan(model_metric)
    model_metric = model_metric[~nan_mask]
    parameter_matrix = parameter_matrix[~nan_mask, :]

    def output_transform(gp_output, gp_se):
        scaled_out = (gp_output * np.std(model_metric)) + np.mean(model_metric)
        return scaled_out, gp_se * np.std(model_metric)
    return (
        base_metric,
        base_pm,
        model_metric,
        model_metric_predictions,
        nan_mask,
        output_transform,
        parameter_matrix,
    )


@app.cell
def _(base_pm, nan_mask, plt):
    nan_parameters = base_pm[nan_mask, :]

    def plot_nan_mcf():
        fig, ax = plt.subplots()
        ax.scatter(nan_parameters[:, 0], nan_parameters[:, 1], s=1)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect("equal")
        plt.show()

    plot_nan_mcf()
    return


@app.cell
def _(nan_mask, np):
    np.count_nonzero(nan_mask)
    return


@app.cell
def _(np, parameter_matrix):
    rng = np.random.default_rng(0)
    random_indices = rng.choice(parameter_matrix.shape[0], 2048)
    return (random_indices,)


@app.cell
def _(
    cc,
    model_metric,
    model_metric_predictions,
    np,
    plt,
    random_indices,
    scipy,
):
    def plot_predictions():
        # Set up plot:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        x = model_metric
        print(np.count_nonzero(np.isnan(np.log(x))))
        y = model_metric_predictions[:, 0]
        print(np.count_nonzero(np.isnan(np.log(y))))
        full_dataset = np.stack([x, y], axis=1)

        # Estimate density for scatter plot colouring:
        density = scipy.stats.gaussian_kde(full_dataset[random_indices, :].T)(full_dataset.T)
        density_sort = np.argsort(density)
        ax.scatter(
            full_dataset[density_sort, 0],
            full_dataset[density_sort, 1],
            c=density[density_sort],
            s=1, alpha=0.25, cmap=cc.m_CET_L20
        )
        # Plot linear guideline:
        limits = [np.min(full_dataset), np.max(full_dataset)]
        ax.plot(limits, limits, c='r', ls='--')

        # Format axes:
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal")
        plt.show()

    plot_predictions()
    return


@app.cell
def _(model_metric, model_metric_predictions, np):
    broken_mask = np.logical_and(model_metric < 0.05, model_metric_predictions[:, 0] > 0.75)
    return


@app.cell
def _(base_metric, cc, model_metric_predictions, nan_mask, np, plt, scipy):
    def plot_noise_predictions():
        # Set up plot:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))

        # Estimate density for scatter plot colouring:
        x = np.std(base_metric[~nan_mask, :], axis=1) / np.sqrt(16)
        print(np.count_nonzero(np.isnan(x)))
        # x = np.log(x)
        y = np.sqrt(model_metric_predictions[:, 1])
        # y = np.log(y)
        full_dataset = np.stack([x, y], axis=1)
        density = scipy.stats.gaussian_kde(full_dataset[::32].T)(full_dataset.T)
        density_sort = np.argsort(density)
        ax.scatter(
            x[density_sort],
            y[density_sort],
            c=density[density_sort],
            s=1, alpha=0.25, cmap=cc.m_CET_L20
        )
        # Plot linear guideline:
        limits = [np.min([x, y]), np.max([x, y])]
        ax.plot(limits, limits, c='r', ls='--')

        # Format axes:
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal")
        plt.show()

    plot_noise_predictions()
    return


@app.cell
def _(np):
    np.exp(2.5)
    return


@app.cell
def _(base_metric, np, plt):
    plt.hist(np.log(np.std(base_metric, axis=1) / np.sqrt(16)), bins=100);
    plt.show()
    return


@app.cell
def _(model_metric_predictions, np, plt):
    plt.hist(np.log(np.sqrt(model_metric_predictions[:, 1])), bins=100);
    plt.show()
    return


@app.cell
def _(cc, model_metric, np, parameter_matrix, plt, random_indices, scipy):
    plt.rcParams.update({'font.size': 8})

    def plot_parameter_scatter(parameter_index, output_metric):
        # Set up density estimation and subsampling:
        density_dataset = np.stack([parameter_matrix[:, parameter_index], output_metric], axis=0)
        density = scipy.stats.gaussian_kde(density_dataset[:, random_indices])(density_dataset)
        density_sort = np.argsort(density)

        # Plot scatterplot
        fig, ax = plt.subplots(figsize=(2.5, 1.5))
        ax.scatter(
            parameter_matrix[density_sort, parameter_index],
            output_metric[density_sort],
            c=density[density_sort], cmap=cc.m_CET_L20, s=0.5, alpha=0.5
        )
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        plt.show()

    plot_parameter_scatter(0, model_metric)
    return


@app.cell
def _(torch):
    # Need a higher precision for computing double derivatives:
    torch.set_default_dtype(torch.float64)
    return


@app.cell
def _(DataLoader, TensorDataset, gpytorch, os, torch):
    class DeepInputTransformation(torch.nn.Module):
        def __init__(self, dimension, hidden_layer_neuron_count=16):
            # Run general initialisation of the nn.Module base class:
            super().__init__()

            # Record parameters:
            self.dimension = dimension
            self.hl_neuron_count = hidden_layer_neuron_count

            # Set up layers:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(dimension, self.hl_neuron_count),
                torch.nn.SiLU(),
                torch.nn.Linear(self.hl_neuron_count, self.hl_neuron_count),
                torch.nn.SiLU(),
                torch.nn.Linear(self.hl_neuron_count, dimension)
            )

            # Initialise weights:
            with torch.no_grad():
                self.apply(self.initialise)

        def forward(self, x):
            return self.mlp.forward(x)

        def initialise(self, m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)


    class SparseGPModel(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points, dimensions):
            # Set up distribution:
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(0)
            )

            # Set up variational strategy:
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution,
                learn_inducing_locations=True
            )

            # Inherit rest of init logic from approximate GP:
            super().__init__(variational_strategy)

            # Instantiate input transform:
            print(f"Using dimensions: {dimensions}")
            self.input_transform = DeepInputTransformation(dimensions)

            # Define mean and additive covariance functions:
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=dimensions)
            )

        def forward(self, x):
            # Warp input:
            warped_x = self.input_transform(x)

            # Calculate mean of input:
            mean_x = self.mean_module(warped_x)
            covar_x = self.covar_module(warped_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    class ModelManager:

        def __init__(self, inducing_points, learning_rate):
            # Set up model:
            inducing_points = torch.tensor(inducing_points)
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.model = SparseGPModel(inducing_points, inducing_points.shape[1])

            # The default noise constraint sets the minimum too high,
            # we need the more permissive constraint of positivity:
            self.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Positive())

            # Set up optimisation - Adam seems to work best (need to properly test this):
            self.optimizer = torch.optim.Adam([
                {'params': self.model.parameters()},
                {'params': self.likelihood.parameters()},
            ], lr=learning_rate)

            self.loss_history = []

        def train_epoch(self, dataloader, num_data):
            # Ensure parameters are trainable:
            self.model.train()
            self.likelihood.train()

            # Set up loss:
            mll = gpytorch.mlls.PredictiveLogLikelihood(self.likelihood, self.model, num_data=num_data)

            # Run through entire dataset:
            for batch_index, (x_batch, y_batch) in enumerate(dataloader):
                self.optimizer.zero_grad()
                output_distribution = self.model(x_batch)
                loss = -mll(output_distribution, y_batch)
                loss.backward()

                # Step through optimisers:
                self.optimizer.step()
                if (batch_index + 1) % 10 == 0:
                    print(batch_index, loss.item(), flush=True)

                # Ensure inducing points don't go out of bounds (implicitly
                # imposing constraints with transforms degrades performance):
                with torch.no_grad():
                    inducing_points = self.model.variational_strategy.inducing_points.detach()
                    self.model.variational_strategy.inducing_points[inducing_points > 1] = 1
                    self.model.variational_strategy.inducing_points[inducing_points < 0] = 0

                self.loss_history.append(loss.detach())

        def train(self, x, y, batch_size, epochs=1):
            # Convert datasets to pytorch:
            x_tensor = torch.tensor(x)
            y_tensor = torch.tensor(y)
            dataset = TensorDataset(x_tensor, y_tensor)

            for _ in range(epochs):
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                self.train_epoch(dataloader, len(y))

        def save(self, experiment_dirpath, metric_name):
            # Generate GP model folder if not present:
            model_dirpath = os.path.join(experiment_dirpath, "gaussian_process_models")
            if not os.path.exists(model_dirpath):
                os.mkdir(model_dirpath)

            # Generate folder for given metric:
            metric_folderpath = os.path.join(model_dirpath, metric_name)
            if not os.path.exists(metric_folderpath):
                os.mkdir(metric_folderpath)

            # Save model components:
            model_filepath = os.path.join(metric_folderpath, "model.pth")
            torch.save(self.model, model_filepath)
            likelihood_filepath = os.path.join(metric_folderpath, "likelihood.pth")
            torch.save(self.likelihood, likelihood_filepath)
            optimiser_filepath = os.path.join(metric_folderpath, "optimiser.pth")
            torch.save(self.optimizer, optimiser_filepath)

        def load(self, experiment_dirpath, metric_name):
            id_folderpath = os.path.join(experiment_dirpath, "gaussian_process_models", metric_name)
            self.model = torch.load(os.path.join(id_folderpath, "model.pth"), weights_only=False)
            self.likelihood = torch.load(os.path.join(id_folderpath, "likelihood.pth"), weights_only=False)
            self.optimizer = torch.load(os.path.join(id_folderpath, "optimiser.pth"), weights_only=False)
    return (ModelManager,)


@app.cell
def _(torch):
    def emulate(manager, x):
        manager.likelihood.eval()
        manager.model.eval()
        tensor_input = torch.tensor(x)
        if len(tensor_input.shape) == 1:
            tensor_input = torch.unsqueeze(tensor_input, 0)
        with torch.no_grad():
            prediction = manager.likelihood(manager.model(tensor_input))
            prediction_mean = prediction.mean.detach().numpy()
            prediction_std = prediction.stddev.detach().numpy()
        return prediction_mean, prediction_std
    return (emulate,)


@app.cell
def _(ModelManager, np, parameter_matrix):
    # Load estimator:
    # -- Dummy points for instantiation:
    inducing_points = np.zeros((5, parameter_matrix.shape[1]))
    metric_model_manager = ModelManager(inducing_points, 0.003)
    metric_model_manager.load("model_experiments/2026-05-20-collisions_shape", "speeds")
    return (metric_model_manager,)


@app.cell
def _(np, pd):
    from statsmodels.regression import mixed_linear_model
    CAT_VAR = "C(phenotype, Treatment(reference='CTL'))[T.RD]"
    QUERY_COUNTS = [75, 250, 375]

    # Load wet lab data:
    site_dataframe = pd.read_csv("wetlab_data/site_dataframe.csv")
    particle_counts = np.array(site_dataframe["particle_count"])
    site_dataframe["scaled_particle_count"] = (particle_counts - np.mean(particle_counts)) / np.std(particle_counts)
    site_dataframe["mean_speed"] *= 60  # Get speed in µm/h.

    # Load regression results:
    regression_results = mixed_linear_model.MixedLMResults.load("wetlab_data/mean_speed.res")

    def get_fit_target(phenotype):
        # Get base parameters without phenotype interaction:
        base_intercept = regression_results.params["Intercept"]
        base_intercept_se = regression_results.bse["Intercept"]
        linear_coeff = regression_results.params["scaled_particle_count"]
        linear_coeff_se = regression_results.bse["scaled_particle_count"]
        square_coeff = regression_results.params["I(scaled_particle_count ** 2)"]
        square_coeff_se = regression_results.bse["I(scaled_particle_count ** 2)"]

        # Get phenotype interactions:
        phenotype_intercept = regression_results.params[f"{CAT_VAR}"]
        phenotype_intercept_se = regression_results.bse[f"{CAT_VAR}"]
        linear_interaction = regression_results.params[f"{CAT_VAR}:scaled_particle_count"]
        linear_interaction_se = regression_results.bse[f"{CAT_VAR}:scaled_particle_count"]
        square_interaction = regression_results.params[f"{CAT_VAR}:I(scaled_particle_count ** 2)"]
        square_interaction_se = regression_results.bse[f"{CAT_VAR}:I(scaled_particle_count ** 2)"]

        # Construct quadratic:
        a = square_coeff + (phenotype * square_interaction)
        b = linear_coeff + (phenotype * linear_interaction)
        c = base_intercept + (phenotype * phenotype_intercept)

        # Do error propagation to get standard errors in coefficients (multiplication preserves percentage errors):
        sq_int_se = np.abs((phenotype * square_interaction) * (square_interaction_se / square_interaction))
        lin_int_se = np.abs((phenotype * linear_interaction) * (linear_interaction_se / linear_interaction))
        base_int_se = np.abs((phenotype * phenotype_intercept) * (phenotype_intercept_se / phenotype_intercept))

        a_se = np.sqrt(square_coeff_se**2 + sq_int_se**2)
        b_se = np.sqrt(linear_coeff_se**2 + lin_int_se**2)
        c_se = np.sqrt(base_intercept_se**2 + base_int_se**2)

        # Get scaled points to query:
        x = (QUERY_COUNTS - np.mean(particle_counts)) / np.std(particle_counts)

        # Get regression prediction:
        y = a*(x**2) + b*x + c

        # Get regression standard error:
        quad_error = (a_se / a) * (a*(x**2))
        linear_error = (b_se / b) * (b*x)
        se = np.sqrt(quad_error**2 + linear_error**2 + c_se**2)
        return y, se

    fit_target, fit_se = get_fit_target(-0.5)
    fit_target /= 60
    fit_se /= 60
    return QUERY_COUNTS, fit_se, fit_target


@app.cell
def _(QUERY_COUNTS, emulate, metric_model_manager, np, output_transform):
    from scipy.stats import qmc
    GRIDSEARCH_COUNT_INDEX = 5

    def generate_sobol_sequence(dimension, exponent, seed):
        sobol_sampler = qmc.Sobol(d=dimension, scramble=True, rng=0)
        sample_matrix = sobol_sampler.random_base2(m=exponent)
        return sample_matrix

    def get_fit_distribution():
        query_space = generate_sobol_sequence(11, 16, 0)
        query_outputs = []
        query_errors = []
        for query_count in QUERY_COUNTS:
            input_count = (query_count - 50) / 350
            input_space = np.copy(query_space)
            input_space = np.insert(input_space, GRIDSEARCH_COUNT_INDEX, input_count, axis=1)
            metric_outputs, metric_se = emulate(metric_model_manager, input_space)
            query_outputs.append(metric_outputs)
            query_errors.append(metric_se)
        query_outputs = np.stack(query_outputs, axis=1)
        query_errors = np.stack(query_errors, axis=1)
        query_outputs, query_errors = output_transform(query_outputs, query_errors)
        return query_outputs, query_errors, query_space

    query_outputs, query_errors, query_space = get_fit_distribution()
    return (
        GRIDSEARCH_COUNT_INDEX,
        generate_sobol_sequence,
        query_errors,
        query_outputs,
        query_space,
    )


@app.cell
def _(fit_se, fit_target, np, query_errors, query_outputs):
    implausibilities = np.sqrt(((fit_target - query_outputs) ** 2) / (query_errors**2 + fit_se**2))
    implausibilities = np.sqrt(np.sum(implausibilities**2, axis=1))
    return (implausibilities,)


@app.cell
def _(fit_target, np, query_errors, query_outputs):
    def get_log_likelihood(x, mu, sigma):
        scale = 1 / np.sqrt(2 * np.pi * sigma**2)
        exponent = - (x - mu)**2 / (2 * sigma**2)
        return np.log(scale) + exponent

    log_likelihoods = get_log_likelihood(fit_target, query_outputs, query_errors)
    log_likelihoods = np.sum(log_likelihoods, axis=1)
    return get_log_likelihood, log_likelihoods


@app.cell
def _(fit_target, log_likelihoods, np, query_errors, query_outputs):
    print(fit_target)
    print(query_outputs[np.argmax(log_likelihoods), :])
    print(query_errors[np.argmax(log_likelihoods), :])
    return


@app.cell
def _(
    GRIDSEARCH_COUNT_INDEX,
    QUERY_COUNTS,
    emulate,
    fit_target,
    generate_sobol_sequence,
    get_log_likelihood,
    metric_model_manager,
    np,
    output_transform,
):
    SCALED_QUERY_COUNTS = (np.array(QUERY_COUNTS) - 50) / 350

    def get_proposal(x, rng):
        candidate_components = x + rng.normal(loc=0.0, scale=0.1, size=x.shape)
        acceptance_mask = ~np.logical_or(candidate_components < 0, candidate_components > 1)
        sampled_x = np.copy(x)
        sampled_x[acceptance_mask] = candidate_components[acceptance_mask]
        return sampled_x


    def get_likelihood(x):
        input_x = []
        for count in SCALED_QUERY_COUNTS:
            input_space = np.copy(x)
            input_space = np.insert(input_space, GRIDSEARCH_COUNT_INDEX, count, axis=1)
            input_x.append(input_space)
        input_x = np.concatenate(input_x, axis=0)
        metric_outputs, metric_se = emulate(metric_model_manager, input_x)
        query_outputs, query_errors = output_transform(metric_outputs, metric_se)
        query_outputs = np.reshape(query_outputs, (3, -1)).T
        query_errors = np.reshape(query_errors, (3, -1)).T
        log_likelihood = get_log_likelihood(fit_target, query_outputs, query_errors)
        return np.sum(log_likelihood, axis=1)


    def quick_mcmc(batch_size=512):
        # Generate first candidate point & likelihood:
        rng = np.random.default_rng(0)
        current_x = generate_sobol_sequence(11, 9, 0)
        current_likelihood = get_likelihood(current_x)

        # Iterate through chain:
        chain = [current_x]
        likelihoods = [current_likelihood]
        for i in range(8):
            # Sample new x:
            proposal_x = get_proposal(current_x, rng)
            proposal_likelihood = get_likelihood(proposal_x)

            # Calculate acceptance criterion:
            acceptance_ratio = np.exp(proposal_likelihood - current_likelihood)
            uniform_sample = rng.uniform(size=(batch_size))
            acceptance_mask = uniform_sample < acceptance_ratio

            # Update population:
            current_x[acceptance_mask, :] = proposal_x[acceptance_mask, :]
            current_likelihood[acceptance_mask] = proposal_likelihood[acceptance_mask]

            # Add to chain:
            chain.append(np.copy(current_x))
            likelihoods.append(np.copy(current_likelihood))

        # Return (truncated) chain:
        chain = np.concatenate(chain[64:], axis=0)
        likelihoods = np.concatenate(likelihoods[64:], axis=0).flatten()
        return chain, likelihoods

    mc_distribution, likelihoods = quick_mcmc()
    return SCALED_QUERY_COUNTS, get_likelihood, likelihoods, mc_distribution


@app.cell
def _(get_likelihood, np, parameter_matrix):
    likelihood_parameters = np.concatenate([parameter_matrix[:, :5], parameter_matrix[:, 6:]], axis=1)
    simulation_likelihoods = get_likelihood(likelihood_parameters)
    return likelihood_parameters, simulation_likelihoods


@app.cell
def _(
    GRIDSEARCH_COUNT_INDEX,
    SCALED_QUERY_COUNTS,
    emulate,
    fit_target,
    metric_model_manager,
    np,
    output_transform,
):
    def get_quadratic_loss(x):
        input_x = []
        for count in SCALED_QUERY_COUNTS:
            input_space = np.copy(x)
            input_space = np.insert(input_space, GRIDSEARCH_COUNT_INDEX, count, axis=1)
            input_x.append(input_space)
        input_x = np.concatenate(input_x, axis=0)
        metric_outputs, metric_se = emulate(metric_model_manager, input_x)
        query_outputs, query_errors = output_transform(metric_outputs, metric_se)
        query_outputs = np.reshape(query_outputs, (3, -1)).T
        loss = np.sum((fit_target - query_outputs) ** 2, axis=1)
        return loss
    return (get_quadratic_loss,)


@app.cell
def _(get_quadratic_loss, likelihood_parameters):
    mse_loss = get_quadratic_loss(likelihood_parameters[:, :])
    return (mse_loss,)


@app.cell
def _(likelihood_parameters):
    likelihood_parameters.shape
    return


@app.cell
def _(mse_loss):
    mse_loss.shape
    return


@app.cell
def _(mse_loss, np):
    np.count_nonzero(mse_loss < 0.0005)
    return


@app.cell
def _(cc, likelihood_parameters, mse_loss, np, plt, scipy):
    def test_quadratic_mcf_plot(query_space, mask, index_i, index_j):
        # Get MCF criteria:
        print(np.min(query_space[mask, :], axis=0))
        print(np.max(query_space[mask, :], axis=0))
        # Estimate density for plotting:
        density_dataset = np.stack([query_space[mask, index_i], query_space[mask, index_j]], axis=1)
        estimate_dataset = density_dataset[:, :]
        print(estimate_dataset.shape)
        density = scipy.stats.gaussian_kde(estimate_dataset.T)(density_dataset.T)
        density_sort = np.argsort(density)

        # Do parameter scatter plot:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.scatter(
            density_dataset[density_sort, 0],
            density_dataset[density_sort, 1],
            c=density[density_sort], cmap=cc.m_CET_L20,
            s=7.5
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        plt.show()

    test_quadratic_mcf_plot(likelihood_parameters, mse_loss < 0.0005, 5, 6)
    return


@app.cell
def _(plt, q_loss, simulation_likelihoods):
    plt.scatter(simulation_likelihoods, q_loss, s=1)
    return


@app.cell
def _(model_metric, plt, simulation_likelihoods):
    plt.scatter(model_metric, simulation_likelihoods, s=1)
    return


@app.cell
def _():
    # reduced_parameter_matrix = np.concatenate([base_pm[:, :5], base_pm[:, 6:]], axis=1)
    # sobol_likelihoods = np.load("model_experiments/2026-05-20-collisions_shape/mcmc_results/sobol_likelihoods.npy")
    # check_index = np.argsort(sobol_likelihoods)[-1]
    # print(check_index)
    # print(sobol_likelihoods[check_index])
    # print(reduced_parameter_matrix[check_index])
    return


@app.cell
def _(np):
    1 * (np.sqrt(2) ** 5)
    return


@app.cell
def _(base_metric, np, plt, sobol_likelihoods):
    def plot_likelihood_v_metric():
        fig, ax = plt.subplots()
        ax.scatter(np.nanmean(base_metric, axis=1), np.exp(sobol_likelihoods), s=1)
        # ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        plt.show()

    plot_likelihood_v_metric()
    return


@app.cell
def _(plt, reduced_parameter_matrix, sobol_likelihoods):
    def reference_table_mcf(i, j):
        selected_sobol_sets = reduced_parameter_matrix[sobol_likelihoods > 8.5, :]
        fig, ax = plt.subplots()
        ax.scatter(selected_sobol_sets[:, i], selected_sobol_sets[:, j], c=sobol_likelihoods[sobol_likelihoods > 8.5])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        plt.show()

    reference_table_mcf(0, 1)
    return


@app.cell
def _(np, os, plt, sobol_likelihoods):
    from PIL import Image

    def plot_sim_range(grid_size):
        fig, ax = plt.subplots(grid_size, grid_size, figsize=(7.5, 7.5), layout="constrained")
        data_dirpath = "model_experiments/2026-05-20-collisions_shape/run_data"

        count = 1
        for i in range(grid_size):
            for j in range(grid_size):
                plot_index = np.argsort(sobol_likelihoods)[-count]
                print(sobol_likelihoods[plot_index])
                image_filepath = os.path.join(
                    data_dirpath,
                    f"{int(np.floor(plot_index / 1000))}/{plot_index}/trajectory.png"
                )
                pil_image = Image.open(image_filepath)
                ax[i, j].imshow(pil_image)
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                ax[i, j].set_axis_off()
                count += 1

        plt.show()

    plot_sim_range(7)
    return


@app.cell
def _(likelihoods, plt):
    plt.hist(likelihoods, bins=100);
    plt.show()
    return


@app.cell
def _(likelihoods, plt):
    for i in range(512):
        plt.plot(likelihoods[i::512])

    plt.show()
    return


@app.cell
def _(cc, mc_distribution, np, plt, scipy):
    def plot_mc_distribution(i, j):
        # Estimate density for plotting:
        density_dataset = np.stack([mc_distribution[:, i], mc_distribution[:, j]], axis=1)

        # Subsample so it doesn't take years:
        rng = np.random.default_rng(0)
        estimate_indices = rng.choice(density_dataset.shape[0], 4092)
        estimate_dataset = density_dataset[estimate_indices, :]
        kde = scipy.stats.gaussian_kde(estimate_dataset.T)

        # Get mesh over which to evaluate KDE:
        points = np.linspace(0, 1, 101)[1:-1]
        X, Y = np.meshgrid(points, points)
        eval_points = np.stack([X.ravel(), Y.ravel()])
        densities = kde(eval_points)
        densities = np.reshape(densities, (99, 99))[::-1, :]
        densities = densities / np.sum(densities)

        # Get effective display range:
        stddev = np.std(densities)
        uniform_density = 1 / 99**2
        vmin = uniform_density - (3 * stddev)
        vmax = uniform_density + (3 * stddev)

        # Plot sample:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.imshow(
            densities,
            extent=[0, 1, 0, 1],
            vmin=vmin, vmax=vmax,
            cmap=cc.m_CET_D3
        )
        ax.set_aspect("equal")
        plt.show()

    plot_mc_distribution(9, 10)
    return


@app.cell
def _(mc_distribution, plt):
    plt.hist(mc_distribution[:, 0], bins=100);
    plt.show()
    return


@app.cell
def _(log_likelihoods, plt):
    plt.hist(log_likelihoods, bins=100);
    plt.show()
    return


@app.cell
def _(implausibilities, plt):
    plt.hist(implausibilities, bins=100);
    plt.show()
    return


@app.cell
def _(fit_target, implausibilities, np, query_errors, query_outputs):
    print(fit_target)
    print(query_outputs[np.argmin(implausibilities), :])
    print(query_errors[np.argmin(implausibilities), :])
    return


@app.cell
def _(cc, log_likelihoods, np, plt, query_space, scipy):
    def test_mcf_plot(index_i, index_j):
        # Get MCF criteria:
        implausible_mask = log_likelihoods > 0
        print(np.min(query_space[implausible_mask, :], axis=0))
        print(np.max(query_space[implausible_mask, :], axis=0))

        # Estimate density for plotting:
        density_dataset = np.stack([query_space[implausible_mask, index_i], query_space[implausible_mask, index_j]], axis=1)
        estimate_dataset = density_dataset[:, :]
        density = scipy.stats.gaussian_kde(estimate_dataset.T)(density_dataset.T)
        density_sort = np.argsort(density)

        # Do parameter scatter plot:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.scatter(
            density_dataset[density_sort, 0],
            density_dataset[density_sort, 1],
            c=density[density_sort], cmap=cc.m_CET_L20,
            s=7.5
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        plt.show()

    test_mcf_plot(0, 1)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
