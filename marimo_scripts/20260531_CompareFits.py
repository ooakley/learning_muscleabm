import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import json

    import torch
    import gpytorch

    import numpy as np
    import colorcet as cc

    import matplotlib.pyplot as plt
    return cc, gpytorch, json, np, os, plt, torch


@app.cell(hide_code=True)
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
def _(json, os):
    EXPERIMENT_DIRPATH = "model_experiments/2026-05-31-collisions_shape"
    with open(os.path.join(EXPERIMENT_DIRPATH, 'config.json')) as json_file:
        config_dictionary = json.load(json_file)
    return EXPERIMENT_DIRPATH, config_dictionary


@app.cell
def _(config_dictionary):
    gs_parameters = config_dictionary["gridsearch_parameters"]
    gs_parameters = gs_parameters[:5] + gs_parameters[6:]
    return (gs_parameters,)


@app.cell
def _(EXPERIMENT_DIRPATH, np, os):
    control_mcmc_chain = np.load(os.path.join(EXPERIMENT_DIRPATH, "mcmc_results", "wt_mcmc_chain.npy"))
    rd_mcmc_chain = np.load(os.path.join(EXPERIMENT_DIRPATH, "mcmc_results", "rd_mcmc_chain.npy"))

    ctl_fit = control_mcmc_chain[2048::8, :, 0, :].reshape(-1, 11)
    rd_fit = rd_mcmc_chain[2048::8, :, 0, :].reshape(-1, 11)
    return ctl_fit, rd_fit


@app.cell
def _(ctl_fit):
    ctl_fit.shape
    return


@app.cell
def _(EXPERIMENT_DIRPATH, np, os):
    control_mcmc_likelihoods = np.load(os.path.join(EXPERIMENT_DIRPATH, "mcmc_results", "wt_mcmc_likelihoods.npy"))
    rd_mcmc_likelihoods = np.load(os.path.join(EXPERIMENT_DIRPATH, "mcmc_results", "rd_mcmc_likelihoods.npy"))

    ctl_likelihoods = control_mcmc_likelihoods[4092::4, :, 0].flatten()
    rd_likelihoods = rd_mcmc_likelihoods[4092::4, :, 0].flatten()
    return


@app.cell
def _(ctl_fit, gs_parameters, np, plt, rd_fit):
    def plot_parameter_histogram(index):
        print(gs_parameters[index][0])
        bins = np.linspace(0, 1, 51)
        fig, ax = plt.subplots(figsize=(2.5, 1.75))
        ax.hist(ctl_fit[:, index], bins=bins, histtype="step", color="#1A85FF", density=True)
        ax.hist(rd_fit[:, index], bins=bins, histtype="step", color="#D41159", density=True)
        ax.set_xlim(0, 1)
        plt.show()

    plot_parameter_histogram(6)
    return


@app.cell
def _(plt):
    def plot_joint_distribution(distribution, i, j):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.scatter(distribution[:, i], distribution[:, j], s=0.1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        plt.show()
    return (plot_joint_distribution,)


@app.cell
def _(ctl_fit, plot_joint_distribution):
    plot_joint_distribution(ctl_fit, 6, 7)
    return


@app.cell
def _(plot_joint_distribution, rd_fit):
    plot_joint_distribution(rd_fit, 6, 7)
    return


@app.cell
def _(ModelManager, np):
    matrix_experiment_dirpath = "model_experiments/2026-06-03-matrix_shape"
    inducing_points = np.zeros((32, 13))
    model_manager = ModelManager(inducing_points, 0.003)
    model_manager.load(matrix_experiment_dirpath, "op65")
    return matrix_experiment_dirpath, model_manager


@app.cell
def _(json, matrix_experiment_dirpath, model_manager, np, os, torch):
    def get_matrix_parameters():
        with open(os.path.join(matrix_experiment_dirpath, "config.json")) as json_file:
            matrix_config_dictionary = json.load(json_file)
        return matrix_config_dictionary["gridsearch_parameters"]

    matrix_parameters = get_matrix_parameters()

    order_parameters = np.load(os.path.join(matrix_experiment_dirpath, "summary_data", "matrix_order_parameters.npy"))
    print(order_parameters.shape)

    op65 = np.nanmean(order_parameters[:, :, 2], axis=1)
    mean_op65 = np.mean(op65)
    std_op65 = np.std(op65)

    def add_scaled_matrix_parameters(x, m_advection_rate, m_sample_rate, cell_number):
        sample_size = x.shape[0]

        # Scale advection rate:
        m_ar_min = matrix_parameters[-3][1][0]
        m_ar_max = matrix_parameters[-3][1][1]
        scaled_m_ar = (m_advection_rate - m_ar_min) / (m_ar_max - m_ar_min)
        scaled_m_ar = np.ones((sample_size, 1)) * scaled_m_ar

        # Scale sample rate:
        m_sr_min = matrix_parameters[-2][1][0]
        m_sr_max = matrix_parameters[-2][1][1]
        scaled_m_sr = (m_sample_rate - m_sr_min) / (m_sr_max - m_sr_min)
        scaled_m_sr = np.ones((sample_size, 1)) * scaled_m_sr

        # Scale sample rate:
        cell_no_min = matrix_parameters[-1][1][0]
        cell_no_mmax = matrix_parameters[-1][1][1]
        scaled_cell_no = (cell_number - cell_no_min) / (cell_no_mmax - cell_no_min)
        scaled_cell_no = np.ones((sample_size, 1)) * scaled_cell_no

        return np.concatenate([x, scaled_m_ar, scaled_m_sr, scaled_cell_no], axis=1)

    def matrix_estimate(x, m_advection_rate, m_sample_rate, cell_number):
        np_input = add_scaled_matrix_parameters(x, m_advection_rate, m_sample_rate, cell_number)
        tensor_input = torch.tensor(np_input)

        prediction = model_manager.likelihood(
            model_manager.model(tensor_input)
        )

        return (prediction.mean.detach().numpy() * std_op65) + mean_op65
    return add_scaled_matrix_parameters, matrix_estimate


@app.cell
def _():
    # ctl_sr_fit = ctl_bt_fit[ctl_bt_fit[:, 6] < 0.55, :]
    # rd_sr_fit = rd_bt_fit[rd_bt_fit[:, 6] < 0.55, :]
    return


@app.cell
def _(ctl_fit, matrix_estimate, rd_fit):
    advection_rate = 3.0
    wt_organisation = matrix_estimate(ctl_fit, advection_rate, 1.0, 375)
    rd_organisation = matrix_estimate(rd_fit, advection_rate, 15.0, 375)
    return rd_organisation, wt_organisation


@app.cell
def _(plt, rd_organisation, wt_organisation):
    def plot_op65_histogram():
        fig, ax = plt.subplots(figsize=(2.5, 1.75))
        ax.hist(wt_organisation, bins=50, histtype="step", color="#1A85FF", density=True)
        ax.vlines(wt_organisation.mean(), 0, 10, color="#1A85FF")
        ax.hist(rd_organisation, bins=50, histtype="step", color="#D41159", density=True)
        ax.vlines(rd_organisation.mean(), 0, 10, color="#D41159")
        plt.show()

    plot_op65_histogram()
    return


@app.cell
def _(ctl_fit, np, plt, wt_organisation):
    test_sort = np.argsort(wt_organisation)
    plt.scatter(ctl_fit[test_sort, 3], ctl_fit[test_sort, -3], c=wt_organisation[test_sort], s=1)
    return


@app.cell
def _(np):
    op_hessian_estimate = np.load("model_experiments/2026-05-22-matrix_shape/gaussian_process_models/op31/hessian_estimate.npy")
    return (op_hessian_estimate,)


@app.cell
def _(np):
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
    return (get_eigenvectors,)


@app.cell
def _(get_eigenvectors, op_hessian_estimate):
    eigenvalues, eigenvectors = get_eigenvectors(op_hessian_estimate)
    return eigenvalues, eigenvectors


@app.cell
def _(eigenvectors, plt):
    plt.hist(eigenvectors[:, 0, 10], bins=100);
    plt.show()
    return


@app.cell
def _(eigenvalues, eigenvectors, np):
    def calculate_geometric_median(key_eigenvectors, key_eigenvalues, weighting=None):
        if weighting is None:
            weighting = np.ones(key_eigenvectors.shape[0])
        proposal_median = np.mean(key_eigenvectors, axis=0)
        converged = False
        while not converged:
            # Get distances:
            cosine_similarities = key_eigenvectors @ np.expand_dims(proposal_median, axis=1)
            nematic_distances = 1 - np.abs(cosine_similarities)

            # Get weighted average of appropriately flipped vectors:
            flipped_vectors = np.sign(cosine_similarities) * key_eigenvectors
            combined_weights = (weighting) / np.squeeze(nematic_distances)
            combined_weights = np.expand_dims(combined_weights, axis=1)
            new_proposal = np.sum(flipped_vectors * combined_weights, axis=0) / np.sum(combined_weights)
            new_proposal /= np.linalg.norm(new_proposal)
            epsilon = 1 - np.dot(proposal_median, new_proposal)
            if epsilon < 1e-8:
                converged = True
            proposal_median = new_proposal

        # Get average distance from median:
        cosine_similarities = key_eigenvectors @ np.expand_dims(proposal_median, axis=1)
        nematic_distances = 1 - np.abs(cosine_similarities)

        return proposal_median, flipped_vectors, np.mean(nematic_distances)

    proposal_median, flipped_vectors, _ = calculate_geometric_median(eigenvectors[:, 0, :], eigenvalues[:, 0])
    return (proposal_median,)


@app.cell
def _(add_scaled_matrix_parameters, np):
    def estimate_projected_fit(fit_distribution, transform, matrix_advection_rate=1.0):
        mat_distribution = add_scaled_matrix_parameters(fit_distribution, matrix_advection_rate, 5)
        # Return global eigenparameter values:
        ge_values = np.sum(np.log(mat_distribution) * transform, axis=1)
        return ge_values
    return (estimate_projected_fit,)


@app.cell
def _(ctl_fit, estimate_projected_fit, proposal_median, rd_fit):
    wt_ge_values = estimate_projected_fit(ctl_fit, proposal_median)
    rd_ge_values = estimate_projected_fit(rd_fit, proposal_median)
    return rd_ge_values, wt_ge_values


@app.cell
def _(plt, rd_ge_values, wt_ge_values):
    plt.hist(wt_ge_values, bins=100, histtype="step");
    plt.hist(rd_ge_values, bins=100, histtype="step");
    plt.show()
    return


@app.cell
def _(eigenvectors, np):
    # Get estimate of distance on manifold of primary eigenparameters:
    import scipy.sparse
    from scipy.sparse.csgraph import dijkstra
    from sklearn.neighbors import NearestNeighbors

    # First get nematic similarity:
    similarity_matrix = np.abs(eigenvectors[:, 0, :] @ eigenvectors[:, 0, :].T)
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
    return (dijkstra_distance_matrix,)


@app.cell
def _(np, torch):
    PARAMETER_DIMENSION = 13

    def log_sammon_mapping(input_data, distance_matrix, n_components, batch_size=512, seed=0):
        # Get relevant dimensions:
        sample_count = distance_matrix.shape[0]

        # Convert to torch:
        input_tensor = torch.from_numpy(input_data)
        # input_tensor = (input_tensor * 0.95) + 0.05

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

        print("Performing gradient descent for transform...")
        loss_history = []
        for step_index in range(20000):
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
            axis_similarity = torch.sum(torch.abs(similiarities))

            # Get loading entropies:
            # --- Get entropies by transform dimension:
            absolute_components = torch.abs(log_transform)
            norm_transform_components = absolute_components / torch.linalg.norm(absolute_components, dim=0, keepdims=True)
            transform_entropies = torch.sum(-norm_transform_components * torch.log(norm_transform_components), dim=0)

            # --- Get entropies by parameter dimension:
            norm_param_components = absolute_components / torch.linalg.norm(absolute_components, dim=1, keepdims=True)
            param_entropies = torch.sum(-norm_param_components * torch.log(norm_param_components), dim=1)

            total_entropy = torch.sum(torch.concatenate([transform_entropies, param_entropies]))

            # Run backprop and take update step:
            opt.zero_grad()
            (sammon_error + (5e-3 * total_entropy)).backward()
            opt.step()
            loss_history.append(sammon_error.item())

            # Print progress:
            if (step_index + 1) % 500 == 0:
                print(step_index + 1, sammon_error.item(), scale_factor)

        return loss_history, log_transform.detach().numpy()
    return (log_sammon_mapping,)


@app.cell
def _(dijkstra_distance_matrix, log_sammon_mapping, np):
    hessian_inputs = np.load("model_experiments/2026-05-22-matrix_shape/gaussian_process_models/op31/hessian_inputs.npy")
    loss_history, log_transform = log_sammon_mapping(hessian_inputs, dijkstra_distance_matrix, 2, 64, 0)
    return hessian_inputs, log_transform


@app.cell
def _(ctl_fit, estimate_projected_fit, log_transform, rd_fit):
    mat_advection_rate = 0.1
    ctl_phase_i = estimate_projected_fit(ctl_fit, log_transform[:, 0], mat_advection_rate)
    ctl_phase_j = estimate_projected_fit(ctl_fit, log_transform[:, 1], mat_advection_rate)
    rd_phase_i = estimate_projected_fit(rd_fit, log_transform[:, 0], mat_advection_rate)
    rd_phase_j = estimate_projected_fit(rd_fit, log_transform[:, 1], mat_advection_rate)
    return ctl_phase_i, ctl_phase_j, rd_phase_i, rd_phase_j


@app.cell
def _(hessian_inputs, model_manager, torch):
    input_predictions = model_manager.likelihood(
        model_manager.model(torch.from_numpy(hessian_inputs[:32760, :]))
    )
    input_predictions = input_predictions.mean.detach().numpy()
    return (input_predictions,)


@app.cell
def _(log_transform, np):
    print(np.round(log_transform[:, 0], 2))
    return


@app.cell
def _(log_transform, np):
    print(np.round(log_transform[:, 1], 2))
    return


@app.cell
def _(cc, hessian_inputs, input_predictions, log_transform, np, plt):
    input_distribution = np.log(hessian_inputs) @ log_transform
    pred_sort = np.argsort(input_predictions)

    def plot_distribution(fit_data_i, fit_data_j):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.scatter(
            input_distribution[pred_sort, 0],
            input_distribution[pred_sort, 1],
            c=input_predictions[pred_sort],
            s=1,
            cmap=cc.m_CET_L20
        )
        ax.scatter(np.exp(fit_data_i), np.exp(fit_data_j), s=1, c='r', alpha=0.1)
        # ax.set_xlim(-15, 3)
        # ax.set_ylim(-8, 10)
        plt.show()

    plot_distribution(None, None)
    return input_distribution, plot_distribution, pred_sort


@app.cell
def _(ctl_phase_i, ctl_phase_j, plot_distribution):
    plot_distribution(ctl_phase_i, ctl_phase_j)
    return


@app.cell
def _(plot_distribution, rd_phase_i, rd_phase_j):
    plot_distribution(rd_phase_i, rd_phase_j)
    return


@app.cell
def _(
    cc,
    input_distribution,
    input_predictions,
    plt,
    pred_sort,
    rd_phase_i,
    rd_phase_j,
):
    plt.scatter(
        input_distribution[pred_sort, 0],
        input_distribution[pred_sort, 1],
        c=input_predictions[pred_sort], s=1,
        cmap=cc.m_CET_L20
    )

    plt.scatter(rd_phase_i, rd_phase_j, s=1, alpha=0.1)
    return


app._unparsable_cell(
    r"""
    |# plt.scatter(rd_phase_i, rd_phase_j, s=1, alpha=0.25)
    """,
    name="_"
)


@app.cell
def _(ctl_phase_i, plt, rd_phase_i):
    plt.hist(ctl_phase_i, bins=100, histtype="step");
    plt.hist(rd_phase_i, bins=100, histtype="step");
    plt.show()
    return


@app.cell
def _(ctl_phase_j, plt, rd_phase_j):
    plt.hist(ctl_phase_j, bins=100, histtype="step");
    plt.hist(rd_phase_j, bins=100, histtype="step");
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
