import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    # WEIGHT DISTANCE IMPORTANCE BY SIZE OF EIGENVALUES?
    # RUN MAPPING ON DISTANCE BETWEEN HESSIANS?
    return


@app.cell
def _():
    PARAMETER_DIMENSION = 11
    return (PARAMETER_DIMENSION,)


@app.cell
def _():
    import os
    import json
    import copy

    import gpytorch
    import torch

    import numpy as np
    import matplotlib.pyplot as plt

    from torch.utils.data import TensorDataset, DataLoader
    from dppy.finite_dpps import FiniteDPP

    import colorcet as cc
    return (
        DataLoader,
        TensorDataset,
        cc,
        copy,
        gpytorch,
        json,
        np,
        os,
        plt,
        torch,
    )


@app.cell
def _(torch):
    torch.set_default_dtype(torch.float64)
    return


@app.cell(hide_code=True)
def _(DataLoader, TensorDataset, gpytorch, json, np, os, torch):
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
        speeds = np.load(
            os.path.join(experiment_folderpath, "summary_data", "magnitude_cellmeans.npy")
        )

        # Get gridsearch configuration:
        with open(os.path.join(experiment_folderpath, "config.json")) as json_filestream:
            config_dictionary  = json.load(json_filestream)
        gridsearch_parameters = config_dictionary["gridsearch_parameters"]

        return parameter_matrix, coherency_fractions, ann_indices, speeds, gridsearch_parameters


    class DeepInputTransformation(torch.nn.Module):
        def __init__(self, dimension, hidden_layer_neuron_count=64):
            # Run general initialisation of the nn.Module base class:
            super().__init__()

            # Record parameters:
            self.dimension = dimension
            self.hl_neuron_count = hidden_layer_neuron_count

            # Set up layers:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(dimension, self.hl_neuron_count),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hl_neuron_count, self.hl_neuron_count),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hl_neuron_count, 10)
            )

            # Initialise weights:
            with torch.no_grad():
                self.apply(self.initialise)

        def forward(self, x):
            return self.mlp.forward(x)

        def initialise(self, m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                diagonal_index_object = range(min(m.weight.size()))
                # m.weight[diagonal_index_object, diagonal_index_object] = 1


    class SparseGPModel(gpytorch.models.ApproximateGP):
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

            # Inherit rest of init logic from approximate GP:
            super().__init__(variational_strategy)

            # Instantiate input transform:
            self.input_transform = DeepInputTransformation(dimensions)

            # Define mean and additive covariance functions:
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=10)
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
            inducing_points = torch.tensor(
                inducing_points, dtype=torch.float64
            )
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.model = SparseGPModel(inducing_points, inducing_points.shape[1])

            # The default noise constraint is too high, more permissive constraint of positivity:
            self.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Positive())

            # Set up optimisation:
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
            loss_history = []

            # Run through entire dataset:
            for batch_index, (x_batch, y_batch) in enumerate(dataloader):
                self.optimizer.zero_grad()
                output_distribution = self.model(x_batch)
                loss = -mll(output_distribution, y_batch)
                loss.backward()

                # Step through optimisers:
                self.optimizer.step()
                if (batch_index + 1) % 64 == 0:
                    print(batch_index, loss.item())

                # Ensure inducing points don't go out of bounds:
                with torch.no_grad():
                    inducing_points = self.model.variational_strategy.inducing_points.detach()
                    self.model.variational_strategy.inducing_points[inducing_points > 1] = 1
                    self.model.variational_strategy.inducing_points[inducing_points < 0] = 0

                self.loss_history.append(loss.detach())

        def train(self, x, y, batch_size, epochs=1):
            # Convert datasets to pytorch:
            x_tensor = torch.tensor(x, dtype=torch.float64)
            y_tensor = torch.tensor(y, dtype=torch.float64)
            dataset = TensorDataset(x_tensor, y_tensor)

            for _ in range(epochs):
                print(f"---> Epoch {_ + 1}...")
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                self.train_epoch(dataloader, len(y))

        def save(self, dirpath, id):
            # Generate save folder:
            id_folderpath = os.path.join(dirpath, id)
            if not os.path.exists(id_folderpath):
                os.mkdir(id_folderpath)

            # Save model components:
            model_filepath = os.path.join(id_folderpath, "model.pth")
            torch.save(self.model, model_filepath)
            likelihood_filepath = os.path.join(id_folderpath, "likelihood.pth")
            torch.save(self.likelihood, likelihood_filepath)
            optimiser_filepath = os.path.join(id_folderpath, "optimiser.pth")
            torch.save(self.optimizer, optimiser_filepath)

        def load(self, dirpath, id):
            id_folderpath = os.path.join(dirpath, id)
            self.model = torch.load(os.path.join(id_folderpath, "model.pth"), weights_only=False)
            self.likelihood = torch.load(os.path.join(id_folderpath, "likelihood.pth"), weights_only=False)
            self.optimizer = torch.load(os.path.join(id_folderpath, "optimiser.pth"), weights_only=False)


    def run_inference(model, likelihood, inputs, batch_size=512):
        # Set up dataloading:
        tensor_input = torch.tensor(inputs, dtype=torch.float64)
        inference_dataset = TensorDataset(tensor_input)
        inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

        # Shift to eval mode:
        model.eval()
        likelihood.eval()

        # Set up outputs:
        predictions_array = []
        variance_array = []
        with torch.no_grad():
            for batch_index, inference_batch in enumerate(inference_loader):
                inference_batch = inference_batch[0]
                predictions = likelihood(model(inference_batch))
                predictions_array.append(predictions.mean.detach().numpy())
                variance_array.append(predictions.variance.detach().numpy())
                if (batch_index + 1) % 100 == 0:
                    print(batch_index + 1)

        return np.concatenate(predictions_array), np.concatenate(variance_array)
    return (ModelManager,)


@app.cell(hide_code=True)
def _(np, torch):
    def emulate(manager, x):
        tensor_input = torch.tensor(x, dtype=torch.float32)
        if len(tensor_input.shape) == 1:
            tensor_input = torch.unsqueeze(tensor_input, 0)
        with torch.no_grad():
            prediction = manager.likelihood(manager.model(tensor_input))
            prediction_mean = prediction.mean.detach().numpy()
            prediction_variance = prediction.variance.detach().numpy()
        return prediction_mean, prediction_variance


    def grad_emulate(manager, x):
        # Increase precision:
        manager.model.double()
        manager.likelihood = manager.likelihood.to(torch.float64)

        # Convert to tensor:
        tensor_input = torch.tensor(x, dtype=torch.float64)
        tensor_input.requires_grad_(True)

        # Ensure we have a batch dimension:
        if len(tensor_input.shape) == 1:
            tensor_input = torch.unsqueeze(tensor_input, 0)
        print(tensor_input.dtype)

        # Run emulation:
        manager.model.eval()
        manager.likelihood.eval()
        prediction = manager.likelihood(manager.model(tensor_input))
        prediction_mean = prediction.mean
        phantom_set_point = prediction_mean.detach()
        local_cost = (phantom_set_point - prediction_mean) ** 2
        return local_cost, tensor_input


    def log_grad_emulate(manager, x):
        # Increase precision:
        manager.model = manager.model.to(torch.float64)
        manager.likelihood = manager.likelihood.to(torch.float64)

        # [print(p.dtype) for p in manager.model.parameters()]

        # Convert to tensor:
        tensor_input = torch.tensor(np.log(x), dtype=torch.float64)
        tensor_input.requires_grad_(True)

        # Ensure we have a batch dimension:
        if len(tensor_input.shape) == 1:
            tensor_input = torch.unsqueeze(tensor_input, 0)

        # Run emulation:
        manager.model.eval()
        manager.likelihood.eval()
        prediction = manager.likelihood(manager.model(torch.exp(tensor_input)))
        prediction_mean = prediction.mean
        phantom_set_point = prediction_mean.detach()
        local_cost = (phantom_set_point - prediction_mean) ** 2
        return local_cost, tensor_input


    def calculate_jacobian(y, x, create_graph=False):
        jac = []
        flat_y = y.reshape(-1)
        grad_y = torch.zeros_like(flat_y)
        grad_matrix = torch.eye(len(flat_y))
        for i in range(len(flat_y)):
            grad_x, = torch.autograd.grad(
                flat_y, x, grad_matrix[:, i], create_graph=create_graph, retain_graph=True
            )
            jac.append(grad_x.reshape(x.shape))
        return torch.stack(jac).reshape(y.shape + x.shape)


    def calculate_hessian(y, x, create_graph=False):
        return calculate_jacobian(
            calculate_jacobian(y, x, create_graph=True), x, create_graph=create_graph
        )
    return (emulate,)


@app.cell
def _():
    # # Load data & gaussian process emulators:
    # parameter_matrix, coherency_fractions, ann_indices, speeds, gridsearch_parameters = load_gridsearch_data(
    #     "model_experiments/2025-12-04-collisions_only"
    # )

    # # Get information necessary to transform GP outputs:
    # CF_DIST_MEAN = np.mean(np.mean(coherency_fractions, axis=1))
    # CF_DIST_STD = np.std(np.mean(coherency_fractions, axis=1))

    # ANNI_DIST_MEAN = np.mean(np.mean(ann_indices, axis=1))
    # ANNI_DIST_STD = np.std(np.mean(ann_indices, axis=1))

    # SPEED_DIST_MEAN = np.mean(speeds[:, 0])
    # SPEED_DIST_STD = np.std(speeds[:, 0])


    # # Instantiate then load emulators:
    # cf_model_manager = ModelManager(inducing_points, 0.003)
    # cf_model_manager.load("model_experiments/2025-12-04-collisions_only/gaussian_process_models", "coherency_fraction")
    # anni_model_manager = ModelManager(inducing_points, 0.003)
    # anni_model_manager.load("model_experiments/2025-12-04-collisions_only/gaussian_process_models", "ann_index")
    # speed_model_manager = ModelManager(inducing_points, 0.003)
    # speed_model_manager.load("model_experiments/2025-12-04-collisions_only/gaussian_process_models", "speed")
    return


@app.cell
def _(np):
    # Dummy inducing points:
    inducing_points = np.ones((10, 10))
    return (inducing_points,)


@app.cell
def _(ModelManager, inducing_points):
    model_manager = ModelManager(inducing_points, 0.003)
    model_manager.load("model_experiments/2026-01-26-matrix_collisions/gaussian_process_models", "op17")
    return (model_manager,)


@app.cell
def _(model_manager, torch):
    def localised_cost_function(parameter_input):
        row_input = torch.unsqueeze(parameter_input, 0)
        transformed_input = torch.exp(row_input)
        prediction = model_manager.likelihood(model_manager.model(transformed_input))
        prediction_mean = prediction.mean
        phantom_set_point = prediction_mean.detach()
        localised_cost = (phantom_set_point - prediction_mean) ** 2
        return localised_cost

    sample_input = torch.log(torch.ones((64, 11)) * 0.5)

    # Autograd calculation:
    autograd_hessian = torch.autograd.functional.hessian(localised_cost_function, sample_input[0, :])
    print(autograd_hessian)

    # # torch.func calculations:
    # get_hessian = torch.func.hessian(localised_cost_function)
    # batch_hessian = torch.func.vmap(get_hessian)

    # print(sample_input.shape)
    # single_hessian = get_hessian(sample_input[0, :])
    # # sample_hessians = batch_hessian(sample_input)
    return


@app.cell
def _(gridsearch_parameters, np):
    def convert_inputs_to_original_units(inputs):
        transformed_inputs = np.zeros_like(inputs)
        for i in range(10):
            p_min, p_max = list(gridsearch_parameters.values())[i]
            transformed_inputs[:, i] = (inputs[:, i] * (p_max - p_min)) + p_min
        return transformed_inputs
    return


@app.cell
def _():
    # # Set up sampled inputs across parameter space:
    # rng = np.random.default_rng(0)
    # sampled_inputs = rng.uniform(0, 1, size=(100, 10))
    return


@app.cell
def _(np):
    hessians_filepath = "model_experiments/2026-01-26-matrix_collisions/gaussian_process_models/op17/log_hessian_dataset.npy"
    hessians = np.load(hessians_filepath)
    return (hessians,)


@app.cell
def _(np):
    from scipy.stats import qmc

    inputs_filepath = "model_experiments/2026-01-26-matrix_collisions/gaussian_process_models/op17/hessian_inputs.npy"
    sampled_inputs = np.load(inputs_filepath)

    # sobol_sampler = qmc.Sobol(d=10, scramble=True, rng=0)
    # sampled_inputs = sobol_sampler.random_base2(m=15)  # 32768 samples across dimensions.
    # sampled_inputs = np.log(sampled_inputs)
    return (sampled_inputs,)


@app.cell
def _(hessians, sampled_inputs):
    print(len(hessians))
    print(len(sampled_inputs))
    return


@app.cell
def _(np, sampled_inputs):
    np.min(sampled_inputs)
    return


@app.cell
def _(cc, hessians, plt):
    plt.imshow(hessians[0], cmap=cc.m_CET_D1)
    return


@app.cell
def _(hessians, np, plt):
    plt.hist(np.log(np.abs(hessians.flatten())), bins=100);
    plt.show()
    return


@app.cell
def _(emulate, model_manager, plt, sampled_inputs):
    sampled_outputs, _ = emulate(model_manager, sampled_inputs)
    plt.hist(sampled_outputs, bins=50);
    plt.show()
    return (sampled_outputs,)


@app.cell
def _(cf_model_manager, np, torch):
    def localised_cost_function(input):
        prediction = cf_model_manager.likelihood(cf_model_manager.model(input))
        prediction_mean = prediction.mean
        phantom_set_point = prediction_mean.detach()
        localised_cost = (phantom_set_point - prediction_mean) ** 2
        return localised_cost

    def get_hessians(inputs):
        hessians = []
        for i in range(inputs.shape[0]):
            if (i + 1) % 100 == 0:
                print(i + 1)
            hessian = torch.autograd.functional.hessian(localised_cost_function, torch.from_numpy(inputs[[i], :]))
            hessian = np.squeeze(hessian.detach().numpy())
            hessians.append(hessian)
        return np.stack(hessians, axis=0)
    return


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
def _(get_eigenvectors, hessians):
    sampled_eigenvalues, sampled_eigenvectors = get_eigenvectors(hessians)
    return sampled_eigenvalues, sampled_eigenvectors


@app.cell
def _(sampled_eigenvalues):
    sampled_eigenvalues
    return


@app.cell
def _(np, sampled_eigenvalues):
    np.any(np.isnan(np.log(np.abs(sampled_eigenvalues[:, 1]))))
    return


@app.cell
def _(np, plt, sampled_eigenvalues):
    plt.ecdf(np.log(np.abs(sampled_eigenvalues[:, 1])))
    return


@app.cell
def _(np, plt, sampled_eigenvalues):
    def plot_eigval_ecdf():
        fig, ax = plt.subplots(layout="constrained", figsize=(7, 3.5))
        for i in range(10):
            ax.ecdf(np.log(np.abs(sampled_eigenvalues[:, i])), label=f"Eigvalue {i + 1}", alpha=0.75)
            # if i == 1:
            #     break

        ax.set_xlabel("ln(|λ|)")
        ax.set_ylabel("CDF")
        fig.legend(loc='outside center right')
        plt.show()

    plot_eigval_ecdf()
    return


@app.cell
def _():
    # sampled_eigenvalues[sampled_eigenvalues < 5e-16] = 0
    return


@app.cell
def _(np, plt, sampled_eigenvalues):
    plt.hist(np.log(sampled_eigenvalues[:, 0]), bins=50);
    plt.show()
    return


@app.cell
def _(np, plt, sampled_eigenvalues):
    plt.hist(np.log(sampled_eigenvalues[:, 4]), bins=50);
    plt.show()
    return


@app.cell
def _(np, sampled_eigenvalues, sampled_eigenvectors):
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

    proposal_median, flipped_vectors, _ = calculate_geometric_median(sampled_eigenvectors[:, 0, :], sampled_eigenvalues[:, 0])
    return (flipped_vectors,)


@app.cell
def _(
    PARAMETER_DIMENSION,
    flipped_vectors,
    gridsearch_parameters,
    np,
    plt,
    sampled_inputs,
    sampled_outputs,
):
    import colorstamps

    def format_axes(ax):
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()

    def plot_eigendirections(inputs, eigenvectors, parameter_i, parameter_j, ax=None, s=1, c=None):
        if ax is None:
            fig, ax = plt.subplots()

        # Get 2D colormap from parameters:
        if c is None:
            c, _ = colorstamps.apply_stamp(
                inputs[:, parameter_i], inputs[:, parameter_j],
                'flat',
                vmin_0=0, vmax_0=1,
                vmin_1=0, vmax_1=1,
            )

        # Plot components:
        ax.scatter(
            eigenvectors[:, parameter_i],
            eigenvectors[:, parameter_j],
            s=s, alpha=0.25, c=c
        )

        theta = np.linspace(0, np.pi * 2, 100)
        ax.plot(1.025*np.cos(theta), 1.025*np.sin(theta), c='k', ls="--",  lw=1, alpha=0.5)
        format_axes(ax)
        # plt.show()

    def plot_linear_combinations():
        fig, axs = plt.subplots(PARAMETER_DIMENSION, PARAMETER_DIMENSION, figsize=(10, 10), layout="constrained")
        for i in range(PARAMETER_DIMENSION):
            for j in range(PARAMETER_DIMENSION):
                if i == j:
                    if i == 10:
                        continue
                    axs[i, j].text(
                        0, 0, list(gridsearch_parameters.keys())[i],
                        fontsize=6, horizontalalignment="center",
                        rotation=45, rotation_mode="anchor"
                    )
                    format_axes(axs[i, j])
                    continue
                if i > j:
                    plot_eigendirections(sampled_inputs, flipped_vectors, i, j, ax=axs[i, j], s=0.1)
                if i < j:
                    plot_eigendirections(sampled_inputs, flipped_vectors, i, j, ax=axs[i, j], s=0.1, c=sampled_outputs)

        plt.show()
    return (plot_linear_combinations,)


@app.cell
def _(plot_linear_combinations):
    plot_linear_combinations()
    return


@app.cell
def _(np):
    def fisher_metric(matrix_a, matrix_b):
        eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(matrix_a) @ matrix_b)
        return np.sqrt(np.sum(np.log(eigenvalues) ** 2))
    return


@app.cell
def _(np, sampled_eigenvectors):
    from sklearn.decomposition import KernelPCA

    # Get similarity matrix:
    similarity_matrix = np.abs(sampled_eigenvectors[:, 0, :] @ sampled_eigenvectors[:, 0, :].T)
    kernel_pca = KernelPCA(n_components=3, kernel='precomputed')
    kernel_embeddings = kernel_pca.fit_transform(similarity_matrix)
    return kernel_embeddings, similarity_matrix


@app.cell
def _(np):
    # Adapted (lightly) from https://www.ivoverhoeven.nl/blog/sampling-maximally-diverse-subsets

    def wsp_space_filling_design(min_dist, seed_index, distance_matrix):
        # A point should never be able to choose itself, so set diagonals to nan:
        wsp_distance_matrix = np.copy(distance_matrix)
        np.fill_diagonal(wsp_distance_matrix, np.nan)

        # Add the seed point to the list of chosen points:
        chosen_points = [seed_index]

        # Initialise algorithm
        current_point = np.copy(seed_index)

        # Start the iterations:
        while True:
            # Find all points points within a circle of radius min_dist around the current point
            points_within_circle = (wsp_distance_matrix[current_point, :] < min_dist).squeeze()

            # Eliminate those points from ever being chosen
            wsp_distance_matrix[points_within_circle, :] = np.nan
            wsp_distance_matrix[:, points_within_circle] = np.nan

            # If no points are able to be chosen, stop
            if np.all(np.isnan(wsp_distance_matrix[current_point, :])):
                break

            # Find the nearest neighbour that is not within that circle &
            # choose it as the next point:
            nearest_outside_point = np.nanargmin(wsp_distance_matrix[current_point, :])
            chosen_points.append(nearest_outside_point)

            # Make sure the current point can no longer be chosen
            wsp_distance_matrix[current_point, :] = np.nan
            wsp_distance_matrix[:, current_point] = np.nan

            current_point = nearest_outside_point

        chosen_points = np.stack(chosen_points)

        return chosen_points
    return (wsp_space_filling_design,)


@app.cell
def _(similarity_matrix, wsp_space_filling_design):
    diverse_batches = []
    for _ in range(1):
        chosen_points = wsp_space_filling_design(0.2, _, 1 - similarity_matrix)
        diverse_batches.append(chosen_points)
    return (chosen_points,)


@app.cell
def _(chosen_points):
    print(len(chosen_points))
    return


@app.cell
def _(chosen_points, plt, similarity_matrix):
    plt.hist(similarity_matrix[:, chosen_points][chosen_points, :].flatten(), bins=100);
    plt.show()
    return


@app.cell
def _(cc, kernel_embeddings, plt, sampled_outputs):
    def plot_kernel_embeddings(index_i, index_j):
        fig, ax = plt.subplots()
        ax.scatter(kernel_embeddings[:, index_i], kernel_embeddings[:, index_j], s=1, c=sampled_outputs, cmap=cc.m_CET_R1)
        # ax.scatter(kernel_embeddings[::8][result.medoids, index_i], kernel_embeddings[::8][result.medoids, index_j], c='r')
        ax.set_aspect("equal")
        ax.set_xlabel("KPC1")
        ax.set_ylabel("KPC2")
        ax.set_title("Kernel PCA of Principal Eigendirections")
        plt.show()

    plot_kernel_embeddings(0, 1)
    return


@app.cell
def _(PARAMETER_DIMENSION, np, torch):
    def sammon_mapping(input_data, distance_matrix, n_components=2, batch_size=512, steps=750, reset_count=None):
        # Get relevant dimensions:
        sample_count = input_data.shape[0]
        learning_rate = 0.03
        assert sample_count == distance_matrix.shape[0]

        # Convert to torch:
        input_tensor = torch.from_numpy(input_data)

        # Get distance matrix:
        target_distance_matrix = torch.from_numpy(distance_matrix)

        # Set up initial linear transformation:
        rng = np.random.default_rng(0)
        linear_transform = rng.uniform(-1, 1, (PARAMETER_DIMENSION, n_components))
        linear_transform = torch.from_numpy(linear_transform)
        linear_transform.requires_grad_()
        opt = torch.optim.Adam([linear_transform], lr=learning_rate)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

        # Get indices so we only calculate over upper triangle of matrices (we divide by 0 with the diagonal otherwise):
        upper_indices = torch.triu_indices(batch_size, batch_size, offset=1)
        axis_indices = torch.triu_indices(n_components, n_components, offset=1)

        print("Performing gradient descent for transform...")
        error_history = []
        for step_index in range(steps):
            # Randomly sample indices:
            # db_index = step_index % len(diverse_batches)
            # batch_indices = diverse_batches[rng.integers(len(diverse_batches))]
            batch_indices = rng.permutation(np.arange(sample_count))[:batch_size]

            if reset_count is not None:
                if (step_index + 1) % reset_count == 0:
                    print("Resetting optimiser...")
                    opt = torch.optim.Adam([linear_transform], lr=learning_rate)

            # Do linear transformation:
            transformed_data = input_tensor[batch_indices, :] @ linear_transform
            transform_distance_matrix = torch.functional.cdist(transformed_data, transformed_data)

            # Get Sammon's error:
            batch_target_distance_matrix = target_distance_matrix[batch_indices, :][:, batch_indices]
            triu_eig = batch_target_distance_matrix[upper_indices[0], upper_indices[1]]
            triu_transform = transform_distance_matrix[upper_indices[0], upper_indices[1]]

            # Mask out disconnected areas:
            disconnection_mask = torch.isinf(triu_eig)
            clean_eig = triu_eig[~disconnection_mask]
            clean_transform = triu_transform[~disconnection_mask]

            # sammon_error = torch.sum(((triu_eig - triu_transform) ** 2) / triu_eig) / torch.sum(triu_eig)
            sammon_error = torch.mean((clean_transform * torch.log(clean_transform / clean_eig)) - clean_transform + clean_eig)

            # Get similarity between transformation vectors:
            normalised_transform = \
                linear_transform / torch.linalg.norm(linear_transform, dim=0, keepdims=True)
            similiarities = (normalised_transform.T @ normalised_transform)[axis_indices[0], axis_indices[1]]
            axis_similarity = torch.linalg.norm(torch.abs(similiarities))

            # Run backprop and take update step:
            opt.zero_grad()
            (sammon_error).backward(inputs=linear_transform)
            opt.step()

            # Print progress:
            error_history.append(sammon_error.item())
            if (step_index + 1) % 100 == 0:
                print(step_index + 1, sammon_error.item())

        return error_history, linear_transform.detach().numpy()
    return (sammon_mapping,)


@app.cell
def _(np, plt, similarity_matrix):
    distance_test_matrix = np.copy(1 - similarity_matrix)
    np.fill_diagonal(distance_test_matrix, 100)
    plt.hist(np.min(distance_test_matrix, axis=1), bins=100);
    plt.show()
    print(np.max(np.min(distance_test_matrix, axis=1)))
    return


@app.cell
def _():
    from sklearn.neighbors import NearestNeighbors

    cosine_neighbours = NearestNeighbors(n_neighbors=5, metric="precomputed")
    return (cosine_neighbours,)


@app.cell
def _(np, similarity_matrix):
    nc_distance_matrix = 1 - similarity_matrix
    nc_distance_matrix = np.clip(nc_distance_matrix, 0, None)
    return (nc_distance_matrix,)


@app.cell
def _():
    # dijkstra_indices = wsp_space_filling_design(0.005, 0, nc_distance_matrix)
    # print(len(dijkstra_indices))
    return


@app.cell
def _(cosine_neighbours, nc_distance_matrix):
    cosine_neighbours.fit(nc_distance_matrix);
    return


@app.cell
def _(cosine_neighbours, nc_distance_matrix):
    kneighbours_graph = cosine_neighbours.kneighbors_graph(nc_distance_matrix, mode='distance')
    kneighbours_graph = kneighbours_graph.toarray()
    return (kneighbours_graph,)


@app.cell
def _():
    # epsilon = 0.05
    # e_neighbour_matrix = np.copy(nc_distance_matrix)
    # e_neighbour_matrix[e_neighbour_matrix > epsilon] = 0
    # np.fill_diagonal(e_neighbour_matrix, 0)
    return


@app.cell
def _(kneighbours_graph, np):
    symmetric_neighbours = np.stack([kneighbours_graph, kneighbours_graph.T], axis=0)
    symmetric_neighbours = np.max(symmetric_neighbours, axis=0)
    return (symmetric_neighbours,)


@app.cell
def _(symmetric_neighbours):
    import scipy.sparse

    csr_sn = scipy.sparse.csr_matrix(symmetric_neighbours)
    return (csr_sn,)


@app.cell
def _(csr_sn):
    from scipy.sparse.csgraph import dijkstra

    dijkstra_distance_matrix = dijkstra(csr_sn, directed=False)
    return (dijkstra_distance_matrix,)


@app.cell
def _(dijkstra_distance_matrix):
    dijkstra_distance_matrix[100, 50]
    return


@app.cell
def _(dijkstra_distance_matrix):
    dijkstra_distance_matrix[50, 100]
    return


@app.cell
def _(nc_distance_matrix):
    nc_distance_matrix[0, 0]
    return


@app.cell
def _(dijkstra_distance_matrix, np):
    print(np.count_nonzero(np.isinf(dijkstra_distance_matrix)))
    print(np.count_nonzero(~np.isinf(dijkstra_distance_matrix)))
    return


@app.cell
def _(dijkstra_distance_matrix, np):
    # Get off-diagonals:
    dm_size = dijkstra_distance_matrix.shape[0]
    od_indices = np.triu_indices(dm_size, k=1)
    off_diagonals = dijkstra_distance_matrix[od_indices]
    off_diagonals = off_diagonals[off_diagonals != np.inf]
    return od_indices, off_diagonals


@app.cell
def _(nc_distance_matrix, od_indices, off_diagonals, plt):
    plt.hist(off_diagonals, bins=100);
    plt.hist(nc_distance_matrix[od_indices], bins=100, alpha=0.5);
    plt.show()
    return


@app.cell
def _(dijkstra_distance_matrix, sammon_mapping, sampled_inputs):
    error_history, linear_transform = sammon_mapping(
        sampled_inputs, dijkstra_distance_matrix,
        n_components=2, batch_size=512, steps=500, reset_count=None
    )
    return error_history, linear_transform


@app.cell
def _(error_history, plt):
    plt.plot(error_history);
    plt.show()
    return


@app.cell
def _(linear_transform, np):
    transform_principal_indices = np.argsort(np.linalg.norm(linear_transform.T, axis=1, keepdims=False))[::-1]
    sorted_transform = linear_transform[:, transform_principal_indices]
    return


@app.cell
def _(linear_transform, np):
    print(np.linalg.norm(linear_transform.T, axis=1, keepdims=False))
    return


@app.cell
def _(copy, linear_transform, np):
    unit_transform = linear_transform.T / np.linalg.norm(linear_transform.T, axis=1, keepdims=True)
    simplified_linear_transform = copy.deepcopy(unit_transform)
    # simplified_linear_transform[np.abs(simplified_linear_transform) < 0.1] = 0
    simplified_linear_transform /= np.max(np.abs(simplified_linear_transform), axis=1, keepdims=True)
    print(np.round(simplified_linear_transform, 2))
    return


@app.cell
def _(PARAMETER_DIMENSION, emulate, model_manager, np):
    # Set up sampled inputs across parameter space:
    rng = np.random.default_rng(0)
    sammon_inputs = rng.uniform(0.02, 0.98, size=(500000, PARAMETER_DIMENSION))
    sammon_outputs, _ = emulate(model_manager, sammon_inputs)
    return sammon_inputs, sammon_outputs


@app.cell
def _(linear_transform, np, sammon_inputs, sammon_outputs):
    import sklearn.decomposition
    import sklearn.cross_decomposition

    def get_axis_aligned_transform(n_dim=2):
        # Transform data and linearise:
        # normalised_transform = linear_transform / np.linalg.norm(linear_transform, axis=0, keepdims=True)
        normalised_transform = linear_transform
        sammon_transformed_data = sammon_inputs @ normalised_transform

        # Perform PCA:
        pca_object = sklearn.decomposition.PCA(n_components=n_dim)
        pca_object.fit(sammon_transformed_data)
        components = pca_object.components_

        axis_aligned_transform = components @ normalised_transform.T
        axis_aligned_transform /= np.linalg.norm(axis_aligned_transform, axis=1, keepdims=True)
        print(components)

        # Standardise direction:
        aa_transformed_data = sammon_inputs @ axis_aligned_transform.T
        for i in range(n_dim):
            axis_aligned_transform[i, :] *= np.sign(np.corrcoef(aa_transformed_data[:, i], sammon_outputs, rowvar=False)[0, 1])

        # Standardise order:
        axis_aligned_transform = axis_aligned_transform[np.argsort(pca_object.explained_variance_)[::-1], :]

        return axis_aligned_transform
    return get_axis_aligned_transform, sklearn


@app.cell
def _(get_axis_aligned_transform):
    axis_aligned_transform = get_axis_aligned_transform(n_dim=2)
    return (axis_aligned_transform,)


@app.cell
def _(axis_aligned_transform, np):
    np.round(axis_aligned_transform, 2)
    return


@app.cell
def _(cc, linear_transform, np, plt, sammon_inputs, sammon_outputs):
    def plot_sammon_transformation():
        # sammon_transformed_data = sammon_inputs @ axis_aligned_transform.T
        sammon_transformed_data = sammon_inputs @ linear_transform
        color = sammon_outputs

        # Plot scatter:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(
            sammon_transformed_data[:, 0], sammon_transformed_data[:, 1],
            s=1, c=sammon_outputs, alpha=0.1, cmap=cc.m_CET_R1,
            vmin=np.quantile(sammon_outputs, 0.02),
            vmax=np.quantile(sammon_outputs, 0.98)
        )
        plt.show()

    plot_sammon_transformation()
    return


@app.cell
def _(
    KNeighborsRegressor,
    axis_aligned_transform,
    cc,
    np,
    plt,
    sammon_inputs,
    sammon_outputs,
):
    def knn_phase_plot(i_component, j_component, mesh_points=100):
        # Get predictions:
        sammon_transformed_data = sammon_inputs @ axis_aligned_transform.T
        knn_regressor = KNeighborsRegressor(n_neighbors=10, weights='distance')
        knn_regressor.fit(sammon_transformed_data[:, [i_component, j_component]], sammon_outputs[:]);

        # Get points to predict:
        low_x, upper_x = np.quantile(sammon_transformed_data[:, i_component], [0.01, 0.99])
        low_y, upper_y = np.quantile(sammon_transformed_data[:, j_component], [0.01, 0.99])
        x = np.linspace(low_x, upper_x, mesh_points)
        y = np.linspace(low_y, upper_y, mesh_points)
        xv, yv = np.meshgrid(x, y)
        query_points = np.stack([xv.flatten(), yv.flatten()], axis=1)
        knn_predictions = knn_regressor.predict(query_points)
        knn_mesh = knn_predictions.reshape((mesh_points, mesh_points))

        # Plot:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(
            knn_mesh, origin='lower',
            cmap=cc.m_CET_R1,
            vmin=np.quantile(sammon_outputs, 0.01),
            vmax=np.quantile(sammon_outputs, 0.99),
            extent=[low_x, upper_x, low_y, upper_y]
        )
        ax.set_xticks([])
        ax.set_xlabel("$\mathregular{\pi_1}$", fontsize=12)
        ax.set_yticks([])
        ax.set_ylabel("$\mathregular{\pi_2}$", fontsize=12)
        plt.show()

    knn_phase_plot(0, 1)
    return


@app.cell
def _(axis_aligned_transform, np, sammon_inputs):
    def sammon_phase_plot(inputs, outputs, dims, mesh_count=50):
        # Get transformed data:
        sammon_transformed_data = sammon_inputs @ axis_aligned_transform.T

        # Get relevant quantiles:
        dim_quantiles = []
        bin_edges = np.linspace(0, 1, mesh_count + 1)
        for dim_index in range(dims):
            dim_quantiles.append(np.quantile(sammon_transformed_data[:, dim_index], bin_edges))

        # Get bin masks:
        dim_masks = []
        for dim_index in range(dims):
            mask_list = []
            for bin_index in range(mesh_count):
                # Get mask for the row in this dimension:
                bin_mask = np.logical_and(
                    sammon_transformed_data[:, dim_index] >= dim_quantiles[dim_index][bin_index],
                    sammon_transformed_data[:, dim_index] <  dim_quantiles[dim_index][bin_index + 1]
                )
                mask_list.append(bin_mask)
            dim_masks.append(mask_list)

        # Generate necessary indices:
        multidim_indices = np.indices([mesh_count]*dims)
        flat_indices = []
        for dim_index in range(dims):
            flat_indices.append(multidim_indices[dim_index, :].flatten())
        flat_indices = np.stack(flat_indices, axis=1)

        # Get bins:
        phase_array = np.zeros([mesh_count]*dims)
        bin_regression = np.zeros_like(outputs)
        for index in flat_indices:
            # Collate masks:
            cell_mask_array = []
            for dim_index in range(dims):
                cell_mask_array.append(dim_masks[dim_index][index[dim_index]])
            cell_mask = np.all(np.stack(cell_mask_array, axis=1), axis=1)
            cell_mean = np.mean(outputs[cell_mask])
            phase_array[*index] = cell_mean
            bin_regression[cell_mask] = cell_mean

        return phase_array, bin_regression
    return


@app.cell
def _():
    # phase_array, bin_regression = sammon_phase_plot(sammon_inputs, sammon_outputs, 2)
    return


@app.cell
def _():
    # plt.imshow(
    #     phase_array[:, :], cmap=cc.m_CET_R1,
    #     vmin=np.quantile(sammon_outputs, 0.02), vmax=np.quantile(sammon_outputs, 0.98),
    #     origin='lower'
    # )
    return


@app.cell
def _(linear_transform, plt, sammon_inputs, sammon_outputs):
    from sklearn.neighbors import KNeighborsRegressor

    def plot_knn_reg():
        # Get predictions:
        # sammon_transformed_data = sammon_inputs @ axis_aligned_transform.T
        sammon_transformed_data = sammon_inputs @ linear_transform
        knn_regressor = KNeighborsRegressor(n_neighbors=10, weights='distance')
        knn_regressor.fit(sammon_transformed_data[::2, :], sammon_outputs[::2]);
        knn_predictions = knn_regressor.predict(sammon_transformed_data[1::2, :])

        # Plot:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(sammon_outputs[1::2], knn_predictions, s=0.1, alpha=0.5)
        ax.set_aspect("equal")
        # ax.plot([-1, 4], [-1, 4], c='r')
        # ax.set_xlim(-1, 4)
        # ax.set_ylim(-1, 4)
        plt.show()

    plot_knn_reg()
    return KNeighborsRegressor, plot_knn_reg


@app.cell
def _(plot_knn_reg):
    plot_knn_reg()
    return


@app.cell
def _(
    KNeighborsRegressor,
    axis_aligned_transform,
    coherency_fractions,
    np,
    parameter_matrix,
    plt,
):
    def plot_knn_reg_sim(split_ratio=0.8):
        # Do test/train split:
        rng = np.random.default_rng(0)
        permuted_indices = rng.permutation(np.arange(parameter_matrix.shape[0]))
        split_index = int(parameter_matrix.shape[0] * split_ratio)
        train_indices = permuted_indices[:split_index]
        test_indices = permuted_indices[split_index:]

        # Transform and regress:
        sim_transformed_data = parameter_matrix @ axis_aligned_transform.T
        knn_regressor = KNeighborsRegressor(n_neighbors=10, weights='distance')
        mean_cf = np.mean(coherency_fractions, axis=1)
        knn_regressor.fit(sim_transformed_data[train_indices, :], mean_cf[train_indices]);
        knn_predictions = knn_regressor.predict(sim_transformed_data[test_indices, :])

        # Plot:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(mean_cf[test_indices], knn_predictions, s=1, alpha=0.5)
        ax.set_aspect("equal")
        ax.plot([-0.005, 0.15], [-0.005, 0.15], c='r')
        ax.set_xlim(-0.005, 0.15)
        ax.set_ylim(-0.005, 0.15)
        plt.show()

    plot_knn_reg_sim()
    return


@app.cell
def _(bin_regression, np, plt, sammon_outputs):
    def binreg_histogram2d():
        fig, ax = plt.subplots()
        binned_array, _, _ = np.histogram2d(sammon_outputs, bin_regression, bins=100, range=[[-1, 4], [-1, 4]])
        ax.imshow(np.log(binned_array).T, origin="lower", extent=(-1, 4, -1, 4))
        ax.plot([-1, 4], [-1, 4], c='r', alpha=0.5)
        ax.plot()
        plt.show()

    binreg_histogram2d()
    return


@app.cell
def _(PARAMETER_DIMENSION, np, torch):
    def log_sammon_mapping(input_data, distance_matrix, n_components, batch_size=512, seed=0):
        # Get relevant dimensions:
        sample_count = input_data.shape[0]

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
def _(dijkstra_distance_matrix, log_sammon_mapping, sampled_inputs):
    loss_history, log_transform = log_sammon_mapping(sampled_inputs, dijkstra_distance_matrix, 2, 32, 14)
    return log_transform, loss_history


@app.cell
def _(log_transform, np):
    def print_log_components():
        norm_transform = log_transform / np.linalg.norm(log_transform, axis=0, keepdims=True)
        print(np.round(norm_transform, 2).T)

    print_log_components()
    return


@app.cell
def _(loss_history, np, plt):
    plt.plot(loss_history, lw=0.1)
    plt.plot(np.convolve(loss_history, np.ones(100), mode='valid') / 100)
    return


@app.cell
def _(cc, log_transform, np, plt, sammon_inputs, sammon_outputs):
    def plot_log_transform(downsample):
        # Transform input data:
        sammon_transformed_data = np.log(sammon_inputs) @ log_transform
        sammon_transformed_data = sammon_transformed_data[::downsample, :]
        sammon_dims = sammon_transformed_data.shape[1]

        fig, axs = plt.subplots(sammon_dims, sammon_dims)

        for i in range(sammon_dims):
            for j in range(sammon_dims):
                if j > i:
                    axs[i, j].scatter(
                        sammon_transformed_data[:, i],
                        sammon_transformed_data[:, j],
                        s=1,
                        c=sammon_outputs[::downsample], cmap=cc.m_CET_R1,
                        vmin=np.quantile(sammon_outputs, 0.02),
                        vmax=np.quantile(sammon_outputs, 0.98),
                        alpha=0.1
                    )

        plt.show()

    plot_log_transform(10)
    return


@app.cell
def _(log_transform, np):
    # Adapted for readability from https://en.wikipedia.org/wiki/Talk:Varimax_rotation:

    def get_varimax(input_factors, gamma=1.0, iterations=1000, tol=1e-10):
        # Get input information:
        normalised_factors = input_factors / np.linalg.norm(input_factors, axis=1, keepdims=True)
        num_dims, num_factors = input_factors.shape
        rotation = np.eye(num_factors)

        # Iterate to maximise the sum of the variances of the squared loadings:
        score = 0
        for step_index in range(iterations):
            # Get factors rotated by proposal rotation:
            rotated_factors = input_factors @ rotation

            # Simplicity of factors defined as variance of squared loadings:
            sum_of_squared_loadings = np.sum(rotated_factors**2, axis=0)
            c1 = np.diag(sum_of_squared_loadings) / num_dims
            c3 = rotated_factors**3 - np.dot(rotated_factors, c1)
            B = np.dot(input_factors.T, c3)
            U, S, V= np.linalg.svd(B, full_matrices=True)
            rotation = np.dot(U, V)
            new_score = np.sum(S)

            # Break if converged:
            if (new_score - score) < tol:
                break

            score = new_score
            print(score)

        return rotation

    qr_transform, r = np.linalg.qr(log_transform)
    qr_rotation = get_varimax(qr_transform)
    return qr_rotation, qr_transform


@app.cell
def _(np, qr_rotation, qr_transform):
    varimax_transform = np.round(qr_transform @ qr_rotation, 2)
    print(np.round(varimax_transform.T, 2))
    return


@app.cell
def _(log_transform, np):
    normalised_log_transform = log_transform / np.linalg.norm(log_transform, axis=0, keepdims=True)
    return (normalised_log_transform,)


@app.cell
def _(normalised_log_transform, np):
    print(np.round(normalised_log_transform[:, 2], 2))
    return


@app.cell
def _(
    KNeighborsRegressor,
    log_transform,
    np,
    plt,
    sammon_inputs,
    sammon_outputs,
):
    def plot_log_knn_reg():
        # Get predictions:
        # normalised_transform, r = np.linalg.qr(log_transform)
        # sammon_log_transformed_data = np.log(sammon_inputs) @ (log_transform @ rotation)
        # sammon_log_transformed_data = np.log(sammon_inputs) @ normalised_transform
        # sammon_log_transformed_data = np.log(sammon_inputs) @ varimax_transform
        sammon_log_transformed_data = np.log(sammon_inputs) @ log_transform

        log_knn_regressor = KNeighborsRegressor(n_neighbors=3, weights='distance')
        log_knn_regressor.fit(sammon_log_transformed_data[::2, :], sammon_outputs[::2]);
        log_knn_predictions = log_knn_regressor.predict(sammon_log_transformed_data[1::2, :])

        # Plot:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(sammon_outputs[1::2], log_knn_predictions, s=0.1, alpha=0.5)
        ax.set_aspect("equal")

        lower_bound = np.min([np.min(sammon_outputs[1::2]), np.min(log_knn_predictions)])
        upper_bound = np.max([np.max(sammon_outputs[1::2]), np.max(log_knn_predictions)])
        ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], c='r')
        ax.set_xlim(lower_bound, upper_bound)
        ax.set_ylim(lower_bound, upper_bound)
        plt.show()


    plot_log_knn_reg()
    return


@app.cell
def _(
    KNeighborsRegressor,
    log_transform,
    np,
    plt,
    sammon_inputs,
    sammon_outputs,
):
    def log_sammon_histogram2d():
        # Get transformed data:
        sammon_log_transformed_data = np.log(sammon_inputs) @ log_transform

        # Get k-NN regressor:
        log_knn_regressor = KNeighborsRegressor(n_neighbors=3, weights='distance')
        log_knn_regressor.fit(sammon_log_transformed_data[::2, :], sammon_outputs[::2]);
        log_knn_predictions = log_knn_regressor.predict(sammon_log_transformed_data[1::2, :])

        # Plot 2D histogram:
        fig, ax = plt.subplots(figsize=(5, 5))
        lower_bound = np.min([np.min(sammon_outputs[1::2]), np.min(log_knn_predictions)])
        upper_bound = np.max([np.max(sammon_outputs[1::2]), np.max(log_knn_predictions)])
        count_array, _, _ = np.histogram2d(
            sammon_outputs[1::2],
            log_knn_predictions,
            bins=75,
            range=[[lower_bound, upper_bound], [lower_bound, upper_bound]]
        )
        ax.imshow(
            np.log(count_array), origin="lower",
            extent=(lower_bound, upper_bound, lower_bound, upper_bound)
        )
        ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], c='r')
        plt.show()

    log_sammon_histogram2d()
    return


@app.cell
def _(log_transform, torch):
    def random_invert(point, max_iter=10000):
        # Get system information:
        torch_point = torch.from_numpy(point)
        torch_transform = torch.from_numpy(log_transform)

        # Random suggestion:
        input_parameters = torch.rand(10)
        input_parameters.requires_grad_()
        opt = torch.optim.SGD([input_parameters], lr=0.03)

        # Iterate:
        converged = False
        iteration_count = 0
        while not converged:
            # Transform parameters:
            proposal_point = input_parameters @ torch_transform
            loss = torch.linalg.norm(torch_point - proposal_point) + torch.linalg.norm(torch_transform)
            print(loss.item())
            if loss.item() < 0.001:
                converged = True

            # Update proposal:
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Log iterations:
            iteration_count += 1
            if iteration_count >= max_iter:
                print("Convergence failed...")
                converged = True

        return input_parameters.detach().numpy()
    return (random_invert,)


@app.cell
def _(np, random_invert):
    np.exp(random_invert(np.array([-4, -2])))
    return


@app.cell
def _():
    # from sklearn.manifold import MDS

    # mds_manager = MDS(n_components=2, metric="precomputed")
    # mds_embeddings = mds_manager.fit_transform(1- similarity_matrix)
    return


@app.cell
def _():
    # def plot_mds_embeddings():
    #     fig, ax = plt.subplots()
    #     ax.scatter(mds_embeddings[:, 0], mds_embeddings[:, 1], s=1, c=sampled_outputs, cmap=cc.m_CET_R1)
    #     ax.set_aspect("equal")
    #     plt.show()

    # plot_mds_embeddings()
    return


@app.cell
def _(flipped_vectors):
    from sklearn.decomposition import PCA

    pca = PCA()
    pca_embeddings = pca.fit_transform(flipped_vectors)
    return pca, pca_embeddings


@app.cell
def _(pca):
    pca.components_[0, :]
    return


@app.cell
def _(pca, plt):
    plt.plot(pca.explained_variance_ratio_)
    return


@app.cell
def _(cc, np, pca_embeddings, plt, result, sampled_outputs):
    def plot_pca(index_i, index_j):
        fig, ax = plt.subplots()
        lb, ub = np.quantile(sampled_outputs, [0.05, 0.95])
        ax.scatter(
            pca_embeddings[:, index_i], pca_embeddings[:, index_j],
            s=0.1, alpha=0.75, c=sampled_outputs, cmap=cc.m_CET_R1, vmin=lb, vmax=ub
        )

        ax.scatter(pca_embeddings[::8][result.medoids, 0], pca_embeddings[::8][result.medoids, 1], c='r')
        # ax.scatter(pca.components_[0, :], pca.components_[1, :], c='r')

        xlow = np.min(pca_embeddings[:, index_i]) - 0.05
        xhigh = np.max(pca_embeddings[:, index_i]) + 0.05
        ax.set_xlim(xlow, xhigh)

        ylow = np.min(pca_embeddings[:, index_j]) - 0.05
        yhigh = np.max(pca_embeddings[:, index_j]) + 0.05
        ax.set_ylim(ylow, yhigh)

        ax.hlines(0, xlow, xhigh, color='k', alpha=0.25)
        ax.vlines(0, ylow, yhigh, color='k', alpha=0.25)
        ax.set_aspect("equal")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.show()

    plot_pca(0, 1)
    return


@app.cell
def _(sklearn):
    sklearn.neighbors.KernelDensity
    return


@app.cell
def _():
    # import kmedoids

    # def get_vector_distance_matrix(eigenvectors):
    #     return 1 - np.abs(eigenvectors @ eigenvectors.T)

    # vector_distance_matrix = get_vector_distance_matrix(sampled_eigenvectors[:, 0, :])
    # result = kmedoids.fasterpam(vector_distance_matrix[::8, ::8], 3)
    return


@app.cell
def _(np, result, sampled_eigenvectors, sampled_inputs):
    selected_eigenparameters = sampled_eigenvectors[::8][result.medoids, 0, :]
    cluster_distances = 1 - np.abs(np.sum(sampled_eigenvectors[:, [0], :] * selected_eigenparameters, axis=2))

    ep_values = np.sum(sampled_inputs[:, :, np.newaxis] * selected_eigenparameters.T, axis=1)
    ep_values.shape
    return ep_values, selected_eigenparameters


@app.cell
def _(np, selected_eigenparameters):
    import copy

    for i in range(3):
        simplified_parameter = copy.deepcopy(selected_eigenparameters[i, :])
        simplified_parameter[np.abs(simplified_parameter) < 0.1] = 0
        simplified_parameter *= 1  / np.max(np.abs(simplified_parameter))
        simplified_parameter = np.round(simplified_parameter, 2)
        print(simplified_parameter)
    return (copy,)


@app.cell
def _(ep_values, plt):
    def plot_ep_values():
        fig, ax = plt.subplots()
        ax.scatter(ep_values[:, 0], ep_values[:, 1], s=1)
        ax.set_aspect("equal")
        plt.show()

    plot_ep_values()
    return


@app.cell
def _(np, plt, sampled_inputs, sampled_outputs, selected_eigenparameters):
    def plot_eigenparameter(eig_index):
        # Convert relevant inputs into eigenparameter space:
        ep_values = -np.sum(sampled_inputs * selected_eigenparameters[eig_index, :], axis=1)

        # Plot binscatter:
        fig, ax = plt.subplots()
        ax.scatter(ep_values, sampled_outputs, s=1, alpha=1)

        ventiles = np.linspace(0, 1, 21)
        bin_x = []
        bin_y = []
        for index in range(20):
            lq = ventiles[index]
            uq = ventiles[index + 1]
            lb = np.quantile(ep_values, lq)
            ub = np.quantile(ep_values, uq)
            bin_x.append((lb + ub) / 2)

            param_mask = np.logical_and(ep_values > lb, ep_values < ub)
            bin_y.append(np.mean(sampled_outputs[param_mask]))

        # ax.scatter(bin_x, bin_y, s=1)
        # ax.plot(bin_x, bin_y)
        # ax.set_xlabel(f"Cluster {cluster_index} Eigenparameter")
        # ax.set_ylabel("ANNI")
        plt.show()
    return (plot_eigenparameter,)


@app.cell
def _(plot_eigenparameter):
    plot_eigenparameter(0)
    return


@app.cell
def _(hessians, np):
    np.all(np.isclose(hessians[0], hessians[0].T))
    return


if __name__ == "__main__":
    app.run()
