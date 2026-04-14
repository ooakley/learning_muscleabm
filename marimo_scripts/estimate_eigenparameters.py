import marimo

__generated_with = "0.19.7"
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
    from dppy.finite_dpps import FiniteDPP

    import colorcet as cc
    return (
        DataLoader,
        FiniteDPP,
        TensorDataset,
        cc,
        gpytorch,
        json,
        np,
        os,
        plt,
        torch,
    )


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
                inducing_points, dtype=torch.float32
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
            x_tensor = torch.tensor(x, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
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
        tensor_input = torch.tensor(inputs, dtype=torch.float32)
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
    return ModelManager, load_gridsearch_data


@app.cell
def _(FiniteDPP, ModelManager, load_gridsearch_data, np):
    # Load data & gaussian process emulators:
    parameter_matrix, coherency_fractions, ann_indices, speeds, gridsearch_parameters = load_gridsearch_data(
        "model_experiments/2025-12-04-collisions_only"
    )

    # Get information necessary to transform GP outputs:
    CF_DIST_MEAN = np.mean(np.mean(coherency_fractions, axis=1))
    CF_DIST_STD = np.std(np.mean(coherency_fractions, axis=1))

    ANNI_DIST_MEAN = np.mean(np.mean(ann_indices, axis=1))
    ANNI_DIST_STD = np.std(np.mean(ann_indices, axis=1))

    SPEED_DIST_MEAN = np.mean(speeds[:, 0])
    SPEED_DIST_STD = np.std(speeds[:, 0])

    # Get likelihood matrix:
    print("Getting squared exponential likelihood matrix...", flush=True)
    distance_matrix = 1 - np.matmul(parameter_matrix[::7, :], parameter_matrix[::7, :].T).astype(np.float32)
    likelihood_matrix = np.exp(distance_matrix ** 2)

    # Set up determinantal point process:
    print("Setting up point process...")
    DPP = FiniteDPP('likelihood', **{'L': likelihood_matrix})

    k = 256
    DPP.sample_mcmc_k_dpp(size=k, random_state=None)

    # Get inducing points:
    inducing_indices = DPP.list_of_samples[0][-1]
    inducing_points = parameter_matrix[::7, :][inducing_indices, :]

    # Instantiate then load emulators:
    cf_model_manager = ModelManager(inducing_points, 0.003)
    cf_model_manager.load("model_experiments/2025-12-04-collisions_only/gaussian_process_models", "coherency_fraction")
    anni_model_manager = ModelManager(inducing_points, 0.003)
    anni_model_manager.load("model_experiments/2025-12-04-collisions_only/gaussian_process_models", "ann_index")
    speed_model_manager = ModelManager(inducing_points, 0.003)
    speed_model_manager.load("model_experiments/2025-12-04-collisions_only/gaussian_process_models", "speed")
    return cf_model_manager, gridsearch_parameters


@app.cell
def _(cf_model_manager):
    print(f'Actual noise value: {cf_model_manager.likelihood.noise}')
    return


@app.cell
def _(cf_model_manager):
    cf_model_manager
    return


@app.cell
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
        # Convert to tensor:
        tensor_input = torch.tensor(x, dtype=torch.float32)
        tensor_input.requires_grad_(True)

        # Ensure we have a batch dimension:
        if len(tensor_input.shape) == 1:
            tensor_input = torch.unsqueeze(tensor_input, 0)

        # Run emulation:
        manager.model.eval()
        manager.likelihood.eval()
        prediction = manager.likelihood(manager.model(tensor_input))
        prediction_mean = prediction.mean
        phantom_set_point = prediction_mean.detach()
        local_cost = (phantom_set_point - prediction_mean) ** 2
        return local_cost, tensor_input


    def log_grad_emulate(manager, x):
        # Convert to tensor:
        tensor_input = torch.tensor(np.log(x), dtype=torch.float32)
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
    return (
        calculate_hessian,
        calculate_jacobian,
        emulate,
        grad_emulate,
        log_grad_emulate,
    )


@app.cell
def _(cf_model_manager, emulate, np, plt):
    def plot_oat(num_points, origin):
        for dimension_index in range(10):
            inputs = np.repeat(np.expand_dims(origin, axis=0), num_points, 0)
            inputs[:, dimension_index] = np.linspace(0, 1, num_points)
            estimated_mean, estimated_variance = emulate(cf_model_manager, inputs)
            plt.plot(np.linspace(0, 1, num_points), estimated_mean, alpha=0.8)
        plt.show()

    plot_oat(5000, np.ones(10) * 0.5)
    return (plot_oat,)


@app.cell
def _(np):
    # Set up sampled inputs across parameter space:
    rng = np.random.default_rng(0)
    sample_inputs = rng.uniform(0, 1, size=(10000, 10))
    # sample_inputs = (sample_inputs * 0.9) + 0.1
    return (sample_inputs,)


@app.cell
def _(cf_model_manager, emulate, plt, sample_inputs):
    sample_outputs, _ = emulate(cf_model_manager, sample_inputs)
    plt.hist(sample_outputs, bins=100);
    plt.show()
    return (sample_outputs,)


@app.cell
def _(calculate_jacobian, cf_model_manager, grad_emulate, np):
    def sample_jacobians(inputs):
        jacobians = []
        for i in range(inputs.shape[0]):    
            # Estimate Hessian in local area:
            prediction_mean, tensor_input = grad_emulate(cf_model_manager, inputs[i, :])
            estimated_jacobian = calculate_jacobian(prediction_mean, tensor_input)
            estimated_jacobian = np.squeeze(estimated_jacobian.detach().numpy())
            jacobians.append(estimated_jacobian)

        return np.stack(jacobians, axis=0)
    return


@app.cell
def _():
    # sampled_jacobians = sample_jacobians(sample_inputs)
    return


@app.cell
def _():
    # plt.hist(np.linalg.norm(sampled_jacobians, axis=1), bins=100);
    # plt.show()
    return


@app.cell
def _():
    # min_j_point = sample_inputs[np.argmin(np.linalg.norm(sampled_jacobians, axis=1)), :]
    # plot_oat(1000, min_j_point)
    return


@app.cell
def _(calculate_hessian, cf_model_manager, log_grad_emulate, np):
    def sample_hessians(inputs):
        hessians = []
        for i in range(inputs.shape[0]):    
            # Estimate Hessian in local area:
            prediction_mean, tensor_input = log_grad_emulate(cf_model_manager, inputs[i, :])
            estimated_hessian = calculate_hessian(prediction_mean, tensor_input)
            estimated_hessian = np.squeeze(estimated_hessian.detach().numpy())
            hessians.append(estimated_hessian)

        return np.stack(hessians, axis=0)
    return (sample_hessians,)


@app.cell
def _(np):
    def batch_eigen_calc(hessians):
        calculated_eigvals = []
        calculated_eigvectors = []
        for i in range(hessians.shape[0]):
            eigvals, eigvectors = np.linalg.eig(hessians[i, :, :])
            calculated_eigvals.append(eigvals)
            calculated_eigvectors.append(eigvectors.T)

        calculated_eigvals = np.stack(calculated_eigvals, axis=0)
        calculated_eigvectors = np.stack(calculated_eigvectors, axis=0)
        return np.real(calculated_eigvals), np.real(calculated_eigvectors)
    return (batch_eigen_calc,)


@app.cell
def _(plt, sample_inputs):
    import colorstamps

    def plot_eigendirections(eigenvectors, parameter_i, parameter_j, eigenvector_index=0):
        fig, ax = plt.subplots()

        # Get 2D colormap from parameters:
        rgb, _ = colorstamps.apply_stamp(
            sample_inputs[:, parameter_i], sample_inputs[:, parameter_j],
            'flat',
            vmin_0=0, vmax_0=1,
            vmin_1=0, vmax_1=1,
        )


        ax.scatter(
            eigenvectors[:, eigenvector_index, parameter_i],
            eigenvectors[:, eigenvector_index, parameter_j],
            s=1, alpha=0.25, c=rgb
        )
        # ax.scatter(
        #     proposal_median[parameter_i],
        #     proposal_median[parameter_j],
        #     c='r'
        # )
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        plt.show()
    return colorstamps, plot_eigendirections


@app.cell
def _(sample_hessians, sample_inputs):
    sampled_hessians = sample_hessians(sample_inputs)
    return (sampled_hessians,)


@app.cell
def _(batch_eigen_calc, np, sampled_hessians):
    sampled_eigvals, sampled_eigvectors = batch_eigen_calc(sampled_hessians.astype(np.double))
    return sampled_eigvals, sampled_eigvectors


@app.cell
def _(np, plt, sampled_hessians):
    # Weight by spectral norm?
    mean_hessian = np.mean(sampled_hessians, axis=0)
    mean_eigval, mean_eigvector = np.linalg.eig(mean_hessian)
    plt.imshow(np.abs(mean_eigvector.T))
    return mean_eigvector, mean_hessian


@app.cell
def _(mean_eigvector):
    mean_eigvector[:, 0]
    return


@app.cell
def _(mean_eigvector):
    mean_eigvector[:, 1]
    return


@app.cell
def _(np, plt, sampled_eigvals):
    def plot_distribution_of_spectra(eigenvalues):
        fig, ax = plt.subplots(figsize=(5, 3))
        for eigenvalue_index in range(10):
            ax.hist(np.log(np.abs(eigenvalues)[:, eigenvalue_index].flatten()), bins=25, histtype='step');
        plt.show()

    plot_distribution_of_spectra(sampled_eigvals)
    return (plot_distribution_of_spectra,)


@app.cell
def _(np, sampled_eigvals):
    np.abs(sampled_eigvals)[:, 0]
    return


@app.cell
def _(np, sampled_eigvals, sampled_eigvectors):
    def calculate_geometric_median(key_eigenvectors, key_eigenvalues):
        proposal_median = np.mean(key_eigenvectors, axis=0)    
        converged = False
        while not converged:
            # Get distances:
            cosine_similarities = key_eigenvectors @ np.expand_dims(proposal_median, axis=1)
            nematic_distances = 1 - np.abs(cosine_similarities)

            # Get weighted average of appropriately flipped vectors:
            flipped_vectors = np.sign(cosine_similarities) * key_eigenvectors
            combined_weights = np.abs(key_eigenvalues) / np.squeeze(nematic_distances)
            combined_weights = np.expand_dims(combined_weights, axis=1)
            new_proposal = np.sum(flipped_vectors * combined_weights, axis=0) / np.sum(combined_weights)
            new_proposal /= np.linalg.norm(new_proposal)
            epsilon = 1 - np.dot(proposal_median, new_proposal)
            print(epsilon)
            if epsilon < 1e-6:
                print("Converged!")
                print(f"Mean nematic distance: {np.mean(nematic_distances)}")
                converged = True
            proposal_median = new_proposal

        return proposal_median, flipped_vectors

    proposal_median, flipped_vectors = calculate_geometric_median(sampled_eigvectors[:, 0, :], sampled_eigvals[:, 0])
    return calculate_geometric_median, flipped_vectors, proposal_median


@app.cell
def _(np):
    def run_deflation(deflation_vector, input_hessians, input_eigenvalues):
        deflation_components = np.expand_dims(deflation_vector, axis=1) @ np.expand_dims(deflation_vector, axis=1).T
        deflated_hessians = []
        for i in range(input_hessians.shape[0]):
            # Get current eigenvalues:
            eigvals, eigvectors = np.linalg.eig(input_hessians[i, :, :])
            cosine_similarity = np.dot(deflation_vector, eigvectors[:, 0])
            deflation_matrix = deflation_components * cosine_similarity * eigvals[0]
            deflated_hessians.append(input_hessians[i, :, :] - deflation_matrix)

        return np.real(np.stack(deflated_hessians, axis=0))
    return (run_deflation,)


@app.cell
def _(proposal_median, run_deflation, sampled_eigvals, sampled_hessians):
    deflated_hessians = run_deflation(proposal_median, sampled_hessians, sampled_eigvals)
    return (deflated_hessians,)


@app.cell
def _(deflated_hessians, np):
    md_eigval, md_eigvector = np.linalg.eig(np.mean(deflated_hessians, axis=0))
    md_eigvector = md_eigvector.T
    return (md_eigvector,)


@app.cell
def _(md_eigvector):
    md_eigvector[0, :]
    return


@app.cell
def _(batch_eigen_calc, deflated_hessians):
    deflated_eigvals, deflated_eigvectors = batch_eigen_calc(deflated_hessians)
    return deflated_eigvals, deflated_eigvectors


@app.cell
def _(calculate_geometric_median, deflated_eigvals, deflated_eigvectors):
    deflated_median, deflated_vectors = calculate_geometric_median(deflated_eigvectors[:, 0, :], deflated_eigvals[:, 0])
    return deflated_median, deflated_vectors


@app.cell
def _(deflated_median):
    deflated_median
    return


@app.cell
def _(proposal_median):
    proposal_median
    return


@app.cell
def _(deflated_eigvals, plot_distribution_of_spectra):
    plot_distribution_of_spectra(deflated_eigvals)
    return


@app.cell
def _(cc, flipped_vectors, plt, proposal_median, sample_outputs):
    def plot_referenced_distribution(component_i, component_j, flipped_vectors, proposal_median):
        fig, ax = plt.subplots()
        ax.scatter(
            flipped_vectors[:, component_i], flipped_vectors[:, component_j],
            s=1, alpha=0.5,
            c=sample_outputs, cmap=cc.m_CET_L20, vmin=-1, vmax=1
        )
        ax.scatter(proposal_median[component_i], proposal_median[component_j], c='r')
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        # ax.set_facecolor('k')
        plt.show()

    plot_referenced_distribution(0, 9, flipped_vectors, proposal_median)
    return (plot_referenced_distribution,)


@app.cell
def _(deflated_median, deflated_vectors, plot_referenced_distribution):
    plot_referenced_distribution(0, 3, deflated_vectors, deflated_median)
    return


@app.cell
def _(mean_eigvector, mean_hessian):
    (mean_eigvector @ mean_hessian).shape
    return


@app.cell
def _():
    # resigned_vectors = np.expand_dims(np.sign(sampled_eigvals), axis=2) * sampled_eigvectors
    # resigned_vectors = sampled_eigvectors * np.expand_dims(np.sign(sampled_eigvectors[:, 0, 0]), axis=(1, 2))
    return


@app.cell
def _(np, sampled_hessians):
    def calculate_effective_ranks(hessians):
        effective_ranks = []
        for hessian_index in range(hessians.shape[0]):
            svd_values = np.linalg.svdvals(hessians[hessian_index, :, :])
            svd_spectrum = (1 / svd_values) / np.sum(1 / svd_values)
            erank = np.exp(np.sum(-svd_spectrum*np.log(svd_spectrum)))
            effective_ranks.append(erank)
        return np.array(effective_ranks)

    sampled_ers = calculate_effective_ranks(sampled_hessians)
    return (sampled_ers,)


@app.cell
def _(np, plt, sampled_ers):
    plt.hist(np.log(sampled_ers), bins=100);
    plt.show()
    return


@app.cell
def _(plot_eigendirections, sampled_eigvectors):
    plot_eigendirections(sampled_eigvectors, 5, 6, 1)
    return


@app.cell
def _(np, sampled_eigvectors):
    import umap
    import numba

    @numba.njit()
    def nematic_cosine(a, b):
        return 1 - np.abs(np.dot(a, b))

    @numba.njit()
    def nematic_matrix_cosine(a, b):
        component_distances = 1 - np.abs(np.vecdot(a, b))
        return np.sqrt(np.sum(component_distances**2))


    def run_umap(input_vectors):
        # fit = umap.UMAP(metric=nematic_matrix_cosine, n_neighbors=25, min_dist=0, random_state=0)
        fit = umap.UMAP(metric=nematic_cosine, n_neighbors=25, min_dist=0, random_state=0)
        model_embeddings = fit.fit_transform(input_vectors)
        return model_embeddings

    model_embeddings = run_umap(sampled_eigvectors[:, 0, :])
    # model_embeddings = run_umap(sampled_eigvectors)
    # model_embeddings = run_umap(sampled_jacobians)
    return (model_embeddings,)


@app.cell
def _():
    # plt.scatter(sampled_jacobians[:, 8], sampled_jacobians[:, 9], s=1)
    return


@app.cell
def _(sampled_eigvals):
    sampled_eigvals.shape
    return


@app.cell
def _(np, plt, sampled_eigvals):
    plt.hist(np.log(np.abs(sampled_eigvals[:, 0])), bins=100);
    plt.show()
    return


@app.cell
def _(np, sampled_eigvectors):
    abs_key_ev = np.abs(sampled_eigvectors[:, 0, :])
    parameter_entropy = np.sum(-abs_key_ev*np.log(abs_key_ev), axis=1)
    return (parameter_entropy,)


@app.cell
def _(np, proposal_median, sampled_eigvectors):
    deviation = 1 - np.abs(np.dot(sampled_eigvectors[:, 0, :], proposal_median))
    return


@app.cell
def _(cc, model_embeddings, parameter_entropy, plt):
    def plot_umap():
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        pos = ax.scatter(
            model_embeddings[:, 0], model_embeddings[:, 1],
            # c=np.log(np.abs(sampled_eigvals[:, 0])),
            c=parameter_entropy,
            # c=deviation,
            s=0.1, alpha=0.5, cmap=cc.m_CET_L20
        )
        # cbar = fig.colorbar(pos, ax=ax, shrink=0.6, label="ln(|λ|)")
        cbar = fig.colorbar(pos, ax=ax, shrink=0.6, label="Parameter Entropy")
        ax.set_aspect("equal")
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        fig.tight_layout()
        plt.show()

    plot_umap()
    return


@app.cell
def _(gridsearch_parameters, model_embeddings, np, plt, sampled_eigvectors):
    def plot_component_umap():
        fig, ax = plt.subplots(figsize=(8, 8))

        largest_components = np.argmax(np.abs(sampled_eigvectors[:, 0, :]), axis=1)
        largest_components = np.argsort(np.abs(sampled_eigvectors[:, 0, :]), axis=1)
        largest_components = largest_components[:, -1]
        parameter_entropy = -np.sum(np.abs(sampled_eigvectors[:, 0, :]) * np.log(np.abs(sampled_eigvectors[:, 0, :])), axis=1)
        parameter_entropy = (parameter_entropy - np.min(parameter_entropy)) / (np.max(parameter_entropy) - np.min(parameter_entropy))

        for p_i in range(10):
            component_mask = largest_components == p_i
            if np.count_nonzero(component_mask) == 0:
                continue
            ax.scatter(
                model_embeddings[component_mask, 0],
                model_embeddings[component_mask, 1],
                alpha=0.5,
                s=0.5, label=list(gridsearch_parameters.keys())[p_i]
            )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        # Set up legend:
        legend = ax.legend()
        for lh in legend.legend_handles:
            lh.set_alpha(1)
            lh.set_sizes([10])

        fig.tight_layout()
        plt.show()

    plot_component_umap()
    return


@app.cell
def _(gridsearch_parameters, model_embeddings, np, plt, sampled_eigvectors):
    def plot_unit_umap():
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(model_embeddings[:, 0], model_embeddings[:, 1], c='k', alpha=0.1, s=0.1)

        for p_i in range(10):
            component_mask = np.abs(sampled_eigvectors[:, 0, p_i]) > 0.95
            if np.count_nonzero(component_mask) == 0:
                continue
            ax.scatter(
                model_embeddings[component_mask, 0],
                model_embeddings[component_mask, 1],
                alpha=1.0,
                s=0.5, label=list(gridsearch_parameters.keys())[p_i]
            )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        # Set up legend:
        legend = ax.legend()
        for lh in legend.legend_handles:
            lh.set_alpha(1)
            lh.set_sizes([10])

        fig.tight_layout()
        plt.show()

    plot_unit_umap()
    return


@app.cell
def _(np, plt, sample_inputs, sampled_eigvectors):
    # cluster_mask = np.logical_and(model_embeddings[:, 0] < 5, model_embeddings[:, 1] < 1)
    # cluster_mask = model_embeddings[:, 0] < -2
    cluster_mask = np.abs(sampled_eigvectors[:, 0, 3]) > 0.9

    def plot_cluster_points():
        fig, axs = plt.subplots(10, 10)

        for i in range(10):
            for j in range(10):
                axs[i, j].scatter(sample_inputs[cluster_mask, i], sample_inputs[cluster_mask, j], s=0.1)
                axs[i, j].set_xlim(0, 1)
                axs[i, j].set_ylim(0, 1)
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                axs[i, j].set_aspect("equal")
        fig.tight_layout()
        plt.show()

    plot_cluster_points()
    return (cluster_mask,)


@app.cell
def _(cluster_mask, colorstamps, plt, sample_inputs, sampled_eigvectors):
    def plot_masked_eigencomponents(mask, eigenvectors, eigenvector_index, parameter_i, parameter_j):
        fig, ax = plt.subplots()

        # Get 2D colormap from parameters:
        rgb, _ = colorstamps.apply_stamp(
            sample_inputs[mask, parameter_i], sample_inputs[mask, parameter_j],
            'flat',
            vmin_0=0, vmax_0=1,
            vmin_1=0, vmax_1=1,
        )
        ax.scatter(
            eigenvectors[mask, eigenvector_index, parameter_i],
            eigenvectors[mask, eigenvector_index, parameter_j],
            s=1, alpha=0.25, c=rgb
        )
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        plt.show()

    plot_masked_eigencomponents(cluster_mask, sampled_eigvectors, 0, 1, 0)
    return


@app.cell
def _(np, torch):
    def grad_random_emulate(manager, x):
        # Convert to tensor:
        tensor_input = torch.tensor(x, dtype=torch.float32)
        tensor_input.requires_grad_(True)

        # Ensure we have a batch dimension:
        if len(tensor_input.shape) == 1:
            tensor_input = torch.unsqueeze(tensor_input, 0)

        # Run emulation:
        manager.model.eval()
        manager.likelihood.eval()
        prediction = manager.likelihood(manager.model(tensor_input))
        rng = np.random.default_rng()
        reparameterised_sample = torch.tensor(rng.normal(0, 1, 1)).to(torch.float32)
        prediction_sample = (reparameterised_sample * prediction.stddev) + prediction.mean
        return prediction_sample, tensor_input
    return (grad_random_emulate,)


@app.cell
def _(calculate_hessian, cf_model_manager, grad_random_emulate, np):
    def random_hessians(input):
        hessians = []
        for i in range(1000):
            # Estimate Hessian in local area, sampling from same point:
            prediction_mean, tensor_input = grad_random_emulate(cf_model_manager, input)
            estimated_hessian = calculate_hessian(prediction_mean, tensor_input)
            estimated_hessian = np.squeeze(estimated_hessian.detach().numpy())
            hessians.append(estimated_hessian)
        return np.stack(hessians, axis=0)

    random_hessians = random_hessians(np.ones(10) * 0.25)
    return (random_hessians,)


@app.cell
def _(batch_eigen_calc, random_hessians):
    random_eigvals, random_eigvectors = batch_eigen_calc(random_hessians)
    return (random_eigvectors,)


@app.cell
def _(plot_eigendirections, random_eigvectors):
    plot_eigendirections(random_eigvectors, 0, 1, 0)
    return


@app.cell
def _(np, sampled_eigvals):
    minima_mask = np.all(sampled_eigvals[:, :2] > 0, axis=1)
    np.count_nonzero(minima_mask)
    return (minima_mask,)


@app.cell
def _(minima_mask, plot_oat, sample_inputs):
    plot_oat(1000, sample_inputs[minima_mask, :][9, :])
    return


@app.cell
def _(plt, sampled_eigvals):
    plt.hist(sampled_eigvals[:, 0], bins=100, density=True);
    plt.show()
    return


@app.cell
def _(hessians, np):
    def get_distance_matrix():
        num_sampled_points = len(hessians)
        hessian_distances = np.empty((num_sampled_points, num_sampled_points))
        for i in range(num_sampled_points):
            for j in range(num_sampled_points):
                # Flip signs if necessary:
                distance = np.min(
                    [
                        np.mean(np.abs(hessians[i] - hessians[j])),
                        np.mean(np.abs(hessians[i] + hessians[j]))
                    ]
                )
                hessian_distances[i, j] = distance

        return hessian_distances

    hessian_distances = get_distance_matrix()
    return (hessian_distances,)


@app.cell
def _(hessian_distances, plt):
    plt.imshow(hessian_distances)
    return


@app.cell
def _(hessian_distances, np):
    import sklearn

    # Mask for bizarre areas:
    # normal_mask = np.mean(hessian_distances, axis=1) < 5

    mds_manager = sklearn.manifold.MDS(dissimilarity="precomputed")
    metric_space = mds_manager.fit_transform(np.sqrt(hessian_distances))
    return (metric_space,)


@app.cell
def _():
    # Sammon mapping w linear transformation from input parameters - allows us to batch & converge that way (hopefully less slow):
    return


@app.cell
def _(hessian_distances, np, plt):
    plt.hist(np.sqrt(hessian_distances.flatten()), bins=100);
    plt.show()
    return


@app.cell
def _(metric_space, plt):
    def plot_metric_space():
        fig, ax = plt.subplots()
        ax.scatter(metric_space[:, 0], metric_space[:, 1], s=1, alpha=0.25);
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect(1)
        plt.show()

    plot_metric_space()
    return


@app.cell
def _(hessians, plt):
    plt.imshow(hessians[99])
    return


@app.cell
def _(np, test_hessian):
    eig_result = np.linalg.eig(test_hessian)
    eigenvector_matrix = eig_result.eigenvectors
    print(eigenvector_matrix * np.expand_dims(eig_result.eigenvalues, axis=0))
    return eig_result, eigenvector_matrix


@app.cell
def _(eigenvector_matrix, plt):
    plt.imshow(eigenvector_matrix)
    return


@app.cell
def _(np):
    np.ones(10)*0.5
    return


@app.cell
def _(eig_result, np, plt):
    def plot_log_eigenvalues():
        fig, ax = plt.subplots()
        for eigval in eig_result.eigenvalues:
            ax.hlines(np.log(np.abs(eigval)), 0, 1)
        ax.set_aspect(0.4)
        plt.show()

    plot_log_eigenvalues()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
