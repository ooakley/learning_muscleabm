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
    return ModelManager, load_gridsearch_data


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
    return calculate_hessian, emulate, grad_emulate, log_grad_emulate


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
    return (
        anni_model_manager,
        cf_model_manager,
        gridsearch_parameters,
        inducing_points,
    )


@app.cell
def _(ModelManager, inducing_points):
    idr_model_manager = ModelManager(inducing_points, 0.003)
    idr_model_manager.load("model_experiments/2026-01-26-matrix_collisions/gaussian_process_models", "density_idr")

    op3_model_manager = ModelManager(inducing_points, 0.003)
    op3_model_manager.load("model_experiments/2026-01-26-matrix_collisions/gaussian_process_models", "op3")

    op17_model_manager = ModelManager(inducing_points, 0.003)
    op17_model_manager.load("model_experiments/2026-01-26-matrix_collisions/gaussian_process_models", "op17")
    return


@app.cell
def _(np):
    # Set up sampled inputs across parameter space:
    rng = np.random.default_rng(0)
    sample_inputs = rng.uniform(0, 1, size=(20000, 10))
    return (sample_inputs,)


@app.cell
def _(anni_model_manager, emulate, plt, sample_inputs):
    sampled_outputs, _ = emulate(anni_model_manager, sample_inputs)
    plt.hist(sampled_outputs, bins=50);
    plt.show()
    return (sampled_outputs,)


@app.cell
def _(calculate_hessian, grad_emulate, log_grad_emulate, np):
    def sample_hessians(inputs, manager, log_parameters=False):
        hessians = []
        for i in range(inputs.shape[0]):
            if (i + 1) % 100 == 0:
                print(i + 1)
            # Estimate Hessian in local area:
            if log_parameters:
                prediction_mean, tensor_input = log_grad_emulate(manager, inputs[i, :])
            else:
                prediction_mean, tensor_input = grad_emulate(manager, inputs[i, :])
            estimated_hessian = calculate_hessian(prediction_mean, tensor_input)
            estimated_hessian = np.squeeze(estimated_hessian.detach().numpy())
            hessians.append(estimated_hessian)

        return np.stack(hessians, axis=0)

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
    return batch_eigen_calc, sample_hessians


@app.cell
def _(cf_model_manager, sample_hessians, sample_inputs):
    sampled_hessians = sample_hessians(sample_inputs, cf_model_manager, log_parameters=True)
    return (sampled_hessians,)


@app.cell
def _(np):
    def symmetrise_hessians(sampled_hessians):
        symmetrised_hessians = []
        for i in range(sampled_hessians.shape[0]):
            symmetrised_hessians.append((sampled_hessians[i, :, :] + sampled_hessians[i, :, :].T) / 2)
        return np.stack(symmetrised_hessians, axis=0)
    return (symmetrise_hessians,)


@app.cell
def _(plt, sampled_hessians):
    plt.imshow(sampled_hessians[9, :, :])
    return


@app.cell
def _(sampled_hessians, symmetrise_hessians):
    symmetrised_hessians = symmetrise_hessians(sampled_hessians)
    return


@app.cell
def _(np, sampled_hessians):
    eigvals, eigvecs = np.linalg.eig(sampled_hessians[0, :, :])
    print(eigvals)
    # eigvals, eigvecs = np.linalg.eig(symmetrised_hessians[0, :, :])
    # print(eigvals)
    return


@app.cell
def _(np, plt):
    def riemannian_psd_distance(matrix_a, matrix_b):
        # Get sqrt of matrix:
        jitter = np.eye(11) * 1e-5
        a_eigenvalues, a_eigenvectors = np.linalg.eig(matrix_a)
        print(a_eigenvalues)
        sqrt_a = a_eigenvectors * (np.sqrt(a_eigenvalues) @ a_eigenvectors.T)
        print()
        plt.imshow(sqrt_a @ matrix_b @ sqrt_a)
        plt.show()
    return


@app.cell
def _():
    # riemannian_psd_distance(symmetrised_hessians[10, :, :], symmetrised_hessians[0, :, :])
    return


@app.cell
def _(np, sampled_hessians):
    mean_hessian = np.mean(sampled_hessians, axis=0)
    mean_eigenvalues, mean_eigenvectors = np.linalg.eig(mean_hessian)
    mean_eigenvectors = mean_eigenvectors.T
    return mean_eigenvalues, mean_eigenvectors, mean_hessian


@app.cell
def _(mean_eigenvalues):
    mean_eigenvalues
    return


@app.cell
def _(mean_eigenvalues, np):
    def get_deflation_matrix(eigenvectors):
        dimension = eigenvectors.shape[0]
        deflation_array = np.zeros((dimension, dimension, dimension))
        for i in range(dimension):
            indexed_eigenvector = eigenvectors[i, :]
            indexed_eigenvector = np.expand_dims(indexed_eigenvector, axis=1)
            deflation_component = mean_eigenvalues[i] * (indexed_eigenvector @ indexed_eigenvector.T)
            deflation_array[i, :, :] = deflation_component
        deflation_matrix = np.sum(deflation_array, axis=0)
        return deflation_matrix
    return (get_deflation_matrix,)


@app.cell
def _(candidate_eigenvectors, get_deflation_matrix):
    deflation_matrix = get_deflation_matrix(candidate_eigenvectors)
    return (deflation_matrix,)


@app.cell
def _(deflation_matrix, np, plt, sampled_hessians):
    plt.hist(np.linalg.norm(sampled_hessians - deflation_matrix, axis=(1, 2)), bins=50);
    plt.show()
    return


@app.cell
def _(deflation_matrix, mean_hessian, np):
    np.linalg.norm(mean_hessian - deflation_matrix)
    return


@app.cell
def _(cc, deflation_matrix, mean_hessian, np, plt):
    plt.imshow(mean_hessian, vmin=-1, vmax=1, cmap=cc.m_CET_D1)
    plt.show()
    print(np.linalg.norm(deflation_matrix))
    return


@app.cell
def _(np, torch):
    def ensemble_deflation(hessian_sample):
        dimension = hessian_sample.shape[-1]
        # Get initial candidates:
        mean_hessian = np.mean(hessian_sample, axis=0)
        mean_eigenvalues, mean_eigenvectors = np.linalg.eig(mean_hessian)
        candidate_eigenvectors = torch.from_numpy(mean_eigenvectors.T)
        candidate_eigenvalues = torch.from_numpy(mean_eigenvalues)
        candidate_eigenvectors.requires_grad_()
        candidate_eigenvalues.requires_grad_()

        # Get dataset:
        batch_hessians = torch.from_numpy(hessian_sample)

        # Iterate:
        loss_history = []
        for _ in range(1000):
            # Get deflation loss:
            deflation_tensor = torch.zeros((dimension, dimension, dimension))
            for eig_index in range(dimension):
                indexed_eigenvector = candidate_eigenvectors[eig_index, :]
                indexed_eigenvector = torch.unsqueeze(indexed_eigenvector, dim=1)
                deflation_component = candidate_eigenvalues[eig_index] * (indexed_eigenvector @ indexed_eigenvector.T)
                deflation_tensor[eig_index, :, :] = deflation_component
            deflation_matrix = torch.sum(deflation_tensor, axis=0)
            deflation_loss = torch.mean(torch.linalg.norm(batch_hessians - deflation_matrix, axis=(1, 2)))

            # Get off diagonal loss:
            orthogonality_loss = torch.linalg.norm(torch.triu(candidate_eigenvectors @ candidate_eigenvectors.T, 1))

            # Parameter entropy loss:
            normalised_vectors = torch.abs(candidate_eigenvectors) / torch.linalg.norm(candidate_eigenvectors, axis=1, keepdims=True)
            vector_entropies = torch.sum(-normalised_vectors * torch.log(normalised_vectors), axis=1)
            entropy_loss = torch.mean(vector_entropies)

            # Backwards pass:
            losses = np.array([entropy_loss.item(), deflation_loss.item(), orthogonality_loss.item()])
            loss_history.append(losses)
            combined_loss = entropy_loss + deflation_loss + orthogonality_loss
            vector_step = torch.autograd.grad(combined_loss, candidate_eigenvectors, retain_graph=True)[0]
            value_step = torch.autograd.grad(combined_loss, candidate_eigenvalues, retain_graph=False)[0]

            # Update parameters:
            with torch.no_grad():
                candidate_eigenvectors -= 0.003*vector_step
                candidate_eigenvalues -= 0.003*value_step

        return candidate_eigenvalues.detach().numpy(), candidate_eigenvectors.detach().numpy(), np.stack(loss_history, axis=0)
    return (ensemble_deflation,)


@app.cell
def _(ensemble_deflation, sampled_hessians):
    candidate_eigenvalues, candidate_eigenvectors, loss_history = ensemble_deflation(sampled_hessians)
    return candidate_eigenvalues, candidate_eigenvectors, loss_history


@app.cell
def _(candidate_eigenvectors, np):
    np.linalg.norm(candidate_eigenvectors[0, :])
    return


@app.cell
def _(candidate_eigenvectors):
    candidate_eigenvectors.shape
    return


@app.cell
def _(candidate_eigenvalues):
    candidate_eigenvalues
    return


@app.cell
def _(candidate_eigenvalues, plt):
    plt.plot(candidate_eigenvalues)
    return


@app.cell
def _(candidate_eigenvectors, cc, plt):
    plt.imshow(candidate_eigenvectors, vmin=-1, vmax=1, cmap=cc.m_CET_D2)
    return


@app.cell
def _(candidate_eigenvectors, gridsearch_parameters, np):
    def print_components(query_index):
        out_eigenvector = candidate_eigenvectors[query_index, :]
        out_eigenvector /= np.linalg.norm(out_eigenvector)
        for i in range(10):
            if np.abs(out_eigenvector[i]) < 0.1:
                continue
            print(list(gridsearch_parameters.keys())[i], out_eigenvector[i])       
    return (print_components,)


@app.cell
def _(print_components):
    print_components(0)
    return


@app.cell
def _(loss_history, plt):
    plt.plot(loss_history[:, 1])
    return


@app.cell
def _(batch_eigen_calc, sampled_hessians):
    # jitter_matrix = np.expand_dims(np.eye(11) * 3e-6, axis=0)
    sampled_eigenvalues, sampled_eigenvectors = batch_eigen_calc(sampled_hessians)
    return sampled_eigenvalues, sampled_eigenvectors


@app.cell
def _(np, sampled_eigenvalues):
    condition_numbers = np.abs(sampled_eigenvalues[:, 0]) / np.abs(sampled_eigenvalues[:, -1])
    return (condition_numbers,)


@app.cell
def _(condition_numbers, np, plt, sampled_outputs):
    plt.scatter(sampled_outputs, np.log10(condition_numbers), s=1);
    plt.show()
    return


@app.cell
def _(np, plt, sampled_eigenvalues):
    plt.hist(np.log(sampled_eigenvalues[:, 0] + 1e-5), bins=100);
    plt.show()
    return


@app.cell
def _(np, sampled_eigenvalues):
    np.max(sampled_eigenvalues[:, 0])
    return


@app.cell
def _(np, plt, sampled_eigenvalues, sampled_outputs):
    def plot_eigvalue_dist():
        eig_magnitude = np.abs(sampled_eigenvalues)
        plt.scatter(sampled_outputs, np.log(eig_magnitude[:, 0]), s=1, alpha=0.25)
        # plt.scatter(sampled_outputs, np.log(eig_magnitude[:, 1]), s=1, alpha=0.25)
        # plt.scatter(sampled_outputs, np.log(eig_magnitude[:, 9]), s=1, alpha=0.25)
        plt.show()

    plot_eigvalue_dist()
    return


@app.cell
def _(np, sampled_eigenvectors):
    eigenvector_components = np.abs(sampled_eigenvectors[:, 0, :])
    parameter_entropies = np.sum(-eigenvector_components * np.log(eigenvector_components), axis=1)
    return (parameter_entropies,)


@app.cell
def _(parameter_entropies, plt, sampled_outputs):
    plt.scatter(sampled_outputs, parameter_entropies, s=1);
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
    return calculate_geometric_median, flipped_vectors, proposal_median


@app.cell
def _(flipped_vectors):
    import sklearn.cluster

    kmeans_manager = sklearn.cluster.KMeans(3)
    cluster_space = kmeans_manager.fit_transform(flipped_vectors[:, :])
    cluster_labels = kmeans_manager.fit_predict(flipped_vectors[:, :])
    return cluster_labels, cluster_space, kmeans_manager


@app.cell
def _(
    cluster_labels,
    cluster_space,
    kmeans_manager,
    np,
    plt,
    sample_inputs,
    sampled_outputs,
):
    def plot_cluster_ep(cluster_index):
        # Mask over relevant cluster:
        cluster_mask = cluster_labels == cluster_index
        cluster_eigenvector = kmeans_manager.cluster_centers_[cluster_index, :]
        print(np.linalg.norm(cluster_eigenvector))
        cluster_eigenvector = cluster_eigenvector / np.linalg.norm(cluster_eigenvector)

        # Convert relevant inputs into eigenparameter space:
        cluster_ep_values = -np.sum(np.log(sample_inputs[cluster_mask]) * cluster_eigenvector, axis=1)

        # Plot binscatter:
        fig, ax = plt.subplots()
        ax.scatter(cluster_ep_values, sampled_outputs[cluster_mask], s=1, alpha=1, c=cluster_space[cluster_mask, cluster_index])

        ventiles = np.linspace(0, 1, 21)
        bin_x = []
        bin_y = []
        for index in range(20):
            lq = ventiles[index]
            uq = ventiles[index + 1]
            lb = np.quantile(cluster_ep_values, lq)
            ub = np.quantile(cluster_ep_values, uq)
            bin_x.append((lb + ub) / 2)

            param_mask = np.logical_and(cluster_ep_values > lb, cluster_ep_values < ub)
            bin_y.append(np.mean(sampled_outputs[cluster_mask][param_mask]))

        ax.scatter(bin_x, bin_y, s=1)
        ax.plot(bin_x, bin_y)
        ax.set_xlabel(f"Cluster {cluster_index} Eigenparameter")
        ax.set_ylabel("ANNI")
        plt.show()
    return (plot_cluster_ep,)


@app.cell
def _(plot_cluster_ep):
    plot_cluster_ep(2)
    return


@app.cell
def _(cluster_labels, cluster_space, plt):
    plt.scatter(cluster_space[:, 1], cluster_space[:, 2], s=0.1, c=cluster_labels)
    return


@app.cell
def _(np, proposal_median):
    median_components = np.abs(proposal_median)
    median_entropy = np.sum(-median_components * np.log(median_components))
    return


@app.cell
def _(flipped_vectors, np, plt, sample_inputs, sampled_outputs):
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
            s=s, alpha=0.5, c=c
        )

        theta = np.linspace(0, np.pi * 2, 100)
        ax.plot(1.025*np.cos(theta), 1.025*np.sin(theta), c='k', ls="--",  lw=1, alpha=0.5)
        format_axes(ax)
        # plt.show()

    def plot_linear_combinations():
        fig, axs = plt.subplots(10, 10, figsize=(10, 10))
        for i in range(10):
            for j in range(10):
                if i == j:
                    axs[i, j].text(0, 0, "Placeholder", fontsize=5, horizontalalignment="center")
                    format_axes(axs[i, j])
                    continue
                if i > j:
                    plot_eigendirections(sample_inputs, flipped_vectors, i, j, ax=axs[i, j], s=0.01)
                if i < j:
                    plot_eigendirections(sample_inputs, flipped_vectors, i, j, ax=axs[i, j], s=0.01, c=sampled_outputs)

        fig.tight_layout()
        plt.show()
    return plot_eigendirections, plot_linear_combinations


@app.cell
def _(plot_linear_combinations):
    plot_linear_combinations()
    return


@app.cell
def _(
    flipped_vectors,
    np,
    plot_eigendirections,
    plt,
    sample_inputs,
    sampled_outputs,
):
    lb = 2.5
    test_mask = np.logical_and(sampled_outputs > lb, sampled_outputs < lb + 0.5)
    plot_eigendirections(sample_inputs[test_mask], flipped_vectors[test_mask], 0, 3)
    plt.show()
    return


@app.cell
def _(
    calculate_geometric_median,
    np,
    sample_inputs,
    sampled_eigenvalues,
    sampled_eigenvectors,
    sampled_outputs,
):
    def generate_primary_eigenvector_trajectory(lengthscale=1):
        # mean_points = np.linspace(-1, 3, 100)
        # eigenvector_trajectory = []
        # for t in range(len(mean_points)):
        #     distance = np.abs(mean_points[t] - sampled_outputs)
        #     weighting = np.exp(-(distance / lengthscale)**2)
        #     primary_eigenvector, _ = calculate_geometric_median(sampled_eigenvectors[:, 0, :], sampled_eigenvalues[:, 0], weighting)
        #     eigenvector_trajectory.append(primary_eigenvector)

        weights = (1 / np.sqrt(2*np.pi)) * np.exp(((np.linspace(-2, 2, 500)**2) / 2))
        sorted_indices = np.argsort(sampled_outputs)
        sorted_indices = np.argsort(sample_inputs[:, 0])
        eigenvector_trajectory = []
        distance_from_median = []
        for t in range(250, 19750):
            idx = sorted_indices[t-250:t+250]
            primary_eigenvector, _, dfm = calculate_geometric_median(
                sampled_eigenvectors[idx, 0, :], sampled_eigenvalues[idx, 0],
                # weighting=weights
            )
            eigenvector_trajectory.append(primary_eigenvector)
            distance_from_median.append(dfm)
            #  * np.mean(np.abs(sampled_eigenvalues[idx, 0]))
            # mean_hessian = np.mean(sampled_hessians[idx, :, :], axis=0)
            # eigvals, eigvectors = np.linalg.eig(mean_hessian)
            # eigenvector_trajectory.append(eigvectors.T[0, :])

        return np.stack(eigenvector_trajectory, axis=0), np.stack(distance_from_median)

    eigenvector_trajectory, distance_from_median = generate_primary_eigenvector_trajectory(lengthscale=0.25)
    return distance_from_median, eigenvector_trajectory


@app.cell
def _(distance_from_median, np, plt, sampled_outputs):
    def plot_error_of_median(distance_from_median):
        fig, ax = plt.subplots()
        # Get x values:
        fractional_ranks = np.arange(250, 19750) / 20000
        data_values = np.quantile(sampled_outputs, fractional_ranks)
        ax.plot(data_values, distance_from_median)
        ax.set_xlim(data_values[0], data_values[-1])
        # ax.set_ylim(0, 0.5)
        plt.show()

    plot_error_of_median(distance_from_median)
    return


@app.cell
def _(eigenvector_trajectory, np):
    fixed_trajectory = []
    fixed_trajectory.append(eigenvector_trajectory[0, :])
    for index in range(eigenvector_trajectory.shape[0] - 1):
        dot_product = np.dot(fixed_trajectory[-1], eigenvector_trajectory[index + 1])
        fixed_trajectory.append(eigenvector_trajectory[index + 1] * np.sign(dot_product))
    fixed_trajectory = np.stack(fixed_trajectory, axis=0)
    return (fixed_trajectory,)


@app.cell
def _(fixed_trajectory, gridsearch_parameters, np, plt, sampled_outputs):
    def plot_control_trajectory(eigenvector_trajectory):
        fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')

        # Get x values:
        fractional_ranks = np.arange(250, 19750) / 20000
        data_values = np.quantile(sampled_outputs, fractional_ranks)
        ax.hlines([-1, 0, 1], data_values[0], data_values[-1], color='k', alpha=0.5)

        # # Twin axis for noise value:
        # twin_axis = ax.twinx() 
        # twin_axis.plot(data_values, distance_from_median)

        for i in range(eigenvector_trajectory.shape[-1]):
            if i == 10:
                ax.plot(data_values, eigenvector_trajectory[:, i], label="matrixAdvectionRate", lw=1, ls=":")
                continue
            ax.plot(data_values, eigenvector_trajectory[:, i], label=list(gridsearch_parameters.keys())[i], lw=1)

        ax.set_xlim(data_values[0], data_values[-1])
        # ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Order Parameter 3")
        ax.set_ylabel("Eigenvalue Component - λ1")
        fig.legend(loc='outside center right')
        plt.show()

    plot_control_trajectory(fixed_trajectory)
    return


@app.cell
def _(fixed_trajectory, kmeans_manager, mean_eigenvectors, np):
    approximate_eigenparameter = fixed_trajectory[15000, :]
    # approximate_eigenparameter = candidate_eigenvectors[0, :] / np.linalg.norm(candidate_eigenvectors[0, :])
    approximate_eigenparameter = mean_eigenvectors[0, :]
    approximate_eigenparameter = kmeans_manager.cluster_centers_[2, :] / np.linalg.norm(kmeans_manager.cluster_centers_[2, :])
    return (approximate_eigenparameter,)


@app.cell
def _(approximate_eigenparameter, gridsearch_parameters, np):
    for i in range(10):
        if np.abs(approximate_eigenparameter[i]) < 0.1:
            continue
        print(list(gridsearch_parameters.keys())[i], approximate_eigenparameter[i])
    return


@app.cell
def _(approximate_eigenparameter, np, sample_inputs):
    ep_values = -np.sum(np.log(sample_inputs) * approximate_eigenparameter, axis=1)
    return (ep_values,)


@app.cell
def _(ep_values, plt):
    plt.hist(ep_values, bins=100)
    return


@app.cell
def _(ep_values, np, plt, sampled_outputs):
    OP3_DIST_MEAN = 0.513909147045695
    OP3_DIST_STD = 0.17544598929394573

    def plot_ep_val():
        fig, ax = plt.subplots()
        ax.scatter(ep_values, (sampled_outputs * OP3_DIST_STD) + OP3_DIST_MEAN, s=0.1, alpha=0.1, c='k')

        # Binscatter:
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

        ax.scatter(bin_x, (np.array(bin_y) * OP3_DIST_STD) + OP3_DIST_MEAN, s=10)
        ax.set_xlabel("Eigenparameter λ2 Values")
        ax.set_ylabel("Order Parameter 3")
        plt.show()

    plot_ep_val()
    return (OP3_DIST_MEAN,)


@app.cell
def _(np):
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

    def run_hessian_umap(distance_matrix):
        fit = umap.UMAP(metric="precomputed", n_neighbors=5, min_dist=0, random_state=0)
        model_embeddings = fit.fit_transform(distance_matrix)
        return model_embeddings
    return run_hessian_umap, run_umap


@app.cell
def _(run_umap, sampled_eigenvectors):
    eigenvector_embeddings = run_umap(sampled_eigenvectors[:, 0, :])
    return (eigenvector_embeddings,)


@app.cell
def _(np, sampled_hessians):
    hessian_distance_matrix = (sampled_hessians[:, np.newaxis] - sampled_hessians[np.newaxis, :])
    hessian_distance_matrix = np.sqrt(np.sum(hessian_distance_matrix**2, axis=(2, 3)))
    return (hessian_distance_matrix,)


@app.cell
def _(hessian_distance_matrix):
    import kmedoids
    km = kmedoids.KMedoids(5, method='fasterpam')
    km_fit = km.fit(hessian_distance_matrix)
    return


@app.cell
def _(hessian_distance_matrix, run_hessian_umap):
    hessian_embeddings = run_hessian_umap(hessian_distance_matrix)
    return (hessian_embeddings,)


@app.cell
def _(cc, hessian_embeddings, plt, sampled_outputs):
    plt.scatter(hessian_embeddings[:, 0], hessian_embeddings[:, 1], s=1, c=sampled_outputs, alpha=0.5, cmap=cc.m_CET_D1)
    return


@app.cell
def _(cc, cluster_labels, eigenvector_embeddings, plt):
    plt.scatter(eigenvector_embeddings[:, 0], eigenvector_embeddings[:, 1], s=1, c=cluster_labels, alpha=0.5, cmap=cc.m_CET_D1)
    return


@app.cell
def _(OP3_DIST_MEAN, wtype):
    wtype(OP3_DIST_MEAN)
    return


@app.cell
def _():
    # find thing that's unfixed in fit, that IS fixed for order parameter / matrix organisation; go from there
    return


@app.cell
def _(eigenvector_trajectory, plt):
    def plot_dimensional_trajectory(i, j, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
        ax.scatter(eigenvector_trajectory[0, i], eigenvector_trajectory[0, j], s=5, c='r')
        ax.scatter(eigenvector_trajectory[-1, i], eigenvector_trajectory[-1, j], s=5, c='g')
        ax.plot([0, 0], [-1.05, 1.05], c='k', alpha=0.5)
        ax.plot([-1.05, 1.05], [0, 0], c='k', alpha=0.5)
        ax.plot(eigenvector_trajectory[:, i], eigenvector_trajectory[:, j])
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    def plot_full_trajectory():
        fig, axs = plt.subplots(7, 7, figsize=(5, 5))
        for i in range(7):
            for j in range(7):
                plot_dimensional_trajectory(i, j, axs[i, j])

        fig.tight_layout()
        plt.show()

    plot_full_trajectory()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
