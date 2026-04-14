import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import json
    import copy

    import gpytorch
    import torch
    import scipy.spatial

    import numpy as np
    import matplotlib.pyplot as plt

    from torch.utils.data import TensorDataset, DataLoader
    from dppy.finite_dpps import FiniteDPP

    import colorcet as cc
    return (
        DataLoader,
        TensorDataset,
        cc,
        gpytorch,
        json,
        np,
        os,
        plt,
        scipy,
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
    return ModelManager, emulate, load_gridsearch_data


@app.cell
def _(ModelManager, load_gridsearch_data, np):
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

    # Dummy inducing points:
    inducing_points = np.ones((10, 10))

    # Instantiate then load emulators:
    cf_model_manager = ModelManager(inducing_points, 0.003)
    cf_model_manager.load("model_experiments/2025-12-04-collisions_only/gaussian_process_models", "coherency_fraction")
    anni_model_manager = ModelManager(inducing_points, 0.003)
    anni_model_manager.load("model_experiments/2025-12-04-collisions_only/gaussian_process_models", "ann_index")
    speed_model_manager = ModelManager(inducing_points, 0.003)
    speed_model_manager.load("model_experiments/2025-12-04-collisions_only/gaussian_process_models", "speed")
    return (cf_model_manager,)


@app.cell
def _(np):
    hessians_filepath = "model_experiments/2025-12-04-collisions_only/gaussian_process_models/coherency_fraction/log_hessian_dataset.npy"
    hessians = np.load(hessians_filepath)

    inputs_filepath = "model_experiments/2025-12-04-collisions_only/gaussian_process_models/coherency_fraction/hessian_inputs.npy"
    sampled_inputs = np.load(inputs_filepath)
    return hessians, sampled_inputs


@app.cell
def _(cf_model_manager, emulate, plt, sampled_inputs):
    sampled_outputs, _ = emulate(cf_model_manager, sampled_inputs)
    plt.hist(sampled_outputs, bins=50);
    plt.show()
    return (sampled_outputs,)


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
    return (sampled_eigenvectors,)


@app.cell
def _(np, sampled_eigenvectors):
    eig_distance_matrix = (1 - np.abs(sampled_eigenvectors[:, 0, :] @ sampled_eigenvectors[:, 0, :].T))
    return (eig_distance_matrix,)


@app.cell
def _(np, scipy, torch):
    def sammon_mapping_simulated_annealing(inputs, eigenvectors, steps, initial_temperature, cooling_rate):
        # Precompute necessary distances, indices etc.
        rng = np.random.default_rng(0)
        data_size = inputs.shape[0]
        eig_distance_matrix = (1 - np.abs(eigenvectors @ eigenvectors.T))
        upper_indices = torch.triu_indices(data_size, data_size, offset=1)

        # Initialise the transform (our model state):
        linear_transform = rng.normal(0, 1, (10, 2))

        # Get initial Sammon stress:
        transformed_data = inputs @ linear_transform
        transform_distance_matrix = scipy.spatial.distance.cdist(transformed_data, transformed_data)
        triu_eig = eig_distance_matrix[upper_indices[0], upper_indices[1]]
        triu_transform = transform_distance_matrix[upper_indices[0], upper_indices[1]]
        stress = np.mean(((triu_eig - triu_transform) ** 2) / triu_eig) / np.sum(triu_eig)

        # Iterate through solution:
        stress_history = []
        for step_index in range(steps):
            # Record stress history:
            stress_history.append(np.copy(stress))

            # Get current temperature:
            current_temperature = initial_temperature * np.exp(-cooling_rate * step_index)

            # Create candidate solution:
            component_updates = rng.normal(0, 0.1, (10, 2))
            component_mask = rng.binomial(1, 0.2, (10, 2))
            component_updates[component_mask] = 0
            candidate_solution = linear_transform + component_updates

            # Get stress:
            transformed_data = inputs @ candidate_solution
            transform_distance_matrix = scipy.spatial.distance.cdist(transformed_data, transformed_data)
            triu_transform = transform_distance_matrix[upper_indices[0], upper_indices[1]]
            candidate_stress = np.mean(((triu_eig - triu_transform) ** 2) / triu_eig) / np.sum(triu_eig)

            # If stress is lower, immediately accept candidate:
            if candidate_stress < stress:
                linear_transform = candidate_solution
                stress = candidate_stress
                continue

            # Otherwise calculate acceptance probability based on temperature:
            else:
                acceptance_probability = np.exp(-(candidate_stress - stress) / current_temperature)
                sample = rng.uniform()
                if sample < acceptance_probability:
                    linear_transform = candidate_solution
                    stress = candidate_stress

        # Return iterated solutions:
        return candidate_solution, stress_history
    return (sammon_mapping_simulated_annealing,)


@app.cell
def _(np):
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
def _(eig_distance_matrix, wsp_space_filling_design):
    selected_points = wsp_space_filling_design(0.05, 1, eig_distance_matrix)
    return (selected_points,)


@app.cell
def _(eig_distance_matrix, np, selected_points):
    def get_pairs():
        # Get minimum after distance to self:
        dist_matrix = np.copy(eig_distance_matrix)
        np.fill_diagonal(dist_matrix, np.nan)
        pair_points = np.nanargmin(dist_matrix[selected_points, :], axis=1)
        return pair_points

    pair_points = get_pairs()
    return (pair_points,)


@app.cell
def _(np, pair_points, selected_points):
    wsp_pairs = np.unique(np.concatenate([selected_points, pair_points]))
    return (wsp_pairs,)


@app.cell
def _(wsp_pairs):
    len(wsp_pairs)
    return


@app.cell
def _(
    sammon_mapping_simulated_annealing,
    sampled_eigenvectors,
    sampled_inputs,
    wsp_pairs,
):
    transform, stress_history = sammon_mapping_simulated_annealing(
        sampled_inputs[wsp_pairs], sampled_eigenvectors[wsp_pairs, 0, :],
        2500, 0.1, 0.01
    )
    return stress_history, transform


@app.cell
def _(stress_history):
    stress_history[-3]
    return


@app.cell
def _(plt, stress_history):
    plt.plot(stress_history)
    return


@app.cell
def _(transform):
    transform.T
    return


@app.cell
def _(cc, np, plt, sampled_inputs, sampled_outputs, transform):
    def plot_sammon_transformation(transform):
        sammon_transformed_data = sampled_inputs @ transform

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(
            sammon_transformed_data[:, 0], sammon_transformed_data[:, 1],
            s=1, c=sampled_outputs, alpha=0.5, cmap=cc.m_CET_R1,
            vmin=np.quantile(sampled_outputs, 0.02),
            vmax=np.quantile(sampled_outputs, 0.98)
        )
        # ax.set_aspect("equal")
        plt.show()

    plot_sammon_transformation(transform)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
