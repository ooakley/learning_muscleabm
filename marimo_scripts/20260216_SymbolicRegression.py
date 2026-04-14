import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    # Needs to be imported first for memory management with Julia to not break:
    # import pysr

    import os
    import json

    import gpytorch
    import torch

    import numpy as np
    import matplotlib.pyplot as plt

    from torch.utils.data import TensorDataset, DataLoader
    from dppy.finite_dpps import FiniteDPP

    import colorcet as cc

    torch.set_default_dtype(torch.float64)
    return DataLoader, TensorDataset, gpytorch, json, np, os, plt, torch


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


@app.cell
def _(torch):
    def emulate(manager, x):
        tensor_input = torch.tensor(x, dtype=torch.float32)
        if len(tensor_input.shape) == 1:
            tensor_input = torch.unsqueeze(tensor_input, 0)
        with torch.no_grad():
            prediction = manager.likelihood(manager.model(tensor_input))
            prediction_mean = prediction.mean.detach().numpy()
            prediction_variance = prediction.variance.detach().numpy()
        return prediction_mean, prediction_variance
    return (emulate,)


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
    return (cf_model_manager,)


@app.cell
def _(np):
    hessians_filepath = "model_experiments/2025-12-04-collisions_only/gaussian_process_models/coherency_fraction/log_hessian_dataset.npy"
    hessians = np.load(hessians_filepath)

    inputs_filepath = "model_experiments/2025-12-04-collisions_only/gaussian_process_models/coherency_fraction/hessian_inputs.npy"
    sampled_inputs = np.load(inputs_filepath)
    return hessians, sampled_inputs


@app.cell
def _(np, sampled_inputs):
    low_trim_mask = ~np.any(sampled_inputs < 0.05, axis=1)
    high_trim_mask = ~np.any(sampled_inputs > 0.95, axis=1)
    trim_mask = np.logical_and(low_trim_mask, high_trim_mask)
    print(np.count_nonzero(trim_mask))
    return (trim_mask,)


@app.cell
def _(cf_model_manager, emulate, sampled_inputs):
    sampled_outputs, _ = emulate(cf_model_manager, sampled_inputs)
    return (sampled_outputs,)


@app.cell
def _(plt, sampled_outputs):
    plt.hist(sampled_outputs);
    plt.show()
    return


@app.cell
def _(np, sampled_outputs):
    output_lq = np.quantile(sampled_outputs, 0.01)
    output_uq = np.quantile(sampled_outputs, 0.99)

    rng = np.random.default_rng(0)
    output_resampling = rng.uniform(output_lq, output_uq, size=10000)
    return output_lq, output_resampling


@app.cell
def _(output_resampling):
    output_resampling
    return


@app.cell
def _(output_lq):
    output_lq
    return


@app.cell
def _(output_resampling, plt, sampled_outputs):
    plt.hist(output_resampling, alpha=0.5, density=True, bins=25);
    plt.hist(sampled_outputs, alpha=0.5, density=True, bins=50);
    plt.show()
    return


@app.cell
def _(np, output_resampling, sampled_outputs):
    resampled_indices = np.argmin(np.abs(output_resampling[:, np.newaxis] - sampled_outputs[np.newaxis, :]), axis=1)
    return (resampled_indices,)


@app.cell
def _(plt, resampled_indices, sampled_outputs):
    plt.hist(sampled_outputs[resampled_indices], alpha=0.5, density=True, bins=25);
    plt.show()
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
def _(np, plt, sampled_eigenvalues):
    magnitude_mask = np.log(sampled_eigenvalues[:, 0]) > 0
    plt.hist(np.log(sampled_eigenvalues[:, 0]), bins=100);
    plt.show()
    return


@app.cell
def _(np, plt, sampled_eigenvalues, sampled_inputs, sampled_outputs):
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

        # ax.scatter(
        #     eigenvectors[sampled_dpp, parameter_i],
        #     eigenvectors[sampled_dpp, parameter_j],
        #     s=1, alpha=0.5, c="r"
        # )

        theta = np.linspace(0, np.pi * 2, 100)
        ax.plot(1.025*np.cos(theta), 1.025*np.sin(theta), c='k', ls="--",  lw=1, alpha=0.5)
        format_axes(ax)
        # plt.show()

    def plot_linear_combinations(vectors):
        fig, axs = plt.subplots(10, 10, figsize=(10, 10))
        for i in range(10):
            for j in range(10):
                if i == j:
                    axs[i, j].text(0, 0, "Placeholder", fontsize=5, horizontalalignment="center")
                    format_axes(axs[i, j])
                    continue
                if i > j:
                    plot_eigendirections(sampled_inputs, vectors, i, j, ax=axs[i, j], s=0.1, c=np.log(sampled_eigenvalues[:, 0]))
                if i < j:
                    plot_eigendirections(sampled_inputs, vectors, i, j, ax=axs[i, j], s=0.1, c=sampled_outputs)

        fig.tight_layout()
        plt.show()
    return (plot_linear_combinations,)


@app.cell
def _(plot_linear_combinations, sampled_eigenvectors):
    plot_linear_combinations(sampled_eigenvectors[:, 0, :])
    return


@app.cell
def _():
    # eigenparameter_values = np.sum(np.log(sampled_inputs)[:, :, np.newaxis] * sampled_eigenvectors[:, 0, :].T, axis=1)
    return


@app.cell
def _():
    # ep_correlation_matrix = np.corrcoef(eigenparameter_values, rowvar=False)
    return


@app.cell
def _(ep_correlation_matrix, plt):
    plt.hist(ep_correlation_matrix.flatten(), bins=100, density=True);
    plt.show()
    return


@app.cell
def _(eigenparameter_values, plt):
    plt.scatter(eigenparameter_values[:, 0], eigenparameter_values[:, 1], s=1)
    return


@app.cell
def _():
    # |# Get likelihood matrix:
    # print("Getting squared exponential likelihood matrix...", flush=True)
    # # distance_matrix = 1 - np.abs(sampled_eigenvectors[trim_mask, 0, :] @ sampled_eigenvectors[trim_mask, 0, :].T)
    # # distance_matrix = np.abs(sampled_outputs[:, np.newaxis] - sampled_outputs[np.newaxis, :])
    # distance_matrix = 1 - np.abs(ep_correlation_matrix)

    # likelihood_matrix = np.exp(distance_matrix ** 2)

    # # Set up determinantal point process:
    # print("Setting up point process...")
    # DPP = FiniteDPP('likelihood', **{'L': likelihood_matrix})

    # k = 3
    # sampled_dpp = DPP.sample_mcmc_k_dpp(size=k, random_state=0)
    return


@app.cell
def _(sampled_eigenvectors):
    dpp_eigenparameters = sampled_eigenvectors[0:5, 0, :]
    return (dpp_eigenparameters,)


@app.cell
def _(dpp_eigenparameters, np, sampled_inputs):
    dpp_ep_values = np.sum(np.log(sampled_inputs)[:, :, np.newaxis] * dpp_eigenparameters.T, axis=1)
    return (dpp_ep_values,)


@app.cell
def _(dpp_ep_values):
    dpp_ep_values
    return


@app.cell
def _(dpp_ep_values, plt, trim_mask):
    plt.scatter(dpp_ep_values[trim_mask, 0], dpp_ep_values[trim_mask, 1], s=1)
    return


@app.cell
def _(dpp_ep_values, plt, sampled_outputs, trim_mask):
    def plot_eigenparameter(index):
        # Plot binscatter:
        fig, ax = plt.subplots()
        ax.scatter(dpp_ep_values[trim_mask, index], sampled_outputs[trim_mask], s=1, alpha=1)

        # ventiles = np.linspace(0, 1, 21)
        # bin_x = []
        # bin_y = []
        # for index in range(20):
        #     lq = ventiles[index]
        #     uq = ventiles[index + 1]
        #     lb = np.quantile(eigenparameter_values, lq)
        #     ub = np.quantile(eigenparameter_values, uq)
        #     bin_x.append((lb + ub) / 2)

        #     param_mask = np.logical_and(eigenparameter_values > lb, eigenparameter_values < ub)
        #     bin_y.append(np.mean(sampled_outputs[param_mask]))

        # ax.scatter(bin_x, bin_y, s=1)
        # ax.plot(bin_x, bin_y)
        # ax.set_xlabel(f"Cluster {cluster_index} Eigenparameter")
        # ax.set_ylabel("ANNI")
        plt.show()
    return (plot_eigenparameter,)


@app.cell
def _(plot_eigenparameter):
    plot_eigenparameter(3)
    return


@app.cell
def _():
    # from sklearn.linear_model import LinearRegression

    # regression = LinearRegression().fit(eigenparameter_values, sampled_outputs)
    return


@app.cell
def _():
    # predictions = regression.predict(eigenparameter_values)
    return


@app.cell
def _():
    # def plot_linear_regression():
    #     fig, ax = plt.subplots()
    #     ax.scatter(sampled_outputs, predictions, s=1)
    #     ax.set_aspect("equal")
    #     plt.show()

    # plot_linear_regression()
    return


@app.cell
def _(sampled_inputs, sampled_outputs, torch):
    def optimise_eigenparameters(pytorch_func, eigenparameters):
        torch_eigenparameters = torch.from_numpy(eigenparameters)
        torch_eigenparameters.requires_grad_()
        torch_inputs = torch.from_numpy(sampled_inputs)

        # Calculate eigenparameter values:
        for _ in range(5):
            torch_evs = torch.sum(torch.log(torch_inputs)[:, :, None] * torch_eigenparameters[:, :].T, dim=1)

            # Estimate outputs with symbolic regression function:
            outputs = pytorch_func(torch_evs)
            loss = torch.mean((outputs - torch.from_numpy(sampled_outputs)) ** 2)
            print(loss.item())

            eigenparameter_gradient = torch.autograd.grad(loss, torch_eigenparameters)
            print(eigenparameter_gradient[0].shape)
            with torch.no_grad():
                torch_eigenparameters -= 0.001*eigenparameter_gradient

        return torch_eigenparameters.detach().numpy()
    return


@app.cell
def _(resampled_indices, sampled_inputs, sampled_outputs):
    import gplearn.genetic
    # import gplearn.functions

    # def gp_pow(a, b):
    #     out = np.sign(a) * (np.abs(a) ** b)
    #     if np.any(np.isnan(out)):
    #         return np.ones_like(out)
    #     else:
    #         return out

    # wrapped_pow = gplearn.functions.make_function(
    #     function=gp_pow,
    #     name='gp_pow',
    #     arity=2
    # )

    function_set = [
        'add', 'sub', 'mul', 'div', 'inv', 'neg', 'log'
    ]

    gp = gplearn.genetic.SymbolicRegressor(
        generations=50, population_size=3000,
        tournament_size=10,
        p_crossover=0.85,
        p_hoist_mutation=0.05,
        p_subtree_mutation=0.05,
        p_point_mutation=0.05, metric='mse',
        # hall_of_fame=100, n_components=5,
        function_set=function_set,
        parsimony_coefficient=5e-3,
        verbose=1,
        random_state=0, n_jobs=1
    )

    gp.fit(sampled_inputs[resampled_indices], sampled_outputs[resampled_indices])
    return (gp,)


@app.cell
def _(gp):
    print(gp._program)
    return


@app.cell
def _(gp, sampled_inputs):
    predicted_outputs = gp.predict(sampled_inputs)
    return (predicted_outputs,)


@app.cell
def _(np, plt, predicted_outputs, sampled_outputs, trim_mask):
    def plot_sr_preds():
        fig, ax = plt.subplots()
        lb = np.min(sampled_outputs[trim_mask])
        ub = np.max(sampled_outputs[trim_mask])
        ax.scatter(sampled_outputs[trim_mask], predicted_outputs[trim_mask], s=1)
        ax.set_xlim(lb, ub)
        ax.plot([lb, ub], [lb, ub], c='r')
        ax.set_aspect("equal")
        plt.show()

    plot_sr_preds()
    return


@app.cell
def _(plt, transformed_inputs, trim_mask):
    plt.scatter(transformed_inputs[trim_mask, 0], transformed_inputs[trim_mask, 1], s=1)
    return


@app.cell
def _(plt, sampled_outputs, transformed_inputs, trim_mask):
    plt.scatter(transformed_inputs[trim_mask, 0], sampled_outputs[trim_mask], s=1)
    return


@app.cell
def _(gp):
    gp.get_params()
    return


@app.cell
def _():
    # sr_model = pysr.PySRRegressor(
    #     binary_operators=["+", "*", "-", "^"],
    #     unary_operators=[
    #         "exp",
    #         "log",
    #         "inv"
    #     ],
    #     constraints={
    #         "^": (-1, 1),
    #         "exp": 5,
    #         "log": 5,
    #     },
    #     niterations=100,
    #     warm_start=True
    # )
    return


@app.cell
def _():
    # sr_model.fit(eigenparameter_values[trim_mask][::7, :], sampled_outputs[trim_mask][::7])
    return


@app.cell
def _():
    # import copy

    # iterated_eigenparameters = copy.deepcopy(eigenparameters)

    # for step_index in range(5):
    #     print(step_index)

    #     # Generate eigenparameters:
    #     evs = np.sum(np.log(sampled_inputs)[:, :, np.newaxis] * iterated_eigenparameters[:, :].T, axis=1)

    #     # Fit model to these eigenparameters:
    #     sr_model.fit(evs[::9, :], sampled_outputs[::9]);

    #     # Get pytorch function:
    #     pytorch_func = sr_model.pytorch(-1)

    #     # Iterate over eigenparameters:
    #     iterated_eigenparameters = optimise_eigenparameters(pytorch_func, iterated_eigenparameters)
    return


@app.cell
def _():
    # sr_model.fit(sampled_inputs[::9, :], sampled_outputs[::9]);
    return


@app.cell
def _(sr_model):
    print(sr_model)
    return


@app.cell
def _(eigenparameter_values, plt, sampled_outputs, sr_model):
    def plot_sr_outputs():
        fig, ax = plt.subplots()
        ax.scatter(sampled_outputs, sr_model.predict(eigenparameter_values, -1), s=1)
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.set_aspect("equal")
        plt.show()

    plot_sr_outputs()
    return


@app.cell
def _(sr_model):
    sr_model.sympy(-1)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
