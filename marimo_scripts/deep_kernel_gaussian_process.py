import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


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
    return (
        DataLoader,
        FiniteDPP,
        TensorDataset,
        gpytorch,
        json,
        np,
        os,
        plt,
        torch,
    )


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
        speeds = np.load(
            os.path.join(experiment_folderpath, "summary_data", "magnitude_cellmeans.npy")
        )

        # Get gridsearch configuration:
        with open(os.path.join(experiment_folderpath, "config.json")) as json_filestream:
            config_dictionary  = json.load(json_filestream)
        gridsearch_parameters = config_dictionary["gridsearch_parameters"]

        return parameter_matrix, coherency_fractions, ann_indices, speeds, gridsearch_parameters
    return (load_gridsearch_data,)


@app.cell
def _(load_gridsearch_data):
    parameter_matrix, coherency_fractions, ann_indices, speeds, gridsearch_parameters = load_gridsearch_data(
        "model_experiments/2025-12-04-collisions_only"
    )
    return ann_indices, coherency_fractions, parameter_matrix, speeds


@app.cell
def _(torch):
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
    return (DeepInputTransformation,)


@app.cell
def _(DeepInputTransformation, gpytorch):
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
                gpytorch.kernels.RBFKernel(ard_num_dims=None)
            )

        def forward(self, x):
            # Warp input:
            warped_x = self.input_transform(x)

            # Calculate mean of input:
            mean_x = self.mean_module(warped_x)
            covar_x = self.covar_module(warped_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    return (SparseGPModel,)


@app.cell
def _(DataLoader, SparseGPModel, TensorDataset, gpytorch, os, torch):
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
            x_tensor = torch.tensor(x)
            y_tensor = torch.tensor(y)
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
    return (ModelManager,)


@app.cell
def _(DataLoader, TensorDataset, np, torch):
    def run_inference(model, likelihood, inputs, batch_size=512):
        # Set up dataloading:
        tensor_input = torch.tensor(inputs)
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
    return (run_inference,)


@app.cell
def _(np, plt):
    def plot_variance(data_variance, predicted_variance):
        fig, ax = plt.subplots()
        ax.scatter(data_variance, predicted_variance, s=0.01)
        # Plot linear trend:
        upper_extent = np.max([data_variance, predicted_variance])
        lower_extent = np.min([data_variance, predicted_variance])
        ax.plot(
            [lower_extent, upper_extent],
            [lower_extent, upper_extent],
            c='r', alpha=0.5
        )
        ax.set_xlim(lower_extent, upper_extent)
        ax.set_ylim(lower_extent, upper_extent)
        ax.set_aspect("equal")
        plt.show()

    def plot_gp_mean_predictions(data_mean, predicted_mean):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(
            data_mean,
            predicted_mean,
            s=0.1
        )
        # Plot linear trend:
        upper_extent = np.max([data_mean, predicted_mean])
        lower_extent = np.min([data_mean, predicted_mean])
        ax.plot(
            [lower_extent, upper_extent],
            [lower_extent, upper_extent],
            c='r', alpha=0.5
        )
        ax.set_xlim(lower_extent, upper_extent)
        ax.set_ylim(lower_extent, upper_extent)
        ax.set_aspect("equal")
        plt.show()
    return plot_gp_mean_predictions, plot_variance


@app.cell
def _(FiniteDPP, np, parameter_matrix):
    # Get likelihood matrix:
    print("Getting squared exponential likelihood matrix...", flush=True)
    distance_matrix = 1 - np.matmul(parameter_matrix[::7, :], parameter_matrix[::7, :].T).astype(np.float32)
    likelihood_matrix = np.exp(distance_matrix ** 2)

    # Set up determinantal point process:
    print("Setting up point process...")
    DPP = FiniteDPP('likelihood', **{'L': likelihood_matrix})

    k = 1024
    DPP.sample_mcmc_k_dpp(size=k, random_state=None)

    # Get inducing points:
    inducing_indices = DPP.list_of_samples[0][-1]
    inducing_points = parameter_matrix[::7, :][inducing_indices, :]
    return (inducing_points,)


@app.cell
def _(
    ModelManager,
    coherency_fractions,
    inducing_points,
    np,
    parameter_matrix,
):
    # Get information necessary to transform GP inputs/outputs:
    CF_DIST_MEAN = np.mean(np.mean(coherency_fractions, axis=1))
    CF_DIST_STD = np.std(np.mean(coherency_fractions, axis=1))
    whitened_cf = (np.mean(coherency_fractions, axis=1) - CF_DIST_MEAN) / CF_DIST_STD
    cf_model_manager = ModelManager(inducing_points, 0.003)
    # cf_model_manager.load("model_experiments/2025-12-04-collisions_only/gaussian_process_models", "coherency_fraction")
    cf_model_manager.train(parameter_matrix, whitened_cf, 512, epochs=15)
    cf_model_manager.save("model_experiments/2025-12-04-collisions_only/gaussian_process_models", "coherency_fraction")
    return CF_DIST_MEAN, CF_DIST_STD, cf_model_manager


@app.cell
def _(cf_model_manager):
    # 7.7375e-05
    print(f'Actual noise value: {cf_model_manager.likelihood.noise}')
    return


@app.cell
def _(cf_model_manager, parameter_matrix, run_inference):
    cf_predictions, cf_variance = run_inference(cf_model_manager.model, cf_model_manager.likelihood, parameter_matrix, 512)
    return cf_predictions, cf_variance


@app.cell
def _(
    CF_DIST_MEAN,
    CF_DIST_STD,
    cf_predictions,
    coherency_fractions,
    np,
    plot_gp_mean_predictions,
):
    plot_gp_mean_predictions(np.mean(coherency_fractions, axis=1), (cf_predictions * CF_DIST_STD) + CF_DIST_MEAN)
    return


@app.cell
def _(CF_DIST_STD, cf_variance, coherency_fractions, np, plot_variance):
    plot_variance(np.std(coherency_fractions, axis=1) / np.sqrt(32), np.sqrt(cf_variance) * CF_DIST_STD)
    return


@app.cell
def _(ModelManager, ann_indices, inducing_points, np, parameter_matrix):
    ANNI_DIST_MEAN = np.mean(np.mean(ann_indices, axis=1))
    ANNI_DIST_STD = np.std(np.mean(ann_indices, axis=1))
    whitened_anni = (np.mean(ann_indices, axis=1) - ANNI_DIST_MEAN) / ANNI_DIST_STD
    anni_model_manager = ModelManager(inducing_points, 0.003)
    anni_model_manager.train(parameter_matrix, whitened_anni, 512, epochs=15)
    anni_model_manager.save("model_experiments/2025-12-04-collisions_only/gaussian_process_models", "ann_index")
    return


@app.cell
def _(ModelManager, inducing_points, np, parameter_matrix, speeds):
    SPEED_DIST_MEAN = np.mean(speeds[:, 0])
    SPEED_DIST_STD = np.std(speeds[:, 0])
    whitened_speed = (speeds[:, 0] - SPEED_DIST_MEAN) / SPEED_DIST_STD
    speed_model_manager = ModelManager(inducing_points, 0.003)
    speed_model_manager.train(parameter_matrix, whitened_speed, 512, epochs=15)
    speed_model_manager.save("model_experiments/2025-12-04-collisions_only/gaussian_process_models", "speed")
    return (speed_model_manager,)


@app.cell
def _(speed_model_manager):
    print(f'Actual noise value: {speed_model_manager.likelihood.noise}')
    return


@app.cell
def _(np, speeds):
    simulation_means = speeds[:, 0]
    target_mean = np.mean(simulation_means)
    target_stddev = np.std(simulation_means)
    normalised_data = (simulation_means - np.mean(simulation_means)) / np.std(simulation_means)
    return normalised_data, target_mean, target_stddev


@app.cell
def _(normalised_data, plt):
    plt.hist(normalised_data, bins=250);
    plt.show()
    return


@app.cell
def _(model_manager, normalised_data, parameter_matrix):
    model_manager.train(parameter_matrix, normalised_data, 512, epochs=20)
    return


@app.cell
def _(model_manager):
    print(f'Actual noise value: {model_manager.likelihood.noise}')
    print(f'Noise constraint: {model_manager.likelihood.noise_covar.raw_noise_constraint}')
    return


@app.cell
def _(model_manager):
    model_manager.save("model_experiments/2025-12-04-collisions_only/gaussian_process_models", "speed")
    return


@app.cell
def _():
    # model_manager.load("model_experiments/2025-12-04-collisions_only/gaussian_process_models", "test")
    return


@app.cell
def _(model_manager):
    model_manager.likelihood
    return


@app.cell
def _(model_manager, plt):
    plt.plot(model_manager.loss_history)
    return


@app.cell
def _(cf_model_manager, parameter_matrix, torch):
    warped_matrix = cf_model_manager.model.input_transform(torch.from_numpy(parameter_matrix))
    return (warped_matrix,)


@app.cell
def _(np, plt, warped_matrix):
    def plot_warping():
        fig, ax = plt.subplots()
        for i in range(warped_matrix.shape[1]):
            ax.hist(warped_matrix.detach().numpy()[:, i], bins=100, alpha=0.5, density=True);
            lq, uq = np.quantile(warped_matrix.detach().numpy()[:, i], [0.05, 0.95])
            print(uq - lq)
        plt.show()

    plot_warping()
    return


@app.cell
def _(cf_model_manager, plt, torch, warped_matrix):
    learnt_inducing_points = cf_model_manager.model.variational_strategy.inducing_points.detach().numpy()

    def plot_inducing_points(points):
        fig, ax = plt.subplots(figsize=(7, 7))
        x_axis_index = 0
        y_axis_index = 6

        # Plot dataset:
        ax.scatter(
            warped_matrix.detach().numpy()[:, x_axis_index],
            warped_matrix.detach().numpy()[:, y_axis_index],
            s=0.01
        )

        # Plot inducing points:
        points = cf_model_manager.model.input_transform(torch.from_numpy(points))
        points = points.detach().numpy()
        ax.scatter(points[:, x_axis_index], points[:, y_axis_index], s=1)
        # # Plot bounding area for input space:
        # ax.hlines(0, -0.05, 1.05, alpha=0.5)
        # ax.hlines(1, -0.05, 1.05, alpha=0.5)
        # ax.vlines(0, -0.05, 1.05, alpha=0.5)
        # ax.vlines(1, -0.05, 1.05, alpha=0.5)
        # ax.set_xlim(-0.05, 1.05)
        # ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        plt.show()

    plot_inducing_points(learnt_inducing_points)
    return


@app.cell
def _(plt, predictions):
    plt.hist(predictions, bins=100);
    plt.show()
    return


@app.cell
def _(np, plt, variance):
    plt.hist(np.sqrt(variance), bins=100);
    plt.show()
    return


@app.cell
def _(coherency_fractions, np, predictions):
    mae = np.mean(np.abs(np.mean(coherency_fractions, axis=1) - predictions))
    mae
    return


@app.cell
def _(np):
    WT_1_CF_MEAN = 0.03646833938262533
    WT_1_CF_VAR = (np.sqrt(1.289952251152215e-05) / np.sqrt(12)) ** 2
    return WT_1_CF_MEAN, WT_1_CF_VAR


@app.cell
def _(
    WT_1_CF_MEAN,
    WT_1_CF_VAR,
    np,
    predictions,
    target_mean,
    target_stddev,
    variance,
):
    rescaled_predictions = (predictions * target_stddev) + target_mean
    rescaled_variance = (np.sqrt(variance) * target_stddev)**2
    implausibility = np.abs(rescaled_predictions - WT_1_CF_MEAN) / np.sqrt(WT_1_CF_VAR + rescaled_variance)
    return (implausibility,)


@app.cell
def _(WT_1_CF_VAR, np, variance):
    np.sqrt(WT_1_CF_VAR + variance)
    return


@app.cell
def _(variance):
    variance
    return


@app.cell
def _(implausibility, plt):
    plt.hist(implausibility, bins=100);
    plt.show()
    return


@app.cell
def _(implausibility, np):
    np.count_nonzero(implausibility < 3)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
