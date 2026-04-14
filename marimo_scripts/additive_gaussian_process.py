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
    return


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
        "model_experiments/2025-12-04-collisions_only"
    )
    return coherency_fractions, parameter_matrix


@app.cell
def _(coherency_fractions, np):
    import scipy.stats
    ecdf_result = scipy.stats.ecdf(np.mean(coherency_fractions, axis=1))
    return ecdf_result, scipy


@app.cell
def _(coherency_fractions, ecdf_result, np):
    uniformly_distributed_cf = ecdf_result.cdf.evaluate(np.mean(coherency_fractions, axis=1))
    inverse_transform_cf = np.quantile(np.mean(coherency_fractions, axis=1), uniformly_distributed_cf)
    return (uniformly_distributed_cf,)


@app.cell
def _(scipy):
    standard_normal = scipy.stats.norm()
    return (standard_normal,)


@app.cell
def _(standard_normal, uniformly_distributed_cf):
    normal_cf_values = standard_normal.ppf(uniformly_distributed_cf / (1 + 1e-5))
    return (normal_cf_values,)


@app.cell
def _(normal_cf_values, standard_normal):
    denormalised_values = standard_normal.cdf(normal_cf_values)
    return (denormalised_values,)


@app.cell
def _(denormalised_values, plt):
    plt.hist(denormalised_values);
    plt.show()
    return


@app.cell
def _(torch):
    class WarpingTransformation(torch.nn.Module):
        def __init__(self, sum_count=10):
            # Parameterised monotonic function:
            super().__init__()
            self.a_raw = torch.nn.Parameter(torch.rand(sum_count))
            self.b_raw = torch.nn.Parameter(torch.rand(sum_count))
            self.c = torch.nn.Parameter(torch.rand(sum_count))

        def forward(self, x):
            # Generate batch dimension:
            batch_x = torch.unsqueeze(x, dim=1)
            a = torch.exp(self.a_raw)
            b = torch.exp(self.b_raw)
            tanh_array = a * torch.tanh(b * (batch_x + self.c))
            return torch.squeeze(batch_x + torch.sum(tanh_array, dim=1, keepdim=True))
    return


@app.cell
def _():
    return


@app.cell
def _(np, plt, torch):
    class InputTransformation(torch.nn.Module):
        def __init__(self, dimension, sum_count=50):
            # Record parameters:
            self.dimension = dimension
            self.sum_count = sum_count

            # Parameterised monotonic function:
            super().__init__()
            self.a_raw = torch.nn.Parameter(torch.log(torch.rand((dimension, sum_count))))
            self.b_raw = torch.nn.Parameter(torch.log(torch.rand((dimension, sum_count))))
            self.c = torch.nn.Parameter(torch.rand((dimension, sum_count)))

        def forward(self, x):
            # Generate batch dimension:
            x = torch.unsqueeze(x, dim=-1)
            a = torch.exp(self.a_raw)
            b = torch.exp(self.b_raw)
            tanh_array = a * torch.tanh(b * (x - self.c))
            return torch.squeeze(x + torch.sum(tanh_array, dim=-1, keepdim=True))

        def plot_transform(self, index=0):
            with torch.no_grad():
                # Get input space:
                x = np.linspace(0, 1, 100)
                x = np.expand_dims(x, 1)

                # Get warp parameters:
                a = np.exp(self.a_raw.detach().numpy()[index, :])
                b = np.exp(self.b_raw.detach().numpy()[index, :])
                c = self.c.detach().numpy()[index, :]

                tanh_array = a * np.tanh(b * (x - c))
                warped_x = np.squeeze(x + np.sum(tanh_array, axis=1, keepdims=True))

                plt.plot(warped_x)
                plt.show()
    return (InputTransformation,)


@app.cell
def _(torch):
    class DeepInputTransformation(torch.nn.Module):
        def __init__(self, dimension, hidden_layer_neuron_count=50):
            # Run general initialisation of the nn.Module base class:
            super().__init__()

            # Record parameters:
            self.dimension = dimension
            self.hidden_layer_neuron_count = hidden_layer_neuron_count

            # Set up layers:
            self.linear_layer_one = torch.nn.Linear(dimension, hidden_layer_neuron_count)
            torch.nn.init.xavier_normal_(self.linear_layer_one.weight)
            self.linear_layer_two = torch.nn.Linear(hidden_layer_neuron_count, dimension)
            torch.nn.init.xavier_normal_(self.linear_layer_two.weight)
            self.activation = torch.nn.ReLU()

        def forward(self, x):
            # Generate batch dimension:
            return self.linear_layer_two(self.activation(self.linear_layer_one(x)))
    return (DeepInputTransformation,)


@app.cell
def _(InputTransformation):
    test_transform = InputTransformation(10)
    for _ in range(10):
        test_transform.plot_transform(_)
    return


@app.cell
def _(DeepInputTransformation, gpytorch, torch):
    class SparseAdditiveGPModel(gpytorch.models.ApproximateGP):
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
            super(SparseAdditiveGPModel, self).__init__(variational_strategy)

            # Instantiate input transform:
            self.input_transform = DeepInputTransformation(dimensions)

            # Define mean and additive covariance functions:
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = \
                gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(batch_shape=torch.Size([dimensions]), ard_num_dims=None)
                ) \
                + gpytorch.kernels.ConstantKernel()

        def forward(self, x):
            # Warp input:
            warped_x = self.input_transform(x)

            # Calculate mean of input:
            mean_x = self.mean_module(warped_x)

            # Calculate interaction covariance matrix:
            batched_dimensions_of_x = warped_x.mT.unsqueeze(-1)  # Now a d x n x 1 tensor
            univariate_covars = self.covar_module(batched_dimensions_of_x)
            covar_x = gpytorch.utils.sum_interaction_terms(
                univariate_covars, max_degree=2, dim=-3
            )
            conditioned_covar_x = gpytorch.add_jitter(covar_x,  jitter_val=0.003)

            return gpytorch.distributions.MultivariateNormal(mean_x, conditioned_covar_x)
    return (SparseAdditiveGPModel,)


@app.cell
def _(DataLoader, SparseAdditiveGPModel, TensorDataset, gpytorch, torch):
    def instantiate_model(inducing_points):
        # Convert inducing points to torch:
        inducing_points = torch.tensor(
            inducing_points, dtype=torch.float32
        )

        # Set up likelihoods:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = SparseAdditiveGPModel(inducing_points, inducing_points.shape[1])
        return model, likelihood

    def train_model(model, likelihood, x_dataset, y_means, epochs=1, batch_size=64):
        # Convert datasets to torch:
        train_x = torch.tensor(x_dataset, dtype=torch.float32)
        y_means = torch.tensor(y_means, dtype=torch.float32)

        # Initialise dataloaders:
        train_dataset = TensorDataset(train_x, y_means)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Set up training process:
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.01)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 512
        )

        # Set up loss:
        mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=y_means.size(0))
        loss_history = []
        for i in range(epochs):
            # Run through entire dataset:
            for batch_index, (x_batch, y_mean_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                output_distribution = model(x_batch)
                loss = -mll(output_distribution, y_mean_batch)
                loss.backward()

                # Step through optimisers:
                optimizer.step()
                # scheduler.step()
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
    return instantiate_model, train_model


@app.cell
def _(FiniteDPP, np, parameter_matrix):
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
        parameter_matrix,
        np.mean(coherency_fractions, axis=1),
        epochs=5, batch_size=512
    )
    return


@app.cell
def _(model):
    for i in range(10):
        model.input_transform.plot_transform(i)
    return


@app.cell
def _(model, np, parameter_matrix, torch):
    with torch.no_grad():
        # Get univariate function means:
        input_space = torch.from_numpy(np.linspace(0, 1, 200)).to(torch.float32)
        input_space = input_space.unsqueeze(1)
        input_space = input_space.repeat(1, parameter_matrix.shape[1])
        # input_space = torch.from_numpy(parameter_matrix[:100, :])

        # Calculate mean kernel value:
        output_distributions = model(input_space, prior=False)

        # # Calculate interaction covariance matrix:
        # univariate_means = model(input_space)
        # batched_dimensions = input_space.mT.unsqueeze(-1)  # Now a d x n x 1 tensor
        # tensor_covariances = model.covar_module(batched_dimensions)
        # univariate_covariances = gpytorch.add_jitter(tensor_covariances.evaluate()).sum(dim=0)

        # # Get distributions:
        # output_distributions = gpytorch.distributions.MultivariateNormal(univariate_means, univariate_covariances)
    return input_space, output_distributions


@app.cell
def _(input_space, plt):
    plt.plot(input_space[:, 0])
    return


@app.cell
def _(output_distributions, plt):
    plt.plot(output_distributions.mean.detach().numpy())
    return


@app.cell
def _(input_space):
    input_space.shape
    return


@app.cell
def _(model, plt):
    learnt_inducing_points = model.variational_strategy.inducing_points.detach().numpy()

    def plot_inducing_points(points):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(points[:, 0], points[:, 4], s=5)
        # Plot bounding area for input space:
        ax.hlines(0, -0.05, 1.05, alpha=0.5)
        ax.hlines(1, -0.05, 1.05, alpha=0.5)
        ax.vlines(0, -0.05, 1.05, alpha=0.5)
        ax.vlines(1, -0.05, 1.05, alpha=0.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        # ax.set_aspect("equal")
        plt.show()

    plot_inducing_points(learnt_inducing_points)
    return


@app.cell
def _(DataLoader, TensorDataset, np, torch):
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
        stddev_array = []
        with torch.no_grad():
            for batch_index, inference_batch in enumerate(inference_loader):
                inference_batch = inference_batch[0]
                predictions = likelihood(model(inference_batch))
                predictions_array.append(predictions.mean.detach().numpy())
                stddev_array.append(predictions.stddev.detach().numpy())
                if (batch_index + 1) % 100 == 0:
                    print(batch_index + 1)

        return np.concatenate(predictions_array), np.concatenate(stddev_array)**2

    def run_grad_inference(model, likelihood, inputs):
        tensor_input = torch.tensor(inputs, dtype=torch.float32, requires_grad=True)
        tensor_input = torch.unsqueeze(tensor_input, 0)
        model.eval()
        likelihood.eval()
        predictions = likelihood(model(tensor_input))
        return tensor_input, predictions.mean
    return (run_inference,)


@app.cell
def _(likelihood, model, parameter_matrix, run_inference):
    predictions, variance = run_inference(model, likelihood, parameter_matrix, 128)
    return predictions, variance


@app.cell
def _(plt, predictions):
    plt.hist(predictions, bins=100);
    plt.show()
    return


@app.cell
def _(coherency_fractions, np, plt, variance):
    plt.scatter(np.var(coherency_fractions, axis=1), variance);
    plt.show()
    return


@app.cell
def _(parameter_matrix):
    parameter_matrix
    return


@app.cell
def _(coherency_fractions, np, plt, predictions, standard_normal, variance):
    def plot_gp_mean_predictions():
        fig, ax = plt.subplots(figsize=(5, 5))
        uniform_predictions = standard_normal.cdf(predictions) * (1 + 1e-5)
        iecdf_predictions = np.quantile(np.mean(coherency_fractions, axis=1), uniform_predictions)
        ax.scatter(
            np.mean(coherency_fractions, axis=1),
            predictions,
            c=variance,
            s=0.01
        )
        ax.plot([-0.05, 0.16], [-0.05, 0.16], c='r')
        ax.set_xlim(0, 0.16)
        ax.set_ylim(0, 0.16)
        ax.set_aspect("equal")
        plt.show()

    plot_gp_mean_predictions()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
