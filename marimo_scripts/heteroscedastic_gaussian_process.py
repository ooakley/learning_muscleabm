import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    # https://github.com/cornellius-gp/gpytorch/issues/1158#issuecomment-1739668889
    import os
    import json

    import torch
    import pyro
    import gpytorch

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
        pyro,
        torch,
    )


@app.cell
def _():
    WT_1_CF_MEAN = 0.03646833938262533
    WT_1_CF_VAR = 1.289952251152215e-05
    return WT_1_CF_MEAN, WT_1_CF_VAR


@app.cell
def _(np, plt, torch):
    class InputTransformation(torch.nn.Module):
        def __init__(self, dimension, sum_count=100):
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
    return


@app.cell
def _(torch):
    class DeepInputTransformation(torch.nn.Module):
        def __init__(self, dimension, hidden_layer_neuron_count=25):
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
                torch.nn.Linear(self.hl_neuron_count, dimension),
                # torch.nn.Sigmoid()  # Ensures output space remains bounded to [0, 1]
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
                m.weight[diagonal_index_object, diagonal_index_object] = 1
    return (DeepInputTransformation,)


@app.cell
def _(DeepInputTransformation, gpytorch, pyro, torch):
    class HeteroscedasticGP(gpytorch.models.ApproximateGP):

        def __init__(self, inducing_points, name_prefix="heteroskedastic_gp"):
            # Set prefix for pyro work:
            self.name_prefix = name_prefix

            # Stack the inducing points - for both mean and noise estimation:
            stacked_inducing_points = torch.stack([inducing_points, inducing_points], dim=0)

            # We have to mark the CholeskyVariationalDistribution as batch
            # so that we learn a variational distribution for each task (we have 2):
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                num_inducing_points=inducing_points.size(0), 
                batch_shape=torch.Size([2])
            )

            single_variational_strategy = gpytorch.variational.VariationalStrategy(
                self,
                stacked_inducing_points,
                variational_distribution,
                learn_inducing_locations=True
            )

            # Wrap the single variational strategy into a
            # Linear Model of Coregionalization one, so the two
            # tasks (and therefore latent GPs) are assumed to be
            # somehow related (and we learn such relationship):
            variational_strategy = gpytorch.variational.LMCVariationalStrategy(
                single_variational_strategy,
                num_tasks=2,
                num_latents=2
            )

            # Standard initializtation
            super().__init__(variational_strategy)

            # The mean and covariance modules should be marked as batch
            # so we learn a different set of hyperparameters:
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([2]))
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    batch_shape=torch.Size([2]),
                    ard_num_dims=None
                ),
                batch_shape=torch.Size([2])
            )

            # Get basic input warping:
            print(inducing_points.size(1))
            self.input_transform = DeepInputTransformation(inducing_points.size(1))

        def forward(self, x):
            # The forward function should be written as if we were dealing with each output
            # dimension in batch
            warped_x = self.input_transform(x)
            mean_x = self.mean_module(warped_x)
            covar_x = self.covar_module(warped_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        def guide(self, x, y):
            # Get q(f) - variational (guide) distribution of latent function
            function_dist = self.pyro_guide(x)

            # Use a plate here to mark conditional independencies.
            # Our samples are of shape (nobs, 2), therefore the dim=-2
            # in the plate:
            with pyro.plate(self.name_prefix + ".data_plate", dim=-2):
                # Sample from latent function distribution
                function_samples = pyro.sample(self.name_prefix + ".f(x)", function_dist)

        def model(self, x, y):
            # Define pyro module:
            pyro.module(self.name_prefix + ".gp", self)

            # Get p(f) - prior distribution of latent function
            function_dist = self.pyro_model(x)

            # Use a plate here to mark conditional independencies.
            # Our samples are of shape (nobs, 2), therefore the dim=-2
            # in the plate:
            with pyro.plate(self.name_prefix + ".data_plate", dim=-2):
                # Sample from latent function distribution
                function_samples = pyro.sample(self.name_prefix + ".f(x)", function_dist)
                mean_samples = function_samples[..., 0]
                std_samples = function_samples[..., 1]

                # Exp to force always nonnegative stddevs:
                transformed_std_samples = torch.exp(std_samples)

                # Sample from observed distribution
                return pyro.sample(self.name_prefix + ".y",
                                   pyro.distributions.Normal(mean_samples, transformed_std_samples),
                                   obs=y)
    return (HeteroscedasticGP,)


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
def _(WT_1_CF_MEAN, WT_1_CF_VAR, coherency_fractions, np):
    difference_metrics = \
        np.abs(np.mean(coherency_fractions, axis=1) - WT_1_CF_MEAN) \
        / np.sqrt(WT_1_CF_VAR + np.var(coherency_fractions, axis=1))
    return (difference_metrics,)


@app.cell
def _(coherency_fractions, np):
    import scipy.stats
    ecdf_result = scipy.stats.ecdf(np.mean(coherency_fractions, axis=1))
    cf_uniform_values = ecdf_result.cdf.evaluate(np.mean(coherency_fractions, axis=1))
    standard_normal = scipy.stats.norm()
    cf_normal_values = standard_normal.ppf(cf_uniform_values / (1 + 1e-5))
    return cf_normal_values, standard_normal


@app.cell
def _(cf_normal_values, plt):
    plt.hist(cf_normal_values, bins=100);
    plt.show()
    return


@app.cell
def _(FiniteDPP, np, parameter_matrix):
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
    dpp_inducing_points = parameter_matrix[::7, :][inducing_indices, :]
    return (dpp_inducing_points,)


@app.cell
def _(HeteroscedasticGP, dpp_inducing_points, pyro, torch):
    # Instantiate model:
    pyro.clear_param_store()  # Good practice

    inducing_points = torch.from_numpy(dpp_inducing_points).to(torch.float32)
    model = HeteroscedasticGP(inducing_points)
    return (model,)


@app.cell
def _(DataLoader, TensorDataset, model, pyro, torch):
    # Pyro hyperparameters:
    NUM_PARTICLES = 256

    # Set up training routine:
    def train_model(train_x, train_y, learning_rate, num_epochs=1):
        # Set up optimisation:
        optimizer = pyro.optim.Adam({"lr": learning_rate})
        elbo = pyro.infer.Trace_ELBO(num_particles=NUM_PARTICLES, vectorize_particles=True, retain_graph=True)
        svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)

        # Convert datasets to torch:
        train_x = torch.tensor(train_x, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.float32)

        for epoch_index in range(num_epochs):
            print(f"---> Running epoch: {epoch_index + 1}")
            # Initialise dataloaders:
            train_dataset = TensorDataset(train_x, train_y)
            train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

            # Train model:
            model.train()
            for batch_index, (x_batch, y_batch) in enumerate(train_loader):
                model.zero_grad()
                loss = svi.step(x_batch, y_batch)
                if (batch_index + 1) % 25 == 0:
                    print(f"Batch: {batch_index + 1}, Loss: {loss}")

                # Constrain inducing points to input region:
                with torch.no_grad():
                    inducing_points = model.variational_strategy.base_variational_strategy.inducing_points.detach().numpy()
                    model.variational_strategy.base_variational_strategy.inducing_points[inducing_points > 1] = 1
                    model.variational_strategy.base_variational_strategy.inducing_points[inducing_points < 0] = 0
    return (train_model,)


@app.cell
def _(coherency_fractions, np, parameter_matrix, train_model):
    train_model(parameter_matrix, np.mean(coherency_fractions, axis=1), 0.003, 10)
    return


@app.cell
def _():
    # for index in range(10):
    #     model.input_transform.plot_transform(index)
    return


@app.cell
def _():
    # torch.save(model, os.path.join("model_experiments/2025-12-04-collisions_only", "cf_gp_model.pth"))
    return


@app.cell
def _(model):
    model.covar_module.base_kernel.lengthscale.detach().numpy()[0]
    return


@app.cell
def _(model, np, parameter_matrix, torch):
    # Predict:
    model.eval()
    with torch.no_grad():
        output_dist = model(torch.from_numpy(parameter_matrix[:, :]).to(torch.float32))
        gp_mean = output_dist.mean[:, 0]
        epistemic_variance = output_dist.stddev[:, 0].detach().numpy() ** 2
        aleatoric_variance = np.exp(output_dist.mean[:, 1]) ** 2
    return aleatoric_variance, epistemic_variance, gp_mean, output_dist


@app.cell
def _(coherency_fractions, gp_mean, np, standard_normal):
    predictions = gp_mean.detach().numpy()
    uniform_predictions = standard_normal.cdf(predictions) * (1 + 1e-5)
    iecdf_predictions = np.quantile(np.mean(coherency_fractions, axis=1), uniform_predictions)
    return iecdf_predictions, predictions


@app.cell
def _(iecdf_predictions, plt):
    plt.hist(iecdf_predictions, bins=100);
    plt.show()
    return


@app.cell
def _(model, np, parameter_matrix, torch):
    def test_sampling():
        # Get distribution and samples:
        test_distribution = model(torch.from_numpy(parameter_matrix[:5, :]).to(torch.float32))
        samples = test_distribution.sample(torch.Size([10000]))
        samples = samples.detach().numpy()

        # Compare sample variance to GP output variance:
        print(samples.shape)
        print("Sampled variance:")
        print(np.var(samples, axis=0)[:, 0])
        print("GP output variance:")
        print(np.exp(np.mean(samples, axis=0)[:, 1]) ** 2)

    test_sampling()
    return


@app.cell
def _(epistemic_variance, plt):
    plt.hist(epistemic_variance, bins=100);
    plt.show()
    return


@app.cell
def _(aleatoric_variance, plt):
    plt.hist(aleatoric_variance, bins=100);
    plt.show()
    return


@app.cell
def _(coherency_fractions, np, plt):
    plt.hist(np.var(coherency_fractions[:, :], axis=1), bins=100);
    plt.show()
    return


@app.cell
def _(coherency_fractions):
    coherency_fractions.shape
    return


@app.cell
def _(coherency_fractions, np):
    np.min(np.var(coherency_fractions[:, :], axis=1))
    return


@app.cell
def _(aleatoric_variance, coherency_fractions, np, plt):
    def plot_variance_predictions():
        fig, ax = plt.subplots()
        ax.scatter(np.var(coherency_fractions[:, :], axis=1), aleatoric_variance, s=0.1)
        ax.plot([-0, 0.005], [-0, 0.005], c='r')
        ax.set_xlim(-0, None)
        ax.set_ylim(-0, None)
        ax.set_aspect("equal")
        plt.show()

    plot_variance_predictions()
    return


@app.cell
def _(coherency_fractions, np, plt, predictions):
    def plot_predictions():
        fig, ax = plt.subplots()
        ax.scatter(np.mean(coherency_fractions[:, :], axis=1), predictions, s=0.01)
        # ax.scatter(difference_metrics, output_dist.mean[:, 0], c=1/output_dist.mean[:, 1], s=0.1)
        ax.plot([-0.05, 0.15], [-0.05, 0.15], c='r')
        ax.set_xlim(0, 0.15)
        ax.set_ylim(0, 0.15)
        ax.set_aspect("equal")
        plt.show()

    plot_predictions()
    return


@app.cell
def _(cf_normal_values, plt, predictions):
    def plot_normalised_predictions():
        fig, ax = plt.subplots()
        ax.scatter(cf_normal_values, predictions, s=0.001)
        # ax.scatter(difference_metrics, output_dist.mean[:, 0], c=1/output_dist.mean[:, 1], s=0.1)
        ax.plot([-4, 4], [-4, 4], c='r')
        # ax.set_xlim(0, 0.15)
        # ax.set_ylim(0, 0.15)
        ax.set_aspect("equal")
        plt.show()

    plot_normalised_predictions()
    return


@app.cell
def _(model, plt):
    learnt_inducing_points = model.variational_strategy.base_variational_strategy.inducing_points.detach().numpy()

    def plot_inducing_points(points):
        fig, ax = plt.subplots(figsize=(7, 7))
        dim_x = 0
        dim_y = 1

        # Inducing points for mean estimation:
        ax.scatter(points[0, :, dim_x], points[0, :, dim_y], s=5)

        # Inducing points for noise estimation:
        ax.scatter(points[1, :, dim_x], points[1, :, dim_y], s=5)

        # Plot bounding area for input space:
        ax.hlines(0, -0.05, 1.05, alpha=0.5)
        ax.hlines(1, -0.05, 1.05, alpha=0.5)
        ax.vlines(0, -0.05, 1.05, alpha=0.5)
        ax.vlines(1, -0.05, 1.05, alpha=0.5)
        # ax.set_xlim(-0.05, 1.05)
        # ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        plt.show()

    plot_inducing_points(learnt_inducing_points)
    return


@app.cell
def _(WT_1_CF_MEAN, WT_1_CF_VAR, model, np, torch):
    def get_implausibility_metric(input_parameters):
        # Convert to tensor:
        if not torch.is_tensor(input_parameters):
            input_parameters = torch.from_numpy(input_parameters).to(torch.float32)

        # Call emulator:
        with torch.no_grad():
            output_distribution = model(input_parameters)

        # Get relevant data:
        gp_estimate = output_distribution.mean[:, 0].detach().numpy()
        epistemic_variance = output_distribution.stddev[:, 0].detach().numpy() ** 2
        aleatoric_variance = np.exp(output_distribution.mean[:, 1].detach().numpy()) ** 2
        total_variance = epistemic_variance + aleatoric_variance
        implausibility_metric = np.abs(gp_estimate - WT_1_CF_MEAN) / np.sqrt(WT_1_CF_VAR + epistemic_variance)

        return implausibility_metric
    return (get_implausibility_metric,)


@app.cell
def _(get_implausibility_metric, parameter_matrix):
    implausibility_metrics = get_implausibility_metric(parameter_matrix)
    return (implausibility_metrics,)


@app.cell
def _(difference_metrics, plt):
    plt.hist(difference_metrics, bins=100);
    plt.show()
    return


@app.cell
def _(difference_metrics, np):
    np.count_nonzero(difference_metrics < 4) / len(difference_metrics)
    return


@app.cell
def _(difference_metrics, parameter_matrix, plt):
    plt.hist(parameter_matrix[difference_metrics < 4, 9], bins=100);
    plt.show()
    return


@app.cell
def _(implausibility_metrics, plt):
    plt.hist(implausibility_metrics, bins=100);
    plt.show()
    return


@app.cell
def _(implausibility_metrics, np):
    np.count_nonzero(implausibility_metrics > 2) / len(implausibility_metrics)
    return


@app.cell
def _(coherency_fractions, implausibility_metrics, np, output_dist, plt):
    def plot_masked_predictions(mask):
        fig, ax = plt.subplots()
        ax.scatter(np.mean(coherency_fractions[mask, :], axis=1), output_dist.mean[mask, 0], c=1/output_dist.mean[mask, 1], s=0.1)
        ax.plot([-0.05, 0.15], [-0.05, 0.15], c='r')
        ax.set_xlim(-0.01, None)
        ax.set_ylim(-0.01, None)
        ax.set_aspect("equal")
        plt.show()

    plot_masked_predictions(implausibility_metrics > 2)
    return


@app.cell
def _():
    WT_1_SPEED_MEAN = 0.23917124350731356
    WT_1_SPEED_VAR = 0.004417720510108223
    return


if __name__ == "__main__":
    app.run()
