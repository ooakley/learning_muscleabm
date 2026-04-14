import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import json

    import gpytorch
    import torch

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from torch.utils.data import TensorDataset, DataLoader
    from dppy.finite_dpps import FiniteDPP
    return DataLoader, TensorDataset, gpytorch, json, np, os, pd, plt, torch


@app.cell
def _(DataLoader, TensorDataset, gpytorch, os, torch):
    class DeepInputTransformation(torch.nn.Module):
        def __init__(self, dimension, hidden_layer_neuron_count=150):
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
                if (batch_index + 1) % 10 == 0:
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
def _():
    EXPERIMENT_DIRPATH = "model_experiments/2026-03-20-collisions_shape"
    return (EXPERIMENT_DIRPATH,)


@app.cell
def _(EXPERIMENT_DIRPATH, json, np, os):
    def load_gridsearch_data(experiment_folderpath):
        # Load numpy data:
        parameter_matrix = np.load(
            os.path.join(experiment_folderpath, "sample_matrix.npy")
        )
        coherency_fractions = np.load(
            os.path.join(experiment_folderpath, "summary_data", "com_coherency_fractions.npy")
        )
        ann_indices = np.load(
            os.path.join(experiment_folderpath, "summary_data", "com_ann_indices.npy")
        )
        speeds = np.load(
            os.path.join(experiment_folderpath, "summary_data", "com_speeds.npy")
        )

        # Get gridsearch configuration:
        with open(os.path.join(experiment_folderpath, "config.json")) as json_filestream:
            config_dictionary  = json.load(json_filestream)
        gridsearch_parameters = config_dictionary["gridsearch_parameters"]

        return parameter_matrix, coherency_fractions, ann_indices, speeds, gridsearch_parameters

    # Load data:
    parameter_matrix, coherency_fractions, ann_indices, speeds, gridsearch_parameters = load_gridsearch_data(
        EXPERIMENT_DIRPATH
    )

    # Get information necessary to transform GP inputs/outputs:
    SPEED_DIST_MEAN = np.nanmean(speeds[:, 0])
    SPEED_DIST_STD = np.nanstd(speeds[:, 0])

    CF_DIST_MEAN = np.nanmean(np.nanmean(coherency_fractions, axis=1))
    CF_DIST_STD = np.nanstd(np.nanmean(coherency_fractions, axis=1))
    return (
        CF_DIST_MEAN,
        CF_DIST_STD,
        SPEED_DIST_MEAN,
        SPEED_DIST_STD,
        gridsearch_parameters,
        speeds,
    )


@app.cell
def _(np, plt, speeds):
    plt.hist(np.log(speeds[:, 0]), bins=100);
    plt.show()
    return


@app.cell
def _(SPEED_DIST_MEAN, SPEED_DIST_STD):
    print(SPEED_DIST_MEAN, SPEED_DIST_STD)
    return


@app.cell
def _(json, np, os, pd):
    # Get wet lab data:
    def load_wetlab_data(experiment_folderpath):
        # Get average nearest neighbour index:
        with open(os.path.join(experiment_folderpath, "anni_dictionary.json")) as f:
            anni_dictionary = json.load(f)

        # Get coherency fraction:
        with open(os.path.join(experiment_folderpath, "cf_dictionary.json")) as f:
            cf_dictionary = json.load(f)

        # Get speeds:
        fitting_dataframe = pd.read_csv(os.path.join(experiment_folderpath, "fitting_dataset.csv"))

        return anni_dictionary, cf_dictionary, fitting_dataframe

    anni_dictionary, cf_dictionary, fitting_dataframe = load_wetlab_data("wetlab_data/OEO20241206")

    def get_speed_data(column):
        column_mask = fitting_dataframe["column"] == column
        return np.array(fitting_dataframe.loc[column_mask, "speed"])

    def get_cf_data(column):
        return cf_dictionary[f"{column}"]

    # # Get targets for implausibility metric:
    # COLUMN = 6
    # ANNI_MEAN = np.mean(anni_dictionary[f"{COLUMN}"])
    # ANNI_SEM = np.std(anni_dictionary[f"{COLUMN}"]) / np.sqrt(12)

    # CF_MEAN = np.mean(cf_dictionary[f"{COLUMN}"])
    # CF_SEM = np.std(cf_dictionary[f"{COLUMN}"]) / np.sqrt(12)

    # column_mask = fitting_dataframe["column"] == COLUMN
    # SPEED_MEAN = np.mean(fitting_dataframe.loc[column_mask, "speed"])
    # SPEED_SEM = np.std(fitting_dataframe.loc[column_mask, "speed"]) / np.sqrt(np.count_nonzero(column_mask))
    return get_cf_data, get_speed_data


@app.cell
def _(ModelManager, np, os):
    gp_dirpath = os.path.join("model_experiments/2026-03-07-matrix_collisions", "gaussian_process_models")

    # Dummy points for instantiation:
    inducing_points = np.zeros((10, 10))

    # # Instantiate then load emulators:
    # idr_model_manager = ModelManager(inducing_points, 0.003)
    # idr_model_manager.load(gp_dirpath, "density_idr")

    # op3_model_manager = ModelManager(inducing_points, 0.003)
    # op3_model_manager.load(gp_dirpath, "op3")

    # op17_model_manager = ModelManager(inducing_points, 0.003)
    # op17_model_manager.load(gp_dirpath, "op17")

    op_model_manager = ModelManager(inducing_points, 0.003)
    op_model_manager.load(gp_dirpath, "matrix_order_parameters")
    return (op_model_manager,)


@app.cell
def _(EXPERIMENT_DIRPATH, np, os):
    subset_dirpath = os.path.join(EXPERIMENT_DIRPATH, "subset_estimation")

    estimated_parameter_array = []
    for column in range(1, 7):
        estimated_parameter_array.append(
            np.load(os.path.join(subset_dirpath, f"folder_{column}/developed_estimate.npy"))
        )

    speed_array = []
    for column in range(1, 7):
        speed_array.append(
            np.load(os.path.join(subset_dirpath, f"folder_{column}/subset_speed_values.npy"))
        )

    cf_array = []
    for column in range(1, 7):
        cf_array.append(
            np.load(os.path.join(subset_dirpath, f"folder_{column}/subset_cf_values.npy"))
        )

    imp_array = []
    for column in range(1, 7):
        imp_array.append(
            np.load(os.path.join(subset_dirpath, f"folder_{column}/subset_implausibility.npy"))
        )

    distance_array = []
    for column in range(1, 7):
        distance_array.append(
            np.load(os.path.join(subset_dirpath, f"folder_{column}/subset_distance.npy"))
        )

    for idx, distance_matrix in enumerate(distance_array):
        # distance_matrix[:, 0] *= 3
        distance_array[idx] = distance_matrix
    return (
        cf_array,
        distance_array,
        estimated_parameter_array,
        imp_array,
        speed_array,
    )


@app.cell
def _(np, speed_array):
    np.min(speed_array)
    return


@app.cell
def _(SPEED_DIST_MEAN, SPEED_DIST_STD, np, speed_array):
    np.min((speed_array[0] * SPEED_DIST_STD) + SPEED_DIST_MEAN)
    return


@app.cell
def _(distance_array, estimated_parameter_array, np, plt):
    def test_plot():
        index = 0
        distance_matrix = np.copy(distance_array[index])
        distance_norm = np.linalg.norm(distance_matrix, axis=1)

        mask = distance_norm < np.quantile(distance_norm, 0.01)
        parameters = estimated_parameter_array[index]

        fig, ax = plt.subplots()
        ax.scatter(parameters[mask, 0], parameters[mask, 1], s=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        plt.show()

    test_plot()
    return


@app.cell
def _(distance_array, np, plt):
    def pareto_plot(index, ax):
        distance_matrix = np.copy(distance_array[index])
        distance_norm = np.linalg.norm(distance_matrix, axis=1)
        mask = distance_norm < np.quantile(distance_norm, 0.005)
        fit_distances = np.log(distance_matrix[mask, :])

        ax.scatter(fit_distances[:, 0], fit_distances[:, 1], s=1, c=distance_norm[mask])
        ax.set_xlim(-10, 0)
        ax.set_ylim(-10, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    def plot_all_pareto():
        fig, axs = plt.subplots(6, 1, figsize=(1.5, 9), sharex=True, sharey=True, layout="constrained")
        for i in range(6):
            pareto_plot(i, axs[i])
        plt.show()

    plot_all_pareto()
    return


@app.cell
def _(
    distance_array,
    estimated_parameter_array,
    gridsearch_parameters,
    np,
    plt,
):
    def plot_full_distribution(index):
        # Get distances and distance entropies:
        distance_norms = np.linalg.norm(distance_array[index], axis=1)
        unit_distances = distance_array[index] / np.expand_dims(distance_norms, axis=1)
        distance_entropies = np.sum(-unit_distances * np.log(unit_distances + 1e-6), axis=1)
        distance_mask = distance_norms < np.quantile(distance_norms, 0.005)

        parameters = estimated_parameter_array[index]
        parameter_labels = [p_label for p_label, p_range in gridsearch_parameters]
        fig, axs = plt.subplots(10, 10, figsize=(10, 10), layout="constrained")

        for i in range(10):
            for j in range(10):
                ax = axs[i, j]
                if i == j:
                    parameter_label = parameter_labels[i]
                    ax.text(
                        0.5, 0.5, parameter_label,
                        fontsize=5, horizontalalignment="center",
                        rotation=45, rotation_mode="anchor"
                    )
                    ax.set_aspect("equal")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_axis_off()
                    continue

                # ax.scatter(parameters[mask, i], parameters[mask, j], s=1, c="tab:orange")
                ax.scatter(parameters[distance_mask, i], parameters[distance_mask, j], s=1, c='k')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect("equal")

        # fig.tight_layout()
        plt.show()
    return (plot_full_distribution,)


@app.cell
def _(plot_full_distribution):
    plot_full_distribution(1)
    return


@app.cell
def _(imp_array, plt):
    def plot_implausibilities():
        fig, ax = plt.subplots()
        for imp_vals in imp_array[0:3]:
            ax.hist(imp_vals, histtype="step", color='tab:blue', bins=100);
        for imp_vals in imp_array[3:]:
            ax.hist(imp_vals, histtype="step", color='tab:orange', bins=100);
        plt.show()

    plot_implausibilities()
    return


@app.cell
def _(imp_array, plt):
    def plot_ecdf_implausibilities():
        fig, ax = plt.subplots()
        for imp_vals in imp_array[0:3]:
            ax.ecdf(imp_vals, color='tab:blue', lw=1, alpha=0.5);
        for imp_vals in imp_array[3:]:
            ax.ecdf(imp_vals, color='tab:orange', lw=1, alpha=0.5);
        plt.show()

    plot_ecdf_implausibilities()
    return


@app.cell
def _(distance_array, np, plt, speed_array):
    def plot_speeds():
        fig, ax = plt.subplots()
        for index, speeds in enumerate(speed_array):
            if index < 3:
                color = "tab:blue"
            else:
                color = "tab:orange"
            distance_norm = np.linalg.norm(distance_array[index], axis=1)
            mask = distance_norm < np.quantile(distance_norm, 0.01)
            ax.hist(speeds[0, mask], histtype="step", color=color, bins=15, alpha=0.75);
        plt.show()

    plot_speeds()
    return


@app.cell
def _(
    SPEED_DIST_MEAN,
    SPEED_DIST_STD,
    distance_array,
    get_speed_data,
    np,
    plt,
    speed_array,
):
    import seaborn as sns

    def plot_speed_swarm():
        fig, ax = plt.subplots()

        swarm_data = []
        for index, speeds in enumerate(speed_array):
            if index < 3:
                color = "tab:blue"
            else:
                color = "tab:orange"

            distance_norms = np.linalg.norm(distance_array[index], axis=1)
            unit_distances = distance_array[index] / np.expand_dims(distance_norms, axis=1)
            distance_entropies = np.sum(-unit_distances * np.log(unit_distances + 1e-6), axis=1)

            norm_cutoff = np.quantile(distance_norms, 0.001)
            entropy_cutoff = np.quantile(-distance_entropies, 0.0025)
            norm_mask = distance_norms < norm_cutoff
            entropy_mask = -distance_entropies < entropy_cutoff
            mask = np.logical_and(norm_mask, entropy_mask)

            # Get GP estimated data:
            gp_estimate = (speeds[0, mask] * SPEED_DIST_STD) + SPEED_DIST_MEAN
            swarm_data.append(gp_estimate * (577 / 1440))

            site_mean_estimates = []
            for i in range(12):
                gm_speed = np.exp(np.mean(np.log(get_speed_data(index + 1)[i::12])))
                site_mean_estimates.append(gm_speed)
            swarm_data.append(np.array(site_mean_estimates))

        # Plot data:
        sns.violinplot(data=swarm_data, fill=False)

        # Label data:
        ax.set_xticklabels([
            # CTRL:
            "CTRL1 Fit",
            "CTRL1 Data",
            "CTRL2 Fit",
            "CTRL2 Data",
            "CTRL3 Fit",
            "CTRL3 Data",
            # RD:
            "RD1 Fit",
            "RD1 Data",
            "RD2 Fit",
            "RD2 Data",
            "RD3 Fit",
            "RD3 Data",
        ])

        ax.set_ylabel("Speed")
        ax.tick_params(axis='x', labelrotation=70)

        plt.show()

    plot_speed_swarm()
    return (sns,)


@app.cell
def _(
    CF_DIST_MEAN,
    CF_DIST_STD,
    cf_array,
    distance_array,
    get_cf_data,
    np,
    plt,
    sns,
):
    def plot_cf_swarm():
        fig, ax = plt.subplots()

        swarm_data = []
        for index, cf_vals in enumerate(cf_array):
            if index < 3:
                color = "tab:blue"
            else:
                color = "tab:orange"

            distance_norms = np.linalg.norm(distance_array[index], axis=1)
            unit_distances = distance_array[index] / np.expand_dims(distance_norms, axis=1)
            distance_entropies = np.sum(-unit_distances * np.log(unit_distances + 1e-6), axis=1)

            norm_cutoff = np.quantile(distance_norms, 0.001)
            entropy_cutoff = np.quantile(-distance_entropies, 0.0025)
            norm_mask = distance_norms < norm_cutoff
            entropy_mask = -distance_entropies < entropy_cutoff
            mask = np.logical_and(norm_mask, entropy_mask)

            swarm_data.append((cf_vals[0, mask] * CF_DIST_STD) + CF_DIST_MEAN)
            swarm_data.append(get_cf_data(index + 1))

        # Plot data:
        sns.violinplot(data=swarm_data, fill=False)

        # Label data:
        ax.set_xticklabels([
            # CTRL:
            "CTRL1 Fit",
            "CTRL1 Data",
            "CTRL2 Fit",
            "CTRL2 Data",
            "CTRL3 Fit",
            "CTRL3 Data",
            # RD:
            "RD1 Fit",
            "RD1 Data",
            "RD2 Fit",
            "RD2 Data",
            "RD3 Fit",
            "RD3 Data",
        ])

        ax.set_ylabel("Coherency Fraction")
        ax.tick_params(axis='x', labelrotation=70)
        plt.show()

    plot_cf_swarm()
    return


@app.cell
def _(cf_array, distance_array, np, plt):
    def plot_coherency_fractions():
        fig, ax = plt.subplots()
        for index, cfs in enumerate(cf_array):
            if index < 3:
                color = "tab:blue"
            else:
                color = "tab:orange"
            distance_norm = np.linalg.norm(distance_array[index], axis=1)
            mask = distance_norm < np.quantile(distance_norm, 0.05)
            ax.hist(cfs[0, mask], histtype="step", color=color, bins=20);
        plt.show()

    plot_coherency_fractions()
    return


@app.cell
def _(torch):
    def emulate(manager, x):
        manager.likelihood.eval()
        manager.model.eval()
        tensor_input = torch.tensor(x, dtype=torch.float32)
        if len(tensor_input.shape) == 1:
            tensor_input = torch.unsqueeze(tensor_input, 0)
        with torch.no_grad():
            prediction = manager.likelihood(manager.model(tensor_input))
            prediction_mean = prediction.mean.detach().numpy()
            prediction_std = prediction.stddev.detach().numpy()
        return prediction_mean, prediction_std
    return (emulate,)


@app.cell
def _(
    distance_array,
    emulate,
    estimated_parameter_array,
    np,
    op_model_manager,
):
    CELL_COUNT = 80
    ADVECTION_PARAMETER = 0.5
    SAMPLE_RATE = 0.5
    cell_count_parameter = (CELL_COUNT - 50) / 75

    # idr_estimates = []
    # op3_estimates = []
    # op17_estimates = []
    op_estimates = []
    for index, parameters in enumerate(estimated_parameter_array):
        # Add matrix advection rate:
        advection_parameter_array = np.ones((parameters.shape[0], 1)) * ADVECTION_PARAMETER
        sample_rate_parameter_array = np.ones((parameters.shape[0], 1)) * SAMPLE_RATE
        matrix_parameters = np.concatenate([parameters[:, :-2], advection_parameter_array, sample_rate_parameter_array], axis=1)
        # matrix_parameters[:, 5] = cell_count_parameter

        # Get distance information:
        distance_norms = np.linalg.norm(distance_array[index], axis=1)
        unit_distances = distance_array[index] / np.expand_dims(distance_norms, axis=1)
        distance_entropies = np.sum(-unit_distances * np.log(unit_distances + 1e-6), axis=1)

        # Estimate tradeoff points:
        norm_cutoff = np.quantile(distance_norms, 0.01)
        entropy_cutoff = np.quantile(-distance_entropies, 0.01)
        norm_mask = distance_norms < norm_cutoff
        entropy_mask = -distance_entropies < entropy_cutoff
        joint_mask = np.logical_and(norm_mask, entropy_mask)
        print(index, np.count_nonzero(joint_mask))

        # idr, _ = emulate(idr_model_manager, matrix_parameters[joint_mask, :])
        # op3, _ = emulate(op3_model_manager, matrix_parameters[joint_mask, :])
        # op17, _ = emulate(op17_model_manager, matrix_parameters[joint_mask, :])

        # idr_estimates.append(idr)
        # op3_estimates.append(op3)
        # op17_estimates.append(op17)

        op_values, _ = emulate(op_model_manager, matrix_parameters[joint_mask, :])
        op_estimates.append(op_values)
    return (op_estimates,)


@app.cell
def _(op_estimates, plt):
    def plot_op_histograms():
        fig, ax = plt.subplots()
        for index, op_vals in enumerate(op_estimates):
            if index < 3:
                color = "tab:blue"
            else:
                color = "tab:orange"
            ax.hist(op_vals, histtype="step", color=color, bins=10);
        plt.show()

    plot_op_histograms()
    return


@app.cell
def _(op_estimates, plt, sns):
    def plot_op_violin():
        fig, ax = plt.subplots()

        swarm_data = []
        for index, op_vals in enumerate(op_estimates):
            swarm_data.append(op_vals)

        # Plot data:
        sns.violinplot(data=swarm_data, fill=False)

        # Label data:
        ax.set_xticklabels([
            # CTRL:
            "CTRL1 Fit",
            "CTRL2 Fit",
            "CTRL3 Fit",
            # RD:
            "RD1 Fit",
            "RD2 Fit",
            "RD3 Fit",
        ])

        ax.set_ylabel("Order Parameter 31")
        ax.tick_params(axis='x', labelrotation=70)
        # ax.set_ylim(-1.5, 1)

        plt.show()

    plot_op_violin()
    return


@app.cell
def _(idr_estimates, plt, sns):
    def plot_idr_violin():
        fig, ax = plt.subplots()

        swarm_data = []
        for index, idr_vals in enumerate(idr_estimates):
            swarm_data.append(idr_vals)

        # Plot data:
        sns.violinplot(data=swarm_data, fill=False)

        # Label data:
        ax.set_xticklabels([
            # CTRL:
            "CTRL1 Fit",
            "CTRL2 Fit",
            "CTRL3 Fit",
            # RD:
            "RD1 Fit",
            "RD2 Fit",
            "RD3 Fit",
        ])

        ax.set_ylabel("Fibre Density Interdecile Range")
        ax.tick_params(axis='x', labelrotation=70)
        plt.show()

    plot_idr_violin()
    return


@app.cell
def _(op3_estimates, plt):
    def plot_op3_ecdf():
        fig, ax = plt.subplots()
        for index, op3_vals in enumerate(op3_estimates):
            if index < 3:
                color = "tab:blue"
            else:
                color = "tab:orange"
            ax.ecdf(op3_vals, color=color);
        plt.show()

    plot_op3_ecdf()
    return


@app.cell
def _(op17_estimates, plt):
    def plot_op17_ecdf():
        fig, ax = plt.subplots()
        for index, op17_vals in enumerate(op17_estimates):
            if index < 3:
                color = "tab:blue"
            else:
                color = "tab:orange"
            ax.ecdf(op17_vals, color=color);
        plt.show()

    plot_op17_ecdf()
    return


@app.cell
def _(np, plt):
    from sklearn.neighbors import KernelDensity

    def plot_kde_estimates(inputs):
        # Determine range of plot:
        estimate_array = np.stack(inputs, axis=0)
        plot_points = np.linspace(np.min(estimate_array), np.max(estimate_array), 500)

        for i in [0, 2]:
            # Get KDE estimate:
            kde_manager = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(estimate_array[i, :].flatten().reshape(-1, 1))
            estimated_pdf = np.exp(kde_manager.score_samples(plot_points.reshape(-1, 1)))
            plt.plot(plot_points, estimated_pdf, color="tab:blue")

        for i in [3, 4]:
            # Get KDE estimate:
            kde_manager = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(estimate_array[i, :].flatten().reshape(-1, 1))
            estimated_pdf = np.exp(kde_manager.score_samples(plot_points.reshape(-1, 1)))
            plt.plot(plot_points, estimated_pdf, color="tab:orange")

        plt.show()
    return (plot_kde_estimates,)


@app.cell
def _(idr_estimates, plot_kde_estimates):
    plot_kde_estimates(idr_estimates)
    return


@app.cell
def _(op3_estimates, plot_kde_estimates):
    plot_kde_estimates(op3_estimates)
    return


@app.cell
def _(op17_estimates, plot_kde_estimates):
    plot_kde_estimates(op17_estimates)
    return


@app.cell
def _(np, torch):
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
        phantom_mean = prediction_mean.detach()
        prediction_cost = (phantom_mean - prediction_mean)**2
        return prediction_cost, tensor_input


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


    def calculate_hessians(manager, inputs):
        hessians = []
        for i in range(inputs.shape[0]):    
            # Estimate Hessian in local area:
            prediction_mean, tensor_input = grad_emulate(manager, inputs[i, :])
            estimated_hessian = calculate_hessian(prediction_mean, tensor_input)
            estimated_hessian = np.squeeze(estimated_hessian.detach().numpy())
            hessians.append(estimated_hessian)

        return np.stack(hessians, axis=0)


    def calculate_eigens(sampled_hessians):
        sampled_eigvals = []
        sampled_eigvectors = []
        for _i in range(sampled_hessians.shape[0]):
            eigvals, eigvectors = np.linalg.eig(sampled_hessians[_i, :, :])
            sampled_eigvals.append(eigvals)
            sampled_eigvectors.append(eigvectors.T)

        sampled_eigvals = np.stack(sampled_eigvals, axis=0)
        sampled_eigvectors = np.stack(sampled_eigvectors, axis=0)
        return sampled_eigvals, sampled_eigvectors
    return calculate_eigens, calculate_hessians


@app.cell
def _(np):
    sample_rng = np.random.default_rng(0)
    random_sample = sample_rng.uniform(0, 1, (5000, 11))
    return (random_sample,)


@app.cell
def _(calculate_hessians, op3_model_manager, random_sample):
    sampled_hessians = calculate_hessians(op3_model_manager, random_sample)
    return (sampled_hessians,)


@app.cell
def _(calculate_eigens, sampled_hessians):
    sampled_eigvals, sampled_eigvectors = calculate_eigens(sampled_hessians)
    return (sampled_eigvectors,)


@app.cell
def _(plt):
    import colorstamps

    def plot_eigendirections(inputs, eigenvectors, parameter_i, parameter_j, eigenvector_index=0):
        fig, ax = plt.subplots()

        # Get 2D colormap from parameters:
        rgb, _ = colorstamps.apply_stamp(
            inputs[:, parameter_i], inputs[:, parameter_j],
            'flat',
            vmin_0=0, vmax_0=1,
            vmin_1=0, vmax_1=1,
        )

        # Plot components:
        ax.scatter(
            eigenvectors[:, eigenvector_index, parameter_i],
            eigenvectors[:, eigenvector_index, parameter_j],
            s=1, alpha=0.25, c=rgb
        )

        # Format plot:
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        plt.show()
    return colorstamps, plot_eigendirections


@app.cell
def _(np, plot_eigendirections, random_sample, sampled_eigvectors):
    plot_eigendirections(random_sample, np.real(sampled_eigvectors), 0, 4)
    return


@app.cell
def _(np, sampled_eigvectors):
    def calculate_geometric_median(key_eigenvectors):
        proposal_median = np.mean(key_eigenvectors, axis=0)    
        converged = False
        while not converged:
            # Get distances:
            cosine_similarities = key_eigenvectors @ np.expand_dims(proposal_median, axis=1)
            nematic_distances = 1 - np.abs(cosine_similarities)

            # Get weighted average of appropriately flipped vectors:
            flipped_vectors = np.sign(cosine_similarities) * key_eigenvectors
            new_proposal = np.sum(flipped_vectors / nematic_distances, axis=0) / np.sum(1 / nematic_distances)
            new_proposal /= np.linalg.norm(new_proposal)
            epsilon = 1 - np.dot(proposal_median, new_proposal)
            # print(epsilon)
            if epsilon < 1e-8:
                print("Converged!")
                print(f"Mean nematic distance: {np.mean(nematic_distances)}")
                converged = True
            proposal_median = new_proposal

        return proposal_median, flipped_vectors

    proposal_median, flipped_vectors = calculate_geometric_median(np.real(sampled_eigvectors)[:, 0, :])
    print(proposal_median)
    return flipped_vectors, proposal_median


@app.cell
def _(colorstamps, flipped_vectors, plt, proposal_median, random_sample):
    def plot_referenced_distribution(component_i, component_j):
        fig, ax = plt.subplots()
        # Get 2D colormap from parameters:
        rgb, _ = colorstamps.apply_stamp(
            random_sample[:, component_i], random_sample[:, component_j],
            'flat',
            vmin_0=0, vmax_0=1,
            vmin_1=0, vmax_1=1,
        )
        ax.scatter(flipped_vectors[:, component_i], flipped_vectors[:, component_j], c=rgb, s=0.1, alpha=0.5)
        ax.scatter(proposal_median[component_i], proposal_median[component_j], s=10)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        plt.show()

    plot_referenced_distribution(7, 10)
    return


@app.cell
def _(flipped_vectors, np):
    import umap
    import numba

    @numba.njit()
    def nematic_cosine(a, b):
        return 1 - np.abs(np.dot(a, b))

    # Select largest eigenvector as position:
    umap_model = umap.UMAP(metric=nematic_cosine, n_neighbors=25, min_dist=0, random_state=0)
    sampled_embeddings = umap_model.fit_transform(
        flipped_vectors, axis=1
    )
    return (sampled_embeddings,)


@app.cell
def _(flipped_vectors, np, plt, sampled_embeddings):
    import colorcet as cc

    def plot_umap():
        fig, ax = plt.subplots(figsize=(10, 10))
        parameter_entropy = -np.sum(np.abs(flipped_vectors) * np.log(np.abs(flipped_vectors)), axis=1)
        # parameter_entropy = (parameter_entropy - np.min(parameter_entropy)) / (np.max(parameter_entropy) - np.min(parameter_entropy))
        ax.scatter(sampled_embeddings[:, 0], sampled_embeddings[:, 1], alpha=0.75, s=0.5, c=parameter_entropy)
        # ax.scatter(
        #     subset_embeddings[:, 0],
        #     subset_embeddings[:, 1],
        #     alpha=0.5, s=1,
        #     c=np.log(effective_ranks),
        #     cmap=cc.m_CET_D1A
        # )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        fig.tight_layout()
        plt.show()

    plot_umap()
    return


@app.cell
def _():
    # # Get gridsearch configuration:
    # with open("model_experiments/2026-01-26-matrix_collisions/config.json") as json_filestream:
    #     config_dictionary  = json.load(json_filestream)
    # gridsearch_parameters = config_dictionary["gridsearch_parameters"]

    # def plot_component_umap():
    #     fig, ax = plt.subplots(figsize=(10, 10))
    #     largest_components = np.argmax(np.abs(flipped_vectors), axis=1)
    #     largest_components = np.argsort(np.abs(flipped_vectors), axis=1)
    #     largest_components = largest_components[:, -1]
    #     for p_i in range(10):
    #         component_mask = largest_components == p_i
    #         if np.count_nonzero(component_mask) == 0:
    #             continue
    #         ax.scatter(
    #             sampled_embeddings[component_mask, 0],
    #             sampled_embeddings[component_mask, 1],
    #             alpha=0.5,
    #             s=0.5, label=list(gridsearch_parameters.keys())[p_i]
    #         )
    #     ax.set_aspect("equal")
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_xlabel("UMAP 1")
    #     ax.set_ylabel("UMAP 2")

    #     # Set up legend:
    #     legend = ax.legend()
    #     for lh in legend.legend_handles:
    #         lh.set_alpha(1)
    #         lh.set_sizes([10])

    #     fig.tight_layout()
    #     plt.show()

    # plot_component_umap()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
