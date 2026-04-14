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
    return (
        DataLoader,
        FiniteDPP,
        TensorDataset,
        gpytorch,
        json,
        np,
        os,
        pd,
        plt,
        torch,
    )


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
    EXPERIMENT_FOLDERPATH = "model_experiments/2026-03-20-collisions_shape"
    return (EXPERIMENT_FOLDERPATH,)


@app.cell
def _(EXPERIMENT_FOLDERPATH, json, np, os):
    # Load input data:
    parameter_matrix = np.load(
        os.path.join(EXPERIMENT_FOLDERPATH, "sample_matrix.npy")
    )

    # Load metrics
    ann_indices = np.load(
        os.path.join(EXPERIMENT_FOLDERPATH, "summary_data", "com_ann_indices.npy")
    )
    coherency_fractions = np.load(
        os.path.join(EXPERIMENT_FOLDERPATH, "summary_data", "com_coherency_fractions.npy")
    )
    meander_ratios = np.load(
        os.path.join(EXPERIMENT_FOLDERPATH, "summary_data", "com_meander_ratios.npy")
    )
    speeds = np.load(
        os.path.join(EXPERIMENT_FOLDERPATH, "summary_data", "com_speeds.npy")
    )


    # Get gridsearch configuration:
    with open(os.path.join(EXPERIMENT_FOLDERPATH, "config.json")) as json_filestream:
        config_dictionary  = json.load(json_filestream)
    gridsearch_parameters = config_dictionary["gridsearch_parameters"]
    return (
        ann_indices,
        coherency_fractions,
        gridsearch_parameters,
        meander_ratios,
        parameter_matrix,
        speeds,
    )


@app.cell
def _(meander_ratios, np, plt):
    plt.hist(np.mean(meander_ratios, axis=1), bins=100);
    plt.show()
    return


@app.cell
def _(plt, speeds):
    plt.hist(speeds[:, 0], bins=100);
    plt.show()
    return


@app.cell
def _(np):
    def get_transforms(output_metric):
        if output_metric.shape[1] == 2:
            metric_mean = np.nanmean(output_metric[:, 0])
            metric_std = np.nanstd(output_metric[:, 0])
        else:
            metric_mean = np.nanmean(np.nanmean(output_metric, axis=1))
            metric_std = np.nanstd(np.nanmean(output_metric, axis=1))
        return metric_mean, metric_std
    return (get_transforms,)


@app.cell
def _(
    EXPERIMENT_FOLDERPATH,
    ModelManager,
    ann_indices,
    coherency_fractions,
    get_transforms,
    meander_ratios,
    np,
    os,
    speeds,
):
    # Get information necessary to transform GP outputs:
    CF_DIST_MEAN, CF_DIST_STD = get_transforms(coherency_fractions)
    ANNI_DIST_MEAN, ANNI_DIST_STD = get_transforms(ann_indices)
    SPEED_DIST_MEAN, SPEED_DIST_STD = get_transforms(speeds)
    MR_DIST_MEAN, MR_DIST_STD = get_transforms(meander_ratios)

    # Instantiate then load emulators:
    inducing_points = np.ones((10, 10))
    gp_dirpath = os.path.join(EXPERIMENT_FOLDERPATH, "gaussian_process_models")

    anni_model_manager = ModelManager(inducing_points, 0.003)
    anni_model_manager.load(gp_dirpath, "com_ann_indices")

    cf_model_manager = ModelManager(inducing_points, 0.003)
    cf_model_manager.load(gp_dirpath, "com_coherency_fractions")

    mr_model_manager = ModelManager(inducing_points, 0.003)
    mr_model_manager.load(gp_dirpath, "com_meander_ratios")

    speed_model_manager = ModelManager(inducing_points, 0.003)
    speed_model_manager.load(gp_dirpath, "com_speeds")
    return (
        ANNI_DIST_MEAN,
        ANNI_DIST_STD,
        CF_DIST_MEAN,
        CF_DIST_STD,
        MR_DIST_MEAN,
        MR_DIST_STD,
        SPEED_DIST_MEAN,
        SPEED_DIST_STD,
        anni_model_manager,
        cf_model_manager,
        mr_model_manager,
        speed_model_manager,
    )


@app.cell
def _(
    ANNI_DIST_MEAN,
    ANNI_DIST_STD,
    CF_DIST_MEAN,
    CF_DIST_STD,
    MR_DIST_MEAN,
    MR_DIST_STD,
    SPEED_DIST_MEAN,
    SPEED_DIST_STD,
):
    print(CF_DIST_MEAN, CF_DIST_STD)
    print(ANNI_DIST_MEAN, ANNI_DIST_STD)
    print(SPEED_DIST_MEAN, SPEED_DIST_STD)
    print(MR_DIST_MEAN, MR_DIST_STD)
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

    # Get targets for implausibility metric:
    COLUMN = 6
    ANNI_MEAN = np.mean(anni_dictionary[f"{COLUMN}"])
    ANNI_SEM = np.std(anni_dictionary[f"{COLUMN}"]) / np.sqrt(12)

    CF_MEAN = np.mean(cf_dictionary[f"{COLUMN}"])
    CF_SEM = np.std(cf_dictionary[f"{COLUMN}"]) / np.sqrt(12)

    # Attempt to simulate process:
    column_mask = fitting_dataframe["column"] == COLUMN
    site_speed_means = []
    speed_array = fitting_dataframe.loc[column_mask, "speed"]
    for site_index in range(12):
        site_speed_mean = np.exp(np.mean(np.log(speed_array[site_index::12])))
        site_speed_means.append(site_speed_mean)
    SPEED_MEAN = np.mean(site_speed_means)
    SPEED_SEM = np.std(site_speed_means) / np.sqrt(12)

    MR_MEAN = np.mean(fitting_dataframe.loc[column_mask, "meander_ratio"])
    MR_SEM = np.std(fitting_dataframe.loc[column_mask, "meander_ratio"]) / np.sqrt(np.count_nonzero(column_mask))
    return (
        ANNI_MEAN,
        ANNI_SEM,
        CF_MEAN,
        CF_SEM,
        COLUMN,
        MR_MEAN,
        MR_SEM,
        SPEED_MEAN,
        SPEED_SEM,
    )


@app.cell
def _(SPEED_MEAN):
    SPEED_MEAN
    return


@app.cell
def _(SPEED_SEM):
    SPEED_SEM
    return


@app.cell
def _(
    ANNI_DIST_MEAN,
    ANNI_DIST_STD,
    ANNI_MEAN,
    ANNI_SEM,
    CF_DIST_MEAN,
    CF_DIST_STD,
    CF_MEAN,
    CF_SEM,
    MR_DIST_MEAN,
    MR_DIST_STD,
    MR_MEAN,
    MR_SEM,
    SPEED_DIST_MEAN,
    SPEED_DIST_STD,
    SPEED_MEAN,
    SPEED_SEM,
    anni_model_manager,
    cf_model_manager,
    mr_model_manager,
    np,
    speed_model_manager,
    torch,
):
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

    def get_implausibility_metric(x):
        # # Constrain number of cells:
        # if x.ndim > 1: 
        #     x[:, 5] = 0.8
        # else:
        #     x[5] = 0.8

        # Emulate coherency fraction implausibility:
        cf_mean, cf_std = emulate(cf_model_manager, x)
        cf_mean = (cf_mean * CF_DIST_STD) + CF_DIST_MEAN
        cf_std *= CF_DIST_STD
        cf_implausibility = np.abs(CF_MEAN - cf_mean) / np.sqrt(CF_SEM**2 + cf_std**2)

         # Emulate ANNI implausibility:
        anni_mean, anni_std = emulate(anni_model_manager, x)
        anni_mean = (anni_mean * ANNI_DIST_STD) + ANNI_DIST_MEAN
        anni_std *= ANNI_DIST_STD
        anni_implausibility = np.abs(ANNI_MEAN - anni_mean) / np.sqrt(ANNI_SEM**2 + anni_std**2)

        # Emulate speed implausibility:
        speed_mean, speed_std = emulate(speed_model_manager, x)
        speed_mean = (speed_mean * SPEED_DIST_STD) + SPEED_DIST_MEAN
        speed_std *= SPEED_DIST_STD

        # Temporary fix for interpolation error:
        speed_mean *= (577 / 1440)
        speed_std *= (577 / 1440)
        speed_implausibility = np.abs(SPEED_MEAN - speed_mean) / np.sqrt(SPEED_SEM**2 + speed_std**2)

        # Emulate meander ratio implausibility:
        mr_mean, mr_std = emulate(mr_model_manager, x)
        mr_mean = (mr_mean * MR_DIST_STD) + MR_DIST_MEAN
        mr_std *= MR_DIST_STD
        mr_implausibility = np.abs(MR_MEAN - mr_mean) / np.sqrt(MR_SEM**2 + mr_std**2)

        implausibilities = np.stack([cf_implausibility, anni_implausibility, speed_implausibility, mr_implausibility], axis=1)

        return np.max(implausibilities, axis=1)
    return emulate, get_implausibility_metric


@app.cell
def _(
    ANNI_DIST_MEAN,
    ANNI_DIST_STD,
    ANNI_MEAN,
    CF_DIST_MEAN,
    CF_DIST_STD,
    CF_MEAN,
    MR_DIST_MEAN,
    MR_DIST_STD,
    MR_MEAN,
    SPEED_DIST_MEAN,
    SPEED_DIST_STD,
    SPEED_MEAN,
    anni_model_manager,
    cf_model_manager,
    emulate,
    mr_model_manager,
    np,
    speed_model_manager,
):
    def get_relative_distance(x):
        # Emulate coherency fraction distance:
        cf_mean, cf_std = emulate(cf_model_manager, x)
        cf_mean = (cf_mean * CF_DIST_STD) + CF_DIST_MEAN
        cf_distance = np.abs(CF_MEAN - cf_mean) / CF_MEAN

         # Emulate ANNI distance:
        anni_mean, anni_std = emulate(anni_model_manager, x)
        anni_mean = (anni_mean * ANNI_DIST_STD) + ANNI_DIST_MEAN
        anni_distance = np.abs(ANNI_MEAN - anni_mean) / ANNI_MEAN

        # Emulate speed distance:
        speed_mean, speed_std = emulate(speed_model_manager, x)
        speed_mean = (speed_mean * SPEED_DIST_STD) + SPEED_DIST_MEAN
        # !!! Temporary fix for interpolation error:
        speed_mean *= (577 / 1440)
        speed_distance = np.abs(SPEED_MEAN - speed_mean) / SPEED_MEAN

        # Emulate MR distance:
        mr_mean, mr_std = emulate(mr_model_manager, x)
        mr_mean = (mr_mean * MR_DIST_STD) + MR_DIST_MEAN
        mr_distance = np.abs(MR_MEAN - mr_mean) / MR_MEAN

        distances = np.stack([cf_distance, anni_distance, speed_distance, mr_distance], axis=1)

        return distances
    return (get_relative_distance,)


@app.cell
def _():
    # rng = np.random.default_rng(0)
    # input_space = rng.uniform(low=0.0, high=1.0, size=(2**17, 10))
    # imp_metrics = get_implausibility_metric(input_space)
    return


@app.cell
def _():
     # def plot_implausibility_metric():
    #     fig, ax = plt.subplots()
    #     ax.hist(imp_metrics, bins=100, density=True);
    #     ax.set_xlabel("Implausibility Metric")
    #     ax.set_ylabel("Density")
    #     plt.show()

    # plot_implausibility_metric()
    return


@app.cell
def _():
    # np.count_nonzero(imp_metrics < 3)
    return


@app.cell
def _(get_implausibility_metric, np):
    # The modified Metropolis-Hastings from [2]:
    def generate_candidates(x, proposal_noise):
        # Generate new candidate point:
        rng = np.random.default_rng()
        candidate_components = x + rng.normal(loc=0.0, scale=proposal_noise, size=x.shape)
        acceptance_mask = ~np.logical_or(candidate_components < 0, candidate_components > 1)

        sampled_x = np.copy(x)
        sampled_x[acceptance_mask] = candidate_components[acceptance_mask]
        assert ~np.any(sampled_x > 1)
        assert ~np.any(sampled_x < 0)
        return sampled_x

    def sample_markov_chain(x, threshold, length, discard_ratio, proposal_noise=0.05):
        # Algorithm doesn't work if input x is already below threshold:
        assert np.all(get_implausibility_metric(x) < threshold)

        # Set up sampling:
        sampled_chain = []
        while len(sampled_chain) < length:
            # Generate proposal points:
            proposal_x = generate_candidates(x, proposal_noise)

            # Check if the proposal point is in the subset as defined by our threshold value:
            plausibility_mask = get_implausibility_metric(proposal_x) < threshold
            x[plausibility_mask] = proposal_x[plausibility_mask]
            sampled_chain.append(np.copy(x))

        return np.concatenate(sampled_chain[discard_ratio-1::discard_ratio], axis=0)

    def subset_simulation(target_threshold=0, dimension=12):
        # Set algorithm hyperparameters:
        level_probability = 1/2  # Often written as p0
        subset_sample_count = 5000  # Often written as N
        level_count = int(level_probability * subset_sample_count)

        # ---> Perform level 0 estimation:
        # Sample input space:
        rng = np.random.default_rng()
        input_space = rng.uniform(low=0.0, high=1.0, size=(subset_sample_count, dimension))
        performance_values = get_implausibility_metric(input_space)

        # Sort according to performance value:
        sorting_indices = np.argsort(performance_values)

        # Get valid points in input space (for this level):
        subset_indices = sorting_indices[:level_count]
        subset_x = input_space[subset_indices, :]

        # Get threshold (for this level):
        threshold_lower_bound = performance_values[sorting_indices[level_count]]
        threshold_higher_bound = performance_values[sorting_indices[level_count + 1]]
        threshold = (threshold_higher_bound + threshold_lower_bound) / 2
        print(f"Initial threshold: {threshold}")

        # ---> Perform iterative subset estimation:
        subset_count = 0
        converged = False
        while not converged:
            # Tick subset count:
            subset_count += 1

            # Sample from each subset point as the seed for our modified MCMC sampling:
            print("Sampling from subset...")
            new_samples = sample_markov_chain(subset_x, threshold, 10, 5)

            # Get performance values for these new samples:
            performance_values = get_implausibility_metric(new_samples)

            # Sort according to performance value:
            sorting_indices = np.argsort(performance_values)

            # Get threshold for this round of subsetting:
            threshold_lower_bound = performance_values[sorting_indices[level_count]]
            threshold_higher_bound = performance_values[sorting_indices[level_count + 1]]
            threshold = (threshold_higher_bound + threshold_lower_bound) / 2
            print(f"Threshold at subset {subset_count}: {threshold}")

            # Exit if converged:
            if threshold < target_threshold:
                converged = True
                continue

            # Get valid points in input space (for this level):
            subset_indices = sorting_indices[:level_count]
            subset_x = new_samples[subset_indices, :]

        # Get final estimate of probability:
        hit_estimate = \
            (level_probability ** subset_count) \
            * (np.count_nonzero(performance_values < target_threshold) / subset_sample_count)

        return subset_x, hit_estimate
    return sample_markov_chain, subset_simulation


@app.cell
def _(subset_simulation):
    subset_result, hit_estimate = subset_simulation(target_threshold=3.0)
    return hit_estimate, subset_result


@app.cell
def _(hit_estimate):
    print(hit_estimate)
    return


@app.cell
def _(get_implausibility_metric, np, sample_markov_chain, subset_result):
    threshold_mask = get_implausibility_metric(subset_result) < 3
    sample_length = np.ceil(100000 / np.count_nonzero(threshold_mask))
    developed_estimate = sample_markov_chain(subset_result[threshold_mask], 3, sample_length, 1)
    developed_estimate = developed_estimate[:100000, :]
    return (developed_estimate,)


@app.cell
def _(developed_estimate, get_implausibility_metric):
    subset_implausibility = get_implausibility_metric(developed_estimate)
    return (subset_implausibility,)


@app.cell
def _(developed_estimate, get_relative_distance):
    subset_distance = get_relative_distance(developed_estimate)
    return (subset_distance,)


@app.cell
def _(cf_model_manager, developed_estimate, emulate, speed_model_manager):
    subset_cf_values = emulate(cf_model_manager, developed_estimate)
    subset_speed_values = emulate(speed_model_manager, developed_estimate)
    return subset_cf_values, subset_speed_values


@app.cell
def _(
    COLUMN,
    EXPERIMENT_FOLDERPATH,
    developed_estimate,
    np,
    os,
    subset_cf_values,
    subset_distance,
    subset_implausibility,
    subset_speed_values,
):
    # Generate overall folder:
    save_dirpath = os.path.join(EXPERIMENT_FOLDERPATH, "subset_estimation")
    if not os.path.exists(save_dirpath):
        os.mkdir(save_dirpath)

    # Generate cell type folder:
    cell_dirpath = os.path.join(save_dirpath, f"folder_{COLUMN}")
    if not os.path.exists(cell_dirpath):
        os.mkdir(cell_dirpath)

    # Save subset & backprop results:
    np.save(os.path.join(cell_dirpath, "developed_estimate.npy"), developed_estimate)
    np.save(os.path.join(cell_dirpath, "subset_implausibility.npy"), subset_implausibility)
    np.save(os.path.join(cell_dirpath, "subset_cf_values.npy"), subset_cf_values)
    np.save(os.path.join(cell_dirpath, "subset_speed_values.npy"), subset_speed_values)
    np.save(os.path.join(cell_dirpath, "subset_distance.npy"), subset_distance)
    return


@app.cell
def _(plt, subset_distance):
    def plot_distance_ecdf():
        fig, ax = plt.subplots()
        for i in range(4):
            ax.ecdf(subset_distance[:, i])
        ax.set_xlim(0, 1)
        plt.show()

    plot_distance_ecdf()
    return


@app.cell
def _(np, subset_distance):
    distance_norms = np.linalg.norm(subset_distance, axis=1)
    unit_distances = subset_distance / np.expand_dims(distance_norms, axis=1)
    print(np.count_nonzero(unit_distances < 0))
    distance_entropies = np.sum(-unit_distances * np.log(unit_distances + 1e-6), axis=1)
    return distance_entropies, distance_norms


@app.cell
def _(distance_entropies, distance_norms, np, plt):
    def plot_ent_v_norms():
        fig, ax = plt.subplots()
        ax.scatter(-distance_entropies, distance_norms, s=1)

        # Estimate tradeoff points:
        norm_cutoff = np.quantile(distance_norms, 0.1)
        entropy_cutoff = np.quantile(-distance_entropies, 0.1)

        norm_mask = distance_norms < norm_cutoff
        entropy_mask = -distance_entropies < entropy_cutoff
        joint_mask = np.logical_and(norm_mask, entropy_mask)
        # print(np.count_nonzero(norm_mask))
        # print(np.count_nonzero(entropy_mask))
        print(np.count_nonzero(joint_mask))

        ax.scatter(-distance_entropies[joint_mask], distance_norms[joint_mask], s=1)
        # ax.set_ylim(-0.05, 2.6)
        plt.show()

        return joint_mask

    joint_mask = plot_ent_v_norms()
    return


@app.cell
def _():
    # def conditioned_distance_descent(initial_x):
    #     candidate_position = np.copy(initial_x)
    #     distance_history = []
    #     for i in range(200):
    #         distance, tensor_input = distance_grad_emulate(candidate_position, bounded=True, bound_alpha=20)
    #         distance_history.append(distance.item())
    #         jacobian, hessian = calculate_hessian(distance, tensor_input)
    #         jacobian = np.squeeze(jacobian.detach().numpy())
    #         hessian = np.squeeze(hessian.detach().numpy())
    #         #  + (np.eye(10) * 1e-6)
    #         conditioned_gradient = np.linalg.inv(hessian) @ jacobian
    #         candidate_position += np.squeeze(1e-3*conditioned_gradient)
    #         # candidate_position -= 1e-3*jacobian
    #     return candidate_position, distance_history
    return


@app.cell
def _(calculate_jacobian, imp_grad_emulate, np, subset_estimate):
    # Attempt gradient descent:
    def logged_implausibility_descent(estimate_index):
        candidate_position = np.copy(subset_estimate[estimate_index, :])
        implausibility_history = []
        parameter_trajectory = []
        for i in range(1000):
            total_imp, tensor_input = imp_grad_emulate(candidate_position)
            jacobian = calculate_jacobian(total_imp, tensor_input).detach().numpy()
            if (i + 1) % 100 == 0:
                print(np.linalg.norm(jacobian))
            jacobian /= np.linalg.norm(jacobian)
            candidate_position -= np.squeeze(1e-6*jacobian)
            implausibility_history.append(total_imp.item())
            parameter_trajectory.append(np.copy(candidate_position))

        return candidate_position, implausibility_history, np.stack(parameter_trajectory, axis=0)

    # candidate_position, implausibility_history, parameter_trajectory = logged_implausibility_descent(1)
    return


@app.cell
def _(FiniteDPP, np):
    def get_dpp_estimate(samples):
        # Get distance matrix:
        reduced_estimate = samples[::4, :]
        estimate_distance_matrix = np.sum((reduced_estimate[:, np.newaxis, :] - reduced_estimate[np.newaxis, :, :]) ** 2, axis = -1)
        likelihood_matrix = np.exp(estimate_distance_matrix ** 2)

        # Set up DPP:
        DPP = FiniteDPP('likelihood', **{'L': likelihood_matrix})
        k = 5000
        DPP.sample_mcmc_k_dpp(size=k, random_state=None)
        dpp_selected_indices = DPP.list_of_samples[0][-1]
        return reduced_estimate[dpp_selected_indices, :]

    # dpp_estimate = get_dpp_estimate(developed_estimate)
    return


@app.cell
def _(parameter_trajectory, plt):
    def plot_trajectory(i, j):
        fig, ax = plt.subplots()
        ax.scatter(parameter_trajectory[0, i], parameter_trajectory[0, j])
        ax.plot(parameter_trajectory[:, i], parameter_trajectory[:, j])
        ax.set_aspect("equal")
        plt.show()

    plot_trajectory(0, 9)
    return


@app.cell
def _(calculate_jacobian, imp_grad_emulate, np, subset_estimate):
    def bare_implausibility_descent(estimate_index):
        candidate_position = np.copy(subset_estimate[estimate_index, :])
        for i in range(100):
            total_imp, tensor_input = imp_grad_emulate(candidate_position)
            jacobian = calculate_jacobian(total_imp, tensor_input).detach().numpy()
            jacobian /= np.linalg.norm(jacobian)
            candidate_position -= np.squeeze(1e-5*jacobian)
        return candidate_position

    def subset_descent(count):
        minimised_samples = []
        for index in range(count):
            minimised_samples.append(bare_implausibility_descent(index))
        return np.stack(minimised_samples)
    return


@app.cell
def _():
    # minimised_samples = subset_descent(300)
    return


@app.cell
def _(
    ANNI_DIST_MEAN,
    ANNI_DIST_STD,
    ANNI_MEAN,
    ANNI_SEM,
    CF_DIST_MEAN,
    CF_DIST_STD,
    CF_MEAN,
    CF_SEM,
    SPEED_DIST_MEAN,
    SPEED_DIST_STD,
    SPEED_MEAN,
    SPEED_SEM,
    anni_model_manager,
    cf_model_manager,
    np,
    speed_model_manager,
    torch,
):
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
        return prediction_mean, tensor_input


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
        return prediction_mean, tensor_input


    def imp_grad_emulate(x, bounded=False, bound_alpha=1e5):
        # Convert to tensor:
        tensor_input = torch.tensor(x, dtype=torch.float32)
        tensor_input.requires_grad_(True)

        # Ensure we have a batch dimension:
        if len(tensor_input.shape) == 1:
            tensor_input = torch.unsqueeze(tensor_input, 0)

        # Run CF emulation:
        cf_model_manager.model.eval()
        cf_model_manager.likelihood.eval()
        cf_prediction = cf_model_manager.likelihood(cf_model_manager.model(tensor_input))
        scaled_mean_cf = (cf_prediction.mean * CF_DIST_STD) + CF_DIST_MEAN
        scaled_std_cf = cf_prediction.stddev * CF_DIST_STD
        cf_implausibility = torch.abs(CF_MEAN - scaled_mean_cf) / torch.sqrt(CF_SEM**2 + scaled_std_cf**2)

        # Run ANNI emulation:
        anni_model_manager.model.eval()
        anni_model_manager.likelihood.eval()
        anni_prediction = anni_model_manager.likelihood(anni_model_manager.model(tensor_input))
        scaled_mean_anni = (anni_prediction.mean * ANNI_DIST_STD) + ANNI_DIST_MEAN
        scaled_std_anni = anni_prediction.stddev * ANNI_DIST_STD
        anni_implausibility = torch.abs(ANNI_MEAN - scaled_mean_anni) / torch.sqrt(ANNI_SEM**2 + scaled_std_anni**2)

        # Run speed emulation:
        speed_model_manager.model.eval()
        speed_model_manager.likelihood.eval()
        speed_prediction = speed_model_manager.likelihood(speed_model_manager.model(tensor_input))
        scaled_mean_speed = (speed_prediction.mean * SPEED_DIST_STD) + SPEED_DIST_MEAN
        scaled_std_speed = speed_prediction.stddev * SPEED_DIST_STD
        speed_implausibility = torch.abs(SPEED_MEAN - scaled_mean_speed) / torch.sqrt(SPEED_SEM**2 + scaled_std_speed**2)

        implausibility = torch.linalg.norm(torch.concatenate([cf_implausibility, anni_implausibility, speed_implausibility]))
        if bounded:
            lb_exceedance = torch.abs(torch.clip(0.0 - tensor_input, 0.0, None))
            ub_exceedance = torch.abs(torch.clip(tensor_input - 1.0, 0.0, None))
            lower_constraint = bound_alpha * torch.sum(lb_exceedance)
            upper_constraint = bound_alpha * torch.sum(ub_exceedance)
            total_imp = implausibility + lower_constraint + upper_constraint
        else:
            total_imp = implausibility
        return total_imp, tensor_input


    def distance_grad_emulate(x, bounded=False, bound_alpha=1e3):
        # Convert to tensor:
        tensor_input = torch.tensor(x, dtype=torch.float32)
        tensor_input.requires_grad_(True)

        # Ensure we have a batch dimension:
        if len(tensor_input.shape) == 1:
            tensor_input = torch.unsqueeze(tensor_input, 0)

        # Run CF emulation:
        cf_model_manager.model.eval()
        cf_model_manager.likelihood.eval()
        cf_prediction = cf_model_manager.likelihood(cf_model_manager.model(tensor_input))
        scaled_mean_cf = (cf_prediction.mean * CF_DIST_STD) + CF_DIST_MEAN
        cf_distance = torch.abs(CF_MEAN - scaled_mean_cf) / CF_MEAN

        # Run ANNI emulation:
        anni_model_manager.model.eval()
        anni_model_manager.likelihood.eval()
        anni_prediction = anni_model_manager.likelihood(anni_model_manager.model(tensor_input))
        scaled_mean_anni = (anni_prediction.mean * ANNI_DIST_STD) + ANNI_DIST_MEAN
        anni_distance = torch.abs(ANNI_MEAN - scaled_mean_anni) / ANNI_MEAN

        # Run speed emulation:
        speed_model_manager.model.eval()
        speed_model_manager.likelihood.eval()
        speed_prediction = speed_model_manager.likelihood(speed_model_manager.model(tensor_input))
        scaled_mean_speed = (speed_prediction.mean * SPEED_DIST_STD) + SPEED_DIST_MEAN
        speed_distance = torch.abs(SPEED_MEAN - scaled_mean_speed) / SPEED_MEAN

        displacement_vector = torch.concatenate([cf_distance, anni_distance, speed_distance])
        distance_norm = torch.linalg.norm(displacement_vector)
        unit_displacement = displacement_vector / distance_norm
        distance_entropy = torch.sum(-unit_displacement*torch.log(unit_displacement))

        composite_loss = distance_norm - distance_entropy
        if bounded:
            lb_exceedance = torch.abs(torch.clip(0.0 - tensor_input, 0.0, None))
            ub_exceedance = torch.abs(torch.clip(tensor_input - 1.0, 0.0, None))
            lower_constraint = bound_alpha * torch.sum(lb_exceedance)
            upper_constraint = bound_alpha * torch.sum(ub_exceedance)
            total_distance = composite_loss + lower_constraint + upper_constraint
        else:
            total_distance = composite_loss
        return total_distance, tensor_input


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
        jacobian = calculate_jacobian(y, x, create_graph=True)
        hessian = calculate_jacobian(jacobian, x, create_graph=create_graph)
        return jacobian, hessian
    return calculate_hessian, calculate_jacobian, imp_grad_emulate


@app.cell
def _(calculate_hessian, imp_grad_emulate, np):
    # def calculate_hessians(inputs):
    #     hessians = []
    #     for i in range(inputs.shape[0]):    
    #         # Estimate Hessian in local area:
    #         prediction_mean, tensor_input = grad_emulate(cf_model_manager, inputs[i, :])
    #         estimated_hessian = calculate_hessian(prediction_mean, tensor_input)
    #         estimated_hessian = np.squeeze(estimated_hessian.detach().numpy())
    #         hessians.append(estimated_hessian)

    #     return np.stack(hessians, axis=0)

    def calculate_hessians(inputs):
        hessians = []
        for i in range(inputs.shape[0]):
            if (i + 1) % 100 == 0:
                print(i + 1)
            # Estimate Hessian in local area:
            prediction_mean, tensor_input = imp_grad_emulate(inputs[i, :])
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
def _(calculate_eigens, calculate_hessians, np):
    # Set up sampled inputs across parameter space:
    input_rng = np.random.default_rng(0)
    sampled_inputs = input_rng.uniform(0, 1, size=(100, 10))
    sampled_hessians = calculate_hessians(sampled_inputs)
    sampled_eigenvalues, sampled_eigenvectors = calculate_eigens(sampled_hessians)
    return sampled_eigenvalues, sampled_eigenvectors, sampled_inputs


@app.cell
def _(cf_model_manager, emulate, sampled_inputs):
    sampled_cf, _ = emulate(cf_model_manager, sampled_inputs)
    return (sampled_cf,)


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
    return (plot_eigendirections,)


@app.cell
def _(
    np,
    plot_eigendirections,
    subset_eigenvalues,
    subset_eigenvectors,
    subset_estimate,
):
    plot_eigendirections(
        subset_estimate, subset_eigenvectors * np.expand_dims(np.sign(subset_eigenvalues), axis=1),
        0, 1, 0
    )
    return


@app.cell
def _(np, plt, sampled_eigenvalues):
    def plot_spectra_distribution():
        for i in range(10):
            plt.hist(np.log(np.abs(sampled_eigenvalues[:, i])), bins=10, alpha=0.5)
        plt.show()

    plot_spectra_distribution()
    return


@app.cell
def _(np, subset_eigenvectors):
    def calculate_geometric_median(key_eigenvectors):
        proposal_median = np.mean(key_eigenvectors, axis=0)    
        converged = False
        while not converged:
            # Get distances:
            cosine_similarities = key_eigenvectors @ np.expand_dims(proposal_median, axis=1)
            nematic_distances = (1 - np.abs(cosine_similarities)) + + 1e-8

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

    proposal_median, flipped_vectors = calculate_geometric_median(subset_eigenvectors[:, 0, :])
    print(proposal_median)
    return flipped_vectors, proposal_median


@app.cell
def _(flipped_vectors, np, proposal_median):
    distances = np.squeeze(1 - np.abs(flipped_vectors @ np.expand_dims(proposal_median, axis=1)))
    return (distances,)


@app.cell
def _(distances, plt):
    plt.hist(distances, bins=100);
    plt.show()
    return


@app.cell
def _(distances, np):
    key_index = np.argmin(distances)
    return (key_index,)


@app.cell
def _(key_index, np, subset_eigenvectors):
    eigenparameter = np.copy(subset_eigenvectors[key_index, 1, :])
    eigenparameter[np.abs(eigenparameter) < 0.1] = 0
    eigenparameter /= np.linalg.norm(eigenparameter)
    eigenparameter = np.round(eigenparameter, decimals=2)
    eigenparameter
    return (eigenparameter,)


@app.cell
def _(eigenparameter, gridsearch_parameters):
    # Print eigenparameter description:
    def print_eigenparameter(eigenparameter):
        for i in range(10):
            if eigenparameter[i] != 0:
                print(list(gridsearch_parameters.keys())[i], eigenparameter[i])

    print_eigenparameter(eigenparameter)
    return


@app.cell
def _(eigenparameter, np, sampled_inputs, subset_estimate):
    sampled_ep_values = sampled_inputs @ np.expand_dims(eigenparameter, axis=1)
    subset_ep_values = subset_estimate @ np.expand_dims(eigenparameter, axis=1)
    return sampled_ep_values, subset_ep_values


@app.cell
def _(plt, sampled_ep_values, subset_ep_values):
    plt.hist(sampled_ep_values, bins=25, alpha=0.5, density=True);
    plt.hist(subset_ep_values, bins=25, alpha=0.5, density=True);
    plt.show()
    return


@app.cell
def _(flipped_vectors, plt, proposal_median):
    def plot_referenced_distribution(component_i, component_j):
        fig, ax = plt.subplots()
        ax.scatter(flipped_vectors[:, component_i], flipped_vectors[:, component_j], s=0.1, alpha=0.5)
        ax.scatter(proposal_median[component_i], proposal_median[component_j], s=10)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        plt.show()

    plot_referenced_distribution(4, 5)
    return


@app.cell
def _():
    # # @numba.jit(nopython=True)
    # def get_undirected_matrix(x, reference_vector=np.array([])):
    #     out = np.zeros_like(x)

    #     for row_index in range(x.shape[0]):
    #         # Compare to reference vector:
    #         cosine_similarity = np.dot(reference_vector, x[row_index, :])

    #         # Flip if necessary:
    #         if cosine_similarity >= 0:
    #             out[row_index, :] = x[row_index, :]
    #         else:
    #             out[row_index, :] = -x[row_index, :]

    #     return out
    return


@app.cell
def _(get_undirected_matrix, np, sampled_eigenvectors):
    reference_vector = np.ones(10) / np.linalg.norm(np.ones(10))
    undirected_sample = get_undirected_matrix(sampled_eigenvectors[:, 0, :], reference_vector)
    return reference_vector, undirected_sample


@app.cell
def _(reference_vector):
    reference_vector
    return


@app.cell
def _(plt, undirected_sample):
    def undirected_plot():
        fig, ax = plt.subplots()
        ax.scatter(undirected_sample[:, 0], undirected_sample[:, 5], s=1)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        plt.show()

    undirected_plot()
    return


@app.cell
def _(
    np,
    plot_eigendirections,
    subset_eigenvalues,
    subset_eigenvectors,
    subset_estimate,
):
    plot_eigendirections(
        subset_estimate, subset_eigenvectors * np.expand_dims(np.sign(subset_eigenvalues), axis=1),
        4, 5, 0
    )
    return


@app.cell
def _(plt, subset_eigenvalues):
    plt.hist(subset_eigenvalues[:, 0], bins=100);
    plt.show()
    return


@app.cell
def _(
    np,
    sampled_eigenvalues,
    sampled_eigenvectors,
    subset_eigenvalues,
    subset_eigenvectors,
):
    import umap
    import numba

    @numba.njit()
    def nematic_cosine(a, b):
        return 1 - np.abs(np.dot(a, b))

    # Select largest eigenvector as position:
    umap_model = umap.UMAP(metric=nematic_cosine, n_neighbors=25, min_dist=0, random_state=0)
    sampled_embeddings = umap_model.fit_transform(
        sampled_eigenvectors[:, 0, :] * np.expand_dims(np.sign(sampled_eigenvalues[:, 0]), axis=1)
    )
    subset_embeddings = umap_model.transform(
        subset_eigenvectors[:, 0, :] * np.expand_dims(np.sign(subset_eigenvalues[:, 0]), axis=1)
    )
    return sampled_embeddings, subset_embeddings


@app.cell
def _(effective_ranks, np, plt, sampled_embeddings, subset_embeddings):
    import colorcet as cc

    def plot_umap():
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(sampled_embeddings[:, 0], sampled_embeddings[:, 1], alpha=0.1, s=0.1, c='k')
        ax.scatter(
            subset_embeddings[:, 0],
            subset_embeddings[:, 1],
            alpha=0.5, s=1,
            c=np.log(effective_ranks),
            cmap=cc.m_CET_D1A
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        fig.tight_layout()
        plt.show()

    plot_umap()
    return (cc,)


@app.cell
def _(
    gridsearch_parameters,
    np,
    plt,
    sampled_eigenvectors,
    sampled_embeddings,
):
    def plot_component_umap():
        fig, ax = plt.subplots(figsize=(10, 10))
        largest_components = np.argmax(np.abs(sampled_eigenvectors[:, 0, :]), axis=1)
        largest_components = np.argsort(np.abs(sampled_eigenvectors[:, 0, :]), axis=1)
        largest_components = largest_components[:, -1]
        parameter_entropy = -np.sum(np.abs(sampled_eigenvectors[:, 0, :]) * np.log(np.abs(sampled_eigenvectors[:, 0, :])), axis=1)
        parameter_entropy = (parameter_entropy - np.min(parameter_entropy)) / (np.max(parameter_entropy) - np.min(parameter_entropy))
        for p_i in range(10):
            component_mask = largest_components == p_i
            if np.count_nonzero(component_mask) == 0:
                continue
            ax.scatter(
                sampled_embeddings[component_mask, 0],
                sampled_embeddings[component_mask, 1],
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
def _(cc, parameter_entropy, plt, sampled_embeddings):
    def plot_entropy_umap():
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(
            sampled_embeddings[:, 0],
            sampled_embeddings[:, 1],
            c=parameter_entropy, cmap=cc.m_CET_L20,
            alpha=0.5,
            s=0.5,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        fig.tight_layout()
        plt.show()

    plot_entropy_umap()
    return


@app.cell
def _(
    CF_MEAN,
    cc,
    np,
    plt,
    sampled_cf,
    sampled_eigenvectors,
    sampled_embeddings,
):
    def plot_cf_umap():
        fig, ax = plt.subplots(figsize=(10, 10))
        parameter_entropy = -np.sum(np.abs(sampled_eigenvectors[:, 0, :]) * np.log(np.abs(sampled_eigenvectors[:, 0, :])), axis=1)
        ax.scatter(
            sampled_embeddings[:, 0],
            sampled_embeddings[:, 1],
            c=np.abs(sampled_cf - CF_MEAN), cmap=cc.m_CET_L20,
            alpha=0.5,
            s=0.5,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        fig.tight_layout()
        plt.show()

    plot_cf_umap()
    return


@app.cell
def _(np, plt, subset_eigenvalues):
    mb_proximity = np.log(np.min(np.abs(subset_eigenvalues), axis=1))
    plt.hist(mb_proximity, bins=100);
    plt.show()
    return (mb_proximity,)


@app.cell
def _(np, subset_eigenvalues):
    norm_eigval = np.abs(subset_eigenvalues) / np.linalg.norm(np.abs(subset_eigenvalues), axis=1, keepdims=True)
    effective_ranks = np.sum(-norm_eigval * np.log(norm_eigval), axis=1)
    return (effective_ranks,)


@app.cell
def _(effective_ranks, plt):
    plt.hist(effective_ranks, bins=100);
    plt.show()
    return


@app.cell
def _(mb_proximity, np, plt, subset_eigenvalues, subset_eigenvectors):
    def plot_colour_eigendirections(c_array, eigenvectors, parameter_i, parameter_j, eigenvector_index=0):
        fig, ax = plt.subplots()

        # Plot components
        ax.scatter(
            eigenvectors[:, eigenvector_index, parameter_i],
            eigenvectors[:, eigenvector_index, parameter_j],
            s=1, alpha=0.5, c=c_array
        )
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        plt.show()

    # manifold_boundary_mask = effective_ranks < np.quantile(effective_ranks, 0.1)
    manifold_boundary_mask = mb_proximity < np.quantile(mb_proximity, 1)

    plot_colour_eigendirections(
        mb_proximity[manifold_boundary_mask],
        subset_eigenvectors[manifold_boundary_mask] * np.expand_dims(np.sign(subset_eigenvalues[manifold_boundary_mask]), axis=1),
        0, 1, 0
    )
    return


@app.cell
def _(mb_proximity):
    mb_proximity.shape
    return


@app.cell
def _():
    # def calculate_effective_ranks(hessians):
    #     effective_ranks = []
    #     for hessian_index in range(hessians.shape[0]):
    #         svd_values = np.linalg.svdvals(hessians[hessian_index, :, :])
    #         svd_spectrum = (1 / svd_values) / np.sum(1 / svd_values)
    #         erank = np.exp(np.sum(-svd_spectrum*np.log(svd_spectrum)))
    #         effective_ranks.append(erank)
    #     return effective_ranks

    # sampled_ers = calculate_effective_ranks(sampled_hessians)
    return


@app.cell
def _():
    # plt.hist(sampled_ers, bins=100);
    # plt.show()
    return


@app.cell
def _():
    # def plot_eigendirections(parameter_i, parameter_j, eigenvector_index=0):
    #     fig, ax = plt.subplots()
    #     ax.scatter(
    #         sampled_eigvectors[:, eigenvector_index, parameter_i],
    #         sampled_eigvectors[:, eigenvector_index, parameter_j],
    #         s=1, alpha=0.5, c=sampled_ers
    #     )
    #     ax.set_xlim(-1.05, 1.05)
    #     ax.set_ylim(-1.05, 1.05)
    #     ax.set_aspect("equal")
    #     plt.show()

    # plot_eigendirections(4, 5, 0)
    return


@app.cell
def _(hessians, np):
    def extract_key_eigenvectors():
        key_eigenvectors = []
        for hessian in hessians:
            eig_result = np.linalg.eig(hessian)
            key_eigenvector = eig_result.eigenvectors[0, :]
            key_eigenvectors.append(key_eigenvector)
        return np.stack(key_eigenvectors, axis=0)

    key_eigenvectors = extract_key_eigenvectors()
    # key_eigenvectors *= np.expand_dims(np.sign(key_eigenvectors[:, 0]), axis=1)
    return (key_eigenvectors,)


@app.cell
def _(key_eigenvectors):
    key_eigenvectors.shape
    return


@app.cell
def _(key_eigenvectors):
    # averaged_parameter = np.mean(key_eigenvectors, axis=0) 
    # averaged_parameter /= np.linalg.norm(averaged_parameter)
    averaged_parameter = key_eigenvectors[0, :]
    return (averaged_parameter,)


@app.cell
def _(averaged_parameter, np, parameter_matrix, plt, subset_estimate):
    all_evs = np.log(parameter_matrix) @ np.expand_dims(averaged_parameter, axis=1)
    subset_evs = np.log(subset_estimate) @ np.expand_dims(averaged_parameter, axis=1)
    plt.hist(all_evs, bins=100, density=True, alpha=0.5);
    plt.hist(subset_evs, bins=100, density=True, alpha=0.5);
    plt.show()
    return (subset_evs,)


@app.cell
def _(plt, subset_cf, subset_evs):
    plt.scatter(subset_evs, subset_cf, s=1)
    return


@app.cell
def _(key_eigenvectors, np):
    np.linalg.norm(key_eigenvectors, axis=1)
    return


@app.cell
def _(key_eigenvectors, np):
    similarity_matrix = np.dot(key_eigenvectors, key_eigenvectors.T)
    flip_mask = similarity_matrix < 0
    similarity_matrix[similarity_matrix < 0] = -similarity_matrix[similarity_matrix < 0]
    return (similarity_matrix,)


@app.cell
def _(plt, similarity_matrix):
    plt.hist(similarity_matrix.flatten(), bins=100);
    plt.show()
    return


@app.cell
def _():
    # import sklearn
    # mds_manager = sklearn.manifold.MDS(dissimilarity="precomputed")
    # metric_space = mds_manager.fit_transform(1 - similarity_matrix)
    return


@app.cell
def _(cf_model_manager, emulate, subset_estimate):
    subset_cf, _ = emulate(cf_model_manager, subset_estimate)
    # plt.scatter(metric_space[:, 0], metric_space[:, 1], c=-np.log(subset_imp), s=1, alpha=0.25)
    return (subset_cf,)


@app.cell
def _(key_eigenvectors, plt):
    plt.hist(key_eigenvectors[:, 0], bins=50);
    plt.show()
    return


@app.cell
def _(key_eigenvectors, plt, subset_imp):
    def plot_eigenvector_components():
        fig, ax = plt.subplots()
        ax.scatter(key_eigenvectors[:, 5], key_eigenvectors[:, 7], c=subset_imp, s=1)
        ax.set_aspect("equal")
        plt.show()

    plot_eigenvector_components()
    return


@app.cell
def _(subset_estimate):
    subset_estimate.shape
    return


@app.cell
def _(sample_markov_chain, subset_estimate):
    populated_subset = sample_markov_chain(subset_estimate, 0.05, 100, 1, 0.1)
    return (populated_subset,)


@app.cell
def _(populated_subset):
    populated_subset.shape
    return


@app.cell
def _(plt, populated_subset):
    plt.hist(populated_subset[:, 8], bins=50);
    plt.show()
    return


@app.cell
def _(get_implausibility_metric, minimised_samples):
    subset_imp = get_implausibility_metric(minimised_samples)
    return (subset_imp,)


@app.cell
def _(subset_imp):
    subset_imp
    return


@app.cell
def _(developed_estimate, distance_norms, np, plt):
    def plot_subset_estimate(samples, dimensions=12):
        fig, axs = plt.subplots(dimensions, dimensions, figsize=(10, 10), sharex=True, sharey=True, layout="constrained")
        for i in range(dimensions):
            for j in range(dimensions):
                axs[i, j].scatter(
                    samples[:, i], samples[:, j],
                    alpha=0.5,
                    s=0.05
                )
                axs[i, j].set_xlim(0, 1)
                axs[i, j].set_ylim(0, 1)
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                axs[i, j].set_aspect("equal")
        plt.show()

    # plot_subset_estimate(populated_subset[:, :])
    distance_mask = distance_norms < np.quantile(distance_norms, 0.01)
    plot_subset_estimate(developed_estimate[distance_mask])
    return (plot_subset_estimate,)


@app.cell
def _(dpp_estimate, plot_subset_estimate):
    plot_subset_estimate(dpp_estimate)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
