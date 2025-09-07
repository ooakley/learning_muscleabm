import marimo

__generated_with = "0.12.10"
app = marimo.App(width="columns")


@app.cell
def _():
    import os
    import ot
    import json
    import skimage
    import torch
    import gpytorch

    import scipy.stats
    import scipy.spatial

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import colorcet as cc

    import matplotlib.pyplot as plt 

    from torch.utils.data import TensorDataset, DataLoader
    from scipy.stats import qmc

    DATA_DIRECTORY = "./wetlab_data/OEO20241206"
    ROWS = ["A", "B", "C"]
    COLUMNS = ["1", "2", "3", "4", "5", "6"]
    return (
        COLUMNS,
        DATA_DIRECTORY,
        DataLoader,
        ROWS,
        TensorDataset,
        cc,
        gpytorch,
        json,
        np,
        os,
        ot,
        pd,
        plt,
        qmc,
        scipy,
        skimage,
        sns,
        torch,
    )


@app.cell
def _(np, skimage):
    def find_coherency_fraction(site_data):
        # Get valid cell indices:
        cell_index_list = list(set(list(site_data["tree_id"])))

        # Estimate from trajectory dataframe as test:
        line_array = np.zeros((1024, 1024))
        for cell_index in cell_index_list:
            particle_mask = site_data["tree_id"] == cell_index
            particle_data = site_data[particle_mask].sort_values("frame")
            xy_data = np.array(particle_data.loc[:, ["x", "y"]])
            for frame_index in range(len(xy_data) - 1):
                # Get indices of line:
                xy_t = xy_data[frame_index, :].astype(int)
                xy_t1 = xy_data[frame_index+1, :].astype(int)

                # Account for periodic boundaries:
                distance = np.sqrt(np.sum((xy_t - xy_t1)**2, axis=0))
                if distance > 1024:
                    continue

                # Plot line indices on matrix:
                _rr, _cc = skimage.draw.line(*xy_t, *xy_t1)
                line_array[_rr, _cc] += 1

        # Find orientations of lines:
        structure_tensor = skimage.feature.structure_tensor(
            line_array, sigma=32,
            mode='constant', cval=0,
            order='rc'
        )

        eigenvalues = skimage.feature.structure_tensor_eigenvalues(structure_tensor)
        coherency_numerator = eigenvalues[0, :, :] - eigenvalues[1, :, :]
        coherency_denominator = eigenvalues[0, :, :] + eigenvalues[1, :, :]
        coherency = coherency_numerator / coherency_denominator

        line_array_mask = line_array > 0
        coherency_fraction = np.sum(coherency[line_array_mask]) / np.sum(line_array)
        return coherency_fraction
    return (find_coherency_fraction,)


@app.cell
def _(DATA_DIRECTORY, os, pd):
    processed_dataset_filepath = os.path.join(DATA_DIRECTORY, "fitting_dataset.csv")
    experiment_dataframe = pd.read_csv(processed_dataset_filepath, index_col=0)
    return experiment_dataframe, processed_dataset_filepath


@app.cell
def _(experiment_dataframe):
    experiment_dataframe
    return


@app.cell
def _(experiment_dataframe, plt, sns):
    _fig, _axs = plt.subplots(3, 1, figsize=(4, 10))
    sns.violinplot(data=experiment_dataframe, x="column", y="speed", ax=_axs[0])
    sns.violinplot(data=experiment_dataframe, x="column", y="meander_ratio", ax=_axs[1])
    sns.violinplot(data=experiment_dataframe, x="column", y="mean_nn_distance", ax=_axs[2])
    return


@app.cell
def _(COLUMNS, DATA_DIRECTORY, ROWS, os, pd):
    # Get wet lab data:
    print("Loading and processing wet lab trajectory data...")
    trajectory_folderpath = os.path.join(DATA_DIRECTORY, "trajectories")
    trajectory_dictionary = {column: [] for column in COLUMNS}

    # Each different cell type is contained in different columns of 3 wells,
    # with four sites each:
    for row in ROWS:
        for column in COLUMNS:
            for site in range(4):
                csv_filename = f"{row}{column}-Site_{site}.csv"
                site_dataframe = pd.read_csv(
                    os.path.join(trajectory_folderpath, csv_filename), index_col=0
                )
                trajectory_dictionary[column].append(site_dataframe)
    return (
        column,
        csv_filename,
        row,
        site,
        site_dataframe,
        trajectory_dictionary,
        trajectory_folderpath,
    )


@app.cell
def _(find_coherency_fraction, trajectory_dictionary):
    def get_cf_dictionary(trajectory_dictionary):
        cf_dictionary = {}
        for _column in ["1", "2", "3", "4", "5", "6"]:
            coherency_fractions = []
            for _i in range(12):
                site_data = trajectory_dictionary[_column][_i]
                coherency_fractions.append(find_coherency_fraction(site_data))
            cf_dictionary[_column] = coherency_fractions
        return cf_dictionary

    cf_dictionary = get_cf_dictionary(trajectory_dictionary)
    return cf_dictionary, get_cf_dictionary


@app.cell
def _(np):
    def get_frame_anni(frame_positions):
        # Get distance matrix, taken from:
        # https://stackoverflow.com/questions/22720864/efficiently-calculating-a-euclidean-distance-matrix-using-numpy
        displacements = \
            frame_positions[:, np.newaxis, :] - frame_positions[np.newaxis, :, :]
        distance_sq = np.sum(displacements ** 2, axis=-1)
        distance_matrix = np.sqrt(distance_sq) * 2 # Account for binning

        # Set all diagonal entries to a large number, so minimum func can be broadcast:
        diagonal_idx = np.diag_indices(distance_matrix.shape[0], 2)
        distance_matrix[diagonal_idx] = 2048
        minimum_distances = np.min(distance_matrix, axis=1)

        # Get ratio of mean NN distance to expected distance:
        expected_minimum = 0.5 / np.sqrt(len(minimum_distances) / (2048 * 2048))
        anni = np.mean(minimum_distances) / expected_minimum
        return anni

    def get_mean_anni(site_data):
        anni_timeseries = []
        for frame in list(set(list(site_data["frame"]))):
            frame_mask = site_data["frame"] == frame
            frame_positions = np.array(site_data[frame_mask].loc[:, ['x', 'y']])
            anni_timeseries.append(get_frame_anni(frame_positions))
        return np.mean(anni_timeseries)
    return get_frame_anni, get_mean_anni


@app.cell
def _(get_mean_anni, trajectory_dictionary):
    def get_anni_dictionary(trajectory_dictionary):
        anni_dictionary = {}
        for column in ["1", "2", "3", "4", "5", "6"]:
            ann_indices = []
            for i in range(12):
                site_data = trajectory_dictionary[column][i]
                ann_indices.append(get_mean_anni(site_data))
            anni_dictionary[column] = ann_indices
        return anni_dictionary

    anni_dictionary = get_anni_dictionary(trajectory_dictionary)
    return anni_dictionary, get_anni_dictionary


@app.cell
def _(COLUMNS, anni_dictionary, cf_dictionary, experiment_dataframe, np):
    speed_boundaries = []
    for idx in range(1, 7):
        column_mask = experiment_dataframe["column"] == idx
        col_speeds = experiment_dataframe[column_mask].loc[:, "speed"]
        mean_speed = np.mean(col_speeds)
        std_speed = 0.5*np.std(col_speeds)
        speed_boundaries.append([mean_speed - std_speed, mean_speed + std_speed])

    cf_imp_boundaries = []
    for _column in COLUMNS:
        cell_type_cf_mean = np.mean(cf_dictionary[_column])
        cell_type_cf_std = np.std(cf_dictionary[_column])
        cf_imp_boundaries.append([
            cell_type_cf_mean - cell_type_cf_std,
            cell_type_cf_mean + cell_type_cf_std
        ])

    anni_imp_boundaries = []
    for _column in COLUMNS:
        cell_type_anni_mean = np.mean(anni_dictionary[_column])
        cell_type_anni_std = np.std(anni_dictionary[_column])
        anni_imp_boundaries.append([
            cell_type_anni_mean - cell_type_anni_std,
            cell_type_anni_mean + cell_type_anni_std
        ])
    return (
        anni_imp_boundaries,
        cell_type_anni_mean,
        cell_type_anni_std,
        cell_type_cf_mean,
        cell_type_cf_std,
        cf_imp_boundaries,
        col_speeds,
        column_mask,
        idx,
        mean_speed,
        speed_boundaries,
        std_speed,
    )


@app.cell
def _():
    # Interrogate Circular Collision data:
    return


@app.cell
def _():
    CONSTANT_PARAMETERS = {
        "superIterationCount": 12,
        "timestepsToRun": 2880,
        "worldSize": 2048,
        "gridSize": 64,
        "dt": 1,
        "thereIsMatrixInteraction": 1,
        "aspectRatio": 1,
        "stretchFactor": 0.01,
        "slipFactor": 1
    }

    CIRCULAR_GRIDSEARCH_PARAMETERS = {
        "cueDiffusionRate": [0.0001, 2.5],
        "cueKa": [0.0001, 5],
        "fluctuationAmplitude": [1e-5, 0.01],
        "fluctuationTimescale": [1, 75],
        "maximumSteadyStateActinFlow": [0.0, 3],
        "numberOfCells": [75, 175],
        "actinAdvectionRate": [0.0, 3],
        "cellBodyRadius": [5, 75],
        "collisionFlowReductionRate": [0.0, 1],
        "collisionAdvectionRate": [0.0, 1.5]
    }

    TD_GRIDSEARCH_PARAMETERS = {
        "cueDiffusionRate": [0.0001, 2.5],
        "cueKa": [0.0001, 5],
        "fluctuationAmplitude": [1e-5, 1e-2],
        "fluctuationTimescale": [1, 75],
        "maximumSteadyStateActinFlow": [0.1, 3],
        "numberOfCells": [75, 175],
        "actinAdvectionRate": [0.1, 3],
        "cellBodyRadius": [15, 100],
        # Collisions:
        "collisionFlowReductionRate": [0.0, 1],
        "collisionAdvectionRate": [0.0, 1],
        # Shape:
        "stretchFactor": [0.25, 3],
        "slipFactor": [0.0001, 0.1],
    }
    return (
        CIRCULAR_GRIDSEARCH_PARAMETERS,
        CONSTANT_PARAMETERS,
        TD_GRIDSEARCH_PARAMETERS,
    )


@app.cell
def _(
    CIRCULAR_GRIDSEARCH_PARAMETERS,
    TD_GRIDSEARCH_PARAMETERS,
    anni_imp_boundaries,
    cf_imp_boundaries,
    json,
    np,
    speed_boundaries,
):
    def analyse_gridsearch_folder(folder_name, cell_index, circular=False):
        parameters = np.load(
            f"./gridsearch_data/{folder_name}/collated_inputs.npy"
        )
        distances = np.load(
            f"./gridsearch_data/{folder_name}/collated_distances.npy"
        )
        order_parameters = np.load(
            f"./gridsearch_data/{folder_name}/collated_order_parameters.npy"
        )
        speeds = np.load(
            f"./gridsearch_data/{folder_name}/collated_magnitudes.npy"
        )
        coherency_fractions = np.load(
            f"./gridsearch_data/{folder_name}/coherency_fractions.npy"
        )
        ann_indices = np.load(
            f"./gridsearch_data/{folder_name}/ann_indices.npy"
        )

        if circular:
            gridsearch_parameters = CIRCULAR_GRIDSEARCH_PARAMETERS
        else:
            gridsearch_parameters = TD_GRIDSEARCH_PARAMETERS

        # Discard failed simulations:
        nan_mask = np.any(np.isnan(distances), axis=(1, 2))
        nan_parameters = parameters[nan_mask, :]
        parameters = parameters[~nan_mask, :]
        distances = distances[~nan_mask, :]

        order_parameters = order_parameters[~nan_mask, 0]
        speeds = speeds[~nan_mask, 0]

        coherency_fractions = coherency_fractions[~nan_mask, :]
        ann_indices = ann_indices[~nan_mask, :]

        normalised_parameters = np.zeros_like(parameters)
        for index, parameter_range in enumerate(gridsearch_parameters.values()):
            _minimum = parameter_range[0]
            _maximum = parameter_range[1]
            normalised_parameters[:, index] = \
                (parameters[:, index] - _minimum) / (_maximum - _minimum)

        euclidean_distances = np.sqrt(np.sum(distances[:, :, :3]**2, axis=2))

        # For copying to PRISM:
        mean_coherency = np.mean(coherency_fractions, axis=1)
        mean_anni = np.mean(ann_indices, axis=1)

        # plt.hist(distances[:, CELL_INDEX, 0])
        # plt.show()

        # Generate implausibility criterion:
        mask_cf = np.logical_and(
            cf_imp_boundaries[cell_index][0] < mean_coherency,
            mean_coherency < cf_imp_boundaries[cell_index][1]
        )
        mask_anni = np.logical_and(
            anni_imp_boundaries[cell_index][0] < mean_anni,
            mean_anni < anni_imp_boundaries[cell_index][1]
        )
        mask_speed = np.logical_and(
            speed_boundaries[cell_index][0] < speeds,
            speeds < speed_boundaries[cell_index][1]
        )
        mask_speed = distances[:, cell_index, 0] < 1
        implausibility_mask = np.all(
            np.stack([mask_cf, mask_speed], axis=1),
            axis=1
        )
        print(np.count_nonzero(implausibility_mask))
        # print(parameters[implausibility_mask, :])

        optim_params = dict(zip(
            list(gridsearch_parameters.keys()),
            parameters[implausibility_mask, :][0]
        ))
        print(json.dumps(optim_params, sort_keys=False, indent=4))

        # # Apply implausibility criterion:
        # plausible_parameters = normalised_parameters[implausibility_mask, :]
        # plausible_distances = distances[implausibility_mask, :]

        # # Get scores:
        # quantile_boundaries = np.linspace(0, 1.0, 100)
        # quantile_list = []

        # for quantile in quantile_boundaries:
        #     distance_quantile = np.quantile(
        #         distances[:, CELL_INDEX, :3],
        #         quantile, axis=0
        #     )
        #     quantile_list.append(distance_quantile)

        # all_scores = []
        # for idx in range(len(quantile_list)):
        #     # Get masks:
        #     quantile_check = distances[:, CELL_INDEX, :3] <= quantile_list[idx]
        #     mask = np.all(quantile_check, axis=1)
        #     all_scores.extend([[valid_index[0], idx] for valid_index in np.argwhere(mask)])

        # all_scores = np.stack(all_scores, axis=0)
        # best_scores = []
        # for run_index in range(len(distances)):
        #     run_mask = all_scores[:, 0] == run_index
        #     minimum_score = np.min(all_scores[:, 1][run_mask])
        #     best_scores.append(minimum_score)
        # best_scores = np.array(best_scores)

        # plt.hist(best_scores[mask_cf])
        # plt.show()

        # print(coherency_fractions[implausibility_mask][np.argmin(best_scores)].tolist())
        # print(distances[np.argmin(best_scores), CELL_INDEX, :3].tolist())

        # optim_params = dict(zip(
        #     list(gridsearch_parameters.keys()),
        #     parameters[implausibility_mask, :][np.argmin(best_scores), :]
        # ))
        # print(json.dumps(optim_params, sort_keys=False, indent=4))
    return (analyse_gridsearch_folder,)


@app.cell
def _(analyse_gridsearch_folder):
    for _i in range(6):
        analyse_gridsearch_folder("out_20250822CircularCollisions", _i, True)
    return


@app.cell
def _(analyse_gridsearch_folder):
    for _i in range(6):
        analyse_gridsearch_folder("out_20250826TrajectoryCollisions", _i, False)
    return


@app.cell
def _():
    # Get better idea of overlapping regions from GP modelling:
    return


@app.cell
def _(DataLoader, TensorDataset, gpytorch, torch):
    class ApproximateGPModel(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points, dimensions):
            # Set up distribution:
            variational_distribution = \
                gpytorch.variational.MeanFieldVariationalDistribution(
                    inducing_points.shape[0]
            )

            # Set up variational strategy:
            variational_strategy = gpytorch.variational.NNVariationalStrategy(
                self, inducing_points, variational_distribution,
                k=256, training_batch_size=64
            )

            # Inherit rest of init logic from approximate gp:
            super(ApproximateGPModel, self).__init__(variational_strategy)

            # Defining scale parameter prior:
            mu_0 = 0
            sigma_0 = 1
            lognormal_prior = gpytorch.priors.LogNormalPrior(
                mu_0, sigma_0
            )

            # Define mean and convariance functions:
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=dimensions,
                    lengthscale_prior=lognormal_prior
                )
            )

        def forward(self, x):
            mean = self.mean_module(x)
            covars = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean, covars)

        def __call__(self, x, prior=False, **kwargs):
            return self.variational_strategy(x=x, prior=False, **kwargs)

    def instantiate_model(inducing_points, dimensions):
        # Convert inducing points to torch:
        inducing_points = torch.tensor(
            inducing_points, dtype=torch.float32
        )

        # Set up likelihoods:
        likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        model = ApproximateGPModel(inducing_points, dimensions)
        return model, likelihood

    def train_model(model, likelihood, x_dataset, y_dataset, epochs=50):
        # Convert datasets to torch:
        train_x = torch.tensor(x_dataset, dtype=torch.float32)
        train_y = torch.tensor(y_dataset, dtype=torch.float32)

        # Initialise dataloaders:
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

        # Set up training process:
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.1)

        # Set up loss:
        # mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
        mll = gpytorch.mlls.VariationalELBO(
            likelihood, model, num_data=train_y.size(0)
        )
        loss_history = []
        # for i in range(epochs):
        #     # Run through entire dataset:
        #     for x_batch, y_batch in train_loader:
        #         optimizer.zero_grad()
        #         output = model(x_batch)
        #         loss = -mll(output, y_batch)
        #         loss.backward()
        #         optimizer.step()


        for i in range(epochs):
            for _ in range(model.variational_strategy._total_training_batches):
                optimizer.zero_grad()
                output = model(x=None)
                current_training_indices = \
                    model.variational_strategy.current_training_indices
                y_batch = train_y[current_training_indices]
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()

            # Print progress:
            if (i + 1) % 5 == 0:
                print(i + 1)
                print(loss.detach())

            loss_history.append(loss.detach())

        return loss_history
    return ApproximateGPModel, instantiate_model, train_model


@app.cell
def _(TD_GRIDSEARCH_PARAMETERS, np):
    def get_gridsearch_data(folder_name):
        parameters = np.load(
            f"./gridsearch_data/{folder_name}/collated_inputs.npy"
        )
        distances = np.load(
            f"./gridsearch_data/{folder_name}/collated_distances.npy"
        )
        order_parameters = np.load(
            f"./gridsearch_data/{folder_name}/collated_order_parameters.npy"
        )
        speeds = np.load(
            f"./gridsearch_data/{folder_name}/collated_magnitudes.npy"
        )
        coherency_fractions = np.load(
            f"./gridsearch_data/{folder_name}/coherency_fractions.npy"
        )
        ann_indices = np.load(
            f"./gridsearch_data/{folder_name}/ann_indices.npy"
        )

        gridsearch_parameters = TD_GRIDSEARCH_PARAMETERS

        # Discard failed simulations:
        nan_mask = np.any(np.isnan(distances), axis=(1, 2))
        nan_parameters = parameters[nan_mask, :]

        # Apply to remaining matrices:
        parameters = parameters[~nan_mask, :]
        distances = distances[~nan_mask, :]
        order_parameters = order_parameters[~nan_mask, 0]
        speeds = speeds[~nan_mask, 0]
        coherency_fractions = coherency_fractions[~nan_mask, :]
        ann_indices = ann_indices[~nan_mask, :]

        # Get normalised parameters:
        normalised_parameters = np.zeros_like(parameters)
        for index, parameter_range in enumerate(gridsearch_parameters.values()):
            _minimum = parameter_range[0]
            _maximum = parameter_range[1]
            normalised_parameters[:, index] = \
                (parameters[:, index] - _minimum) / (_maximum - _minimum)

        return normalised_parameters, distances, coherency_fractions, ann_indices
    return (get_gridsearch_data,)


@app.cell
def _(get_gridsearch_data):
    normalised_parameters, distances, coherency_fractions, ann_indices = \
        get_gridsearch_data("out_20250826TrajectoryCollisions")
    return ann_indices, coherency_fractions, distances, normalised_parameters


@app.cell
def _(TD_GRIDSEARCH_PARAMETERS, normalised_parameters, qmc):
    exponent = 10
    print(2**exponent)
    # sobol_sampler = qmc.Sobol(d=len(TD_GRIDSEARCH_PARAMETERS), scramble=True, rng=0)
    # sample_matrix = sobol_sampler.random_base2(m=exponent)
    sobol_sampler = qmc.Sobol(d=len(TD_GRIDSEARCH_PARAMETERS), scramble=True, rng=0)
    # inducing_points = sobol_sampler.random_base2(m=exponent) # 9 -> 512 inducing points
    inducing_points = normalised_parameters
    return exponent, inducing_points, sobol_sampler


@app.cell
def _():
    CELL_INDEX = 0
    return (CELL_INDEX,)


@app.cell
def _(
    TD_GRIDSEARCH_PARAMETERS,
    inducing_points,
    instantiate_model,
    normalised_parameters,
    train_model,
):
    def generate_criterion_model(classification_dataset):
        model, likelihood = instantiate_model(
            inducing_points,
            len(TD_GRIDSEARCH_PARAMETERS)
        )
        l_hist = train_model(
            model, likelihood,
            normalised_parameters, classification_dataset,
            epochs=10
        )
        return model, likelihood
    return (generate_criterion_model,)


@app.cell
def _(DataLoader, TensorDataset, np, torch):
    def run_inference(model, likelihood, inputs):
        tensor_input = torch.tensor(inputs, dtype=torch.float32)
        inference_dataset = TensorDataset(tensor_input)
        inference_loader = DataLoader(inference_dataset, batch_size=256, shuffle=False)

        model.eval()
        likelihood.eval()
        predictions = []
        with torch.no_grad():
            for idx, parameter_batch in enumerate(inference_loader):
                batch_predictions = likelihood(model(parameter_batch[0]))
                predictions.append(batch_predictions.mean.detach().numpy())

        return np.concatenate(predictions)
    return (run_inference,)


@app.cell
def _(CELL_INDEX, cf_imp_boundaries, coherency_fractions, np):
    mean_coherency = np.mean(coherency_fractions, axis=1)
    mask_cf = np.logical_and(
        cf_imp_boundaries[CELL_INDEX][0] < mean_coherency,
        mean_coherency < cf_imp_boundaries[CELL_INDEX][1]
    )
    cf_class_dataset = mask_cf.astype(float)
    return cf_class_dataset, mask_cf, mean_coherency


@app.cell
def _(cf_class_dataset, np):
    np.count_nonzero(cf_class_dataset)
    return


@app.cell
def _(cf_class_dataset, generate_criterion_model):
    cf_model, cf_likelihood = generate_criterion_model(cf_class_dataset)
    return cf_likelihood, cf_model


@app.cell
def _(cf_model):
    cf_model.covar_module.base_kernel.lengthscale.detach().numpy()
    return


@app.cell
def _(
    cf_class_dataset,
    cf_likelihood,
    cf_model,
    normalised_parameters,
    train_model,
):
    l_hist = train_model(
        cf_model, cf_likelihood,
        normalised_parameters, cf_class_dataset,
        epochs=100
    )
    return (l_hist,)


@app.cell
def _(cf_model):
    print(cf_model.covar_module.base_kernel.lengthscale.detach().numpy().tolist())
    return


@app.cell
def _(cf_likelihood, cf_model, normalised_parameters, run_inference):
    predictions = run_inference(cf_model, cf_likelihood, normalised_parameters)
    return (predictions,)


@app.cell
def _(cf_class_dataset, plt, predictions):
    plt.hist(predictions[cf_class_dataset == 0.0], alpha=0.5);
    plt.hist(predictions[cf_class_dataset == 1.0], alpha=0.5);
    plt.show()
    return


@app.cell
def _(np, predictions):
    print(np.count_nonzero(predictions > 0.5))
    return


@app.cell
def _(cf_class_dataset, plt, predictions):
    plt.boxplot([
        predictions[cf_class_dataset == 0.0],
        predictions[cf_class_dataset == 1.0]
    ]);
    plt.show()
    return


@app.cell
def _(TD_GRIDSEARCH_PARAMETERS, qmc):
    oos_sampler = qmc.Sobol(d=len(TD_GRIDSEARCH_PARAMETERS), scramble=True, rng=0)
    oos_sample_matrix = oos_sampler.random_base2(m=16)
    bounded_gridsearch =  (oos_sample_matrix * 0.9) + 0.05
    return bounded_gridsearch, oos_sample_matrix, oos_sampler


@app.cell
def _(bounded_gridsearch, cf_likelihood, cf_model, run_inference):
    oos_cf_predictions = run_inference(cf_model, cf_likelihood, bounded_gridsearch)
    return (oos_cf_predictions,)


@app.cell
def _(oos_cf_predictions):
    len(oos_cf_predictions)
    return


@app.cell
def _(np, oos_cf_predictions):
    np.count_nonzero(oos_cf_predictions > 0.5)
    return


@app.cell
def _(bounded_gridsearch, oos_cf_predictions, plt):
    _fig, _axs = plt.subplots(12, 1, figsize=(4, 10))

    for _i in range(12):
        _axs[_i].hist(
            bounded_gridsearch[oos_cf_predictions > 0.5, _i],
            bins=25, range=(0, 1), density=True
        );

    plt.show()
    return


@app.cell
def _(oos_cf_predictions, plt):
    plt.hist(oos_cf_predictions, range=(0, 1), bins=100)
    plt.show()
    return


@app.cell
def _(bounded_gridsearch, oos_cf_predictions, plt):
    _fig, _axs = plt.subplots(12, 1, figsize=(4, 10))

    for _i in range(12):
        _axs[_i].hist(
            bounded_gridsearch[oos_cf_predictions > 0.4, _i],
            bins=25, range=(0, 1), density=True
        );

    plt.show()
    return


@app.cell
def _(bounded_gridsearch, np, oos_cf_predictions, oos_sample_matrix):
    # Transform:
    redistributed_array = []
    for parameter in range(12):
        redistributed_parameter = np.quantile(
            bounded_gridsearch[oos_cf_predictions > 0.5, parameter],
            oos_sample_matrix[:, parameter]
        )
        redistributed_array.append(redistributed_parameter)
    redistributed_array = np.stack(redistributed_array, axis=1)
    return parameter, redistributed_array, redistributed_parameter


@app.cell
def _(cf_likelihood, cf_model, redistributed_array, run_inference):
    redist_cf_predictions = run_inference(cf_model, cf_likelihood, redistributed_array)
    return (redist_cf_predictions,)


@app.cell
def _(plt, redist_cf_predictions):
    plt.hist(redist_cf_predictions, range=(0, 1), bins=100)
    plt.show()
    return


@app.cell
def _(np, redist_cf_predictions):
    np.count_nonzero(redist_cf_predictions > 0.5)
    return


@app.cell
def _(CELL_INDEX, distances, np):
    cell_speed_wd = distances[:, CELL_INDEX, 0]
    mask_speed = cell_speed_wd < 1.0
    print(np.count_nonzero(mask_speed))
    speed_class_dataset = mask_speed.astype(float)
    return cell_speed_wd, mask_speed, speed_class_dataset


@app.cell
def _(generate_criterion_model, speed_class_dataset):
    speed_model, speed_likelihood = generate_criterion_model(speed_class_dataset)
    return speed_likelihood, speed_model


@app.cell
def _(speed_model):
    print(speed_model.covar_module.base_kernel.lengthscale.detach().numpy().tolist())
    return


@app.cell
def _(
    normalised_parameters,
    speed_class_dataset,
    speed_likelihood,
    speed_model,
    train_model,
):
    _ = train_model(
        speed_model, speed_likelihood,
        normalised_parameters, speed_class_dataset,
        epochs=100
    )
    return


@app.cell
def _(normalised_parameters, run_inference, speed_likelihood, speed_model):
    param_speed_predictions = \
        run_inference(speed_model, speed_likelihood, normalised_parameters)
    return (param_speed_predictions,)


@app.cell
def _(param_speed_predictions, plt, speed_class_dataset):
    plt.hist(param_speed_predictions[speed_class_dataset == 0.0], alpha=0.5);
    plt.hist(param_speed_predictions[speed_class_dataset == 1.0], alpha=0.5);
    plt.show()
    return


@app.cell
def _(param_speed_predictions, plt, speed_class_dataset):
    plt.boxplot([
        param_speed_predictions[speed_class_dataset == 0.0],
        param_speed_predictions[speed_class_dataset == 1.0]
    ]);
    plt.show()
    return


@app.cell
def _(bounded_gridsearch, run_inference, speed_likelihood, speed_model):
    redist_speed_predictions = run_inference(
        speed_model, speed_likelihood,
        bounded_gridsearch
    )
    return (redist_speed_predictions,)


@app.cell
def _(bounded_gridsearch, oos_cf_predictions, plt, redist_speed_predictions):
    _fig, _axs = plt.subplots(12, 1, figsize=(4, 10))

    for _i in range(12):
        _axs[_i].hist(
            bounded_gridsearch[oos_cf_predictions > 0.5, _i],
            bins=50, range=(0, 1), density=True, alpha=0.5
        );

    for _i in range(12):
        _axs[_i].hist(
            bounded_gridsearch[redist_speed_predictions > 0.5, _i],
            bins=50, range=(0, 1), density=True, alpha=0.5
        );

    plt.show()
    return


@app.cell
def _(np, redist_cf_predictions, redist_speed_predictions):
    joint_mask = np.logical_and(
        redist_cf_predictions > 0.5,
        redist_speed_predictions > 0.5
    )
    # joint_mask = redist_cf_predictions > 0.99
    return (joint_mask,)


@app.cell
def _(np, redist_speed_predictions):
    np.count_nonzero(redist_speed_predictions > 0.6)
    return


@app.cell
def _(np, redist_cf_predictions):
    np.count_nonzero(redist_cf_predictions > 0.6)
    return


@app.cell
def _(joint_mask, np):
    np.count_nonzero(joint_mask)
    return


@app.cell
def _(joint_mask, plt, redistributed_array):
    _fig, _axs = plt.subplots(12, 1, figsize=(4, 10))

    for _i in range(12):
        _axs[_i].hist(
            redistributed_array[joint_mask, _i],
            bins=25, range=(0, 1), density=True
        );

    plt.show()
    return


@app.cell
def _(TD_GRIDSEARCH_PARAMETERS, json):
    def print_parameter_set(parameter_set):
        ranged_parameters = []
        for idx, parameter_range in enumerate(TD_GRIDSEARCH_PARAMETERS.values()):
            param_min = parameter_range[0]
            param_max = parameter_range[1]
            param_range = param_max - param_min
            value = (parameter_set[idx] * param_range) + param_min
            ranged_parameters.append(value)

        optim_params = dict(zip(
            list(TD_GRIDSEARCH_PARAMETERS.keys()),
            ranged_parameters
        ))
        print(json.dumps(optim_params, sort_keys=False, indent=4))
        return ranged_parameters
    return (print_parameter_set,)


@app.cell
def _(joint_mask, oos_sample_matrix, print_parameter_set):
    test_set = oos_sample_matrix[joint_mask, :][0]
    _ = print_parameter_set(test_set)
    return (test_set,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
