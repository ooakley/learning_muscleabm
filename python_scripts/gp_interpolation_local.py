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


def get_cf_dictionary(trajectory_dictionary):
    cf_dictionary = {}
    for _column in ["1", "2", "3", "4", "5", "6"]:
        coherency_fractions = []
        for _i in range(12):
            site_data = trajectory_dictionary[_column][_i]
            coherency_fractions.append(find_coherency_fraction(site_data))
        cf_dictionary[_column] = coherency_fractions
    return cf_dictionary


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


def get_anni_dictionary(trajectory_dictionary):
    anni_dictionary = {}
    for column in ["1", "2", "3", "4", "5", "6"]:
        ann_indices = []
        for i in range(12):
            site_data = trajectory_dictionary[column][i]
            ann_indices.append(get_mean_anni(site_data))
        anni_dictionary[column] = ann_indices
    return anni_dictionary


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

    # Temporary nan mask for overconfluent simulations:
    cell_number = parameters[:, 5]
    cell_radius = parameters[:, 7]
    cell_area = np.pi * (cell_radius ** 2)
    packing_fraction = (cell_area * cell_number) / (2048**2)
    pf_mask = packing_fraction > 0.8
    nan_mask = np.logical_or(
        nan_mask, pf_mask
    )

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

    return normalised_parameters, distances, coherency_fractions, ann_indices, speeds


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


def generate_criterion_model(inducing_points):
    model, likelihood = instantiate_model(
        inducing_points,
        len(TD_GRIDSEARCH_PARAMETERS)
    )
    return model, likelihood


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


def main():
    # Get previously processed wet lab data:
    processed_dataset_filepath = os.path.join(DATA_DIRECTORY, "fitting_dataset.csv")
    experiment_dataframe = pd.read_csv(processed_dataset_filepath, index_col=0)

    # Summarise speeds:
    speed_means = []
    for _column in COLUMNS:
        column_mask = experiment_dataframe["column"] == int(_column)
        speed_means.append(np.mean(experiment_dataframe.loc[column_mask, "speed"]))

    # Get wet lab trajectory data:
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

    cf_dictionary = get_cf_dictionary(trajectory_dictionary)

    cf_imp_boundaries = []
    for _column in COLUMNS:
        cell_type_cf_mean = np.mean(cf_dictionary[_column])
        cell_type_cf_std = np.std(cf_dictionary[_column])
        cf_imp_boundaries.append([
            cell_type_cf_mean - cell_type_cf_std,
            cell_type_cf_mean + cell_type_cf_std
        ])

    normalised_parameters, distances, coherency_fractions, ann_indices, speeds = \
        get_gridsearch_data("out_20250826TrajectoryCollisions")

    # Ensure output directory is present:
    if not os.path.exists("interpolation_outputs"):
        os.mkdir("interpolation_outputs")

    for cell_index in range(6):
        print(f"        *** *** Analysing cell type {cell_index}... *** ***        ")
        # Fit just to speed and coherency for now:
        # Coherency suitability mask:
        mean_coherency = np.mean(coherency_fractions, axis=1)
        mask_cf = np.logical_and(
            cf_imp_boundaries[cell_index][0] < mean_coherency,
            mean_coherency < cf_imp_boundaries[cell_index][1]
        )
        cf_class_dataset = mask_cf.astype(float)

        # Speed suitability mask:
        speed_imp_mask = np.logical_and(
            speeds > speed_means[cell_index] - 0.025,
            speeds < speed_means[cell_index] + 0.025
        )
        speed_class_dataset = speed_imp_mask.astype(float)

        # Check if we have any current overlap:
        sobol_search_native_fit = np.logical_and(
            mask_cf,
            speed_imp_mask
        )
        print(f"Native fits: {np.count_nonzero(sobol_search_native_fit)}")

        print("--- Training coherency fraction classifier...")
        cf_model, cf_likelihood = generate_criterion_model(
            normalised_parameters
        )
        _ = train_model(
            cf_model, cf_likelihood,
            normalised_parameters, cf_class_dataset,
            epochs=100
        )

        print("--- Training speed classifier...")
        speed_model, speed_likelihood = generate_criterion_model(
            normalised_parameters
        )
        _ = train_model(
            speed_model, speed_likelihood,
            normalised_parameters, speed_class_dataset,
            epochs=100
        )

        # Get space-filling interpolating points:
        oos_sampler = qmc.Sobol(d=len(TD_GRIDSEARCH_PARAMETERS), scramble=True, rng=0)
        oos_sample_matrix = oos_sampler.random_base2(m=21)

        # Ensure interpolating points don't overlap with edges of search domain:
        bounded_gridsearch = (oos_sample_matrix * 0.9) + 0.05

        # Run inference over training set:
        train_cf_predictions = \
            run_inference(cf_model, cf_likelihood, normalised_parameters)
        train_speed_predictions = \
            run_inference(speed_model, speed_likelihood, normalised_parameters)

        np.save(
            os.path.join("interpolation_outputs", f"{cell_index}_cf_train_predictions.npy"),
            train_cf_predictions
        )
        np.save(
            os.path.join("interpolation_outputs", f"{cell_index}_speed_train_predictions.npy"),
            train_speed_predictions
        )

        # Run inference over interpolating points:
        print("--- Running inference over larger gridsearch...")
        oos_cf_predictions = run_inference(cf_model, cf_likelihood, bounded_gridsearch)
        oos_speed_predictions = run_inference(speed_model, speed_likelihood, bounded_gridsearch)

        np.save(
            os.path.join("interpolation_outputs", f"{cell_index}_cf_predictions.npy"),
            oos_cf_predictions
        )
        np.save(
            os.path.join("interpolation_outputs", f"{cell_index}_speed_predictions.npy"),
            oos_speed_predictions
        )


if __name__ == "__main__":
    main()
