import marimo

__generated_with = "0.12.10"
app = marimo.App(width="columns")


@app.cell
def _():
    import os
    import skimage
    import json

    import pandas as pd
    import numpy as np

    from scipy.stats import qmc
    import matplotlib.pyplot as plt

    DATA_DIRECTORY = "./wetlab_data/OEO20241206"
    ROWS = ["A", "B", "C"]
    COLUMNS = ["1", "2", "3", "4", "5", "6"]
    return COLUMNS, DATA_DIRECTORY, ROWS, json, np, os, pd, plt, qmc, skimage


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
def _(find_coherency_fraction):
    def get_cf_dictionary(trajectory_dictionary):
        cf_dictionary = {}
        for _column in ["1", "2", "3", "4", "5", "6"]:
            coherency_fractions = []
            for _i in range(12):
                site_data = trajectory_dictionary[_column][_i]
                coherency_fractions.append(find_coherency_fraction(site_data))
            cf_dictionary[_column] = coherency_fractions
        return cf_dictionary
    return (get_cf_dictionary,)


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
    return (get_frame_anni,)


@app.cell
def _(get_frame_anni, np):
    def get_mean_anni(site_data):
        anni_timeseries = []
        for frame in list(set(list(site_data["frame"]))):
            frame_mask = site_data["frame"] == frame
            frame_positions = np.array(site_data[frame_mask].loc[:, ['x', 'y']])
            anni_timeseries.append(get_frame_anni(frame_positions))
        return np.mean(anni_timeseries)
    return (get_mean_anni,)


@app.cell
def _(get_mean_anni):
    def get_anni_dictionary(trajectory_dictionary):
        anni_dictionary = {}
        for column in ["1", "2", "3", "4", "5", "6"]:
            ann_indices = []
            for i in range(12):
                site_data = trajectory_dictionary[column][i]
                ann_indices.append(get_mean_anni(site_data))
            anni_dictionary[column] = ann_indices
        return anni_dictionary
    return (get_anni_dictionary,)


@app.cell
def _():
    TD_GRIDSEARCH_PARAMETERS = {
        "cueDiffusionRate": [0.0001, 2.5],
        "cueKa": [0.0001, 5],
        "fluctuationAmplitude": [1e-5, 1e-3],
        "fluctuationTimescale": [1, 75],
        "maximumSteadyStateActinFlow": [0.1, 3],
        "numberOfCells": [50, 125],
        "actinAdvectionRate": [0.1, 3],
        "cellBodyRadius": [20, 60],
        # Collisions:
        "collisionFlowReductionRate": [0.0, 1],
        "collisionAdvectionRate": [0.0, 1],
        # Shape:
        "stretchFactor": [0.25, 3],
        "slipFactor": [0.0001, 0.1],
    }
    return (TD_GRIDSEARCH_PARAMETERS,)


@app.cell
def _(TD_GRIDSEARCH_PARAMETERS, np):
    def get_gridsearch_data(folder_name):
        parameters = np.load(
            f"./gridsearch_data/{folder_name}/collated_inputs.npy"
        )
        # distances = np.load(
        #     f"./gridsearch_data/{folder_name}/collated_distances.npy"
        # )
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

        # # Discard failed simulations:
        # nan_mask = np.any(np.isnan(distances), axis=(1, 2))
        # nan_parameters = parameters[nan_mask, :]

        # Temporary nan mask for overconfluent simulations:
        cell_number = parameters[:, 5]
        cell_radius = parameters[:, 7]
        cell_area = np.pi * (cell_radius ** 2)
        packing_fraction = (cell_area * cell_number) / (2048**2)
        pf_mask = packing_fraction > 0.8
        nan_mask = pf_mask
        # nan_mask = np.logical_or(
        #     nan_mask, pf_mask
        # )

        # Apply to remaining matrices:
        parameters = parameters[~nan_mask, :]
        # distances = distances[~nan_mask, :]
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
        # distances,
        return normalised_parameters, coherency_fractions, ann_indices, speeds
    return (get_gridsearch_data,)


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
def _(DATA_DIRECTORY, os, pd):
    processed_dataset_filepath = os.path.join(DATA_DIRECTORY, "fitting_dataset.csv")
    experiment_dataframe = pd.read_csv(processed_dataset_filepath, index_col=0)
    return experiment_dataframe, processed_dataset_filepath


@app.cell
def _(COLUMNS, experiment_dataframe, np):
    speed_means = []
    for _column in COLUMNS:
        column_mask = experiment_dataframe["column"] == int(_column)
        speed_means.append(np.mean(experiment_dataframe.loc[column_mask, "speed"]))
    speed_means
    return column_mask, speed_means


@app.cell
def _(get_anni_dictionary, get_cf_dictionary, trajectory_dictionary):
    cf_dictionary = get_cf_dictionary(trajectory_dictionary)
    anni_dictionary = get_anni_dictionary(trajectory_dictionary)
    return anni_dictionary, cf_dictionary


@app.cell
def _(COLUMNS, cf_dictionary, np):
    cf_imp_boundaries = []
    for _column in COLUMNS:
        cell_type_cf_mean = np.mean(cf_dictionary[_column])
        cell_type_cf_std = np.std(cf_dictionary[_column])
        cf_imp_boundaries.append([
            cell_type_cf_mean - cell_type_cf_std,
            cell_type_cf_mean + cell_type_cf_std
        ])
    return cell_type_cf_mean, cell_type_cf_std, cf_imp_boundaries


@app.cell
def _(COLUMNS, anni_dictionary, np):
    anni_imp_boundaries = []
    for _column in COLUMNS:
        cell_type_anni_mean = np.mean(anni_dictionary[_column])
        cell_type_anni_std = np.std(anni_dictionary[_column])
        anni_imp_boundaries.append([
            cell_type_anni_mean - cell_type_anni_std,
            cell_type_anni_mean + cell_type_anni_std
        ])
    return anni_imp_boundaries, cell_type_anni_mean, cell_type_anni_std


@app.cell
def _(get_gridsearch_data):
    # Get data:
    normalised_parameters, coherency_fractions, ann_indices, speeds = \
        get_gridsearch_data("out_largeSearch")
    return ann_indices, coherency_fractions, normalised_parameters, speeds


@app.cell
def _(coherency_fractions, normalised_parameters, np):
    MESH_COUNT = 20

    grid_boundaries =  np.linspace(0, 1, MESH_COUNT + 1)
    parameter_i = 0
    parameter_j = 6

    phase_array = np.zeros((MESH_COUNT, MESH_COUNT))

    for grid_index_i in range(MESH_COUNT):
        # Get row parameter mask:
        i_threshold_low = grid_boundaries[grid_index_i]
        i_low_mask = i_threshold_low < normalised_parameters[:, parameter_i]
        i_threshold_high = grid_boundaries[grid_index_i+1]
        i_high_mask = normalised_parameters[:, parameter_i] < i_threshold_high
        i_mask = np.logical_and(i_low_mask, i_high_mask)

        for grid_index_j in range(MESH_COUNT):
            # Get column parameter mask:
            j_threshold_low = grid_boundaries[grid_index_j]
            j_low_mask = j_threshold_low < normalised_parameters[:, parameter_j]
            j_threshold_high = grid_boundaries[grid_index_j+1]
            j_high_mask = normalised_parameters[:, parameter_j] < j_threshold_high
            j_mask = np.logical_and(j_low_mask, j_high_mask)

            total_mask = np.logical_and(i_mask, j_mask)
            phase_array[grid_index_i, grid_index_j] = \
                np.mean(coherency_fractions[total_mask])
    return (
        MESH_COUNT,
        grid_boundaries,
        grid_index_i,
        grid_index_j,
        i_high_mask,
        i_low_mask,
        i_mask,
        i_threshold_high,
        i_threshold_low,
        j_high_mask,
        j_low_mask,
        j_mask,
        j_threshold_high,
        j_threshold_low,
        parameter_i,
        parameter_j,
        phase_array,
        total_mask,
    )


@app.cell
def _(phase_array):
    phase_array.shape
    return


@app.cell
def _(phase_array, plt):
    plt.imshow(phase_array, origin='lower', vmin=0.0, vmax=0.08)
    return


@app.cell
def _(grid_boundaries):
    grid_boundaries.shape
    return


@app.cell
def _(ann_indices):
    ann_indices.shape
    return


@app.cell
def _(coherency_fractions):
    print(coherency_fractions.shape)
    return


@app.cell
def _(
    ann_indices,
    anni_imp_boundaries,
    cf_imp_boundaries,
    coherency_fractions,
    normalised_parameters,
    np,
    speed_means,
    speeds,
):
    fit_parameters = {}
    overall_parameters = []
    for cell_index in range(6):
        print(f"--- --- Cell Index: {cell_index} --- ---")
        # Coherency suitability mask:
        mean_coherency = np.mean(coherency_fractions, axis=1)
        mask_cf = np.logical_and(
            cf_imp_boundaries[cell_index][0] < mean_coherency,
            mean_coherency < cf_imp_boundaries[cell_index][1]
        )
        cf_class_dataset = mask_cf.astype(float)
    
        # Speed suitability mask:
        speed_imp_mask = np.logical_and(
            speeds > speed_means[cell_index] - 0.1,
            speeds < speed_means[cell_index] + 0.1
        )
        speed_class_dataset = speed_imp_mask.astype(float)

        # ANNI suitability mask:
        mean_anni = np.mean(ann_indices, axis=1)
        anni_imp_mask = np.logical_and(
            anni_imp_boundaries[cell_index][0] < mean_anni,
            mean_anni < anni_imp_boundaries[cell_index][1]
        )
        anni_class_dataset = anni_imp_mask.astype(float)

        # Check if we have any current overlap:
        # sobol_search_native_fit = np.logical_and(
        #     mask_cf,
        #     speed_imp_mask
        # )
        boolean_dataset = np.stack([mask_cf, speed_imp_mask, anni_imp_mask], axis=1)
        sobol_search_native_fit = np.all(boolean_dataset, axis=1)
        print(np.count_nonzero(sobol_search_native_fit))

        # Get parameters:
        fit_parameters[cell_index] = normalised_parameters[sobol_search_native_fit, :]
        overall_parameters.append(normalised_parameters[sobol_search_native_fit, :])

    overall_parameters = np.concatenate(overall_parameters, axis=0)
    return (
        anni_class_dataset,
        anni_imp_mask,
        boolean_dataset,
        cell_index,
        cf_class_dataset,
        fit_parameters,
        mask_cf,
        mean_anni,
        mean_coherency,
        overall_parameters,
        sobol_search_native_fit,
        speed_class_dataset,
        speed_imp_mask,
    )


@app.cell
def _():
    # Demonstrate collective behaviour harder to reproduce without matrix feedback.
    # Get phase plots for all parameter combinations.
    # independent / dependent varaible scatterplots along diagonal
    return


@app.cell
def _(overall_parameters):
    overall_parameters
    return


@app.cell
def _(fit_parameters, plt):
    _fig, _axs = plt.subplots(12, figsize=(3, 8), sharex=True)
    for _i in range(12):
        _axs[_i].hist(fit_parameters[5][:, _i])

    plt.show()
    return


@app.cell
def _(overall_parameters, plt):
    _fig, _axs = plt.subplots(12, figsize=(3, 8), sharex=True)
    for _i in range(12):
        _axs[_i].hist(overall_parameters[:, _i])

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
def _(fit_parameters, print_parameter_set):
    print_parameter_set(fit_parameters[5][0]);
    return


@app.cell
def _(cell_index, np):
    cf_train_predictions = np.load(
        f"interpolation_outputs/{cell_index}_cf_train_predictions.npy"
    )
    cf_predictions = np.load(
        f"interpolation_outputs/{cell_index}_cf_predictions.npy"
    )
    speed_train_predictions = np.load(
        f"interpolation_outputs/{cell_index}_speed_train_predictions.npy"
    )
    speed_predictions = np.load(
        f"interpolation_outputs/{cell_index}_speed_predictions.npy"
    )

    joint_mask = np.logical_and(cf_predictions > 0.5, speed_predictions > 0.5)
    print(np.count_nonzero(joint_mask))
    return (
        cf_predictions,
        cf_train_predictions,
        joint_mask,
        speed_predictions,
        speed_train_predictions,
    )


@app.cell
def _(cf_class_dataset, cf_train_predictions, plt):
    plt.boxplot([
        cf_train_predictions[cf_class_dataset == 0.0],
        cf_train_predictions[cf_class_dataset == 1.0]
    ]);
    plt.show()
    return


@app.cell
def _(plt, speed_class_dataset, speed_train_predictions):
    plt.boxplot([
        speed_train_predictions[speed_class_dataset == 0.0],
        speed_train_predictions[speed_class_dataset == 1.0]
    ]);
    plt.show()
    return


@app.cell
def _(TD_GRIDSEARCH_PARAMETERS, qmc):
    oos_sampler = qmc.Sobol(d=len(TD_GRIDSEARCH_PARAMETERS), scramble=True, rng=0)
    oos_sample_matrix = oos_sampler.random_base2(m=21)
    bounded_gridsearch =  (oos_sample_matrix * 0.9) + 0.05
    print(bounded_gridsearch.shape)
    return bounded_gridsearch, oos_sample_matrix, oos_sampler


@app.cell
def _(bounded_gridsearch, joint_mask, print_parameter_set):
    test_set = bounded_gridsearch[joint_mask, :][0]
    _ = print_parameter_set(test_set)
    return (test_set,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
