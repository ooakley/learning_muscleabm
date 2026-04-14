import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import json

    import skimage

    import numpy as np
    import pandas as pd

    DATA_DIRECTORY = "./wetlab_data/OEO20241206"
    ROWS = ["A", "B", "C"]
    COLUMNS = ["1", "2", "3", "4", "5", "6"]
    return COLUMNS, DATA_DIRECTORY, ROWS, json, np, os, pd, skimage


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
    return get_anni_dictionary, get_cf_dictionary


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
    return (trajectory_dictionary,)


@app.cell
def _(get_anni_dictionary, get_cf_dictionary, trajectory_dictionary):
    cf_dictionary = get_cf_dictionary(trajectory_dictionary)
    anni_dictionary = get_anni_dictionary(trajectory_dictionary)
    return (anni_dictionary,)


@app.cell
def _(DATA_DIRECTORY, anni_dictionary, json, os):
    with open(os.path.join(DATA_DIRECTORY, "anni_dictionary.json"), 'w') as f:
        json.dump(anni_dictionary, f)
    return


@app.cell
def _(trajectory_dictionary):
    trajectory_dictionary["1"][0]
    return


@app.cell
def _(pd):
    pd.read_csv("wetlab_data/OEO20241206/fitting_dataset.csv")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
