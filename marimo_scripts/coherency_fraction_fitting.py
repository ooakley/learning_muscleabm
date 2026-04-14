import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import skimage

    import numpy as np
    import pandas as pd
    return np, os, pd, skimage


@app.cell
def _():
    DATA_DIRECTORY = "./wetlab_data/OEO20241206"
    ROWS = ["A", "B", "C"]
    COLUMNS = ["1", "2", "3", "4", "5", "6"]
    return COLUMNS, DATA_DIRECTORY, ROWS


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
def _(find_coherency_fraction, os, trajectory_dictionary):
    import json

    def get_cf_dictionary(trajectory_dictionary):
        cf_dictionary = {}
        for _column in ["1", "2", "3", "4", "5", "6"]:
            coherency_fractions = []
            for _i in range(12):
                site_data = trajectory_dictionary[_column][_i]
                coherency_fractions.append(find_coherency_fraction(site_data))
            cf_dictionary[_column] = coherency_fractions
        return cf_dictionary

    if os.path.exists("wetlab_data/OEO20241206/cf_dictionary.json"):
        with open("wetlab_data/OEO20241206/cf_dictionary.json") as json_filestream:
            cf_dictionary = json.load(json_filestream)
    else:
        cf_dictionary = get_cf_dictionary(trajectory_dictionary)
        with open("wetlab_data/OEO20241206/cf_dictionary.json", "w") as json_filestream:
            json.dump(cf_dictionary, json_filestream)
    return (cf_dictionary,)


@app.cell
def _(cf_dictionary, np):
    print(np.mean(cf_dictionary["1"]))
    print(np.var(cf_dictionary["1"]))
    return


@app.cell
def _(experiment_dataframe, np):
    column_mask = experiment_dataframe["column"] == 1
    col_speeds = experiment_dataframe[column_mask].loc[:, "speed"]
    mean_speed = np.mean(col_speeds)
    var_speed = np.var(col_speeds)
    return mean_speed, var_speed


@app.cell
def _(mean_speed, var_speed):
    print(mean_speed)
    print(var_speed)
    return


@app.cell
def _(DATA_DIRECTORY, os, pd):
    processed_dataset_filepath = os.path.join(DATA_DIRECTORY, "fitting_dataset.csv")
    experiment_dataframe = pd.read_csv(processed_dataset_filepath, index_col=0)
    return (experiment_dataframe,)


@app.cell
def _(experiment_dataframe):
    experiment_dataframe
    return


@app.cell
def _():
 
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
