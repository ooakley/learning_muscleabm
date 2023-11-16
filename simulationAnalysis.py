import argparse
import os
import copy
import scipy

import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

SIMULATION_OUTPUTS_FOLDER = "./fileOutputs/"
CSV_COLUMN_NAMES = [
    "frame", "particle", "x", "y", "orientation",
    "polarity_extent", "percept_direction", "percept_intensity",
    "actin_flow", "movement_direction", "turning_angle", "contact_inhibition"
]
GRID_SIZE = 32
AREA_SIZE = 1024
TIMESTEPS = 1000


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a folder with a given integer name.')
    parser.add_argument('--folder_id', type=int, help='Integer corresponding to the folder name')
    args = parser.parse_args()
    return args


def read_matrix_into_numpy(filename, grid_size=GRID_SIZE, timesteps=TIMESTEPS):
    number_of_elements = (grid_size**2)*timesteps
    flattened_matrix = np.loadtxt(filename, delimiter=',', usecols=range(number_of_elements))
    return np.reshape(flattened_matrix, (timesteps, grid_size, grid_size), order='C')


def find_average_rmsd(trajectory_dataframe):
    particles = list(set(list(trajectory_dataframe["particle"])))
    rmsd_list = []
    for particle in particles:
        particle_dataframe = trajectory_dataframe[trajectory_dataframe["particle"] == particle]
        rmsd_list.append(np.mean(particle_dataframe["actin_flow"]) * 0.3264 / 5)
    return rmsd_list


def find_persistence_time(trajectory_dataframe):
    particles = list(set(list(trajectory_dataframe["particle"])))
    pt_list = []

    for particle in particles:
        particle_directions = np.array(
            trajectory_dataframe[trajectory_dataframe["particle"] == particle]["movement_direction"]
        )

        direction_differences = np.subtract.outer(particle_directions, particle_directions)
        direction_differences[direction_differences < -np.pi] += 2*np.pi
        direction_differences[direction_differences > np.pi] -= 2*np.pi
        direction_differences = np.abs(direction_differences)

        persistence_time_list = []
        for timestep in range(len(particle_directions)):
            subsequent_directions = direction_differences[timestep, timestep:]
            persistence_time = np.argmax(subsequent_directions > np.pi/2)
            if persistence_time == 0:
                continue
            persistence_time_list.append(persistence_time)

        pt_list.append(np.mean(np.log(persistence_time_list)))

    return pt_list

def main():
    # Parsing command line arguments:
    args = parse_arguments()
    print(f"Processing folder with name: {args.folder_id}")

    # Getting simulation parameters:
    gridseach_dataframe = pd.read_csv("fileOutputs/gridsearch.txt", delimiter="\t")

    # Finding specified simulation output files:
    subdirectory_path = os.path.join(SIMULATION_OUTPUTS_FOLDER, str(args.folder_id))
    output_files = os.listdir(subdirectory_path)

    matrix_list = []
    sub_dataframes = []
    for seed in range(10):
        # Reading trajectory information:
        csv_filename = f"positions_seed{seed:03d}.csv"
        csv_filepath = os.path.join(subdirectory_path, csv_filename)
        trajectory_dataframe = pd.read_csv(csv_filepath, index_col=None, header=None, names=CSV_COLUMN_NAMES)

        # Calculating derived statistics:
        rmsd_list = find_average_rmsd(trajectory_dataframe)
        pt_list = find_persistence_time(trajectory_dataframe)
        persistence_speed_corrcoef, p_val = scipy.stats.pearsonr(rmsd_list, pt_list)

        # Saving to dataframe row:
        new_info = pd.DataFrame(
            {
                "seed_number": [seed],
                "mean_rmsd": [np.mean(rmsd_list)],
                "mean_persistence_time": [np.mean(pt_list)],
                "speed_persistence_correlation": [persistence_speed_corrcoef],
                "speed_persistence_correlation_pval": [p_val]
            }
        )
        simulation_properties = copy.deepcopy(gridseach_dataframe[gridseach_dataframe["array_id"] == args.folder_id])
        simulation_properties = simulation_properties.reset_index()
        iteration_row = pd.concat([simulation_properties, new_info], axis=1)
        sub_dataframes.append(iteration_row)

        # Reading final-step matrix information into list:
        matrix_filename = f"matrix_seed{seed:03d}.txt"
        matrix_filepath = os.path.join(subdirectory_path, matrix_filename)
        matrix = read_matrix_into_numpy(matrix_filepath)
        matrix_list.append(matrix[-1, :, :])

    # Generating full dataframe and saving to subdirectory:
    summary_dataframe = pd.concat(sub_dataframes).reset_index().drop(columns=["index", "level_0"])
    print(summary_dataframe)
    summary_dataframe.to_csv(os.path.join(subdirectory_path, "summary.csv"))

    # Plotting final orientations of the matrix:
    fig, ax = plt.subplots(figsize=(7, 4))
    distributions = []
    bins = np.arange(0, np.pi, 0.05)
    for matrix in matrix_list:
        ax.hist(matrix.flatten(), bins=bins, alpha=0.3, density=True)
    fig.tight_layout()
    fig.savefig(os.path.join(subdirectory_path, "matrix_orientation.png"))

if __name__ == "__main__":
    main()