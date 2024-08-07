"""
Script to collate the outputs of simulational analysis across a set of cluster runs.

Saves to overall summary .csv file.
"""
import os
import numpy as np

SIMULATION_OUTPUTS_FOLDER = "./fileOutputs/"


def main():
    matrix_list = []
    for folder_id in range(1, 700 + 1):
        # Printing progress:
        if folder_id % 100 == 0:
            print(folder_id)

        # Defining filepaths:
        folder_path = os.path.join(SIMULATION_OUTPUTS_FOLDER, str(folder_id))
        matrix_filepath = os.path.join(folder_path, "feature_matrix.npy")
        try:
            simulation_matrix = np.load(matrix_filepath)
        except OSError as e:
            print(f"Folder {folder_id} not found!")

        if simulation_matrix.shape[0] != 1050:
            print(simulation_matrix.shape)

        matrix_list.append(simulation_matrix)


    matrix_list = np.stack(matrix_list, axis=0)
    np.save("./matrix_list.npy", matrix_list)
    return None


if __name__ == "__main__":
    main()
