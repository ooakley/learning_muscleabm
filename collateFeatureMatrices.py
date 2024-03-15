"""
Script to collate the outputs of simulational analysis across a set of cluster runs.

Saves to overall summary .csv file.
"""
import os
import numpy as np

SIMULATION_OUTPUTS_FOLDER = "./fileOutputs/"


def main():
    matrix_list = []
    for folder_id in range(1, 3 + 1):
        # Printing progress:
        if folder_id % 100 == 0:
            print(folder_id)

        # Defining filepaths:
        folder_path = os.path.join(SIMULATION_OUTPUTS_FOLDER, str(folder_id))
        matrix_filepath = os.path.join(folder_path, "feature_matrix.npy")

        matrix_list.append(np.load(matrix_filepath))

    np.save("./matrix_list.npy", matrix_list)
    return None


if __name__ == "__main__":
    main()
