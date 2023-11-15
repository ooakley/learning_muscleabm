import os
import numpy as np

SIMULATION_OUTPUTS_FOLDER = "./fileOutputs/"
GRID_SIZE = 32
AREA_SIZE = 1024
TIMESTEPS = 1000


def read_matrix_into_numpy(filename, grid_size=GRID_SIZE, timesteps=TIMESTEPS):
    number_of_elements = (grid_size**2)*timesteps
    flattened_matrix = np.loadtxt(filename, delimiter=',', usecols=range(number_of_elements))
    return np.reshape(flattened_matrix, (timesteps, grid_size, grid_size), order='C')


def main():
    # Preparing lists for pandas dataframe:
    array_id = []
    seed_id = []
    matrix_entropy = []
    cell_msd = []
    cell_persistence = []

    # Reading in matrix files:
    for subdirectory in os.listdir(SIMULATION_OUTPUTS_FOLDER):
        print(subdirectory)
        directory_path = os.path.join(SIMULATION_OUTPUTS_FOLDER, subdirectory)
        output_files = os.listdir(directory_path)
        output_files.sort()
        matrix_list = []
        for filename in output_files:
            if filename.endswith('.txt'):
                print(filename)
                filepath = directory_path + filename
                matrix_list.append(read_matrix_into_numpy(filepath))

        frequencies, _ = np.histogram(matrix_list[:, :, -1].flatten(), bins=50, range=(0, np.pi), density=True)
        entropy = -(frequencies*np.log(np.abs(data))).sum()
    
    # Calaculating values:

if __name__ == "__main__":
    main()