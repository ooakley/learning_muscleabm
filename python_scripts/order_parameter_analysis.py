"""Perform only basic order parameter calculations."""
import argparse
import os
import numpy as np

MESH_NUMBER = 64
TIMESTEPS = 1440
WORLD_SIZE = 2048
SIMULATION_OUTPUTS_FOLDER = "./fileOutputs/"
SUPERITERATION_NUMBER = 12
NEIGHBOURHOOD_SIZES = np.arange(3, 64, 4)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a folder with a given integer name.')
    parser.add_argument('--folder_id', type=int, help='Integer corresponding to the folder name')
    args = parser.parse_args()
    return args


def read_matrix_into_numpy(filename, mesh_number, timesteps):
    number_of_elements = (mesh_number**2) * timesteps * 3
    flattened_matrix = np.loadtxt(filename, delimiter=',', usecols=range(number_of_elements))
    return np.reshape(flattened_matrix, (timesteps, 3, mesh_number, mesh_number), order='C')


def get_order_parameter(submatrix):
    # Getting central values:
    central_index = int(np.floor(submatrix.shape[0] / 2))
    central_val = submatrix[central_index, central_index]

    # Getting values in window:
    central_cutoff = int(np.ceil(submatrix.size / 2))
    comparators = submatrix.flatten()
    comparators = np.concatenate([comparators[0:central_cutoff], comparators[central_cutoff + 1:]])
    angle_diff = comparators * 2 - central_val * 2

    # Calculating order parameter:
    order_parameter = np.mean(np.cos(angle_diff * 2))
    return order_parameter


def roll_indices(index, half_index):
    # Need to roll matrix to ensure that the order parameter captures the
    # periodic boundaries - calculating the amount of rolling is a bit
    # fiddly however:
    roll_index = 0
    index_start = index - half_index
    index_end = index + (half_index + 1)
    if index_start < 0:
        roll_index -= index_start
        index_start += roll_index
        index_end += roll_index
    if index_end > MESH_NUMBER:
        roll_index = -(index_end - MESH_NUMBER)
        index_start += roll_index
        index_end += roll_index

    return roll_index, index_start, index_end


def get_order_parameter_distribution(matrix, neighbourhood_size=3):
    order_parameters = []
    half_index = int(np.floor(neighbourhood_size / 2))
    for i in range(MESH_NUMBER):
        # Determining amount of rolling required along row:
        roll_i, i_start, i_end = roll_indices(i, half_index)
        for j in range(MESH_NUMBER):
            # Determining amount of rolling required along column:
            roll_j, j_start, j_end = roll_indices(j, half_index)

            # Rolling matrix:
            rolled_matrix = np.roll(matrix, roll_i, axis=1)
            rolled_matrix = np.roll(rolled_matrix, roll_j, axis=2)

            # Getting submatrix:
            orientation_submatrix = rolled_matrix[0, i_start:i_end, j_start:j_end]
            density_submatrix = rolled_matrix[1, i_start:i_end, j_start:j_end]
            order_parameter = get_order_parameter(orientation_submatrix)
            weighted_order_parameter = np.mean(density_submatrix) * order_parameter
            order_parameters.append(weighted_order_parameter)
    return order_parameters


def generate_order_parameter_scale_curve(matrix):
    order_parameters = []
    for neighbourhood_size in NEIGHBOURHOOD_SIZES:
        mean_order_parameter = np.mean(
            get_order_parameter_distribution(matrix, neighbourhood_size=neighbourhood_size)
        )
        order_parameters.append(mean_order_parameter)
    return order_parameters


def main():
    """Run basic script logic."""
    # Parse arguments:
    args = parse_arguments()

    # Finding specified simulation output files:
    subdirectory_path = os.path.join(SIMULATION_OUTPUTS_FOLDER, str(args.folder_id))
    order_parameter_list = []
    order_parameter_auc_list = []
    for seed in range(SUPERITERATION_NUMBER):
        print(f"Reading subiteration {seed}...")
        # Reading matrix information:
        matrix_filename = f"matrix_seed{seed:03d}.txt"
        matrix_filepath = os.path.join(subdirectory_path, matrix_filename)
        matrix = read_matrix_into_numpy(matrix_filepath, MESH_NUMBER, TIMESTEPS)
        final_matrix = matrix[-1, 0:2, :, :]
        orderparameter_scale = generate_order_parameter_scale_curve(final_matrix)
        orderparameter_auc = np.trapezoid(orderparameter_scale)

        # Accumulating in lists:
        order_parameter_list.append(orderparameter_scale)
        order_parameter_auc_list.append(orderparameter_auc)
        print(f"OP - area under scale curve: {orderparameter_auc}")

    # Recording OP AUC:
    mean_output = np.mean(order_parameter_auc_list)
    std_output = np.std(order_parameter_auc_list)
    auc_output = np.array([mean_output, std_output])
    np.save(os.path.join(subdirectory_path, "auc_output.npy"), auc_output)

    # Recording direct scale curves:
    op_curves = np.stack(order_parameter_list, axis=0)
    np.save(os.path.join(subdirectory_path, "op_scale_output.npy"), op_curves)


if __name__ == "__main__":
    main()
