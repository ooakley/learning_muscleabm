import numpy as np

SAMPLE_COUNT = 13000

def main():
    print('Collating data...')
    order_parameter_data = []
    for folder_id in range(SAMPLE_COUNT):
        if (folder_id + 1) % 500 == 0:
            print(folder_id + 1)
        filepath = f'./fileOutputs/{folder_id}/mean_output.npy'
        order_parameter = np.load(filepath)
        order_parameter_data.append(order_parameter)

    print("Saving data...")
    op_matrix = np.stack(order_parameter_data)
    np.save("./collated_op.npy", op_matrix)


if __name__ == "__main__":
    main()
