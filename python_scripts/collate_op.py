import numpy as np

SAMPLE_COUNT = 10000

def main():
    print('Collating data...')
    auc_data_list = []
    opscale_data_list = []
    for folder_id in range(SAMPLE_COUNT):
        if (folder_id + 1) % 10000 == 0:
            print(folder_id + 1)
        # Getting AUC data:
        auc_filepath = f'./fileOutputs/{folder_id}/auc_output.npy'
        auc_data = np.load(auc_filepath)
        auc_data_list.append(auc_data)
        # Getting OP scaling data:
        opscale_filepath = f'./fileOutputs/{folder_id}/op_scale_output.npy'
        opscale_data = np.load(opscale_filepath)
        opscale_data_list.append(opscale_data)

    print("Saving data...")
    auc_matrix = np.stack(auc_data_list)
    opscale_matrix = np.stack(opscale_data_list)
    np.save("./collated_auc.npy", auc_matrix)
    np.save("./collated_opscale.npy", opscale_matrix)


if __name__ == "__main__":
    main()
