import os
import copy
import pandas as pd

SIMULATION_OUTPUTS_FOLDER = "./fileOutputs/"

def main():
    # Reading in gridsearch dataframe:
    gridseach_dataframe = pd.read_csv("fileOutputs/gridsearch.txt", delimiter="\t")

    # Reading through all iterations:
    summary_dataframe_list = []
    speed_dataframe_list = []
    for folder_id in range(1, 2049):
        if folder_id % 100 == 0:
            print(folder_id)
        folder_path = os.path.join(SIMULATION_OUTPUTS_FOLDER, str(folder_id))
        summary_filepath = os.path.join(folder_path, "summary.csv")
        speeds_filepath = os.path.join(folder_path, "speeds.csv")
        try:
            # Reading in trajectory data:
            trajectory_data = pd.read_csv(summary_filepath, index_col=0)
            summary_dataframe_list.append(trajectory_data)
        except:
            print(f"Folder {folder_id} skipped...")
            continue

        # Reading in speed data:
        speed_data = pd.read_csv(speeds_filepath, index_col=0)
        speed_dataframe = copy.deepcopy(gridseach_dataframe[gridseach_dataframe["array_id"] == folder_id])
        speed_dataframe = pd.concat([speed_dataframe] * speed_data.shape[0], ignore_index=True)
        speed_dataframe["particle_rmsd"] = speed_data
        speed_dataframe_list.append(speed_dataframe)

    # Concatenating summary dataframes and saving:
    summary_dataframe = pd.concat(summary_dataframe_list).reset_index()
    summary_dataframe.to_csv("./summary_dataframe.csv")

    rmsd_dataframe = pd.concat(speed_dataframe_list).reset_index()
    rmsd_dataframe.to_csv("./rmsd_dataframe.csv")

    return None

if __name__ == "__main__":
    main()