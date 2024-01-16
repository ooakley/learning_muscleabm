import os
import copy
import pandas as pd

SIMULATION_OUTPUTS_FOLDER = "./fileOutputs/"


def main():
    # Reading in gridsearch dataframe:
    gridseach_dataframe = pd.read_csv("fileOutputs/gridsearch.txt", delimiter="\t")

    # Reading through all iterations:
    summary_dataframe_list = []
    particle_dataframe_list = []
    for folder_id in range(1, 7200+1):
        if folder_id % 100 == 0:
            print(folder_id)

        # Defining filepaths:
        folder_path = os.path.join(SIMULATION_OUTPUTS_FOLDER, str(folder_id))
        summary_filepath = os.path.join(folder_path, "summary.csv")
        particle_filepath = os.path.join(folder_path, "particle_data.csv")

        # Reading in all dataframes:
        try:
            trajectory_data = pd.read_csv(summary_filepath, index_col=0)
            summary_dataframe_list.append(trajectory_data)
        except:
            print(f"Folder {folder_id} skipped...")
            continue

        # Reading in particle-level data:
        particle_data = pd.read_csv(particle_filepath, index_col=0)
        particle_dataframe = copy.deepcopy(
            gridseach_dataframe[gridseach_dataframe["array_id"] == folder_id]
        )
        particle_dataframe = pd.concat(
            [particle_dataframe] * particle_data.shape[0], ignore_index=True
        )
        particle_dataframe["particle_rmsd"] = particle_data["particle_rmsd"]
        particle_dataframe["particle_pt"] = particle_data["particle_persistence_time"]
        particle_dataframe["nn_distances"] = particle_data["particle_nn_distance"]
        particle_dataframe["seed"] = particle_data["seed"]

        particle_dataframe_list.append(particle_dataframe)

    # Concatenating summary dataframes and saving:
    summary_dataframe = pd.concat(summary_dataframe_list).reset_index()
    summary_dataframe.to_csv("./summary_dataframe.csv")

    particle_dataframe = pd.concat(particle_dataframe_list).reset_index()
    particle_dataframe.to_csv("./particle_dataframe.csv")

    return None


if __name__ == "__main__":
    main()
