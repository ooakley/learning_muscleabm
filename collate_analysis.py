"""
Script to collate the outputs of simulational analysis across a set of cluster runs.

Saves to overall summary .csv file.
"""
import os
import copy
import pandas as pd

SIMULATION_OUTPUTS_FOLDER = "./fileOutputs/"


def main():
    # Reading in gridsearch dataframe:
    gridseach_dataframe = pd.read_csv("fileOutputs/gridsearch.txt", delimiter="\t")

    header = True
    with open("./summary_dataframe.csv", 'w') as summary_buffer, \
         open("./particle_dataframe.csv", 'w') as particle_buffer:

        for folder_id in range(1, 2 + 1):
            # Ensuring correct .csv appending behaviour:
            mode = 'w' if header else 'a'

            # Printing progress:
            if folder_id % 100 == 0:
                print(folder_id)

            # Defining filepaths:
            folder_path = os.path.join(SIMULATION_OUTPUTS_FOLDER, str(folder_id))
            summary_filepath = os.path.join(folder_path, "summary.csv")
            particle_filepath = os.path.join(folder_path, "particle_data.csv")
            vortex_filepath = os.path.join(folder_path, "collagen_vortex_data.csv")

            # Reading in all dataframes:
            try:
                trajectory_data = pd.read_csv(summary_filepath, index_col=0)
                vortex_data = pd.read_csv(vortex_filepath, index_col=0)
                full_data = pd.concat(
                    [trajectory_data, vortex_data],
                    axis=1, ignore_index=False, join='inner'
                )
                full_data.to_csv(
                    summary_buffer,
                    mode=mode, header=header, index=False,
                    float_format='%.8f'
                )
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

            particle_dataframe.to_csv(
                particle_buffer,
                mode=mode, header=header, index=False,
                float_format='%.8f'
            )

            # No header written to csv unless its the first row:
            header = False

    return None


if __name__ == "__main__":
    main()
