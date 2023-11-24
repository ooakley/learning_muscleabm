import os
import pandas as pd

SIMULATION_OUTPUTS_FOLDER = "./fileOutputs/"

def main():
    summary_data_list = []
    for folder_id in range(1, 1025):
        folder_path = os.path.join(SIMULATION_OUTPUTS_FOLDER, str(folder_id))
        summary_filepath = os.path.join(folder_path, "summary.csv")
        try:
            trajectory_data = pd.read_csv(summary_filepath, index_col=0)
            summary_data_list.append(trajectory_data)
        except:
            continue
    
    # Concatenating summary dataframes and saving:
    summary_dataframe = pd.concat(summary_data_list).reset_index()
    summary_dataframe.to_csv("./summary_dataframe.csv")

    return None

if __name__ == "__main__":
    main()