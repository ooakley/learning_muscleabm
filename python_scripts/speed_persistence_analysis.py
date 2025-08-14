"""Perform only basic order parameter calculations."""
import argparse
import os
import json
import numpy as np
import pandas as pd

SIMULATION_OUTPUTS_FOLDER = "./fileOutputs/"

OUTPUT_COLUMN_NAMES = [
    "frame", "particle", "x", "y",
    "shapeDirection",
    "orientation", "polarity_extent",
    "percept_direction", "percept_intensity",
    "actin_flow", "actin_mag",
    "collision_number",
    "cil_x", "cil_y",
    "movement_direction",
    "turning_angle",
    "stadium_x", "stadium_y",
    "sampled_angle"
]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a folder with a given integer name.')
    parser.add_argument('--folder_id', type=int, help='Integer corresponding to the folder name')
    args = parser.parse_args()
    return args


def main():
    """Run basic script logic."""
    # Parse arguments:
    args = parse_arguments()

    # Find specified simulation output files:
    subdirectory_path = os.path.join(SIMULATION_OUTPUTS_FOLDER, str(args.folder_id))

    # Get arguments to simulation:
    json_filepath = os.path.join(subdirectory_path, f"{args.folder_id}_arguments.json")
    with open(json_filepath) as json_file:
        simulation_arguments = json.load(json_file)

    # Define variables needed for matrix reshaping later on:
    timesteps = simulation_arguments["timestepsToRun"]
    cell_number = simulation_arguments["numberOfCells"]
    superiteration_number = simulation_arguments["superIterationCount"]

    # Loop through subiterations:
    magnitude_cellmeans = []
    dtheta_cellmeans = []
    order_parameters = []
    collision_cellmeans = []
    for seed in range(superiteration_number):
        # Read dataframe into memory:
        print(f"Reading subiteration {seed}...")
        filename = f"positions_seed{seed:03d}.csv"
        filepath = os.path.join(subdirectory_path, filename)
        trajectory_dataframe = pd.read_csv(
            filepath, index_col=None, header=None, names=OUTPUT_COLUMN_NAMES
        )

        # Sort by cell and then by frame:
        sorted_dataframe = trajectory_dataframe.sort_values(by=["particle", "frame"])

        # Get speed and angle data:
        actin_magnitude = np.asarray(sorted_dataframe["actin_mag"])
        actin_direction = np.asarray(sorted_dataframe["actin_flow"])
        collisions = np.asarray(sorted_dataframe["collision_number"])

        # Reshape into (CELL_NUM, TIMESTEPS) arrays:
        # Averaging over cells in a particular superiteration:
        try: 
            actin_magnitude = np.reshape(actin_magnitude, (cell_number, timesteps))
            collisions = np.reshape(collisions, (cell_number, timesteps))
        except ValueError as ve:
            print(f"Something has gone wrong with the outputs for this subiteration: {seed}")
            print("Skipping for now...")
            continue

        mean_magnitude = np.mean(actin_magnitude, axis=0)
        mean_collision = np.mean(collisions, axis=0)

        # Get change in angle & average over cells in superiteration:
        actin_direction = np.reshape(actin_direction, (cell_number, timesteps))
        angle_change = np.diff(actin_direction, axis=1)
        angle_change[angle_change > np.pi] -= 2*np.pi
        angle_change[angle_change < -np.pi] += 2*np.pi
        dtheta = np.abs(angle_change)
        mean_dtheta = np.mean(dtheta, axis=0)

        # Get x and y components across cell populations for each timestep:
        x_components = np.cos(actin_direction) * actin_magnitude
        y_components = np.sin(actin_direction) * actin_magnitude

        # Sum components:
        summed_x = np.mean(x_components, axis=0)
        summed_y = np.mean(y_components, axis=0)

        # Get order parameter:
        order_parameter_timeseries = np.sqrt(summed_x**2 + summed_y**2) / mean_magnitude

        # Accumulate to list:
        magnitude_cellmeans.append(mean_magnitude)
        dtheta_cellmeans.append(mean_dtheta)
        order_parameters.append(order_parameter_timeseries)
        collision_cellmeans.append(mean_collision)

    # Save to .npy files as (SUPERITERATIONS, TIMESTEPS) arrays:
    magnitude_cellmeans = np.stack(magnitude_cellmeans, axis=0)
    np.save(os.path.join(subdirectory_path, "magnitude_cellmeans.npy"), magnitude_cellmeans)

    dtheta_cellmeans = np.stack(dtheta_cellmeans, axis=0)
    np.save(os.path.join(subdirectory_path, "dtheta_cellmeans.npy"), dtheta_cellmeans)

    order_parameters = np.stack(order_parameters, axis=0)
    np.save(os.path.join(subdirectory_path, "order_parameters.npy"), order_parameters)

    collision_cellmeans = np.stack(collision_cellmeans, axis=0)
    np.save(os.path.join(subdirectory_path, "collision_cellmeans.npy"), collision_cellmeans)


if __name__ == "__main__":
    main()
