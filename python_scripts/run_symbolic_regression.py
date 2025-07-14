"""
Script to run symbolic regression on outputs of model.
"""
import os
import pysr

import numpy as np

def main():
    """Logic to handle loading and analysis of data."""
    # Load data:
    deterministic_directory_path = "./gridsearch_data/deterministic_collisions"
    det_collisions_array = np.load(
        os.path.join(deterministic_directory_path, "collated_collisions.npy")
    )
    det_speed_array = np.load(
        os.path.join(deterministic_directory_path, "collated_magnitudes.npy")
    )
    det_dtheta_array = np.load(
        os.path.join(deterministic_directory_path, "collated_dtheta.npy")
    )
    det_op_array = np.load(
        os.path.join(deterministic_directory_path, "collated_order_parameters.npy")
    )

    # Take mean over final timepoints (stationarity):
    det_stationary_col_mean = np.mean(det_collisions_array[:, 1000:], axis=1)
    det_stationary_speed_mean = np.mean(det_speed_array[:, 1000:], axis=1)
    det_stationary_dtheta_mean = np.mean(det_dtheta_array[:, 1000:], axis=1)
    det_stationary_op_mean = np.mean(det_op_array[:, 1000:], axis=1)

    # Constructing input dataset for regression:
    measured_properties = np.stack(
        [det_stationary_col_mean,
         det_stationary_speed_mean,
         det_stationary_dtheta_mean],
        axis=1
    )

    # Instantiate model:
    timeout_minutes = 6*60
    sr_model = pysr.PySRRegressor(
        maxsize=35,
        population_size=75,
        niterations=10000,
        batching=True,
        batch_size=1000,
        binary_operators=["+", "*", "-"],
        unary_operators=[
            "exp",
            "log",
            "square",
            "inv",
        ],
        constraints={"pow": (9, 1)},
        elementwise_loss="L1DistLoss()",
        weight_optimize=0.01,
        timeout_in_seconds=timeout_minutes * 60,
        verbosity=0,
        output_directory="sr_outputs"
    )

    # Run genetic algorithm:
    sr_model.fit(
        measured_properties, det_stationary_op_mean,
        weights=det_stationary_op_mean
    )


if __name__ == "__main__":
    main()
