"""
Script to run symbolic regression on outputs of model.
"""
import pysr
import os

import numpy as np

GRIDSEARCH_PARAMETERS = {
    "cueDiffusionRate": [0.005, 0.5],
    "cueKa": [0.25, 1.25],
    "fluctuationAmplitude": [0.01, 0.2],
    "fluctuationTimescale": [1.5, 20],
    "actinAdvectionRate": [0.1, 2],
    "maximumSteadyStateActinFlow": [0.5, 5],
}


def main():
    """Logic to handle loading and analysis of data."""
    # Load data:
    data_directory = "./gridsearch_data/out_abcUniformOP"
    # parameters = np.load(os.path.join(data_directory, "collated_inputs.npy"))
    parameters = np.load("./abc_parameters.npy")
    op_array = np.load(os.path.join(data_directory, "collated_order_parameters.npy"))
    # dtheta_array = np.load(os.path.join(data_directory, "mean_dtheta.npy"))

    # normalised_parameters = np.zeros_like(parameters)
    # for index, parameter_range in enumerate(GRIDSEARCH_PARAMETERS.values()):
    #     _minimum = parameter_range[0]
    #     _maximum = parameter_range[1]
    #     normalised_parameters[:, index] = \
    #         (parameters[:, index] - _minimum) / (_maximum - _minimum)

    # simplified_parameters = np.delete(parameters, [1, 3], axis=1)

    # Instantiate model:
    timeout_hours = 48
    timeout_minutes = timeout_hours * 60
    sr_model = pysr.PySRRegressor(
        niterations=500000,
        maxsize=45,
        weight_optimize=0.001,
        warmup_maxsize_by=0.001,
        batching=True,
        batch_size=1000,
        ncycles_per_iteration=1000,
        binary_operators=["+", "*", "-", "^"],
        unary_operators=[
            "exp",
            "log",
            "inv"
        ],
        constraints={
            "^": (-1, 1),
            "exp": 5,
            "log": 5,
        },
        elementwise_loss="L2DistLoss()",
        timeout_in_seconds=timeout_minutes * 60,
        verbosity=0,
        output_directory="sr_outputs",
        # parsimony=0.00001,
        run_id="20250701_SR_abcOP_DeterministicCollisions"
    )

    # Run genetic algorithm:
    sr_model.fit(
        parameters, op_array[:, 0]
    )


if __name__ == "__main__":
    main()
