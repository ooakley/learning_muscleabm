"""
Script to run symbolic regression on outputs of model.
"""
import pysr
import numpy as np


def main():
    """Logic to handle loading and analysis of data."""
    # Load data:
    parameters = np.load("./collated_inputs.npy")
    speed_array = np.load("./mean_magnitudes.npy")

    # Instantiate model:
    sr_model = pysr.PySRRegressor(
        maxsize=20,
        niterations=40,
        batching=True,
        batch_size=250,
        binary_operators=["+", "*", "-"],
        unary_operators=[
            "exp",
            "inv(x) = 1/x",
        ],
        constraints={"pow": (9, 1)},
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        parsimony=0.0001,
        elementwise_loss="L1DistLoss()",
    )

    # Run genetic algorithm:
    sr_model.fit(parameters, speed_array)


if __name__ == "__main__":
    main()
