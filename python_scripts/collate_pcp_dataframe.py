import json
import numpy as np


GRIDSEARCH_PARAMETERS = {
    "cueDiffusionRate": [0.0001, 2.5],
    "cueKa": [0.0001, 5],
    "fluctuationAmplitude": [5e-5, 5e-3],
    "fluctuationTimescale": [1, 250],
    "maximumSteadyStateActinFlow": [0.0, 3],
    "numberOfCells": [75, 150],
    "actinAdvectionRate": [0.0, 1.5],
    "cellBodyRadius": [15, 75],
    # Collisions:
    "collisionFlowReductionRate": [0.0, 0.25],
    "collisionAdvectionRate": [0.0, 1.5],
    # Shape:
    "stretchFactor": [0.0, 7.5],
    "slipFactor": [1e-5, 1e-2]
}

def main():
    input_matrix = []
    for i in range(int(16384 / 2)):
        with open(f'./fileOutputs/{i}/{i}_arguments.json') as json_data:
            # Reading data:
            parameter_dict = json.load(json_data)
            input_row = []
            for parameter_name in GRIDSEARCH_PARAMETERS.keys():
                input_row.append(parameter_dict[parameter_name])
            input_matrix.append(np.array(input_row))
    input_matrix = np.stack(input_matrix, axis=0)
    np.save("./collated_inputs.npy", input_matrix)


if __name__ == "__main__":
    main()
