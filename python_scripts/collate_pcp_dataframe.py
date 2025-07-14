import json
import numpy as np


GRIDSEARCH_PARAMETERS = {
    "cueDiffusionRate": [0.0025, 2.5],
    "cueKa": [0.1, 5],
    "fluctuationAmplitude": [5e-5, 5e-3],
    "fluctuationTimescale": [1, 150],
    "maximumSteadyStateActinFlow": [0.0, 1.25],
    "numberOfCells": [75, 150],
    "actinAdvectionRate": [0, 1],
    "cellBodyRadius": [75, 125],
    # Collisions:
    "collisionFlowReductionRate": [0.0, 0.25],
    "collisionAdvectionRate": [0.0, 0.25],
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
