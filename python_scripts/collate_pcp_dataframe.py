import json
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
    input_matrix = []
    for i in range(4096):
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
