import json
import numpy as np


GRIDSEARCH_PARAMETERS = {
    "matrixAdditionRate": [0.05, 0.25],
    "matrixTurnoverRate": [0.05, 0.25],
    "maxCellAngularConcentration": [1, 5],
    "maxMeanActinFlow": [1, 5],
    "flowScaling": [0, 0.2],
    "polarityDiffusionRate": [0, 0.05],
    "actinAdvectionRate": [0., 2.],
    "contactAdvectionRate": [0., 2.],
    "maxMatrixAngularConcentration": [1, 5],
    "cellBodyRadius": [60, 200],
    "eccentricity": [0, 0.98],
    "inhibitionStrength": [0.0, 1.0]
}


def main():
    input_matrix = []
    for i in range(150000):
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
