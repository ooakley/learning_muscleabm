import json
import numpy as np


GRIDSEARCH_PARAMETERS = [
    "matrixAdditionRate",
    "matrixTurnoverRate",
    "flowScaling",
    "cellBodyRadius",
    "eccentricity",
    "polarityDiffusionRate",
    "actinAdvectionRate",
    "contactAdvectionRate",
    "inhibitionStrength",
    "maxCellAngularConcentration",
    "maxMatrixAngularConcentration"
]


def main():
    input_matrix = []
    for i in range(13000):
        with open(f'./fileOutputs/{i}/{i}_arguments.json') as json_data:
            # Reading data:
            parameter_dict = json.load(json_data)
            input_row = []
            for parameter_name in GRIDSEARCH_PARAMETERS:
                input_row.append(parameter_dict[parameter_name])
            input_matrix.append(np.array(input_row))
    input_matrix = np.stack(input_matrix, axis=0)
    np.save("./collated_inputs.npy", input_matrix)


if __name__ == "__main__":
    main()
