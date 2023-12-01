import os
import numpy as np

# Defines the parameter ranges, with [start, stop, num_steps]:
GRIDSEARCH = {
    "superIterationCount": [10, 10, 1],
    "numberOfCells": [200, 200, 1],
    "worldSize": [2048, 2048, 1],
    "gridSize": [32, 32, 1],
    "cellTypeProportions": [0, 0, 1],
    "matrixPersistence": [0.99, 0.99, 1],
    "wbK": [2, 2, 1],
    "kappa": [2.5, 2.5, 1],
    "homotypicInhibition": [0.1, 0.9, 5],
    "heterotypicInhibition": [0, 0, 1],
    "polarityPersistence": [0.1, 0.9, 5],
    "polarityTurningCoupling": [0.1, 0.9, 5],
    "flowScaling": [0.2, 3.5, 5],
    "flowPolarityCoupling": [0.2, 3.5, 5],
    "collisionRepolarisation": [0, 0, 1],
    "repolarisationRate": [0.75, 0.75, 1]
}

def main():
    # Generate the grid search variables using the most unholy list comprehension I have written:
    argument_grid = [
        (
            int(superIterationCount), int(numberOfCells), int(worldSize), int(gridSize),
            cellTypeProportions, matrixPersistence,
            wbK, kappa, homotypicInhibition, heterotypicInhibition, polarityPersistence,
            polarityTurningCoupling, flowScaling, flowPolarityCoupling,
            collisionRepolarisation, repolarisationRate
        ) \
        for superIterationCount in np.linspace(*GRIDSEARCH["superIterationCount"]) \
        for numberOfCells in np.linspace(*GRIDSEARCH["numberOfCells"]) \
        for worldSize in np.linspace(*GRIDSEARCH["worldSize"]) \
        for gridSize in np.linspace(*GRIDSEARCH["gridSize"]) \
        for cellTypeProportions in np.linspace(*GRIDSEARCH["cellTypeProportions"]) \
        for matrixPersistence in np.linspace(*GRIDSEARCH["matrixPersistence"]) \
        for wbK in np.linspace(*GRIDSEARCH["wbK"]) \
        for kappa in np.linspace(*GRIDSEARCH["kappa"]) \
        for homotypicInhibition in np.linspace(*GRIDSEARCH["homotypicInhibition"]) \
        for heterotypicInhibition in np.linspace(*GRIDSEARCH["heterotypicInhibition"]) \
        for polarityPersistence in np.linspace(*GRIDSEARCH["polarityPersistence"]) \
        for polarityTurningCoupling in np.linspace(*GRIDSEARCH["polarityTurningCoupling"]) \
        for flowScaling in np.linspace(*GRIDSEARCH["flowScaling"]) \
        for flowPolarityCoupling in np.linspace(*GRIDSEARCH["flowPolarityCoupling"]) \
        for collisionRepolarisation in np.linspace(*GRIDSEARCH["collisionRepolarisation"]) \
        for repolarisationRate in np.linspace(*GRIDSEARCH["repolarisationRate"])
    ]

    print(len(argument_grid))

    # Generating directory and output gridsearch to that directory:
    if not os.path.exists("fileOutputs"):
        os.mkdir("fileOutputs")
    output_filepath = os.path.join("fileOutputs", "gridsearch.txt")

    # Write the grid search variables to a file to be read by a job array on the cluster:
    with open(output_filepath, 'w') as f:
        # Writing column names:
        header_string = ""
        header_string += "array_id"
        for argument_name in GRIDSEARCH.keys():
            header_string += "\t"
            header_string += argument_name
        header_string += "\n"
        f.write(header_string)

        # Writing gridsearch values:
        for i, argtuple in enumerate(argument_grid):
            f.write(f"{i+1}")
            for argument_value in argtuple:
                f.write("\t")
                f.write(f"{argument_value}")
            f.write("\n")

    return None

if __name__ == "__main__":
    main()