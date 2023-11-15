import numpy as np

# Defines the parameter ranges, with [start, stop, num_steps]:
GRIDSEARCH = {
    "superIterationCount": [10, 10, 1],
    "numberOfCells": [150, 150, 1],
    "worldSize": [1024, 1024, 1],
    "gridSize": [32, 32, 1],
    "wbK": [1, 1, 1],
    "kappa": [3, 3, 1],
    "homotypicInhibition": [0.4, 0.6, 2],
    "heterotypicInhibition": [0.8, 0.9, 2],
    "polarityPersistence": [0.3, 0.95, 4],
    "polarityTurningCoupling": [0.1, 0.9, 4],
    "flowScaling": [0.5, 4, 3],
    "flowPolarityCoupling": [0.2, 2, 3],
}

def main():
    # Generate the grid search variables using the most unholy list comprehension I have written:
    argument_grid = [
        (
            int(superIterationCount), int(numberOfCells), int(worldSize), int(gridSize),
            wbK, kappa, homotypicInhibition, heterotypicInhibition, polarityPersistence,
            polarityTurningCoupling, flowScaling, flowPolarityCoupling 
        ) \
        for superIterationCount in np.linspace(*GRIDSEARCH["superIterationCount"]) \
        for numberOfCells in np.linspace(*GRIDSEARCH["numberOfCells"]) \
        for worldSize in np.linspace(*GRIDSEARCH["worldSize"]) \
        for gridSize in np.linspace(*GRIDSEARCH["gridSize"]) \
        for wbK in np.linspace(*GRIDSEARCH["wbK"]) \
        for kappa in np.linspace(*GRIDSEARCH["kappa"]) \
        for homotypicInhibition in np.linspace(*GRIDSEARCH["homotypicInhibition"]) \
        for heterotypicInhibition in np.linspace(*GRIDSEARCH["heterotypicInhibition"]) \
        for polarityPersistence in np.linspace(*GRIDSEARCH["polarityPersistence"]) \
        for polarityTurningCoupling in np.linspace(*GRIDSEARCH["polarityTurningCoupling"]) \
        for flowScaling in np.linspace(*GRIDSEARCH["flowScaling"]) \
        for flowPolarityCoupling in np.linspace(*GRIDSEARCH["flowPolarityCoupling"])
    ]

    print(len(argument_grid))

    # Write the grid search variables to a file to be read by a job array on the cluster:
    with open('gridsearch.txt', 'w') as f:
        # Writing column names:
        header_string = ""
        header_string += "array_id\t"
        for argument_name in GRIDSEARCH.keys():
            header_string += argument_name
            header_string += "\t"
        header_string += "\n"
        f.write(header_string)

        # Writing gridsearch values:
        for i, argtuple in enumerate(argument_grid):
<<<<<<< HEAD
            f.write(f"{i+1},")
=======
            f.write(f"{i}\t")
>>>>>>> bf09d54ccade163391cf894a1e036c572f4a374a
            for argument_value in argtuple:
                f.write(f"{argument_value}\t")
            f.write("\n")

    return None

if __name__ == "__main__":
    main()