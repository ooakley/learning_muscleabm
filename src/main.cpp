#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/fstream.hpp"

#include "world.h"

namespace boostfs = boost::filesystem;

int main(int argc, char** argv) {
    // Creating output directory if not present:
    std::string directoryPath{"./fileOutputs/"};
    if (!boostfs::exists(directoryPath)) {
        boostfs::create_directory(directoryPath);
    }

    // Running mulitple simulations:
    for (int superIteration = 0; superIteration < 10; ++superIteration) {
        // Showing iteration on console:
        std::cout << "Iteration: " << std::to_string(superIteration) << "\n";

        // Generating filename:
        std::string positionsFilename{
            directoryPath + "positions_seed" + std::to_string(superIteration) + ".csv"
        };
        std::string matrixFilename{
            directoryPath + "matrix_seed" + std::to_string(superIteration) + ".txt"
        };

        // Opening filestreams:
        std::ofstream csvFile;
        csvFile.open(positionsFilename);
        std::ofstream matrixFile;
        matrixFile.open(matrixFilename);

        // Running simulation:
        World mainWorld{World(superIteration, 1000, 64, 100)};
        for (int i = 0; i < 1000; ++i) {
            mainWorld.runSimulationStep();
            mainWorld.writePositionsToCSV(csvFile);
            mainWorld.writeMatrixToCSV(matrixFile);
        }
    }

    return 0;
}
