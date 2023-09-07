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
    for (int superIteration = 0; superIteration < 1; ++superIteration) {
        // Getting correct width string representation:
        std::stringstream iterationStringStream;
        iterationStringStream << std::setw(3) << std::setfill('0') << superIteration;
        std::string iterationString{iterationStringStream.str()};

        // Showing iteration on console:
        std::cout << "Iteration: " << iterationString << "\n";

        // Generating filepath & filename:
        std::string positionsFilename{
            directoryPath + "positions_seed" + iterationString + ".csv"
        };
        std::string matrixFilename{
            directoryPath + "matrix_seed" + iterationString + ".txt"
        };

        // Opening filestreams:
        std::ofstream csvFile;
        csvFile.open(positionsFilename);
        std::ofstream matrixFile;
        matrixFile.open(matrixFilename);

        // Running simulation:
        World mainWorld{World(superIteration, 1000, 32, 250)};
        for (int i = 0; i < 3000; ++i) {
            mainWorld.runSimulationStep();
            mainWorld.writePositionsToCSV(csvFile);
            mainWorld.writeMatrixToCSV(matrixFile);
        }
    }

    return 0;
}
