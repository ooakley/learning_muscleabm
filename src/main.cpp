#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <string>

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/fstream.hpp"
#include "boost/program_options.hpp"

#include "world.h"

namespace boostfs = boost::filesystem;
namespace po = boost::program_options;

/*
Hello! There are a lot of command line arguments, for which I apologise. It makes
parallelisation easier. If you'd just like to run the simulation, here's what you can paste into
the terminal after compiling the code:
./main --jobArrayID 0 --superIterationCount 5 --numberOfCells 200 --worldSize 1000 --gridSize 32 \
    --wbK 1.0 --kappa 2.5 --homotypicInhibition 0.8 --heterotypicInhibition 0.5 \
    --polarityPersistence 0.9 --polarityTurningCoupling 0.6 --flowScaling 1.2 \
    --flowPolarityCoupling 0.7
*/

int main(int argc, char** argv) {
    // Declaring variables to be parsed:
    int jobArrayID;
    int superIterationCount;
    int numberOfCells;
    int worldSize;
    int gridSize;
    CellParameters cellParams;

    // Parsing variables from the command line:
    po::options_description desc("Parameters to be set for the simulation.");
    desc.add_options()
        ("jobArrayID", po::value<int>(&jobArrayID)->required(),
            "ID of the job array, to be referenced against gridsearch.txt. Ignore if not using array."
        )
        ("superIterationCount", po::value<int>(&superIterationCount)->required(),
            "Number of iterations with same parameters to run, each with a different seed."
        )
        ("numberOfCells", po::value<int>(&numberOfCells)->required(),
            "Number of cells in the simulation."
        )
        ("worldSize", po::value<int>(&worldSize)->required(),
            "Size of the world - ideally approximating the number of pixels in live imaging data."
        )
        ("gridSize", po::value<int>(&gridSize)->required(),
            "Defines number of cells in grid that defines the ECM & cell interaction neighbourhood."
        )
        ("wbK", po::value<float>(&cellParams.wbK)->required(),
            "Weibull distribution k value."
        )
        ("kappa", po::value<float>(&cellParams.kappa)->required(),
            "Von Mises distribution kappa value."
        )
        ("homotypicInhibition", po::value<float>(&cellParams.homotypicInhibition)->required(),
            "Homotypic inhibition rate."
        )
        ("heterotypicInhibition", po::value<float>(&cellParams.heterotypicInhibition)->required(),
            "Heterotypic inhibition rate."
        )
        ("polarityPersistence", po::value<float>(&cellParams.polarityPersistence)->required(),
            "How quickly the polarity of a cell changes when it moves."
        )
        ("polarityTurningCoupling", po::value<float>(&cellParams.polarityTurningCoupling)->required(),
            "Magnitude of the influence that polarity has on turning probability."
        )
        ("flowScaling", po::value<float>(&cellParams.flowScaling)->required(),
            "The scaling between the amount of actin flow and the cell step size."
        )
        ("flowPolarityCoupling", po::value<float>(&cellParams.flowPolarityCoupling)->required(),
            "Magnitude of the influence that actin flow has on polariy."
        )
    ;

    // Parse the variables from the command line:
    po::variables_map vm;
    try {
        // Parse the command line arguments
        po::store(po::parse_command_line(argc, argv, desc), vm);
        // Notify if there are any unrecognized options
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Creating output directory if not present:
    std::string directoryPath{"./fileOutputs/"};
    if (!boostfs::exists(directoryPath)) {
        boostfs::create_directory(directoryPath);
    }

    // Generating subdirectory to store simulation results:
    std::string subdirectoryPath{directoryPath + std::to_string(jobArrayID) + "/"};
    if (!boostfs::exists(subdirectoryPath)) {
        boostfs::create_directory(subdirectoryPath);
    }

    // Running mulitple simulations:
    for (int superIteration = 0; superIteration < superIterationCount; ++superIteration) {
        // Getting correct width string representation:
        std::stringstream iterationStringStream;
        iterationStringStream << std::setw(3) << std::setfill('0') << superIteration;
        std::string iterationString{iterationStringStream.str()};

        // Showing iteration on console:
        std::cout << "Iteration: " << iterationString << "\n";

        // Generating filepath & filename:
        std::string positionsFilename{
            subdirectoryPath + "positions_seed" + iterationString + ".csv"
        };
        std::string matrixFilename{
            subdirectoryPath + "matrix_seed" + iterationString + ".txt"
        };

        // Opening filestreams:
        std::ofstream csvFile;
        csvFile.open(positionsFilename);
        std::ofstream matrixFile;
        matrixFile.open(matrixFilename);

        // Running simulation:
        World mainWorld{World(superIteration, worldSize, gridSize, numberOfCells, cellParams)};
        for (int i = 0; i < 1000; ++i) {
            mainWorld.runSimulationStep();
            mainWorld.writePositionsToCSV(csvFile);
            mainWorld.writeMatrixToCSV(matrixFile);
        }
    }

    return 0;
}
