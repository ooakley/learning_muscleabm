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
Hello! There are a lot of command line arguments, for which I apologise.If you'd just like to run the simulation,
here's a handy python command that you can paste into the terminal after compiling the code:
python3 ./python_scripts/call_json_parameters.py --path_to_config ./example_config.json
*/

int main(int argc, char** argv) {
    // Declaring variables to be parsed:
    // Simulation structural variables:
    int jobArrayID;
    int superIterationCount;
    int timeStepsToRun;
    int numberOfCells;
    int worldSize;
    int gridSize;
    bool thereIsMatrixInteraction;
    double matrixAdditionRate;
    double matrixTurnoverRate;

    // Cell behaviour parameters:
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
        ("timestepsToRun", po::value<int>(&timeStepsToRun)->required(),
            "Number of timesteps of the simulation to run within in each iteration."
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
        ("matrixTurnoverRate", po::value<double>(&matrixTurnoverRate)->required(),
            "Stability of the matrix under reorientation by cell movement."
        )
        ("matrixAdditionRate", po::value<double>(&matrixAdditionRate)->required(),
            "Stability of the matrix under reorientation by cell movement."
        )
        ("thereIsMatrixInteraction", po::value<bool>(&thereIsMatrixInteraction)->required(),
            "Whether or not cells undergo interaction with the matrx."
        )
        // Cell movement parameters:
        ("dt", po::value<double>(&cellParams.dt)->required(),
            "Length of individual timesteps."
        )
        ("cueDiffusionRate", po::value<double>(&cellParams.cueDiffusionRate)->required(),
            "Degree of polarisation at which cell angular concentration reaches half its saturation value."
        )
        ("cueKa", po::value<double>(&cellParams.cueKa)->required(),
            "Degree of cue concentration at which actin flow induction reaches half maximum."
        )
        ("fluctuationAmplitude", po::value<double>(&cellParams.fluctuationAmplitude)->required(),
            "Degree of polarisation at which cell angular concentration reaches half its saturation value."
        )
        ("fluctuationTimescale", po::value<double>(&cellParams.fluctuationTimescale)->required(),
            "Degree of polarisation at which cell angular concentration reaches half its saturation value."
        )
        ("actinAdvectionRate", po::value<double>(&cellParams.actinAdvectionRate)->required(),
            "Degree of polarisation at which cell angular concentration reaches half its saturation value."
        )
        ("matrixAdvectionRate", po::value<double>(&cellParams.matrixAdvectionRate)->required(),
            "Degree of polarisation at which cell angular concentration reaches half its saturation value."
        )
        ("collisionAdvectionRate", po::value<double>(&cellParams.collisionAdvectionRate)->required(),
            "Degree of polarisation at which cell angular concentration reaches half its saturation value."
        )
        ("maximumSteadyStateActinFlow", po::value<double>(&cellParams.maximumSteadyStateActinFlow)->required(),
            "Degree of polarisation at which cell angular concentration reaches half its saturation value."
        )
        // Collision parameters:
        ("cellBodyRadius", po::value<double>(&cellParams.cellBodyRadius)->required(),
            "Radius of the cell body for collision calculations."
        )
        ("aspectRatio", po::value<double>(&cellParams.aspectRatio)->required(),
            "Degree of polarisation at which cell angular concentration reaches half its saturation value."
        )
        ("collisionFlowReductionRate", po::value<double>(&cellParams.collisionFlowReductionRate)->required(),
            "Rate at which actin flow in the direction of a collision is reduced by a collision."
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
    const std::string directoryPath{"./fileOutputs/"};
    if (!boostfs::exists(directoryPath)) {
        boostfs::create_directory(directoryPath);
    }

    // Generating subdirectory to store simulation results:
    const std::string subdirectoryPath{directoryPath + std::to_string(jobArrayID) + "/"};
    if (!boostfs::exists(subdirectoryPath)) {
        boostfs::create_directory(subdirectoryPath);
    }

    // Defining RNG generator for world seeds:
    std::mt19937 seedGenerator = std::mt19937(jobArrayID);
    std::uniform_int_distribution<unsigned int> seedDistribution = std::uniform_int_distribution<unsigned int>(0, UINT32_MAX);

    // Running mulitple simulations:
    for (int superIteration = 0; superIteration < superIterationCount; ++superIteration) {
        // Getting correct width string representation:
        std::stringstream iterationStringStream;
        iterationStringStream << std::setw(3) << std::setfill('0') << superIteration;
        std::string iterationString{iterationStringStream.str()};

        // Showing iteration on console:
        std::cout << "Iteration: " << iterationString << "\n";

        // Generating filepath & filename:
        const std::string positionsFilename{
            subdirectoryPath + "positions_seed" + iterationString + ".csv"
        };
        const std::string matrixFilename{
            subdirectoryPath + "matrix_seed" + iterationString + ".txt"
        };

        // Opening filestreams:
        std::ofstream csvFile;
        csvFile.open(positionsFilename);
        std::ofstream matrixFile;
        matrixFile.open(matrixFilename);

        // Running simulation:
        World mainWorld{
            World(
                seedDistribution(seedGenerator),
                worldSize,
                gridSize,
                numberOfCells,
                thereIsMatrixInteraction,
                matrixTurnoverRate,
                matrixAdditionRate,
                cellParams
            )
        };
        for (int i = 0; i < timeStepsToRun; ++i) {
            mainWorld.runSimulationStep();
            mainWorld.writePositionsToCSV(csvFile);
            mainWorld.writeMatrixToCSV(matrixFile);
        }

        // We need to close files to flush remaining outputs to buffer.
        csvFile.close();
        matrixFile.close();

    }

    return 0;
}
