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
Hello! There are a lot of command line arguments - it's much easier to play around with the config file,
and run the simulation with this handy python script (after compiling the code):
python3 ./python_scripts/simulation/call_json_parameters.py --path_to_config ./configs/example_config.json
*/

int main(int argc, char** argv) {
    // Declaring variables to be parsed:
    // Simulation structural variables:
    std::string outputFolder;
    int jobArrayID;
    int superIterationCount;
    int timeStepsToRun;
    int numberOfCells;
    int worldSize;
    int gridSize;
    double matrixSampleRate;
    double patternSigma;
    int patternFibreCount;

    // Output options:
    bool verbosePositionOutput;
    bool summariseMatrixOutput;
    bool writeMatrixTimeseries;

    // Cell behaviour parameters:
    CellParameters cellParams;

    // Parsing variables from the command line:
    po::options_description desc("Parameters to be set for the simulation.");
    desc.add_options()
        ("outputFolder", po::value<std::string>(&outputFolder)->required(),
            "Name of the folder in which to place outputs."
        )
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
        ("matrixSampleRate", po::value<double>(&matrixSampleRate)->required(),
            "Rate of matrix sampling by cells."
        )
        // Repatterning parameters:
        ("patternSigma", po::value<double>(&patternSigma)->required(),
            "The standard deviation of the angular distribution of fibres in the existing pattern."
        )
        ("patternFibreCount", po::value<int>(&patternFibreCount)->default_value(0),
            "Number of background fibres placed at each ECM site before the simulation starts, "
            "with headings drawn from a normal distribution of standard deviation patternSigma. "
            "Set to 0 to start with an empty matrix."
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
        ("collisionAdvectionRate", po::value<double>(&cellParams.collisionAdvectionRate)->required(),
            "Degree of polarisation at which cell angular concentration reaches half its saturation value."
        )
        ("maximumSteadyStateActinFlow", po::value<double>(&cellParams.maximumSteadyStateActinFlow)->required(),
            "Degree of polarisation at which cell angular concentration reaches half its saturation value."
        )
        ("matrixCoupling", po::value<double>(&cellParams.matrixCoupling)->required(),
            "The strength to which the diffusion of velocity is limited by a cell's fibre environment."
        )
        // Collision parameters:
        ("cellBodyRadius", po::value<double>(&cellParams.cellBodyRadius)->required(),
            "Radius of the cell body for collision calculations."
        )
        ("collisionFlowReductionRate", po::value<double>(&cellParams.collisionFlowReductionRate)->required(),
            "Rate at which actin flow in the direction of a collision is reduced by a collision."
        )
        ("adhesionReductionRate", po::value<double>(&cellParams.adhesionReductionRate)->required(),
            "Rate at which actin flow in the direction of a collision is reduced by a collision."
        )
        // Shape parameters:
        ("cellStiffness", po::value<double>(&cellParams.cellStiffness)->required(),
            "The k value that determines the spring properties acting on the retracting end of the cell."
        )
        ("surfaceStickiness", po::value<double>(&cellParams.surfaceStickiness)->required(),
            "The Kon rate for stick-slip adhesions at the end of the cell."
        )
        // Output options:
        ("verbosePositionOutput", po::value<bool>(&verbosePositionOutput)->default_value(false),
            "Write internal cell state (polarity, actin flow, percepts etc.) alongside positions."
        )
        ("summariseMatrixOutput", po::value<bool>(&summariseMatrixOutput)->default_value(false),
            "Write the average heading, concentration and fibre count of each ECM site on a single "
            "line, instead of the heading of every fibre on one line per site."
        )
        ("writeMatrixTimeseries", po::value<bool>(&writeMatrixTimeseries)->default_value(false),
            "Write the matrix after every timestep, instead of only at the end of the simulation."
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
    const std::string directoryPath{outputFolder + "/"};
    if (!boostfs::exists(directoryPath)) {
        boostfs::create_directory(directoryPath);
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
            directoryPath + "/" + "positions_seed" + iterationString + ".csv"
        };
        const std::string matrixFilename{
            directoryPath + "/" + "matrix_seed" + iterationString + ".txt"
        };

        // Opening filestreams:
        std::ofstream csvFile;
        csvFile.open(positionsFilename);
        std::ofstream matrixFile;
        matrixFile.open(matrixFilename);

        // Running simulation:
        std::cout << "Instantiating world..." << std::endl;
        World mainWorld{
            World(
                seedDistribution(seedGenerator),
                worldSize,
                gridSize,
                numberOfCells,
                matrixSampleRate,
                patternSigma,
                patternFibreCount,
                cellParams
            )
        };

        auto writeMatrix = [&]() {
            if (summariseMatrixOutput) {
                mainWorld.writeSummarisedMatrixToCSV(matrixFile);
            } else {
                mainWorld.writeMatrixToCSV(matrixFile);
            }
        };

        std::cout << "Running simulation..." << std::endl;
        for (int i = 0; i < timeStepsToRun; ++i) {
            mainWorld.runSimulationStep();
            if (verbosePositionOutput) {
                mainWorld.writeVerbosePositionsToCSV(csvFile);
            } else {
                mainWorld.writePositionsToCSV(csvFile);
            }
            if (writeMatrixTimeseries) {
                writeMatrix();
            }
        }

        // Write final matrix to file:
        if (!writeMatrixTimeseries) {
            std::cout << "Writing matrix to file..." << std::endl;
            writeMatrix();
        }

        // We need to close files to flush remaining outputs to buffer.
        std::cout << "Closing files..." << std::endl;
        csvFile.close();
        matrixFile.close();
    }

    return 0;
}
