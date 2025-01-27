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
parallelisation on the cluster easier. If you'd just like to run the simulation, here's what you can paste into
the terminal after compiling the code:
./build/src/main \
--jobArrayID 1 --superIterationCount 10 --timestepsToRun 576 --numberOfCells 70 \
--worldSize 2048 --gridSize 32 \
--cellTypeProportions 0 \
--matrixAdditionRate 0.155 --matrixTurnoverRate 9.5e-4 \
--cellDepositionSigma 70 --cellSensationSigma 70 \
--halfSatCellAngularConcentration 0.5 --maxCellAngularConcentration 0.5 \
--halfSatMeanActinFlow 0.5 --maxMeanActinFlow 0.5 \
--flowScaling 0.5 --polarityPersistence 0.5 \
--actinPolarityRedistributionRate 0.5 --polarityNoiseSigma 0.5 \
--halfSatMatrixAngularConcentration 0.5 --maxMatrixAngularConcentration 0.5 \
--homotypicInhibitionRate 0.5 --heterotypicInhibitionRate 0.5 \
--cellBodyRadius 20 --maxCellExtension 200 --inhibitionStrength 1 \
--collisionRepolarisation 0.5 --collisionRepolarisationRate 0.5 \
--thereIsMatrixInteraction false
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
    double cellTypeProportions;
    bool thereIsMatrixInteraction;
    double matrixAdditionRate;
    double matrixTurnoverRate;
    double cellDepositionSigma;
    double cellSensationSigma;

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
        ("cellTypeProportions", po::value<double>(&cellTypeProportions)->required(),
            "Proportion of cells that take type 0."
        )
        ("matrixTurnoverRate", po::value<double>(&matrixTurnoverRate)->required(),
            "Stability of the matrix under reorientation by cell movement."
        )
        ("matrixAdditionRate", po::value<double>(&matrixAdditionRate)->required(),
            "Stability of the matrix under reorientation by cell movement."
        )
        ("halfSatCellAngularConcentration", po::value<double>(&cellParams.halfSatCellAngularConcentration)->required(),
            "Degree of polarisation at which cell angular concentration reaches half its saturation value."
        )
        ("maxCellAngularConcentration", po::value<double>(&cellParams.halfSatCellAngularConcentration)->required(),
            "Maximum concentration of the angular distribution from which the cell draws its direction of motion."
        )
        ("halfSatMeanActinFlow", po::value<double>(&cellParams.halfSatMeanActinFlow)->required(),
            "Degree of polarisation at which cell movement per timeframe reaches half its saturation value."
        )
        ("maxMeanActinFlow", po::value<double>(&cellParams.maxMeanActinFlow)->required(),
            "Maximum mean actin flow per simulation step. Determines distribution from which cell step size is drawn."
        )
        ("flowScaling", po::value<double>(&cellParams.flowScaling)->required(),
            "The number by which actin flow is multiplied to get actual distance travelled by the cell."
        )
        ("polarityPersistence", po::value<double>(&cellParams.polarityPersistence)->required(),
            "The rate at which cell polarisation is updated from timestep to timestep."
        )     
        ("actinPolarityRedistributionRate", po::value<double>(&cellParams.actinPolarityRedistributionRate)->required(),
            "The number by which actin flow is multiplied to get effective polarisation update."
        )
        ("polarityNoiseSigma", po::value<double>(&cellParams.polarityNoiseSigma)->required(),
            "Size of random fluctuations in cellular polarisation values."
        )
        ("halfSatMatrixAngularConcentration", po::value<double>(&cellParams.halfSatMatrixAngularConcentration)->required(),
            "The matrix density at which angular concentration conferred by the matrix reaches half its maximum value."
        )
        ("maxMatrixAngularConcentration", po::value<double>(&cellParams.maxMatrixAngularConcentration)->required(),
            "The maximum angular concentration conferred by orientational cues in the matrix."
        )
        ("homotypicInhibitionRate", po::value<double>(&cellParams.homotypicInhibitionRate)->required(),
            "Likelihood that of two cells of the same type will collide if in contact for a given timestep."
        )
        ("heterotypicInhibitionRate", po::value<double>(&cellParams.heterotypicInhibitionRate)->required(),
            "Likelihood that of two cells of different types will collide if in contact for a given timestep."
        )
        ("collisionRepolarisation", po::value<double>(&cellParams.collisionRepolarisation)->required(),
            "The amount of polarisation that is conferred by collisions."
        )
        ("collisionRepolarisationRate", po::value<double>(&cellParams.collisionRepolarisationRate)->required(),
            "The rate at which cellular polarisation is dragged to the collisionRepolarisation value."
        )
        ("cellBodyRadius", po::value<double>(&cellParams.cellBodyRadius)->required(),
            "Radius of the cell body for collision calculations.."
        )
        ("maxCellExtension", po::value<double>(&cellParams.maxCellExtension)->required(),
            "Maximum length of polarisation-induced cell extensions."
        )
        ("inhibitionStrength", po::value<double>(&cellParams.inhibitionStrength)->required(),
            "Rate at which actin flow in the direction of a collision is reduced by such a collision."
        )
        ("cellDepositionSigma", po::value<double>(&cellDepositionSigma)->required(),
            "Standard deviation of gaussian kernel defining spatial extent of matrix deposition."
        )
        ("cellSensationSigma", po::value<double>(&cellSensationSigma)->required(),
            "Standard deviation of gaussian kernel defining spatial extent of matrix sensation."
        )
        ("thereIsMatrixInteraction", po::value<bool>(&thereIsMatrixInteraction)->required(),
            "Whether or not cells undergo interaction with the matrx."
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
                cellTypeProportions,
                thereIsMatrixInteraction,
                matrixTurnoverRate,
                matrixAdditionRate,
                cellDepositionSigma,
                cellSensationSigma,
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
