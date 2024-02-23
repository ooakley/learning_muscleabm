#include "world.h"
#include "agents.h"
#include "ecm.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <tuple>

#include <boost/numeric/ublas/matrix.hpp>
namespace boostMatrix = boost::numeric::ublas;


// Constructor and intialisation:
World::World
(
    int setWorldSeed,
    double setWorldSideLength,
    int setECMElementCount,
    int setNumberOfCells,
    double setCellTypeProportions,
    double setMatrixTurnoverRate,
    double setMatrixAdditionRate,
    double setCellDepositionSigma,
    double setCellSensationSigma,
    CellParameters setCellParameters
)
    : worldSeed{setWorldSeed}
    , worldSideLength{setWorldSideLength}
    , countECMElement{setECMElementCount}
    , lengthECMElement{worldSideLength/countECMElement}
    , ecmField{ECMField(countECMElement, 16, setWorldSideLength, setMatrixTurnoverRate, setMatrixAdditionRate)}
    , numberOfCells{setNumberOfCells}
    , cellTypeProportions{setCellTypeProportions}
    , cellDepositionSigma{setCellDepositionSigma}
    , cellSensationSigma{setCellSensationSigma}
    , simulationTime{0}
    , cellParameters{setCellParameters}
{
    // Initialising randomness:
    seedGenerator = std::mt19937(worldSeed);
    seedDistribution = std::uniform_int_distribution<unsigned int>(0, UINT32_MAX);
    std::cout << seedDistribution(seedGenerator) << "\n";

    // Initialising generators for cell seeds:
    shuffleGenerator = std::mt19937(seedDistribution(seedGenerator));
    cellSeedGenerator = std::mt19937(seedDistribution(seedGenerator));

    // Initialising generators for random selection of position, heading and contact inhibition:
    xPositionGenerator = std::mt19937(seedDistribution(seedGenerator));
    yPositionGenerator = std::mt19937(seedDistribution(seedGenerator));
    headingGenerator = std::mt19937(seedDistribution(seedGenerator));
    contactInhibitionGenerator = std::mt19937(seedDistribution(seedGenerator));

    // Distributions:
    positionDistribution = std::uniform_real_distribution<double>(0, worldSideLength);
    headingDistribution = std::uniform_real_distribution<double>(-M_PI, M_PI);
    contactInhibitionDistribution  = std::uniform_real_distribution<double>(0, 1);

    // Kernels:
    cellSensationKernel = generateGaussianKernel(cellSensationSigma);
    cellDepositionKernel = generateGaussianKernel(cellDepositionSigma);

    // Initialising cells:
    initialiseCellVector();
}

// Getters:
void World::writePositionsToCSV(std::ofstream& csvFile) {
    for (int i = 0; i < numberOfCells; ++i) {
        csvFile << simulationTime << ",";
        csvFile << cellAgentVector[i].getID() << ",";
        csvFile << cellAgentVector[i].getX() << ",";
        csvFile << cellAgentVector[i].getY() << ","; 
        csvFile << cellAgentVector[i].getPolarity() << ",";
        csvFile << cellAgentVector[i].getPolarityExtent() << ",";
        csvFile << cellAgentVector[i].getDirectionalInfluence() << ",";
        csvFile << cellAgentVector[i].getDirectionalIntensity() << ",";
        csvFile << cellAgentVector[i].getActinFlow() << ",";
        csvFile << cellAgentVector[i].getMovementDirection() << ",";
        csvFile << cellAgentVector[i].getDirectionalShift() << ",";
        csvFile << cellAgentVector[i].getSampledAngle() << ",";
        csvFile << cellAgentVector[i].getCellType() << "\n"; 
    }
}

void World::writeMatrixToCSV(std::ofstream& matrixFile) {
    for (int i = 0; i < countECMElement; i++) {
        for (int j = 0; j < countECMElement; j++) {
            matrixFile << ecmField.getHeading(i, j) << ",";
        }
    }
    for (int i = 0; i < countECMElement; i++) {
        for (int j = 0; j < countECMElement; j++) {
            matrixFile << ecmField.getMatrixPresent(i, j) << ",";
        }
    }
}

// Setters:

// Private member functions:

// Initialisation Functions:
void World::initialiseCellVector() {
    for (int i = 0; i < numberOfCells; ++i) {
        // Initialising cell:
        CellAgent newCell{initialiseCell(i)};

        // Putting cell into count matrix:
        ecmField.setCellPresence(
            newCell.getPosition(),
            newCell.getCellType(), newCell.getPolarity()
        );

        // Adding newly initialised cell to CellVector:
        cellAgentVector.push_back(newCell);
    }
}

CellAgent World::initialiseCell(int setCellID) {
    // Generating positions and randomness:
    double startX{positionDistribution(xPositionGenerator)};
    double startY{positionDistribution(yPositionGenerator)};
    double startHeading{headingDistribution(headingGenerator)};
    unsigned int setCellSeed{seedDistribution(cellSeedGenerator)};
    double inhibitionBoolean{contactInhibitionDistribution(contactInhibitionGenerator)};

    return CellAgent(
        startX, startY, startHeading,
        setCellSeed, setCellID, int(inhibitionBoolean < cellTypeProportions),
        cellParameters.poissonLambda, cellParameters.kappa, cellParameters.matrixKappa,
        cellParameters.homotypicInhibition, cellParameters.heterotypicInhibition,
        cellParameters.polarityPersistence, cellParameters.polarityTurningCoupling,
        cellParameters.flowScaling, cellParameters.flowPolarityCoupling,
        cellParameters.collisionRepolarisation, cellParameters.repolarisationRate,
        cellParameters.polarityNoiseSigma
    );
    // Higher alpha = higher concentration
    // Higher beta = lower lambda
}

// Simulation functions:
void World::runSimulationStep() {
    // Shuffling acting order of cells:
    std::shuffle(std::begin(cellAgentVector), std::end(cellAgentVector), shuffleGenerator);

    // Looping through cells and running their behaviour:
    for (int i = 0; i < numberOfCells; ++i) {
        runCellStep(cellAgentVector[i]);
    }

    // Updating ECM:
    ecmField.ageMatrix();

    simulationTime += 1;
}

void World::runCellStep(CellAgent& actingCell) {
    // Getting initial cell position:
    std::tuple<double, double> cellStart{actingCell.getPosition()};

    // // Setting cell percepts:

    // Calculating matrix percept:
    // Setting percepts of local matrix:
    auto [angleAverage, angleIntensity] = getAverageDeltaHeading(actingCell);
    actingCell.setDirectionalInfluence(angleAverage);
    actingCell.setDirectionalIntensity(angleIntensity);

    // actingCell.setLocalMatrixHeading(
    //     ecmField.getLocalMatrixHeading(cellStart)
    // );
    // actingCell.setLocalMatrixPresence(
    //     ecmField.getLocalMatrixPresence(cellStart)
    // );

    // if (actingCell.getID() == 1) {
    //     double angle, intensity;
    //     std::tie(angle, intensity) = ecmField.getAverageDeltaHeadingAroundIndex(
    //         startIndex[0], startIndex[1], actingCell.getPolarity()
    //     );
    //     std::cout << angle << " -- " << intensity << "\n";
    // }


    // Determine contact status of cell (i.e. cells in Moore neighbourhood of current cell):
    // Cell type 0:
    actingCell.setContactStatus(
        ecmField.getCellTypeContactState(cellStart, 0), 0
    );
    // Cell type 1:
    actingCell.setContactStatus(
        ecmField.getCellTypeContactState(cellStart, 1), 1
    );
    // Local cell heading state:
    actingCell.setLocalCellHeadingState(
        ecmField.getLocalCellHeadingState(cellStart)
    );

    // // Run cell intrinsic movement:
    ecmField.removeCellPresence(cellStart, actingCell.getCellType());
    actingCell.takeRandomStep();

    // Calculate and set effects of cell on world:
    std::tuple<double, double> cellFinish{actingCell.getPosition()};
    setMovementOnMatrix(
        cellStart, cellFinish,
        actingCell.getMovementDirection(), actingCell.getPolarityExtent()
    );

    // Rollover the cell if out of bounds:
    actingCell.setPosition(rollPosition(cellFinish));

    // Set new count:
    ecmField.setCellPresence(
        rollPosition(cellFinish),
        actingCell.getCellType(), actingCell.getPolarity()
    );
}

void World::setMovementOnMatrix(
    std::tuple<double, double> cellStart,
    std::tuple<double, double> cellFinish,
    double cellHeading, double cellPolarity
)
{
    // Finding gradient of line:
    double dx{std::get<0>(cellStart) - std::get<0>(cellFinish)};
    double dy{std::get<1>(cellStart) - std::get<1>(cellFinish)};
    double pathLength{std::sqrt(pow(dx, 2) + pow(dy, 2))};

    // Getting increments along path:
    int queryNumber{int(floor(pathLength / (lengthECMElement / 2)))};
    double roundedPathLength{(lengthECMElement / 2) * queryNumber};
    double incrementScaling{(roundedPathLength/pathLength)/queryNumber};
    double xIncrement{dx*incrementScaling};
    double yIncrement{dy*incrementScaling};

    // Getting blocks to set to heading:
    std::vector<std::array<int, 2>> blocksToSet;
    blocksToSet.push_back(getECMIndexFromLocation(cellStart));
    std::tuple<double, double> currentPosition{cellStart};
    for (int i = 0; i < queryNumber; ++i) {
        double newX{std::get<0>(currentPosition) + xIncrement};
        double newY{std::get<1>(currentPosition) + yIncrement};
        currentPosition = std::tuple<double, double>{newX, newY};
        currentPosition = rollPosition(currentPosition);
        blocksToSet.push_back(getECMIndexFromLocation(currentPosition));
    }

    // Setting blocks to heading:
    for (auto& block : blocksToSet) {
        ecmField.setSubMatrix(block[0], block[1],
        cellHeading, cellPolarity,
        cellDepositionKernel);
    }

}

std::array<int, 2> World::getECMIndexFromLocation(std::tuple<double, double> position) {
    int xIndex{int(std::floor(std::get<0>(position) / lengthECMElement))};
    int yIndex{int(std::floor(std::get<1>(position) / lengthECMElement))};
    // Note that the y index goes first here because of how we index matrices:
    return std::array<int, 2>{{yIndex, xIndex}};
}

int World::sign(double value) {
    return (double(0) < value) - (value < double(0));
}

std::tuple<double, double> World::rollPosition(std::tuple<double, double> position) {
    double xPosition = std::get<0>(position);
    double yPosition = std::get<1>(position);

    // Dealing with OOB in the negative numbers:
    while (xPosition < 0) {
        xPosition = worldSideLength + xPosition;
    }
    while (yPosition < 0) {
        yPosition = worldSideLength + yPosition;
    }

    // Dealing with OOB past sidelength boundaries:
    double newX{fmodf(xPosition, worldSideLength)};
    double newY{fmodf(yPosition, worldSideLength)};

    return std::tuple<double, double>{newX, newY};
}

std::tuple<int, int> World::rollIndex(int i, int j) {
    while (i < 0) {
        i += countECMElement;
    }
    while (i >= countECMElement) {
        i -= countECMElement;
    }
    while (j < 0) {
        j += countECMElement;
    }
    while (j >= countECMElement) {
        j -= countECMElement;
    }
    return {i, j};
};


std::tuple<double, double> World::getAverageDeltaHeading(CellAgent queryCell) {
    // Generate kernel:
    int numRows = cellSensationKernel.size1();
    int numCols = cellSensationKernel.size2();
    int center{(numRows - 1) / 2};
    assert(numRows == numCols);

    // Finding centre ECM element:
    double polarityDirection{queryCell.getPolarity()};
    auto [iECM, jECM] = getECMIndexFromLocation(queryCell.getPosition());

    // Getting all headings:
    double sineMean{0};
    double cosineMean{0};

    #pragma omp parallel for reduction(+:sineMean,cosineMean)
    for (int i = 0; i < numRows; ++i)
    {
        for (int j = 0; j < numCols; ++j)
        {
            // Getting rolled-over indices for ECM matrix:
            int rowOffset{i - center};
            int columnOffset{j - center};
            auto [iSafe, jSafe] = rollIndex(iECM + rowOffset, jECM + columnOffset);

            // Getting all weightings from kernel & matrix density:
            double ecmHeading{ecmField.getHeading(iSafe, jSafe)};
            double ecmDensity{ecmField.getMatrixPresent(iSafe, jSafe)};
            double kernelWeighting{cellSensationKernel(i, j)};
            double deltaHeading{
                calculateCellDeltaTowardsECM(ecmHeading, polarityDirection)
            };
            sineMean += std::sin(deltaHeading) * ecmDensity * kernelWeighting;
            cosineMean += std::cos(deltaHeading) * ecmDensity * kernelWeighting;
        }
    }

    assert(std::abs(sineMean) <= 1);
    assert(std::abs(cosineMean) <= 1);
    // assert(sineMean != 0 & cosineMean != 0);
    double angleAverage{std::atan2(sineMean, cosineMean)};
    double angleIntensity{
        std::sqrt(std::pow(sineMean, 2) + std::pow(cosineMean, 2))
    };

    return {angleAverage, angleIntensity};
}


double World::calculateCellDeltaTowardsECM(double ecmHeading, double cellHeading) {
    // Ensuring input values are in the correct range:
    assert((ecmHeading >= 0) & (ecmHeading < M_PI));
    assert((cellHeading >= -M_PI) & (cellHeading < M_PI));

    // Calculating change in theta (ECM is direction agnostic so we have to reverse it):
    double deltaHeading{ecmHeading - cellHeading};
    // while (deltaHeading <= -M_PI) {deltaHeading += M_PI;}
    while (deltaHeading > M_PI) {deltaHeading -= M_PI;}

    double flippedHeading;
    if (deltaHeading < 0) {
        flippedHeading = M_PI + deltaHeading;
    } else {
        flippedHeading = -(M_PI - deltaHeading);
    };

    // Selecting smallest change in theta and ensuring correct range:
    if (std::abs(deltaHeading) < std::abs(flippedHeading)) {
        assert((std::abs(deltaHeading) <= M_PI/2));
        return deltaHeading;
    } else {
        assert((std::abs(flippedHeading) <= M_PI/2));
        return flippedHeading;
    };
}

boostMatrix::matrix<double> World::generateGaussianKernel(double sigma) {
    // Determine size of kernel:
    // -- Getting size of sigma in matrix units:
    double sigmaMatrix{sigma / lengthECMElement};
    int kernelSize{(int(3 * sigmaMatrix)*2) + 1}; // Capturing variance up to 3 sigma.
    if (kernelSize < 3) {
        std::cout << "Warning - Deposition or sensation kernel radius too small for current matrix mesh size." << std::endl;
        kernelSize = 3;
    }

    // Creating matrix:
    boostMatrix::matrix<double> kernel(kernelSize, kernelSize, 0);
    int centerValue{(kernelSize-1) / 2};

    // Setting kernel values:
    double normalisation{0};
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            int y{i - centerValue};
            int x{j - centerValue};
            double exponent{-0.5*(std::pow(x / sigmaMatrix, 2.0) + std::pow(y / sigmaMatrix, 2.0))};
            kernel(i, j) = std::exp(exponent);
            normalisation += kernel(i, j);
        }
    }

    kernel /= normalisation;

    return kernel;
}