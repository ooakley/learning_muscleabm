#include "world.h"
#include "agents.h"
#include "ecm.h"
#include "collision.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <tuple>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
namespace boostMatrix = boost::numeric::ublas;

// Constructor and intialisation:
World::World
(
    int setWorldSeed,
    double setWorldSideLength,
    int setECMElementCount,
    int setNumberOfCells,
    bool setThereIsMatrixInteraction,
    double setMatrixTurnoverRate,
    double setMatrixAdditionRate,
    CellParameters setCellParameters
)
    : worldSeed{setWorldSeed}
    , worldSideLength{setWorldSideLength}
    , countECMElement{setECMElementCount}
    , lengthECMElement{worldSideLength/countECMElement}
    , ecmField{ECMField(countECMElement, 16, setWorldSideLength, setMatrixTurnoverRate, setMatrixAdditionRate)}
    , numberOfCells{setNumberOfCells}
    , thereIsMatrixInteraction{setThereIsMatrixInteraction}
    , simulationTime{0}
    , cellParameters{setCellParameters}
    , collisionCellList{CollisionCellList(16, 2048)}
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

    // Initialising generators for random local sampling of environment by cells:
    kernelSamplingGenerator = std::mt19937(seedDistribution(seedGenerator));

    // Distributions:
    positionDistribution = std::uniform_real_distribution<double>(0, worldSideLength);
    headingDistribution = std::uniform_real_distribution<double>(-M_PI, M_PI);
    contactInhibitionDistribution  = std::uniform_real_distribution<double>(0, 1);

    // Initialising cells:
    initialiseCellVector();
}

// Getters:
void World::writePositionsToCSV(std::ofstream& csvFile) {
    for (int i = 0; i < numberOfCells; ++i) {
        csvFile << simulationTime << ",";
        csvFile << cellAgentVector[i]->getID() << ",";
        csvFile << cellAgentVector[i]->getX() << ",";
        csvFile << cellAgentVector[i]->getY() << ","; 
        csvFile << cellAgentVector[i]->getShapeDirection() << ",";
        csvFile << cellAgentVector[i]->getPolarityDirection() << ",";
        csvFile << cellAgentVector[i]->getPolarityMagnitude() << ",";
        csvFile << cellAgentVector[i]->getDirectionalInfluence() << ",";
        csvFile << cellAgentVector[i]->getDirectionalIntensity() << ",";
        csvFile << cellAgentVector[i]->getActinFlowDirection() << ",";
        csvFile << cellAgentVector[i]->getActinFlowMagnitude() << ",";
        csvFile << cellAgentVector[i]->getMovementDirection() << ",";
        csvFile << cellAgentVector[i]->getDirectionalShift() << ",";
        csvFile << cellAgentVector[i]->getSampledAngle() << "\n";
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
            matrixFile << ecmField.getMatrixDensity(i, j) << ",";
        }
    }
    for (int i = 0; i < countECMElement; i++) {
        for (int j = 0; j < countECMElement; j++) {
            double anisotropy{ecmField.getMatrixAnisotropy(i, j)};
            double weighting{ecmField.getMatrixWeighting(i, j)};
            matrixFile << anisotropy/weighting << ",";
        }
    }
}

// Setters:

// Public simulation functions:

void World::runSimulationStep() {
    // Shuffling acting order of cells:
    std::shuffle(std::begin(cellAgentVector), std::end(cellAgentVector), shuffleGenerator);

    // Looping through cells and running their behaviour:
    for (int i = 0; i < numberOfCells; ++i) {
        runCellStep(cellAgentVector[i]);
    }

    // Updating ECM:
    // ecmField.ageMatrix();

    simulationTime += 1;
}

// Private member functions:

// Initialisation Functions:
void World::initialiseCellVector() {
    for (int cellID = 0; cellID < numberOfCells; ++cellID) {
        // Initialising cell:
        std::shared_ptr<CellAgent> newCell{initialiseCell(cellID)};

        // Putting cell into collisions matrix:
        auto [x, y] = newCell->getPosition();
        collisionCellList.addToCollisionMatrix(x, y, newCell);

        // Adding newly initialised cell to CellVector:
        cellAgentVector.push_back(newCell);
    }
}

std::shared_ptr<CellAgent> World::initialiseCell(int setCellID) {
    // Generating positions and randomness:
    const double startX{positionDistribution(xPositionGenerator)};
    const double startY{positionDistribution(yPositionGenerator)};
    const double startHeading{headingDistribution(headingGenerator)};
    const unsigned int setCellSeed{seedDistribution(cellSeedGenerator)};
    const double inhibitionBoolean{contactInhibitionDistribution(contactInhibitionGenerator)};

    return std::make_shared<CellAgent>(
        // Defined behaviour parameters:
        false, setCellSeed, setCellID,

        // Cell movement parameters:
        cellParameters.halfSatCellAngularConcentration,
        cellParameters.maxCellAngularConcentration,
        cellParameters.halfSatMeanActinFlow,
        cellParameters.maxMeanActinFlow,
        cellParameters.flowScaling,

        // Polarisation system parameters:
        cellParameters.polarityDiffusionRate,
        cellParameters.actinAdvectionRate,
        cellParameters.contactAdvectionRate,

        // Matrix sensation parameters:
        cellParameters.halfSatMatrixAngularConcentration,
        cellParameters.maxMatrixAngularConcentration,

        // Collision parameters:
        cellParameters.cellBodyRadius,
        cellParameters.aspectRatio,
        cellParameters.boundarySharpness,
        cellParameters.inhibitionStrength,

        // Randomised initial state parameters:
        startX, startY, startHeading,

        // Binary simulation parameters:
        cellParameters.actinMagnitudeIsFixed,
        cellParameters.actinDirectionIsFixed,
        cellParameters.thereIsExtensionRepulsion,
        cellParameters.collisionsAreDeterministic,
        cellParameters.matrixAlignmentIsDeterministic
    );
}

void World::runCellStep(std::shared_ptr<CellAgent> actingCell) {
    // Getting initial cell position:
    const std::tuple<double, double> cellStart{actingCell->getPosition()};

    // // Setting cell percepts:
    // Calculating matrix percept:
    std::vector<double> attachmentPoint{actingCell->sampleAttachmentPoint()};
    double cellPolarityDirection{actingCell->getPolarityDirection()};
    const auto [matrixAngle, localDensity] = getPerceptAtAttachment(attachmentPoint, cellPolarityDirection);
    assert(localDensity < 1);
    // Setting percepts of local matrix:
    if (thereIsMatrixInteraction) {
        actingCell->setDirectionalInfluence(matrixAngle);
        actingCell->setDirectionalIntensity(localDensity);
        actingCell->setLocalECMDensity(localDensity);
    }

    // Run cell intrinsic movement:
    auto [startX, startY] = actingCell->getPosition();
    collisionCellList.removeFromCollisionMatrix(startX, startY, actingCell);
    actingCell->setLocalCellList(collisionCellList.getLocalAgents(startX, startY));
    actingCell->takeRandomStep();

    // Calculate and set effects of cell on world:
    const std::tuple<double, double> cellFinish{actingCell->getPosition()};

    // Getting angle from cell to attachment:
    double dx{std::get<0>(cellFinish) - attachmentPoint[0]};
    double dy{std::get<1>(cellFinish) - attachmentPoint[1]};
    double angleFromCellToAttachment{std::atan2(dy, dx)};

    // double attachmentWeighting{1.0 / attachmentNumber};
    depositAtAttachment(
        attachmentPoint,
        actingCell->getActinFlowDirection(), 1,
        1
    );

    // Rollover the cell if out of bounds:
    actingCell->setPosition(rollPosition(cellFinish));

    // Set new count:
    auto [finishX, finishY] = actingCell->getPosition();
    collisionCellList.addToCollisionMatrix(finishX, finishY, actingCell);
}

void World::depositAtAttachment(
    std::vector<double> attachmentPoint,
    double heading, double polarity, double weighting
)
{
    const auto [iECM, jECM] = getECMIndexFromLocation({attachmentPoint[0], attachmentPoint[1]});
    const auto [iSafe, jSafe] = rollIndex(iECM, jECM);
    ecmField.ageIndividualECMLattice(iSafe, jSafe);
    ecmField.setIndividualECMLattice(
        iSafe,  jSafe,
        heading, polarity, weighting
    );
}

// Calculating percepts for cells:
std::tuple<double, double> World::getPerceptAtAttachment(std::vector<double> attachmentPoint, double cellPolarity) {
    // Getting relevant ECM index:
    const auto [iECM, jECM] = getECMIndexFromLocation({attachmentPoint[0], attachmentPoint[1]});
    const auto [iSafe, jSafe] = rollIndex(iECM, jECM);

    // Reading from ECM:
    const double ecmHeading{ecmField.getHeading(iSafe, jSafe)};
    const double ecmDensity{ecmField.getMatrixDensity(iSafe, jSafe)};
    assert(ecmDensity < 1);

    // Calculate change in heading of cell to ECM:
    const double deltaHeading{calculateCellDeltaTowardsECM(ecmHeading, cellPolarity)};

    return {deltaHeading, ecmDensity};
}

double World::calculateCellDeltaTowardsECM(double ecmHeading, double cellHeading) {
    // Ensuring input values are in the correct range:
    assert((ecmHeading >= 0) && (ecmHeading <= M_PI));

    if (! ((cellHeading >= -M_PI) && (cellHeading <= M_PI))) {
        std::cout << "cellHeading: " << cellHeading << std::endl;
        assert((cellHeading >= -M_PI) && (cellHeading <= M_PI));
    }

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

// World utility functions:
std::array<int, 2> World::getECMIndexFromLocation(std::tuple<double, double> position) {
    int xIndex{int(std::floor(std::get<0>(position) / lengthECMElement))};
    int yIndex{int(std::floor(std::get<1>(position) / lengthECMElement))};
    // Note that the y index goes first here because of how we index matrices:
    return std::array<int, 2>{{yIndex, xIndex}};
}


// Basic utility functions:
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