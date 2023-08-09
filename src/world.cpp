#include "world.h"
#include "agents.h"
#include "ecm.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <tuple>

// Constructor and intialisation:
World::World
(
    int setWorldSeed,
    double setWorldSideLength,
    int setECMElementCount,
    int setNumberOfCells
)
    : worldSeed{setWorldSeed}
    , worldSideLength{setWorldSideLength}
    , countECMElement{setECMElementCount}
    , lengthECMElement{worldSideLength/countECMElement}
    , ecmField{ECMField(countECMElement)}
    , numberOfCells{setNumberOfCells}
    , simulationTime{0}
{
    // Initialising randomness:
    seedGenerator = std::mt19937(worldSeed);
    seedDistribution = std::uniform_int_distribution<unsigned int>(0, UINT32_MAX);

    // Cell randomness:
    shuffleGenerator = std::mt19937(seedDistribution(seedGenerator));
    cellSeedGenerator = std::mt19937(seedDistribution(seedGenerator));

    // Cell positions:
    xPositionGenerator = std::mt19937(seedDistribution(seedGenerator));
    yPositionGenerator = std::mt19937(seedDistribution(seedGenerator));
    headingGenerator = std::mt19937(seedDistribution(seedGenerator));

    // Distributions:
    positionDistribution = std::uniform_real_distribution<double>(0, worldSideLength);
    headingDistribution = std::uniform_real_distribution<double>(-M_PI, M_PI);

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
        csvFile << cellAgentVector[i].getHeading(); 
        csvFile << "\n";
    }
}

void World::writeMatrixToCSV(std::ofstream& matrixFile) {
    for (int i = 0; i < countECMElement; i++) {
        for (int j = 0; j < countECMElement; j++) {
            matrixFile << ecmField.getHeading(i, j) << ",";
        }
    }
}

// Setters:

// Private member functions:

// Initialisation Functions:
void World::initialiseCellVector() {
    for (int i = 0; i < numberOfCells; ++i) {
        cellAgentVector.push_back(initialiseCell(i));
    }
}

CellAgent World::initialiseCell(int setCellID) {
    // Generating positions and randomness:
    double startX{positionDistribution(xPositionGenerator)};
    double startY{positionDistribution(yPositionGenerator)};
    double startHeading{headingDistribution(headingGenerator)};
    unsigned int setCellSeed{seedDistribution(cellSeedGenerator)};

    return CellAgent(startX, startY, startHeading, setCellSeed, setCellID, 15, 5, 10);
}

// Simulation functions:
void World::runSimulationStep() {
    // Shuffling acting order of cells:
    std::shuffle(std::begin(cellAgentVector), std::end(cellAgentVector), shuffleGenerator);

    // Looping through cells and running their behaviour:
    for (int i = 0; i < numberOfCells; ++i) {
        runCellStep(cellAgentVector[i]);
    }

    simulationTime += 1;
}

void World::runCellStep(CellAgent& actingCell) {
    // Getting initial cell position:
    std::tuple<double, double> cellStart{actingCell.getPosition()};
    std::array<int, 2> startIndex{getIndexFromLocation(cellStart)};

    // Getting headings of ECM surrounding cell:
    double angle, intensity;
    std::tie(angle, intensity) = ecmField.getAverageDeltaHeadingAroundIndex(
        startIndex[0], startIndex[1], actingCell.getHeading()
    );

    // Calculate and set effects of world on cell:
    actingCell.setDirectionalInfluence(angle, intensity);

    // Take step:
    actingCell.takeRandomStep();

    // Calculate and set effects of cell on world:
    std::tuple<double, double> cellFinish{actingCell.getPosition()};
    setMovementOnMatrix(cellStart, cellFinish, actingCell.getHeading());

    // Rollover the cell if out of bounds:
    actingCell.setPosition(rollPosition(cellFinish));
}

void World::setMovementOnMatrix(
    std::tuple<double, double> cellStart,
    std::tuple<double, double> cellFinish,
    double cellHeading
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
    blocksToSet.push_back(getIndexFromLocation(cellStart));
    std::tuple<double, double> currentPosition{cellStart};
    for (int i = 0; i < queryNumber; ++i) {
        double newX{std::get<0>(currentPosition) + xIncrement};
        double newY{std::get<1>(currentPosition) + yIncrement};
        currentPosition = std::tuple<double, double>{newX, newY};
        currentPosition = rollPosition(currentPosition);
        blocksToSet.push_back(getIndexFromLocation(currentPosition));
    }

    // Setting blocks to heading:
    for (const auto& block : blocksToSet) {
        ecmField.setSubMatrix(block[0], block[1], cellHeading);
    }

}

std::array<int, 2> World::getIndexFromLocation(std::tuple<double, double> position) {
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
    if (xPosition < 0) {
        xPosition = worldSideLength + xPosition;
    }
    if (yPosition < 0) {
        yPosition = worldSideLength + yPosition;
    }

    // Dealing with OOB past sidelength boundaries:
    double newX{fmodf(xPosition, worldSideLength)};
    double newY{fmodf(yPosition, worldSideLength)};

    return std::tuple<double, double>{newX, newY};
}