#include "world.h"
#include "agents.h"
#include "ecm.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <tuple>

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/geometry/geometries/register/point.hpp>


namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

typedef bg::model::point<double, 2, bg::cs::cartesian> Point;

struct CellLocation
{
    double x, y;
    int cellIndex;
};

BOOST_GEOMETRY_REGISTER_POINT_2D(CellLocation, double, bg::cs::cartesian, x, y);

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
    CellParameters setCellParameters
)
    : worldSeed{setWorldSeed}
    , worldSideLength{setWorldSideLength}
    , countECMElement{setECMElementCount}
    , lengthECMElement{worldSideLength/countECMElement}
    , ecmField{ECMField(countECMElement, setMatrixTurnoverRate, setMatrixAdditionRate)}
    , numberOfCells{setNumberOfCells}
    , cellTypeProportions{setCellTypeProportions}
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
        std::array<int, 2> initialIndex{getIndexFromLocation(newCell.getPosition())};
        ecmField.setCellPresence(
            initialIndex[0], initialIndex[1],
            newCell.getCellType(), newCell.getPolarity()
        );

        // Adding newly initialised cell to CellVector:
        cellAgentVector.push_back(newCell);

        // Adding cell to unordered map:
        cellAgentMap[i] = &newCell;
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
        cellParameters.wbK, cellParameters.kappa, cellParameters.matrixKappa,
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

    // Constructing RTree for nearest neighbour calculations:
    bgi::rtree<CellLocation, bgi::rstar<8> > cellTree;
    for (int i = 0; i < numberOfCells; ++i)
    {
        auto [x, y] = cellAgentVector[i].getPosition();
        CellLocation position{x, y, i};
        cellTree.insert(position);

        // Adding "ghost" particles for our periodic boundary conditions:
        bool xReflect{false};
        double boundaryX{x};
        bool yReflect{false};
        double boundaryY{y};

        if (x < 100) {
            boundaryX += worldSideLength;
            xReflect = true;
        }
        else if (x > (worldSideLength - 100)) {
            boundaryX -= worldSideLength;
            xReflect = true;
        }
        if (y < 100) {
            boundaryY += worldSideLength;
            yReflect = true;
        }
        else if (y > (worldSideLength - 100)) {
            boundaryY -= worldSideLength;
            yReflect = true;
        }

        // If particle was adjusted, we need to add it to the RTree:
        if (xReflect) {
            CellLocation ghostPosition{boundaryX, y, i};
            cellTree.insert(ghostPosition);
        }
        if (yReflect) {
            CellLocation ghostPosition{x, boundaryY, i};
            cellTree.insert(ghostPosition);
        }
        if (xReflect && yReflect) {
            CellLocation ghostPosition{boundaryX, boundaryY, i};
            cellTree.insert(ghostPosition);
        }
    }

    // Setting nearest neighbour percepts:
    for (int i = 0; i < numberOfCells; ++i) {
        // Getting nearest neighbour position:
        std::vector<CellLocation> nearestNeighbours;
        auto [currentX, currentY] = cellAgentVector[i].getPosition();
        CellLocation currentPosition{currentX, currentY, i};
        cellTree.query(
            bgi::nearest(currentPosition, 2),
            std::back_inserter(nearestNeighbours)
        );

        // Getting nearest neighbour parameters:
        auto[neighbourX, neighbourY] = cellAgentVector[nearestNeighbours[0].cellIndex].getPosition();
        double neighbourPolarity{cellAgentVector[nearestNeighbours[1].cellIndex].getPolarity()};

        // Getting distance and relative heading:
        double dx{neighbourX - currentX};
        double dy{neighbourY - currentY};
        double neighbourDistance{std::sqrt(
            std::pow(dx, 2) +
            std::pow(dy, 2)
        )};
        double relativeHeading{std::atan2(dy, dx)};

        cellAgentVector[i].setNeighbourPercept(neighbourDistance, neighbourPolarity, relativeHeading);

        // std::cout << neighbourDistance << std::endl;
    }   

    // Looping through cells and running their behaviour:
    for (int i = 0; i < numberOfCells; ++i) {
        // Running cell behaviour:
        runCellStep(cellAgentVector[i]);
    }

    // Updating ECM:
    ecmField.turnoverMatrix();

    simulationTime += 1;
}

void World::runCellStep(CellAgent& actingCell) {
    // Getting initial cell position:
    std::tuple<double, double> cellStart{actingCell.getPosition()};
    // std::array<int, 2> startIndex{getIndexFromLocation(cellStart)};

    // Setting percepts of matrix:
    // actingCell.setLocalMatrixHeading(
    //     ecmField.getLocalMatrixHeading(startIndex[0], startIndex[1])
    // );

    // actingCell.setLocalMatrixPresence(
    //     ecmField.getLocalMatrixPresence(startIndex[0], startIndex[1])
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
    // actingCell.setContactStatus(
    //     ecmField.getCellTypeContactState(startIndex[0], startIndex[1], 0), 0
    // );
    // // Cell type 1:
    // actingCell.setContactStatus(
    //     ecmField.getCellTypeContactState(startIndex[0], startIndex[1], 1), 1
    // );
    // // Local cell heading state:
    // actingCell.setLocalCellHeadingState(
    //     ecmField.getLocalCellHeadingState(startIndex[0], startIndex[1])
    // );

    // // Run cell intrinsic movement:
    // ecmField.removeCellPresence(startIndex[0], startIndex[1], actingCell.getCellType());
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
    // std::array<int, 2> endIndex{getIndexFromLocation(rollPosition(cellFinish))};
    // ecmField.setCellPresence(
    //     endIndex[0], endIndex[1],
    //     actingCell.getCellType(), actingCell.getPolarity()
    // );
}

void World::setMovementOnMatrix(
    std::tuple<double, double> cellStart,
    std::tuple<double, double> cellFinish,
    double cellHeading, double cellPolarityExtent
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
    for (auto& block : blocksToSet) {
        ecmField.setSubMatrix(block[0], block[1], cellHeading, cellPolarityExtent);
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