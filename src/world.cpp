#include "world.h"
#include "agents.h"
#include "ecm.h"
#include "collision.h"
#include "buffered_writer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <tuple>

// Constructor and intialisation:
World::World
(
    unsigned int setWorldSeed,
    double setWorldSideLength,
    int setECMElementCount,
    int setNumberOfCells,
    double setMatrixSampleRate,
    double setPatternSigma,
    int setPatternFibreCount,
    CellParameters setCellParameters
)
    : worldSideLength{setWorldSideLength}
    , simulationTime{0}
    , countECMElement{setECMElementCount}
    , lengthECMElement{worldSideLength/countECMElement}
    , matrixSampleRate{setMatrixSampleRate}
    , patternSigma{setPatternSigma}
    , patternFibreCount{setPatternFibreCount}
    , cellParameters{setCellParameters}
    , collisionCellList{CollisionCellList(4, 2048)}
    , numberOfCells{setNumberOfCells}
{
    // Distributions:
    positionDistribution = std::uniform_real_distribution<double>(0, worldSideLength);
    headingDistribution = std::uniform_real_distribution<double>(-M_PI, M_PI);
    cellSeedDistribution = std::uniform_int_distribution<unsigned int>(0, UINT32_MAX);

    // Seed each generator, and the ECM, with a successive draw from a generator seeded by the
    // world seed:
    std::mt19937 seedGenerator(setWorldSeed);
    std::uniform_int_distribution<unsigned int> seedDistribution(0, UINT32_MAX);
    cellInitialisationGenerator = std::mt19937(seedDistribution(seedGenerator));
    cellOrderGenerator = std::mt19937(seedDistribution(seedGenerator));
    attachmentCountGenerator = std::mt19937(seedDistribution(seedGenerator));

    // Initialise ECM:
    ecmField = ECMField(
        countECMElement, patternSigma, patternFibreCount,
        seedDistribution(seedGenerator)
    );

    // Initialising cells:
    initialiseCellVector();
}

// Getters:
void World::writePositionsToCSV(std::ofstream& csvFile) {
    OutputBuffer output{csvFile};
    for (int i = 0; i < numberOfCells; i++) {
        const CellAgent& cell{*cellAgentVector[i]};
        output.add(simulationTime); output.add(',');
        output.add(cell.getID()); output.add(',');
        output.add(cell.getX()); output.add(',');
        output.add(cell.getY()); output.add(',');
        output.add(cell.getStadiumX()); output.add(',');
        output.add(cell.getStadiumY()); output.add('\n');
    }
}

void World::writeVerbosePositionsToCSV(std::ofstream& csvFile) {
    OutputBuffer output{csvFile};
    for (int i = 0; i < numberOfCells; i++) {
        const CellAgent& cell{*cellAgentVector[i]};
        output.add(simulationTime); output.add(',');
        output.add(cell.getID()); output.add(',');
        output.add(cell.getX()); output.add(',');
        output.add(cell.getY()); output.add(',');
        output.add(cell.getShapeDirection()); output.add(',');
        output.add(cell.getPolarityDirection()); output.add(',');
        output.add(cell.getPolarityMagnitude()); output.add(',');
        output.add(cell.getDirectionalInfluence()); output.add(',');
        output.add(cell.getDirectionalIntensity()); output.add(',');
        output.add(cell.getActinFlowDirection()); output.add(',');
        output.add(cell.getActinFlowMagnitude()); output.add(',');
        output.add(cell.getCollisionNumber()); output.add(',');
        output.add(cell.getTotalCILEffectX()); output.add(',');
        output.add(cell.getTotalCILEffectY()); output.add(',');
        output.add(cell.getMovementDirection()); output.add(',');
        output.add(cell.getDirectionalShift()); output.add(',');
        output.add(cell.getStadiumX()); output.add(',');
        output.add(cell.getStadiumY()); output.add(',');
        output.add(cell.getSampledAngle()); output.add('\n');
    }
}

void World::writeMatrixToCSV(std::ofstream& matrixFile) {
    // One line per ECM site, listing the heading of every fibre at that site:
    OutputBuffer output{matrixFile};
    for (int i = 0; i < countECMElement; i++) {
        for (int j = 0; j < countECMElement; j++) {
            for (float heading : ecmField.getFibreDeque(i, j)) {
                output.add(heading); output.add(',');
            }
            output.add('\n');
        }
    }
}

void World::writeSummarisedMatrixToCSV(std::ofstream& matrixFile) {
    // One line for the whole matrix, listing average heading, concentration and fibre count per site:
    OutputBuffer output{matrixFile};
    for (int i = 0; i < countECMElement; i++) {
        for (int j = 0; j < countECMElement; j++) {
            const auto [heading, concentration, fibreCount] = ecmField.summariseFibreMatrix(i, j);
            output.add(heading); output.add(',');
            output.add(concentration); output.add(',');
            output.add(fibreCount); output.add(',');
        }
    }
    output.add('\n');
}

// Public simulation functions:
void World::runSimulationStep() {
    // Shuffling acting order of cells:
    std::shuffle(std::begin(cellAgentVector), std::end(cellAgentVector), cellOrderGenerator);

    // Looping through cells and running their behaviour:
    for (int i = 0; i < numberOfCells; ++i) {
        runCellStep(*cellAgentVector[i]);
    }

    simulationTime += 1;
}

// Private member functions:

// Initialisation Functions:
void World::initialiseCellVector() {
    for (int cellID = 0; cellID < numberOfCells; ++cellID) {
        // Initialising cell:
        std::unique_ptr<CellAgent> newCell{initialiseCell(cellID)};

        // Putting cell into collisions matrix:
        auto [x, y] = newCell->getPosition();
        collisionCellList.addToCollisionMatrix(x, y, newCell.get());

        // Adding newly initialised cell to CellVector:
        cellAgentVector.push_back(std::move(newCell));
    }
}

std::unique_ptr<CellAgent> World::initialiseCell(int setCellID) {
    // Generating positions and randomness:
    const double startX{positionDistribution(cellInitialisationGenerator)};
    const double startY{positionDistribution(cellInitialisationGenerator)};
    const double startHeading{headingDistribution(cellInitialisationGenerator)};
    const unsigned int setCellSeed{cellSeedDistribution(cellInitialisationGenerator)};

    return std::make_unique<CellAgent>(
        // Defined behaviour parameters:
        setCellSeed, setCellID,
        cellParameters.dt,

        // Cell movement parameters:
        cellParameters.cueDiffusionRate,
        cellParameters.cueKa,
        cellParameters.fluctuationAmplitude,
        cellParameters.fluctuationTimescale,
        cellParameters.actinAdvectionRate,
        cellParameters.collisionAdvectionRate,
        cellParameters.maximumSteadyStateActinFlow,

        // Collision parameters:
        cellParameters.cellBodyRadius,
        cellParameters.collisionFlowReductionRate,
        cellParameters.adhesionReductionRate,

        // Shape parameters:
        cellParameters.cellStiffness,
        cellParameters.surfaceStickiness,

        // Matrix parameters:
        cellParameters.matrixCoupling,

        // Randomised initial state parameters:
        startX, startY, startHeading
    );
}

void World::runCellStep(CellAgent& actingCell) {
    double cellDirection{actingCell.getActinFlowDirection()};

    // Sample attachment points:
    int matrixSampleCount;
    if (matrixSampleRate == 0) {
        matrixSampleCount = 0;
    } else {
        std::poisson_distribution<int> poissonDistribution(matrixSampleRate);
        matrixSampleCount = poissonDistribution(attachmentCountGenerator);
    }

    attachmentPoints.clear();
    for (int i = 0; i < matrixSampleCount; i++) {
        attachmentPoints.push_back(actingCell.sampleAttachmentPoint());
    }

    // Set percepts of local matrix:
    if (matrixSampleCount == 0) {
        actingCell.setDirectionalInfluence(0);
        actingCell.setDirectionalIntensity(0);
    } else {
        // Randomly sample multiple matrix sites:
        double effectiveSampleCount{0};
        double averagedDeltaHeadingX{0};
        double averagedDeltaHeadingY{0};
        double orderParameterX{};
        double orderParameterY{};
        for (int i = 0; i < matrixSampleCount; i++) {
            // Get point to sample:
            const auto& sampledPoint{attachmentPoints[i]};
            const auto [iECM, jECM] = getECMIndexFromLocation({sampledPoint[0], sampledPoint[1]});
            const auto [iSafe, jSafe] = rollIndex(iECM, jECM);
            const auto [ecmHeading, localDensity] = ecmField.sampleFibreMatrix(iSafe, jSafe);

            // Skip accumulation if there are no fibres to sample:
            if (localDensity == 0) {continue;}

            // Accumulate otherwise:
            double deltaHeading{calculateCellDeltaTowardsECM(ecmHeading, cellDirection)};
            averagedDeltaHeadingX += std::cos(deltaHeading);
            averagedDeltaHeadingY += std::sin(deltaHeading);
            effectiveSampleCount += 1;
            // Get basic order parameter calculation:
            orderParameterX += std::cos(2 * ecmHeading);
            orderParameterY += std::sin(2 * ecmHeading);
        }
        if (effectiveSampleCount == 0) {
            actingCell.setDirectionalInfluence(0);
            actingCell.setDirectionalIntensity(0);
        } else {
            // Retrieve direction:
            double deltaHeadingDirection{std::atan2(averagedDeltaHeadingY, averagedDeltaHeadingX)};
            assert(std::abs(deltaHeadingDirection) < (M_PI/2));
            actingCell.setDirectionalInfluence(deltaHeadingDirection);

            // Retrieve nematic order parameter:
            double opNorm{std::sqrt(std::pow(orderParameterX, 2) + std::pow(orderParameterY, 2))};
            double directionalIntensity{opNorm / effectiveSampleCount};
            directionalIntensity = std::clamp(directionalIntensity, 0.0, 1.0 - 1e-4);
            actingCell.setDirectionalIntensity(directionalIntensity);
        }
    }

    // Run cell intrinsic movement:
    auto [startX, startY] = actingCell.getPosition();
    collisionCellList.removeFromCollisionMatrix(startX, startY, &actingCell);
    collisionCellList.getLocalAgents(startX, startY, localAgentBuffer);
    actingCell.setLocalCellList(localAgentBuffer);
    actingCell.takeRandomStep();

    // Deposit fibres at attachment points:
    const std::tuple<double, double> cellFinish{actingCell.getPosition()};
    for (int i = 0; i < matrixSampleCount; i++) {
        const auto& sampledPoint{attachmentPoints[i]};
        const auto [iECM, jECM] = getECMIndexFromLocation({sampledPoint[0], sampledPoint[1]});
        const auto [iSafe, jSafe] = rollIndex(iECM, jECM);
        ecmField.addToFibreMatrix(iSafe, jSafe, actingCell.getActinFlowDirection());
    }

    // Rollover the cell if out of bounds:
    actingCell.setPosition(rollPosition(cellFinish));

    // Set new count:
    auto [finishX, finishY] = actingCell.getPosition();
    collisionCellList.addToCollisionMatrix(finishX, finishY, &actingCell);
}

// Calculating percepts for cells:
double World::calculateCellDeltaTowardsECM(double ecmHeading, double cellHeading) {
    // Ensuring input values are in the correct range:
    if (! ((ecmHeading >= 0) && (ecmHeading <= M_PI))) {
        std::cout << "ecmHeading: " << ecmHeading << std::endl;
        assert((ecmHeading >= 0) && (ecmHeading <= M_PI));
    }
    if (! ((cellHeading >= -M_PI) && (cellHeading <= M_PI))) {
        std::cout << "cellHeading: " << cellHeading << std::endl;
        assert((cellHeading >= -M_PI) && (cellHeading <= M_PI));
    }

    // Calculating change in theta (ECM is direction agnostic so we have to reverse it):
    double deltaHeading{ecmHeading - cellHeading};
    while (deltaHeading <= -M_PI) {deltaHeading += M_PI;}
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