#include "world.h"
#include "agents.h"
#include "ecm.h"

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
    , cellSensationKernel{generateGaussianKernel(setCellSensationSigma)}
    , cellDepositionKernel{generateGaussianKernel(setCellDepositionSigma)}
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
    const double startX{positionDistribution(xPositionGenerator)};
    const double startY{positionDistribution(yPositionGenerator)};
    const double startHeading{headingDistribution(headingGenerator)};
    const unsigned int setCellSeed{seedDistribution(cellSeedGenerator)};
    const double inhibitionBoolean{contactInhibitionDistribution(contactInhibitionGenerator)};

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
    // ecmField.ageMatrix();

    simulationTime += 1;
}

void World::runCellStep(CellAgent& actingCell) {
    // Getting initial cell position:
    const std::tuple<double, double> cellStart{actingCell.getPosition()};

    // // Setting cell percepts:

    // Calculating matrix percept:
    // Setting percepts of local matrix:
    const auto [angleAverage, angleIntensity, localDensity, attachmentIndices] = sampleAttachmentHeadings(actingCell);
    actingCell.setDirectionalInfluence(angleAverage);
    actingCell.setDirectionalIntensity(angleIntensity);
    actingCell.setLocalECMDensity(localDensity);

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
    const std::tuple<double, double> cellFinish{actingCell.getPosition()};
    double attachmentWeighting{1./3.};
    depositAtAttachments(attachmentIndices, actingCell.getMovementDirection(), actingCell.getPolarityExtent(), attachmentWeighting);
    // setMovementOnMatrix(
    //     cellStart, cellFinish,
    //     actingCell.getMovementDirection(), actingCell.getPolarityExtent()
    // );

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

void World::depositAtAttachments(
    std::vector<std::tuple<int, int>> attachmentIndices,
    double heading, double polarity, double weighting
)
{
    for (const std::tuple<int, int>& index: attachmentIndices) {
        ecmField.ageIndividualECMLattice(std::get<0>(index), std::get<1>(index));
        ecmField.setIndividualECMLattice(
            std::get<0>(index), std::get<1>(index),
            heading, polarity, weighting
        );
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


std::tuple<double, double, double> World::getAverageDeltaHeading(CellAgent queryCell) {
    // Generate kernel:
    const int numRows = cellSensationKernel.size1();
    const int numCols = cellSensationKernel.size2();
    const int center{(numRows - 1) / 2};
    assert(numRows == numCols);

    // Finding centre ECM element:
    const double polarityDirection{queryCell.getPolarity()};
    const auto [iECM, jECM] = getECMIndexFromLocation(queryCell.getPosition());

    // Getting all headings:
    double sineMean{0};
    double cosineMean{0};
    double localDensity{0};
    
    #pragma omp parallel for reduction(+:sineMean,cosineMean,localDensity)
    for (int i = 0; i < numRows; ++i)
    {
        for (int j = 0; j < numCols; ++j)
        {
            // Getting rolled-over indices for ECM matrix:
            const int rowOffset{i - center};
            const int columnOffset{j - center};
            const auto [iSafe, jSafe] = rollIndex(iECM + rowOffset, jECM + columnOffset);

            // Getting all weightings from kernel & matrix density:
            const double ecmHeading{ecmField.getHeading(iSafe, jSafe)};
            const double ecmDensity{ecmField.getMatrixDensity(iSafe, jSafe)};
            const double kernelWeighting{cellSensationKernel(i, j)};
            const double deltaHeading{
                calculateCellDeltaTowardsECM(ecmHeading, polarityDirection)
            };

            // Taking reduction:
            sineMean += std::sin(deltaHeading) * ecmDensity * kernelWeighting;
            cosineMean += std::cos(deltaHeading) * ecmDensity * kernelWeighting;
            localDensity += ecmDensity * kernelWeighting;
        }
    }

    assert(std::abs(sineMean) <= 1);
    assert(std::abs(cosineMean) <= 1);
    // assert(sineMean != 0 & cosineMean != 0);
    double angleAverage{std::atan2(sineMean, cosineMean)};
    double angleIntensity{
        std::sqrt(std::pow(sineMean, 2) + std::pow(cosineMean, 2))
    };

    return {angleAverage, angleIntensity, localDensity};
}


std::tuple<double, double, double, std::vector<std::tuple<int, int>>>
World::sampleAttachmentHeadings(CellAgent queryCell) {
    // Getting kernel definition:
    const int numRows = cellSensationKernel.size1();
    const int numCols = cellSensationKernel.size2();
    const int center{(numRows - 1) / 2};
    assert(numRows == numCols);

    // Getting maximum normalised PDF density:
    const boostMatrix::matrix<double> convolvedMatrix{
        boostMatrix::element_prod(cellSensationKernel, cellSensationKernel)
    };
    const double normalisationFactor{
        boostMatrix::sum(
            boostMatrix::prod(
                boostMatrix::scalar_vector<double>(convolvedMatrix.size1()), convolvedMatrix
                )
        )
    };
    const boostMatrix::matrix<double> discreteProbabilityDensity{convolvedMatrix/normalisationFactor};
    const double centralWeighting{discreteProbabilityDensity(center, center)};

    // Defining sampling distribution:
    std::uniform_int_distribution<> indexDistribution(0, cellSensationKernel.size1()-1);
    std::uniform_real_distribution<> acceptanceDistribution(0, centralWeighting);

    // Getting cell location and polarity heading:
    const auto [iECM, jECM] = getECMIndexFromLocation(queryCell.getPosition());
    const double polarityDirection{queryCell.getPolarity()};

    // Sampling from local sites on ECM lattice:
    double sineMean{0};
    double cosineMean{0};
    double localDensity{0};
    int attachmentCount{0};
    std::vector<std::tuple<int, int>> attachmentIndices{};
    while (attachmentCount < 3) {
        // Sampling random site within kernel:
        const int i{indexDistribution(kernelSamplingGenerator)};
        const int j{indexDistribution(kernelSamplingGenerator)};
    
        // Sampling from uniform distribution to see if attachment sticks:
        if (discreteProbabilityDensity(i, j) > acceptanceDistribution(kernelSamplingGenerator)) {
            // Getting ECM data at sampled attachment point:
            const int rowOffset{i - center};
            const int columnOffset{j - center};
            const auto [iSafe, jSafe] = rollIndex(iECM + rowOffset, jECM + columnOffset);
            const double ecmHeading{ecmField.getHeading(iSafe, jSafe)};
            const double ecmDensity{ecmField.getMatrixDensity(iSafe, jSafe)};

            // Calculating cellular change in angle:
            const double deltaHeading{
                calculateCellDeltaTowardsECM(ecmHeading, polarityDirection)
            };

            // Adding to cell-perceptual average:
            sineMean += std::sin(deltaHeading) * ecmDensity;
            cosineMean += std::cos(deltaHeading) * ecmDensity;
            localDensity += ecmDensity;
            attachmentCount += 1;
            attachmentIndices.emplace_back(iSafe, jSafe);
        }
    }

    // Taking averages:
    sineMean /= attachmentCount;
    cosineMean /= attachmentCount;
    localDensity /= attachmentCount;

    // Ensuring values are in the correct range:
    assert(std::abs(sineMean) <= 1);
    assert(std::abs(cosineMean) <= 1);

    // Deriving angle averages from accumulated values:
    double angleAverage{std::atan2(sineMean, cosineMean)};
    double angleIntensity{
        std::sqrt(std::pow(sineMean, 2) + std::pow(cosineMean, 2))
    };

    return {angleAverage, angleIntensity, localDensity, attachmentIndices};
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
    const double sigmaMatrix{sigma / lengthECMElement};
    int kernelSize{(int(3 * sigmaMatrix)*2) + 1}; // Capturing variance up to 3 sigma.
    if (kernelSize < 3) {
        std::cout << "Warning - Deposition or sensation kernel radius too small for current matrix mesh size." << std::endl;
        kernelSize = 3;
    }

    // Creating matrix:
    boostMatrix::matrix<double> kernel(kernelSize, kernelSize, 0);
    const int centerValue{(kernelSize-1) / 2};

    // Setting kernel values:
    double normalisation{0};
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            // Getting location and thus Gaussian value:
            const int y{i - centerValue};
            const int x{j - centerValue};
            const double exponent{-0.5*(std::pow(x / sigmaMatrix, 2.0) + std::pow(y / sigmaMatrix, 2.0))};

            // Assigning value to kernel:
            kernel(i, j) = std::exp(exponent);
            normalisation += kernel(i, j);
        }
    }

    // Normalising kernel so deposition is constant for different sigma values:
    kernel /= normalisation;

    return kernel;
}