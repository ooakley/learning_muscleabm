#include "ecm.h"
#include <cmath>
#include <boost/numeric/ublas/matrix.hpp>
#include <deque>
#include <iostream>
#include <random>

namespace boostMatrix = boost::numeric::ublas;
using FibreUnit = std::deque<float>;
using FibreRow = std::vector<FibreUnit>;
using FibreMatrix = std::vector<FibreRow>;

// Constructor:
ECMField::ECMField(
    int setMatrixElements, int setCollisionElements, double setFieldSize,
    double setMatrixTurnoverRate, double setMatrixAdditionRate
    )
    : matrixElementCount{setMatrixElements}
    , collisionElementCount{setCollisionElements}
    , fieldSize{setFieldSize}
    , ecmElementSize{setFieldSize / setMatrixElements}
    , collisionElementSize{setFieldSize / setCollisionElements}
    , ecmHeadingMatrix{boostMatrix::zero_matrix<double>(matrixElementCount, matrixElementCount)}
    , ecmDensityMatrix{boostMatrix::zero_matrix<double>(matrixElementCount, matrixElementCount)}
    , ecmAnisotropyMatrix{boostMatrix::zero_matrix<double>(matrixElementCount, matrixElementCount)}
    , ecmUpdateWeightMatrix{boostMatrix::zero_matrix<double>(matrixElementCount, matrixElementCount)}
    , cellType0CountMatrix{boostMatrix::zero_matrix<int>(collisionElementCount, collisionElementCount)}
    , cellType1CountMatrix{boostMatrix::zero_matrix<int>(collisionElementCount, collisionElementCount)}
    , cellHeadingMatrix{boostMatrix::zero_matrix<double>(collisionElementCount, collisionElementCount)}
    , matrixTurnoverRate{setMatrixTurnoverRate}
    , matrixAdditionRate{setMatrixAdditionRate}
{
    // Initialise the fibre matrix:
    for (int i = 0; i < matrixElementCount; ++i) {
        FibreRow rowConstruct{};
        for (int j = 0; j < matrixElementCount; ++j) {
            FibreUnit emptyUnit;
            rowConstruct.push_back(emptyUnit);
        }
        fibreMatrix.push_back(rowConstruct);
    }
}

// Getters:
boostMatrix::matrix<double> ECMField::getECMHeadingMatrix() const {
    return ecmHeadingMatrix;
}

double ECMField::getHeading(std::tuple<double, double> position) const {
    auto [i, j] = getMatrixIndexFromLocation(position);
    return ecmHeadingMatrix(i, j);
}
double ECMField::getHeading(int i, int j) const {
    return ecmHeadingMatrix(i, j);
}

double ECMField::getMatrixDensity(std::tuple<double, double> position) const {
    auto [i, j] = getMatrixIndexFromLocation(position);
    double indexedDensity{ecmDensityMatrix(i, j)};
    if (indexedDensity > 1) {
        std::cout << "indexedDensity: " << indexedDensity << std::endl;
        assert(indexedDensity < 1);
    };
    return indexedDensity;
}
double ECMField::getMatrixDensity(int i, int j) const {
    double indexedDensity{ecmDensityMatrix(i, j)};
    if (indexedDensity > 1) {
        std::cout << "indexedDensity: " << indexedDensity << std::endl;
        assert(indexedDensity < 1);
    };
    return indexedDensity;
}

double ECMField::getMatrixAnisotropy(std::tuple<double, double> position) const {
    auto [i, j] = getMatrixIndexFromLocation(position);
    return ecmDensityMatrix(i, j);
}
double ECMField::getMatrixAnisotropy(int i, int j) const {
    return ecmDensityMatrix(i, j);
}

double ECMField::getMatrixWeighting(std::tuple<double, double> position) const {
    auto [i, j] = getMatrixIndexFromLocation(position);
    return ecmUpdateWeightMatrix(i, j);
}
double ECMField::getMatrixWeighting(int i, int j) const {
    return ecmUpdateWeightMatrix(i, j);
}

std::tuple<double, double> ECMField::getAverageDeltaHeadingAroundPosition(
    std::tuple<double, double> position, double cellHeading
) const
{
    auto [i, j] = getMatrixIndexFromLocation(position);
    std::array<int, 3> rowScan = {i-1, i, i+1};
    std::array<int, 3> columnScan = {j-1, j, j+1};

    // Getting all headings:
    std::vector<double> deltaHeadingVector;
    for (auto & row : rowScan)
    {
        int safeRow{rollOverMatrixIndex(row)};
        for (auto & column : columnScan)
        {
            int safeColumn{rollOverMatrixIndex(column)};
            double ecmHeading{getHeading(safeRow, safeColumn)};
            deltaHeadingVector.push_back(calculateCellDeltaTowardsECM(ecmHeading, cellHeading));
        }
    }

    // If no ECM present, return angle with no influence:
    if (deltaHeadingVector.empty()) {
        return {0, 0};
    }

    // Getting the angle average:
    double sineMean{0};
    double cosineMean{0};
    for (auto & heading : deltaHeadingVector) {
        sineMean += std::sin(heading);
        cosineMean += std::cos(heading);
    }

    sineMean /= 9;
    cosineMean /= 9;

    assert(std::abs(sineMean) <= 1);
    assert(std::abs(cosineMean) <= 1);
    double angleAverage{std::atan2(sineMean, cosineMean)};
    double angleIntensity{std::sqrt(std::pow(sineMean, 2) + std::pow(cosineMean, 2))};
    return {angleAverage, angleIntensity};
}

std::tuple<double, double> ECMField::sampleFibreMatrix(int i, int j) {
    // Return 0 density if no fibers present:
    assert(0 <= i & i < matrixElementCount);
    assert(0 <= j & j < matrixElementCount);
    int sampleSize{static_cast<int>(fibreMatrix[i][j].size())};
    if (sampleSize == 0) {
        return {0, 0};
    }

    // Return 1 density if fibers present:
    std::uniform_int_distribution<> indexDistribution(0, sampleSize-1);
    std::random_device randomDeviceInitialiser;
    std::mt19937 generator(randomDeviceInitialiser());
    int sampledIndex{indexDistribution(generator)};
    double sampledFibreHeading{fibreMatrix[i][j][sampledIndex]};
    return {sampledFibreHeading, 1};
};

boostMatrix::matrix<bool> ECMField::getCellTypeContactState(
    std::tuple<double, double> position, int cellType
) const
{
    auto [i, j] = getCollisionIndexFromLocation(position);
    // Asserting counting is working as expected:
    assert((cellType0CountMatrix(i, j) >= 1 ) || (cellType1CountMatrix(i, j) >= 1));

    // Instantiating contact matrix:
    boostMatrix::matrix<bool> cellTypeContactState{boostMatrix::zero_matrix<bool>(3, 3)};

    // Looping through local indices:
    std::array<int, 3> rowScan = {i-1, i, i+1};
    std::array<int, 3> columnScan = {j-1, j, j+1};
    int k{0}; 
    for (auto & row : rowScan)
    {
        int safeRow{rollOverCollisionIndex(row)};
        int l{0};
        for (auto & column : columnScan)
        {
            int safeColumn{rollOverCollisionIndex(column)};
            if (cellType == 0) {
                cellTypeContactState(k, l) = bool(cellType0CountMatrix(safeRow, safeColumn) > 0);
            } else {
                cellTypeContactState(k, l) =  bool(cellType1CountMatrix(safeRow, safeColumn) > 0);
            }

            // Incrementing column index for contact matrix:
            l++;
        }

        // Incrementing row index for contact matrix:
        k++;
    }
    return cellTypeContactState;
}

boostMatrix::matrix<double> ECMField::getLocalCellHeadingState(std::tuple<double, double> position) const {
    auto [i, j] = getCollisionIndexFromLocation(position);

    // Asserting counting is working as expected:
    assert((cellType0CountMatrix(i, j) >= 1 ) || (cellType1CountMatrix(i, j) >= 1));

    // Instantiating heading matrix:
    boostMatrix::matrix<double> localCellHeadingState{
        boostMatrix::zero_matrix<double>(3, 3)
    };

    // Looping through local indices:
    std::array<int, 3> rowScan = {i-1, i, i+1};
    std::array<int, 3> columnScan = {j-1, j, j+1};
    int k{0};
    for (auto & row : rowScan)
    {
        int safeRow{rollOverCollisionIndex(row)};
        int l{0};
        for (auto & column : columnScan)
        {
            int safeColumn{rollOverCollisionIndex(column)};
            localCellHeadingState(k, l) = cellHeadingMatrix(safeRow, safeColumn);

            // Incrementing column index for contact matrix:
            l++;
        }

        // Incrementing row index for contact matrix:
        k++;
    }
    return localCellHeadingState;
}

boostMatrix::matrix<double> ECMField::getLocalMatrixHeading(std::tuple<double, double> position) const {
    auto [i, j] = getMatrixIndexFromLocation(position);
    // Instantiating heading matrix:
    boostMatrix::matrix<double> localMatrix{boostMatrix::zero_matrix<double>(3, 3)};

    // Looping through local indices:
    std::array<int, 3> rowScan = {i-1, i, i+1};
    std::array<int, 3> columnScan = {j-1, j, j+1};
    int k{0}; 
    for (auto & row : rowScan)
    {
        int safeRow{rollOverMatrixIndex(row)};
        int l{0};
        for (auto & column : columnScan)
        {
            int safeColumn{rollOverMatrixIndex(column)};
            localMatrix(k, l) = ecmHeadingMatrix(safeRow, safeColumn);

            // Incrementing column index for local matrix:
            l++;
        }

        // Incrementing row index for local matrix:
        k++;
    }
    return localMatrix;
};

boostMatrix::matrix<double> ECMField::getLocalMatrixDensity(std::tuple<double, double> position) const {
    auto [i, j] = getMatrixIndexFromLocation(position);

    // Instantiating heading matrix:
    boostMatrix::matrix<double> localMatrix{boostMatrix::zero_matrix<double>(3, 3)};

    // Looping through local indices:
    std::array<int, 3> rowScan = {i-1, i, i+1};
    std::array<int, 3> columnScan = {j-1, j, j+1};
    int k{0}; 
    for (auto & row : rowScan)
    {
        int safeRow{rollOverMatrixIndex(row)};
        int l{0};
        for (auto & column : columnScan)
        {
            int safeColumn{rollOverMatrixIndex(column)};
            localMatrix(k, l) = ecmDensityMatrix(safeRow, safeColumn);

            // Incrementing column index for local matrix:
            l++;
        }

        // Incrementing row index for local matrix:
        k++;
    }
    return localMatrix;
};


// Setters:
void ECMField::setSubMatrix(
    int iECM, int jECM, double heading, double polarity,
    const boostMatrix::matrix<double>& kernel
) {
    // auto [i, j] = getMatrixIndexFromLocation(position);
    // Getting kernel dimensions:
    int numRows = kernel.size1();
    int numCols = kernel.size2();
    int center{(numRows - 1) / 2};
    assert(numRows == numCols);

    // Applying heading to matrix in neighbourhood:
    for (int i = 0; i < numRows; ++i)
    {
        for (int j = 0; j < numCols; ++j)
        {
            int rowOffset{i - center};
            int columnOffset{j - center};
            int iSafe{rollOverMatrixIndex(iECM + rowOffset)};
            int jSafe{rollOverMatrixIndex(jECM + columnOffset)};
            double kernelWeighting{kernel(i, j)};
            addToMatrix(iSafe, jSafe, heading, polarity, kernelWeighting);
        }
    }
}

void ECMField::setIndividualECMLattice(int i, int j, double heading, double polarity, double weighting) {
    addToMatrix(i, j, heading, polarity, weighting);
}

void ECMField::addToFibreMatrix(int i, int j, double heading) {
    assert(0 <= i & i < matrixElementCount);
    assert(0 <= j & j < matrixElementCount);
    // Add to fibre matrix:
    double nematicHeading{heading};
    while (nematicHeading < 0) {nematicHeading += M_PI;}
    fibreMatrix[i][j].push_back(nematicHeading);

    // Age fibre unit:
    if (fibreMatrix[i][j].size() >= 1000) {
        fibreMatrix[i][j].pop_front();
    }
}

void ECMField::setCellPresence(std::tuple<double, double> position, int cellType, double cellHeading) {
    auto [i, j] = getCollisionIndexFromLocation(position);
    // Adding to cell count matrices:
    if (cellType == 0) {
        cellType0CountMatrix(i, j) = cellType0CountMatrix(i, j) + 1;
    } else {
        cellType1CountMatrix(i, j) = cellType1CountMatrix(i, j) + 1;
    }

    // Placing into cell heading matrix:
    cellHeadingMatrix(i, j) = cellHeading;
}

void ECMField::removeCellPresence(std::tuple<double, double> position, int cellType) {
    auto [i, j] = getCollisionIndexFromLocation(position);
    // Removing from cell count matrices:
    if (cellType == 0) {
        cellType0CountMatrix(i, j) = cellType0CountMatrix(i, j) - 1;
        assert(cellType0CountMatrix(i, j) >= 0);
    } else {
        cellType1CountMatrix(i, j) = cellType1CountMatrix(i, j) - 1;
        assert(cellType1CountMatrix(i, j) >= 0);
    }

}

// Private member functions:

// Simulation functions:
void ECMField::addToMatrix(int i, int j, double cellHeading, double polarity, double kernelWeighting) {
    // If no matrix present, set to cellHeading:
    if (ecmDensityMatrix(i, j) == 0) {
        // Set heading:
        double newECMHeading{cellHeading};
        if (newECMHeading < 0) {newECMHeading += M_PI;};
        ecmHeadingMatrix(i, j) = newECMHeading;

        // Set density:
        double newDensity{polarity*kernelWeighting*matrixAdditionRate};
        if (newDensity > 1) {
            newDensity = 1 - 1e-5;
        }
        ecmDensityMatrix(i, j) = newDensity;

        // Assign to member variables:
        ecmAnisotropyMatrix(i, j) = 0;
        ecmUpdateWeightMatrix(i, j) += polarity*kernelWeighting*matrixAdditionRate;
        return;
    };

    assert(ecmDensityMatrix(i, j) != 0);
    assert(ecmDensityMatrix(i, j) <= 1);

    // Calculating delta between cell heading and ecm heading:
    double currentECMHeading{ecmHeadingMatrix(i, j)};
    double currentECMDensity{ecmDensityMatrix(i, j)};
    double smallestDeltaInECM{calculateECMDeltaTowardsCell(currentECMHeading, cellHeading)};
    double propsedECMHeading{currentECMHeading + smallestDeltaInECM};

    // Calculaing weighted average of current and proposed ECM:
    double combinedWeighting{polarity * kernelWeighting * matrixAdditionRate};
    double sineMean{0};
    sineMean += currentECMDensity * std::sin(currentECMHeading);
    sineMean += combinedWeighting * std::sin(propsedECMHeading);

    double cosineMean{0};
    cosineMean += currentECMDensity * std::cos(currentECMHeading);
    cosineMean += combinedWeighting * std::cos(propsedECMHeading);

    double newECMHeading{std::atan2(sineMean, cosineMean)};
    double newECMDensity{
        std::sqrt(std::pow(sineMean, 2) + std::pow(cosineMean, 2))
    };

    // Ensuring ECM angles are within [0, pi] range:
    if (newECMHeading < 0) {newECMHeading += M_PI;};
    if (newECMHeading >= M_PI) {newECMHeading -= M_PI;};
    assert((newECMHeading >= 0) & (newECMHeading < M_PI));

    // Updating anisotropy value:
    // Taken from https://math.stackexchange.com/questions/102978/incremental-computation-of-standard-deviation
    // double netWeightedUpdates{std::min(ecmUpdateWeightMatrix(i, j), 100.0)};
    double netWeightedUpdates{ecmUpdateWeightMatrix(i, j)};
    double currentAnisotropy{ecmAnisotropyMatrix(i, j)};
    double headingDifferenceCurrentMean{smallestDeltaInECM};
    double headingDifferenceNewMean{calculateECMDeltaTowardsCell(newECMHeading, cellHeading)};
    double newAnisotropy{currentAnisotropy + combinedWeighting*headingDifferenceCurrentMean*headingDifferenceNewMean};

    // Setting matrix values:
    if (newECMDensity >= 1) {
        const double epsilon{std::numeric_limits<double>::epsilon()};
        newECMDensity = 1 - 1e-5;
    };
    ecmDensityMatrix(i, j) = newECMDensity;
    assert(ecmDensityMatrix(i, j) < 1);
    ecmHeadingMatrix(i, j) = newECMHeading;
    ecmAnisotropyMatrix(i, j) = newAnisotropy;
    ecmUpdateWeightMatrix(i, j) += combinedWeighting;
}

void ECMField::ageMatrix() {
    // Putting basic Michaelis-Menten Kinetics into a form that the Boost matrix library can understand:
    boostMatrix::matrix<double> onesMatrix{boostMatrix::scalar_matrix<double>(matrixElementCount, matrixElementCount, 1)};
    boostMatrix::matrix<double> reactionMatrix{
        boostMatrix::element_div(ecmDensityMatrix, ecmDensityMatrix + (onesMatrix * 0.35))
    };

    // Simulating degradation of the ECM:
    ecmDensityMatrix = ecmDensityMatrix - matrixTurnoverRate*reactionMatrix;
}

void ECMField::ageIndividualECMLattice(int i, int j) {
    double localDensity{ecmDensityMatrix(i, j)};
    double reactionTerm{(matrixTurnoverRate*localDensity) / (0.5 + localDensity)};
    ecmDensityMatrix(i, j) = localDensity - reactionTerm;
    assert(ecmDensityMatrix(i, j) < 1);
}

// Utility functions:
int ECMField::rollOverMatrixIndex(int index) const {
    while (index < 0) {
        index = index + matrixElementCount;
    }
    return index % matrixElementCount;
}

int ECMField::rollOverCollisionIndex(int index) const {
    while (index < 0) {
        index = index + collisionElementCount;
    }
    return index % collisionElementCount;
}

double ECMField::calculateCellDeltaTowardsECM(double ecmHeading, double cellHeading) const {
    // Ensuring input values are in the correct range:
    assert((ecmHeading >= 0) & (ecmHeading <= M_PI));
    assert((cellHeading >= -M_PI) & (cellHeading <= M_PI));

    // Calculating change in theta (ECM is direction agnostic so we have to reverse it):
    double deltaHeading{ecmHeading - cellHeading};
    while (deltaHeading <= -M_PI) {deltaHeading += M_PI;}
    while (deltaHeading > M_PI) {deltaHeading -= M_PI;}

    double flippedHeading;
    if (deltaHeading < 0) {
        flippedHeading = M_PI + deltaHeading;
    } else {
        flippedHeading = -(M_PI - deltaHeading);
    }

    // Selecting smallest change in theta and ensuring correct range:
    if (std::abs(deltaHeading) < std::abs(flippedHeading)) {
        assert((std::abs(deltaHeading) <= M_PI/2));
        return deltaHeading;
    } else {
        assert((std::abs(flippedHeading) <= M_PI/2));
        return flippedHeading;
    }; 
}

double ECMField::calculateECMDeltaTowardsCell(double ecmHeading, double cellHeading) const {
    // Ensuring input values are in the correct range:
    assert((ecmHeading >= 0) & (ecmHeading <= M_PI));
    assert((cellHeading >= -M_PI) & (cellHeading <= M_PI));

    // Calculating change in theta (ECM is direction agnostic so we have to reverse it):
    double deltaHeading{cellHeading - ecmHeading};
    while (deltaHeading <= -M_PI) {deltaHeading += M_PI;}

    double flippedHeading;
    if (deltaHeading < 0) {
        flippedHeading = M_PI + deltaHeading;
    } else {
        flippedHeading = -(M_PI - deltaHeading);
    }

    // Selecting smallest change in theta and ensuring correct range:
    if (std::abs(deltaHeading) < std::abs(flippedHeading)) {
        assert((std::abs(deltaHeading) <= M_PI/2));
        return deltaHeading;
    } else {
        assert((std::abs(flippedHeading) <= M_PI/2));
        return flippedHeading;
    }; 
}

double ECMField::calculateMinimumAngularDistance(double headingA, double headingB) const {
    // Calculating change in theta:
    double deltaHeading{headingA - headingB};
    while (deltaHeading <= -M_PI) {deltaHeading += 2*M_PI;}
    while (deltaHeading > M_PI) {deltaHeading -= 2*M_PI;}

    double flippedHeading;
    if (deltaHeading < 0) {
        flippedHeading = M_PI + deltaHeading;
    } else {
        flippedHeading = -(M_PI - deltaHeading);
    }

    // Selecting smallest change in theta and ensuring correct range:
    if (std::abs(deltaHeading) < std::abs(flippedHeading)) {
        assert((std::abs(deltaHeading) <= M_PI/2));
        return deltaHeading;
    } else {
        assert((std::abs(flippedHeading) <= M_PI/2));
        return flippedHeading;
    };
}

std::array<int, 2> ECMField::getMatrixIndexFromLocation(std::tuple<double, double> position) const {
    auto [xPosition, yPosition] = position;

    int xIndex{int(std::floor(xPosition / ecmElementSize))};
    int yIndex{int(std::floor(yPosition / ecmElementSize))};

    return std::array<int, 2>{{yIndex, xIndex}};
}

std::array<int, 2> ECMField::getCollisionIndexFromLocation(std::tuple<double, double> position) const {
    auto [xPosition, yPosition] = position;

    int xIndex{int(std::floor(xPosition / collisionElementSize))};
    int yIndex{int(std::floor(yPosition / collisionElementSize))};

    return std::array<int, 2>{{yIndex, xIndex}};
}
