#include "ecm.h"
#include <cmath>
#include <boost/numeric/ublas/matrix.hpp>

#include <iostream>
namespace boostMatrix = boost::numeric::ublas;
using std::atan2;

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
    , ecmPresentMatrix{boostMatrix::zero_matrix<double>(matrixElementCount, matrixElementCount)}
    , cellType0CountMatrix{boostMatrix::zero_matrix<int>(collisionElementCount, collisionElementCount)}
    , cellType1CountMatrix{boostMatrix::zero_matrix<int>(collisionElementCount, collisionElementCount)}
    , cellHeadingMatrix{boostMatrix::zero_matrix<double>(collisionElementCount, collisionElementCount)}
    , matrixTurnoverRate{setMatrixTurnoverRate}
    , matrixAdditionRate{setMatrixAdditionRate}
{
}

// Getters:
boostMatrix::matrix<double> ECMField::getECMHeadingMatrix() {
    return ecmHeadingMatrix;
}

double ECMField::getHeading(std::tuple<double, double> position) {
    auto [i, j] = getMatrixIndexFromLocation(position);
    return ecmHeadingMatrix(i, j);
}
double ECMField::getHeading(int i, int j) {
    return ecmHeadingMatrix(i, j);
}

double ECMField::getMatrixPresent(std::tuple<double, double> position) {
    auto [i, j] = getMatrixIndexFromLocation(position);
    return ecmPresentMatrix(i, j);
}
double ECMField::getMatrixPresent(int i, int j) {
    return ecmPresentMatrix(i, j);
}

std::tuple<double, double> ECMField::getAverageDeltaHeadingAroundPosition(
    std::tuple<double, double> position, double cellHeading
)
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
        sineMean += sin(heading);
        cosineMean += cos(heading);
    }

    sineMean /= 9;
    cosineMean /= 9;

    assert(std::abs(sineMean) <= 1);
    assert(std::abs(cosineMean) <= 1);
    double angleAverage{atan2(sineMean, cosineMean)};
    double angleIntensity{sqrt(pow(sineMean, 2) + pow(cosineMean, 2))};
    return {angleAverage, angleIntensity};
}

boostMatrix::matrix<bool> ECMField::getCellTypeContactState(
    std::tuple<double, double> position, int cellType
) {
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

boostMatrix::matrix<double> ECMField::getLocalCellHeadingState(std::tuple<double, double> position) {
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

boostMatrix::matrix<double> ECMField::getLocalMatrixHeading(std::tuple<double, double> position) {
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

boostMatrix::matrix<double> ECMField::getLocalMatrixPresence(std::tuple<double, double> position) {
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
            localMatrix(k, l) = ecmPresentMatrix(safeRow, safeColumn);

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
    boostMatrix::matrix<double> kernel
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
    if (ecmPresentMatrix(i, j) == 0) {
        double newECMHeading{cellHeading};
        if (newECMHeading < 0) {newECMHeading += M_PI;};
        ecmHeadingMatrix(i, j) = newECMHeading;
        ecmPresentMatrix(i, j) = matrixAdditionRate;
        return;
    };

    assert(ecmPresentMatrix(i, j) != 0);
    assert(ecmPresentMatrix(i, j) <= 1);

    // Calculating delta between cell heading and ecm heading:
    double currentECMHeading{ecmHeadingMatrix(i, j)};
    double currentECMDensity{ecmPresentMatrix(i, j)};
    double smallestDeltaInECM{calculateECMDeltaTowardsCell(currentECMHeading, cellHeading)};
    double propsedECMHeading{currentECMHeading + smallestDeltaInECM};

    // Persistence weighting:
    // double currentPersistence{currentECMDensity*matrixPersistence};

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

    // Ensuring ECM angles are not vectors:
    if (newECMHeading < 0) {newECMHeading += M_PI;};
    if (newECMHeading >= M_PI) {newECMHeading -= M_PI;};
    assert((newECMHeading >= 0) & (newECMHeading < M_PI));

    // Setting matrix values:
    if (newECMDensity > 1) {
        const double epsilon{std::numeric_limits<double>::epsilon()};
        newECMDensity = 1 - epsilon;
    };
    ecmPresentMatrix(i, j) = newECMDensity;
    ecmHeadingMatrix(i, j) = newECMHeading;
}

void ECMField::ageMatrix() {
    ecmPresentMatrix = ecmPresentMatrix - matrixTurnoverRate*(ecmPresentMatrix);
}

// Utility functions:
int ECMField::rollOverMatrixIndex(int index) {
    while (index < 0) {
        index = index + matrixElementCount;
    }
    return index % matrixElementCount;
}

int ECMField::rollOverCollisionIndex(int index) {
    while (index < 0) {
        index = index + collisionElementCount;
    }
    return index % collisionElementCount;
}

double ECMField::calculateCellDeltaTowardsECM(double ecmHeading, double cellHeading) {
    // Ensuring input values are in the correct range:
    assert((ecmHeading >= 0) & (ecmHeading < M_PI));
    assert((cellHeading >= -M_PI) & (cellHeading < M_PI));

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

double ECMField::calculateECMDeltaTowardsCell(double ecmHeading, double cellHeading) {
    // Ensuring input values are in the correct range:
    assert((ecmHeading >= 0) & (ecmHeading < M_PI));
    assert((cellHeading >= -M_PI) & (cellHeading < M_PI));

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

std::array<int, 2> ECMField::getMatrixIndexFromLocation(std::tuple<double, double> position) {
    auto [xPosition, yPosition] = position;

    int xIndex{int(std::floor(xPosition / ecmElementSize))};
    int yIndex{int(std::floor(yPosition / ecmElementSize))};

    return std::array<int, 2>{{yIndex, xIndex}};
}

std::array<int, 2> ECMField::getCollisionIndexFromLocation(std::tuple<double, double> position) {
    auto [xPosition, yPosition] = position;

    int xIndex{int(std::floor(xPosition / collisionElementSize))};
    int yIndex{int(std::floor(yPosition / collisionElementSize))};

    return std::array<int, 2>{{yIndex, xIndex}};
}