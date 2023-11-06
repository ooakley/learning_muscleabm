#include "ecm.h"
#include <cmath>
#include <boost/numeric/ublas/matrix.hpp>

#include <iostream>
namespace boostMatrix = boost::numeric::ublas;

// Constructor:
ECMField::ECMField(int setElements)
    : elementCount{setElements}
    , ecmHeadingMatrix{boostMatrix::zero_matrix<double>(elementCount, elementCount)}
    , ecmPresentMatrix{boostMatrix::zero_matrix<bool>(elementCount, elementCount)}
    , cellType0CountMatrix{boostMatrix::zero_matrix<int>(elementCount, elementCount)}
    , cellType1CountMatrix{boostMatrix::zero_matrix<int>(elementCount, elementCount)}
    , cellHeadingMatrix{boostMatrix::zero_matrix<double>(elementCount, elementCount)}
    , matrixPersistence{0.999}
{
}

// Getters:
boostMatrix::matrix<double> ECMField::getECMHeadingMatrix() {
    return ecmHeadingMatrix;
}

double ECMField::getHeading(int i, int j) {
    return ecmHeadingMatrix(i, j);
}

bool ECMField::getMatrixPresent(int i, int j) {
    return ecmPresentMatrix(i, j);
}

std::tuple<double, double> ECMField::getAverageDeltaHeadingAroundIndex(
    int i, int j, double cellHeading
)
{
    std::array<int, 3> rowScan = {i-1, i, i+1};
    std::array<int, 3> columnScan = {j-1, j, j+1};

    // Getting all headings:
    std::vector<double> deltaHeadingVector;
    for (auto & row : rowScan)
    {
        int safeRow{rollOverIndex(row)};
        for (auto & column : columnScan)
        {
            int safeColumn{rollOverIndex(column)};
            if (getMatrixPresent(safeRow, safeColumn)) {
                double ecmHeading{getHeading(safeRow, safeColumn)};
                deltaHeadingVector.push_back(calculateCellDeltaTowardsECM(ecmHeading, cellHeading));
            }
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

    assert(abs(sineMean) <= 1);
    assert(abs(cosineMean) <= 1);
    double angleAverage{atan2(sineMean, cosineMean)};
    double angleIntensity{sqrt(pow(sineMean, 2) + pow(cosineMean, 2))};
    return {angleAverage, angleIntensity};
}

boostMatrix::matrix<bool> ECMField::getCellTypeContactState(int i, int j, int cellType) {
    // Asserting counting is working as expected:
    assert((cellType0CountMatrix(i, j) >= 1 ) || (cellType1CountMatrix(i, j) >= 1));

    // Instantiating contact matrix:
    boostMatrix::matrix<bool> cellTypeContactState{boostMatrix::zero_matrix<bool>(3, 3)};

    // Checking for cells in adjacent spaces:
    std::array<int, 3> rowScan = {i-1, i, i+1};
    std::array<int, 3> columnScan = {j-1, j, j+1};
    int k{0}; 
    for (auto & row : rowScan)
    {
        int safeRow{rollOverIndex(row)};
        int l{0};
        for (auto & column : columnScan)
        {
            int safeColumn{rollOverIndex(column)};
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

boostMatrix::matrix<double> ECMField::getLocalHeadingState(int i, int j) {
    // Asserting counting is working as expected:
    assert((cellType0CountMatrix(i, j) >= 1 ) || (cellType1CountMatrix(i, j) >= 1));

    // Instantiating heading matrix:
    boostMatrix::matrix<double> localHeadingState{boostMatrix::zero_matrix<double>(3, 3)};

    // Checking for cells in adjacent spaces:
    std::array<int, 3> rowScan = {i-1, i, i+1};
    std::array<int, 3> columnScan = {j-1, j, j+1};
    int k{0}; 
    for (auto & row : rowScan)
    {
        int safeRow{rollOverIndex(row)};
        int l{0};
        for (auto & column : columnScan)
        {
            int safeColumn{rollOverIndex(column)};
            localHeadingState(k, l) = cellHeadingMatrix(safeRow, safeColumn);

            // Incrementing column index for contact matrix:
            l++;
        }

        // Incrementing row index for contact matrix:
        k++;
    }
    return localHeadingState;
}

// Setters:
void ECMField::setSubMatrix(int i, int j, double heading) {
    std::array<int, 3> rowScan = {i-1, i, i+1};
    std::array<int, 3> columnScan = {j-1, j, j+1};

    // Applying heading to matrix in neighbourhood:
    for (auto & row : rowScan)
    {
        int safeRow{rollOverIndex(row)};
        for (auto & column : columnScan)
        {
            int safeColumn{rollOverIndex(column)};
            addToMatrix(safeRow, safeColumn, heading);
        }
    }
}

void ECMField::setCellPresence(int i, int j, int cellType, double cellHeading) {
    // Adding to cell count matrices:
    if (cellType == 0) {
        cellType0CountMatrix(i, j) = cellType0CountMatrix(i, j) + 1;
    } else {
        cellType1CountMatrix(i, j) = cellType1CountMatrix(i, j) + 1;
    }

    // Placing into cell heading matrix:
    cellHeadingMatrix(i, j) = cellHeading;
}

void ECMField::removeCellPresence(int i, int j, int cellType) {
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
void ECMField::addToMatrix(int i, int j, double cellHeading) {
    // If no matrix present, set to cellHeading:
    if (ecmPresentMatrix(i, j) == false) {
        double newECMHeading{cellHeading};
        if (newECMHeading < 0) {newECMHeading += M_PI;};
        ecmHeadingMatrix(i, j) = newECMHeading;
        ecmPresentMatrix(i, j) = true;
        return;
    };

    // Calculating delta between cell heading and ecm heading:
    double currentECMHeading{ecmHeadingMatrix(i, j)};
    double smallestDeltaInECM{calculateECMDeltaTowardsCell(currentECMHeading, cellHeading)};
    double propsedECMHeading{currentECMHeading + smallestDeltaInECM};

    // Calculaing weighted average of current and proposed ECM:
    double sineMean{0};
    sineMean += matrixPersistence * sin(currentECMHeading);
    sineMean += (1 - matrixPersistence) * sin(propsedECMHeading);

    double cosineMean{0};
    cosineMean += matrixPersistence * cos(currentECMHeading);
    cosineMean += (1 - matrixPersistence) * cos(propsedECMHeading);

    double newECMHeading{atan2(sineMean, cosineMean)};

    // Ensuring ECM angles are not vectors:
    if (newECMHeading < 0) {newECMHeading += M_PI;};
    if (newECMHeading >= M_PI) {newECMHeading -= M_PI;};
    assert((newECMHeading >= 0) & (newECMHeading < M_PI));

    // Setting matrix values:
    ecmPresentMatrix(i, j) = true;
    ecmHeadingMatrix(i, j) = newECMHeading;
}

// Utility functions:
int ECMField::rollOverIndex(int index) {
    while (index < 0) {
        index = index + elementCount;
    }
    return index % elementCount;
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
    if (abs(deltaHeading) < abs(flippedHeading)) {
        assert((abs(deltaHeading) <= M_PI/2));
        return deltaHeading;
    } else {
        assert((abs(flippedHeading) <= M_PI/2));
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
    while (deltaHeading > M_PI) {deltaHeading -= M_PI;}

    double flippedHeading;
    if (deltaHeading < 0) {
        flippedHeading = M_PI + deltaHeading;
    } else {
        flippedHeading = -(M_PI - deltaHeading);
    }

    // Selecting smallest change in theta and ensuring correct range:
    if (abs(deltaHeading) < abs(flippedHeading)) {
        assert((abs(deltaHeading) <= M_PI/2));
        return deltaHeading;
    } else {
        assert((abs(flippedHeading) <= M_PI/2));
        return flippedHeading;
    }; 
}