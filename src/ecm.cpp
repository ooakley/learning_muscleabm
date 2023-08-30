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
    , cellCountMatrix{boostMatrix::zero_matrix<int>(elementCount, elementCount)}
    , matrixPersistence{0.98}
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

boostMatrix::matrix<bool> ECMField::getCellContactState(int i, int j) {
    // Asserting counting is working as expected:
    assert(cellCountMatrix(i, j) >= 1);

    // Instantiating contact matrix:
    boostMatrix::matrix<bool> contactState{boostMatrix::zero_matrix<bool>(3, 3)};

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
            // Checking for presence of other cells:
            if (cellCountMatrix(safeRow, safeColumn) > 0) {
                contactState(k, l) = true;
            } else {
                contactState(k, l) = false;
            }

            // Incrementing column index for contact matrix:
            l++;
        }

        // Incrementing row index for contact matrix:
        k++;
    }
    return contactState;
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

void ECMField::addToCellCount(int i, int j) {
    cellCountMatrix(i, j) = cellCountMatrix(i, j) + 1;
}

void ECMField::subtractFromCellCount(int i, int j) {
    cellCountMatrix(i, j) = cellCountMatrix(i, j) - 1;
    assert(cellCountMatrix(i, j) >= 0);
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