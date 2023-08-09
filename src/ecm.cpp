#include "ecm.h"
#include <cmath>
#include <boost/numeric/ublas/matrix.hpp>
namespace boostMatrix = boost::numeric::ublas;

// Constructor:
ECMField::ECMField(int setElements)
    : elementCount{setElements}
    , ecmHeadingMatrix{boostMatrix::zero_matrix<double>(elementCount, elementCount)}
    , ecmPresentMatrix{boostMatrix::zero_matrix<bool>(elementCount, elementCount)}
    , matrixPersistence{0}
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
    double sineMean;
    double cosineMean;
    for (auto & heading : deltaHeadingVector) {
        sineMean += sin(heading);
        cosineMean += cos(heading);
    }
    sineMean = sineMean / 9;
    cosineMean = cosineMean / 9;

    double angleAverage{atan2(sineMean, cosineMean)};
    double angleIntensity{sqrt(pow(sineMean, 2) + pow(cosineMean, 2))};
    return {angleAverage, angleIntensity};
}


// Setters:
void ECMField::setSubMatrix(int i, int j, double heading) {
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
            addToMatrix(safeRow, safeColumn, heading);
        }
    }
}

// Private member functions:

// Simulation functions:
void ECMField::addToMatrix(int i, int j, double cellHeading) {
    // If no matrix present, set to cellHeading:
    if (ecmPresentMatrix(i, j) == false) {
        ecmPresentMatrix(i, j) = true;
        double newECMHeading{cellHeading};
        if (newECMHeading < 0) {newECMHeading += M_PI;};
        ecmHeadingMatrix(i, j) = newECMHeading;
        return;
    };

    // Calculating delta between cell heading and ecm heading:
    double ecmHeading{ecmHeadingMatrix(i, j)};
    double deltaInECM{calculateECMDeltaTowardsCell(ecmHeading, cellHeading)};
    double newECMHeading{ecmHeading + ((1 - matrixPersistence)*deltaInECM)};

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
    assert((ecmHeading >= 0) & (ecmHeading < M_PI));
    assert((cellHeading >= -M_PI) & (cellHeading < M_PI));
    assert((ecmHeading >= 0) & (ecmHeading < M_PI));
    double positiveMu{ecmHeading - cellHeading};
    double negativeMu{ecmHeading - M_PI - cellHeading};
    if (abs(positiveMu) < abs(negativeMu)) {
        assert((positiveMu >= -M_PI) & (positiveMu < M_PI));
        return positiveMu;
    };
    assert((negativeMu >= -M_PI) & (negativeMu < M_PI));
    return negativeMu;
}

double ECMField::calculateECMDeltaTowardsCell(double ecmHeading, double cellHeading) {
    assert((ecmHeading >= 0) & (ecmHeading < M_PI));
    assert((cellHeading >= -M_PI) & (cellHeading < M_PI));
    double positiveMu{ecmHeading - cellHeading};
    double negativeMu{ecmHeading - M_PI - cellHeading};
    if (abs(positiveMu) < abs(negativeMu)) {
        assert((positiveMu >= -M_PI) & (positiveMu < M_PI));
        return -positiveMu;
    };
    assert((negativeMu >= -M_PI) & (negativeMu < M_PI));
    return -negativeMu;
}