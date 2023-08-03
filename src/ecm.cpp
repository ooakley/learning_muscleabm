#include "ecm.h"
#include <boost/numeric/ublas/matrix.hpp>
namespace boostMatrix = boost::numeric::ublas;

// Constructor:
ECMField::ECMField(int setElements)
    : elementCount{setElements}
    , ecmHeadingMatrix{boostMatrix::zero_matrix<double>(elementCount, elementCount)}
    , ecmPresentMatrix{boostMatrix::zero_matrix<bool>(elementCount, elementCount)}
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

std::tuple<double, double> ECMField::getAverageHeadingAroundIndex(int i, int j) {
    std::array<int, 3> rowScan = {i-1, i, i+1};
    std::array<int, 3> columnScan = {j-1, j, j+1};

    // Getting all headings:
    std::vector<double> headingVector;
    for (auto & row : rowScan)
    {
        int safeRow{rollOverIndex(row)};
        for (auto & column : columnScan)
        {
            int safeColumn{rollOverIndex(column)};
            if (getMatrixPresent(safeRow, safeColumn)) {
                headingVector.push_back(getHeading(safeRow, safeColumn));
            }
        }
    }

    // If no ECM present, return angle with no influence:
    if (headingVector.empty()) {
        return std::tuple<double, double>{0, 0};
    }

    // Getting the angle average:
    double sineMean;
    double cosineMean;
    for (auto & heading : headingVector) {
        sineMean += sin(heading);
        cosineMean += cos(heading);
    }
    sineMean = sineMean / 9;
    cosineMean = cosineMean / 9;

    double angleAverage{atan2(sineMean, cosineMean)};
    double angleIntensity{sqrt(pow(sineMean, 2) + pow(cosineMean, 2))};
    return std::tuple<double, double>{angleAverage, angleIntensity};
}


// Setters:
void ECMField::setMatrix(int i, int j, double heading) {
    assert((heading >= -M_PI) & (heading < M_PI));
    ecmPresentMatrix(i, j) = true;
    if (heading < 0) {
        heading += M_PI;
    };
    ecmHeadingMatrix(i, j) = heading;
}

// Private member functions:

int ECMField::rollOverIndex(int index) {
    if (index < 0) {
        index = index + elementCount;
    }
    return index % elementCount;
}