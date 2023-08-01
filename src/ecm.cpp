#include "ecm.h"
#include <boost/numeric/ublas/matrix.hpp>
namespace boostMatrix = boost::numeric::ublas;

// Constructor:
ECMField::ECMField(int xWorldBlocks, int yWorldBlocks, double xWorldSize, double yWorldSize)
    : ecmPresentMatrix {boostMatrix::zero_matrix<bool>(xWorldBlocks, yWorldBlocks)}
    , ecmHeadingMatrix {boostMatrix::zero_matrix<double>(xWorldBlocks, yWorldBlocks)}
{
    xBlockSize = xWorldSize / xWorldBlocks;
    yBlockSize = yWorldSize / yWorldBlocks;
}   

// Getters:
boostMatrix::matrix<double> ECMField::getECMHeadingMatrix() {
    return ecmHeadingMatrix;
}

double ECMField::getHeading(double x, double y) {
    std::array<int, 2> setIndex{getIndexFromLocation(x, y)};
    return ecmHeadingMatrix(setIndex[0], setIndex[1]);
}

bool ECMField::getMatrixPresent(double x, double y) {
    std::array<int, 2> setIndex{getIndexFromLocation(x, y)};
    return ecmPresentMatrix(setIndex[0], setIndex[1]);
}

void ECMField::setMatrix(double x, double y, double heading) {
    std::array<int, 2> setIndex{getIndexFromLocation(x, y)};
    ecmPresentMatrix(setIndex[0], setIndex[1]) = true;
    ecmHeadingMatrix(setIndex[0], setIndex[1]) = heading;
}

// Location functions:
std::array<int, 2> ECMField::getIndexFromLocation(double x, double y) {
    int xIndex{int(std::floor(x / xBlockSize))};
    int yIndex{int(std::floor(y / yBlockSize))};
    return std::array<int, 2>{{yIndex, xIndex}};
}