#pragma once
#include <boost/numeric/ublas/matrix.hpp>
namespace boostMatrix = boost::numeric::ublas;

class ECMField{
public:
    // Constructor:
    ECMField(int xWorldBlocks, int yWorldBlocks, double xWorldSize, double yWorldSize);

    // Getters:
    boostMatrix::matrix<double> getECMHeadingMatrix();
    double getHeading(double x, double y);
    bool getMatrixPresent(double x, double y);

    // Setters:
    void setMatrix(double x, double y, double heading);


private:
    double xBlockSize;
    double yBlockSize;
    boostMatrix::matrix<bool> ecmPresentMatrix;
    boostMatrix::matrix<double> ecmHeadingMatrix;

    // Location functions:
    std::array<int, 2> getIndexFromLocation(double x, double y);

};