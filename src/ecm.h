#pragma once
#include <boost/numeric/ublas/matrix.hpp>
namespace boostMatrix = boost::numeric::ublas;

class ECMField{
public:
    // Constructor:
    ECMField(int setElements);

    // Getters:
    boostMatrix::matrix<double> getECMHeadingMatrix();
    double getHeading(int i, int j);
    bool getMatrixPresent(int i, int j);
    std::tuple<double, double> getAverageHeadingAroundIndex(int i, int j);

    // Setters:
    void setMatrix(int i, int j, double heading);

private:
    // Matrix size and matrix definitions:
    int elementCount;
    boostMatrix::matrix<double> ecmHeadingMatrix;
    boostMatrix::matrix<bool> ecmPresentMatrix;

    // Utility function for scanning ECM:
    int rollOverIndex(int index);
};