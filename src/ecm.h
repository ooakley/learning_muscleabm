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
    std::tuple<double, double> getAverageDeltaHeadingAroundIndex(int i, int j, double cellHeading);
    boostMatrix::matrix<bool> getCellContactState(int i, int j);

    // Setters:
    void setSubMatrix(int i, int j, double heading);
    void addToCellCount(int i, int j);
    void subtractFromCellCount(int i, int j);

private:
    // Simulation properties of matrix:
    double matrixPersistence;

    // Matrix size and matrix definitions:
    int elementCount;
    boostMatrix::matrix<double> ecmHeadingMatrix;
    boostMatrix::matrix<bool> ecmPresentMatrix;
    boostMatrix::matrix<int> cellCountMatrix;

    // Simulation functions:
    void addToMatrix(int i, int j, double heading);

    // Utility functions:
    int rollOverIndex(int index);
    double calculateCellDeltaTowardsECM(double ecmHeading, double cellHeading);
    double calculateECMDeltaTowardsCell(double ecmHeading, double cellHeading);
};