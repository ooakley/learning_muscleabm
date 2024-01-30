#pragma once
#include <boost/numeric/ublas/matrix.hpp>
namespace boostMatrix = boost::numeric::ublas;

class ECMField{
public:
    // Constructor:
    ECMField(int setElements, float setMatrixPersistence, float setMatrixAdditionRate);

    // Getters:
    boostMatrix::matrix<double> getECMHeadingMatrix();
    double getHeading(int i, int j);
    double getMatrixPresent(int i, int j);
    std::tuple<double, double> getAverageDeltaHeadingAroundIndex(int i, int j, double cellHeading);
    boostMatrix::matrix<bool> getCellTypeContactState(int i, int j, int cellType);
    boostMatrix::matrix<double> getLocalCellHeadingState(int i, int j);
    boostMatrix::matrix<double> getLocalMatrixHeading(int i, int j);
    boostMatrix::matrix<bool> getLocalMatrixPresence(int i, int j);

    // Setters:
    void setSubMatrix(int i, int j, double heading);
    void setCellPresence(int i, int j, int cellType, double cellHeading);
    void removeCellPresence(int i, int j, int cellType);

    // Simulation code:
    void ageMatrix();

private:
    // Simulation properties of matrix:
    double matrixPersistence;
    double matrixAdditionRate;

    // Matrix size and matrix definitions:
    int elementCount;
    boostMatrix::matrix<double> ecmHeadingMatrix;
    boostMatrix::matrix<double> ecmPresentMatrix;
    boostMatrix::matrix<int> cellType0CountMatrix;
    boostMatrix::matrix<int> cellType1CountMatrix;
    boostMatrix::matrix<double> cellHeadingMatrix;

    // Simulation functions:
    void addToMatrix(int i, int j, double heading);

    // Utility functions:
    int rollOverIndex(int index);
    double calculateCellDeltaTowardsECM(double ecmHeading, double cellHeading);
    double calculateECMDeltaTowardsCell(double ecmHeading, double cellHeading);
};