#pragma once
#include <boost/numeric/ublas/matrix.hpp>
namespace boostMatrix = boost::numeric::ublas;

class ECMField{
public:
    // Constructor:
    ECMField(
        int setMatrixElements, int setCollisionElements, double setFieldSize,
        double setMatrixTurnoverRate, double setMatrixAdditionRate
    );

    // Getters:
    // --- --- Matrix Getters:
    boostMatrix::matrix<double> getECMHeadingMatrix();
    double getHeading(std::tuple<double, double> position);
    double getHeading(int i, int j);
    double getMatrixPresent(std::tuple<double, double> position);
    double getMatrixPresent(int i, int j);
    std::tuple<double, double> getAverageDeltaHeadingAroundPosition(
        std::tuple<double, double> position, double cellHeading
    );

    boostMatrix::matrix<double> getLocalMatrixHeading(std::tuple<double, double> position);
    boostMatrix::matrix<double> getLocalMatrixPresence(std::tuple<double, double> position);

    // --- --- Collision Getters:
    boostMatrix::matrix<bool> getCellTypeContactState(std::tuple<double, double> position, int cellType);
    boostMatrix::matrix<double> getLocalCellHeadingState(std::tuple<double, double> position);

    // Setters:
    // --- --- Matrix Setters:
    void setSubMatrix(int i, int j, double heading, double polarity, boostMatrix::matrix<double> kernel);

    // --- --- Collision Setters:
    void setCellPresence(std::tuple<double, double> position, int cellType, double cellHeading);
    void removeCellPresence(std::tuple<double, double> position, int cellType);

    // Simulation code:
    void ageMatrix();

private:
    // Simulation properties of matrix:
    double matrixTurnoverRate;
    double matrixAdditionRate;

    // Matrix size and matrix definitions:
    int matrixElementCount;
    int collisionElementCount;
    double ecmElementSize; 
    double collisionElementSize;
    double fieldSize;
    boostMatrix::matrix<double> ecmHeadingMatrix;
    boostMatrix::matrix<double> ecmPresentMatrix;
    boostMatrix::matrix<int> cellType0CountMatrix;
    boostMatrix::matrix<int> cellType1CountMatrix;
    boostMatrix::matrix<double> cellHeadingMatrix;

    // Simulation functions:
    void addToMatrix(int i, int j, double heading, double polarity, double kernelWeighting);

    // Utility functions:
    int rollOverMatrixIndex(int index);
    int rollOverCollisionIndex(int index);
    double calculateCellDeltaTowardsECM(double ecmHeading, double cellHeading);
    double calculateECMDeltaTowardsCell(double ecmHeading, double cellHeading);

    std::array<int, 2> getMatrixIndexFromLocation(std::tuple<double, double> position);
    std::array<int, 2> getCollisionIndexFromLocation(std::tuple<double, double> position);
};