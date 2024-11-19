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
    boostMatrix::matrix<double> getECMHeadingMatrix() const;
    double getHeading(std::tuple<double, double> position) const;
    double getHeading(int i, int j) const;

    double getMatrixDensity(std::tuple<double, double> position) const;
    double getMatrixDensity(int i, int j) const;

    double getMatrixAnisotropy(std::tuple<double, double> position) const;
    double getMatrixAnisotropy(int i, int j) const;

    double getMatrixWeighting(std::tuple<double, double> position) const;
    double getMatrixWeighting(int i, int j) const;

    std::tuple<double, double> getAverageDeltaHeadingAroundPosition(
        std::tuple<double, double> position, double cellHeading
    ) const;

    boostMatrix::matrix<double> getLocalMatrixHeading(std::tuple<double, double> position) const;
    boostMatrix::matrix<double> getLocalMatrixDensity(std::tuple<double, double> position) const;

    // --- --- Collision Getters:
    boostMatrix::matrix<bool> getCellTypeContactState(std::tuple<double, double> position, int cellType) const;
    boostMatrix::matrix<double> getLocalCellHeadingState(std::tuple<double, double> position) const;

    // Setters:
    // --- --- Matrix Setters:
    void setSubMatrix(int i, int j, double heading, double polarity, const boostMatrix::matrix<double>& kernel);
    void setIndividualECMLattice(int i, int j, double heading, double polarity, double weighting);

    // --- --- Collision Setters:
    void setCellPresence(std::tuple<double, double> position, int cellType, double cellHeading);
    void removeCellPresence(std::tuple<double, double> position, int cellType);

    // Simulation code:
    void ageMatrix();
    void ageIndividualECMLattice(int i, int j);

private:
    // Simulation properties of matrix:
    double matrixTurnoverRate;
    double matrixAdditionRate;

    // Base matrix properties:
    int matrixElementCount;
    int collisionElementCount;
    double ecmElementSize; 
    double collisionElementSize;
    double fieldSize;

    // Matrix data:
    boostMatrix::matrix<double> ecmHeadingMatrix;
    boostMatrix::matrix<double> ecmDensityMatrix;
    boostMatrix::matrix<double> ecmAnisotropyMatrix;
    boostMatrix::matrix<double> ecmUpdateWeightMatrix;

    boostMatrix::matrix<int> cellType0CountMatrix;
    boostMatrix::matrix<int> cellType1CountMatrix;

    boostMatrix::matrix<double> cellHeadingMatrix;

    // Simulation functions:
    void addToMatrix(int i, int j, double heading, double polarity, double kernelWeighting);

    // Utility functions:
    int rollOverMatrixIndex(int index) const;
    int rollOverCollisionIndex(int index) const;
    double calculateCellDeltaTowardsECM(double ecmHeading, double cellHeading) const;
    double calculateECMDeltaTowardsCell(double ecmHeading, double cellHeading) const;
    double calculateMinimumAngularDistance(double headingA, double headingB) const;

    std::array<int, 2> getMatrixIndexFromLocation(std::tuple<double, double> position) const;
    std::array<int, 2> getCollisionIndexFromLocation(std::tuple<double, double> position) const;
};