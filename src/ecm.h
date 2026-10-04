#pragma once
#include <deque>
#include <random>
#include <tuple>
#include <vector>

using FibreUnit = std::deque<float>;
using FibreRow = std::vector<FibreUnit>;
using FibreMatrix = std::vector<FibreRow>;

class ECMField{
public:
    // Constructor:
    ECMField(
        int setMatrixElements, double setPatternSigma, int setPatternFibreCount,
        int setECMSeed
    );
    ECMField() = default;

    // Getters:
    std::tuple<double, double> sampleFibreMatrix(int i, int j);
    std::tuple<double, double, double> summariseFibreMatrix(int i, int j) const;
    std::deque<float> getFibreDeque(int i, int j) const;

    // Setters:
    void addToFibreMatrix(int i, int j, double heading);

private:
    // Necessary structures for reproducible random sampling:
    std::mt19937 seedGenerator;
    std::uniform_int_distribution<unsigned int> seedDistribution;
    std::mt19937 generatorFibreSampling;

    // RNG for generating pattern:
    std::mt19937 generatorPatterning;
    std::normal_distribution<double> patternDistribution;

    // Initial pattern of matrix:
    double patternSigma;
    int patternFibreCount;

    // Base matrix properties:
    int matrixElementCount;

    // Matrix data:
    FibreMatrix fibreMatrix;
};
