#include "ecm.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <numeric>
#include <random>

// Constructor:
ECMField::ECMField(
    int setMatrixElements, double setPatternSigma, int setPatternFibreCount,
    unsigned int setECMSeed
    )
    : patternSigma{setPatternSigma}
    , patternFibreCount{setPatternFibreCount}
    , matrixElementCount{setMatrixElements}
{
    // Initialise the fibre matrix:
    for (int i = 0; i < matrixElementCount; ++i) {
        FibreRow rowConstruct{};
        for (int j = 0; j < matrixElementCount; ++j) {
            FibreUnit emptyUnit;
            rowConstruct.push_back(emptyUnit);
        }
        fibreMatrix.push_back(rowConstruct);
    }

    // Seed each generator with a successive draw from a generator seeded by the ECM seed:
    std::mt19937 seedGenerator(setECMSeed);
    std::uniform_int_distribution<unsigned int> seedDistribution(0, UINT32_MAX);
    fibreSamplingGenerator = std::mt19937(seedDistribution(seedGenerator));
    patterningGenerator = std::mt19937(seedDistribution(seedGenerator));

    // Distribution of background fibre headings:
    patternDistribution = std::normal_distribution<double>(0, patternSigma);

    // Pattern the fibre matrix with background fibres, with headings drawn from N(0, patternSigma):
    for (int i = 0; i < matrixElementCount; ++i) {
        for (int j = 0; j < matrixElementCount; ++j) {
            for (int n = 0; n < patternFibreCount; ++n) {
                double sampledHeading{patternDistribution(patterningGenerator)};
                addToFibreMatrix(i, j, sampledHeading);
            }
        }
    }
}

// Getters:
std::tuple<double, double> ECMField::sampleFibreMatrix(int i, int j) {
    // Return 0 density if no fibers present:
    assert(0 <= i && i < matrixElementCount);
    assert(0 <= j && j < matrixElementCount);
    int sampleSize{static_cast<int>(fibreMatrix[i][j].size())};
    if (sampleSize == 0) {
        return {0, 0};
    }

    // Return 1 density if fibers present:
    std::uniform_int_distribution<> indexDistribution(0, sampleSize-1);
    int sampledIndex{indexDistribution(fibreSamplingGenerator)};
    double sampledFibreHeading{fibreMatrix[i][j][sampledIndex]};
    while (sampledFibreHeading < 0) {sampledFibreHeading += M_PI;}
    while (sampledFibreHeading >= M_PI) {sampledFibreHeading -= M_PI;}

    return {sampledFibreHeading, 1};
};

std::tuple<double, double, double> ECMField::summariseFibreMatrix(int i, int j) const {
    // Get fibre count and return early if nothing deposited:
    int fibreCount{static_cast<int>(fibreMatrix[i][j].size())};
    if (fibreCount == 0) {
        return {0, 0, 0};
    }
    if (fibreCount == 1) {
        return {fibreMatrix[i][j][0], 1, 1};
    }

    // Accumulate cartesian components of the doubled (nematic) headings:
    double cosineSum{0.0};
    double sineSum{0.0};
    for (double nematicAngle : fibreMatrix[i][j]) {
        const double doubleHeading{2.0*nematicAngle};
        cosineSum += std::cos(doubleHeading);
        sineSum += std::sin(doubleHeading);
    }
    double meanCosine{cosineSum / fibreCount};
    double meanSine{sineSum / fibreCount};

    double averagedHeading{std::atan2(meanSine, meanCosine) / 2};
    double concentration{std::sqrt(std::pow(meanCosine, 2) + std::pow(meanSine, 2))};

    return {averagedHeading, concentration, fibreCount};
};

const std::deque<float>& ECMField::getFibreDeque(int i, int j) const {
    return fibreMatrix[i][j];
};

// Setters:
void ECMField::addToFibreMatrix(int i, int j, double heading) {
    assert(0 <= i && i < matrixElementCount);
    assert(0 <= j && j < matrixElementCount);
    // Add to fibre matrix:
    double nematicHeading{heading};
    while (nematicHeading < 0) {nematicHeading += M_PI;}
    while (nematicHeading >= M_PI) {nematicHeading -= M_PI;}
    fibreMatrix[i][j].push_back(nematicHeading);

    // Age fibre unit:
    if (fibreMatrix[i][j].size() >= 500) {
        fibreMatrix[i][j].pop_front();
    }
}
