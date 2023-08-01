#include "agents.h"

#include <random>
#include <cmath>
#include <limits>
#include <iostream>

// Constructor:
CellAgent::CellAgent(
    double startX, double startY, double startHeading,
    int setCellSeed,
    double setKappa, double setWbLambda, double setWbK
    )
    : x{startX}
    , y{startY}
    , heading{startHeading}
    , directionalInfluence{M_PI / 2}
    , cellSeed{setCellSeed}
    , kappa{setKappa}
{
    // Initialising randomness:
    seedGenerator = std::mt19937(cellSeed);
    seedDistribution = std::uniform_int_distribution<unsigned int>(0, UINT32_MAX);

    // Initialising von Mises distribution:
    vm_tau = 1 + sqrt(1 + (4 * pow(kappa, 2)));
    vm_rho = (vm_tau - sqrt(2 * vm_tau)) / (2 * kappa);
    vm_r = (1 + pow(vm_rho, 2)) / (2 * vm_rho);

    generatorU1 = std::mt19937(seedDistribution(seedGenerator));
    generatorU2 = std::mt19937(seedDistribution(seedGenerator));
    generatorB = std::mt19937(seedDistribution(seedGenerator));

    uniformDistribution = std::uniform_real_distribution<double>(0, 1);
    bernoulliDistribution = std::bernoulli_distribution();

    // Initialising Weibull distribution:
    generatorWeibull = std::mt19937(seedDistribution(seedGenerator));
    weibullDistribution = std::weibull_distribution<double>(setWbLambda, setWbK);
}

// Public Definitions:

// Getters:
double CellAgent::getX() {
    return x; 
}

double CellAgent::getY() {
    return y; 
}

double CellAgent::getID() {
    return cellSeed;
}

double CellAgent::getHeading() {
    return heading;
}

// Setters:
void CellAgent::setDirectionalInfluence(double newDirectionalInfluence) {
    directionalInfluence = newDirectionalInfluence;
}

// Simulation code:
void CellAgent::takeRandomStep() {
    // Sampling 0-centred change in heading;
    double angleDelta{sampleVonMises()};

    // Adding environmental effects:
    double positiveMu{angleMod(directionalInfluence - heading)};
    double negativeMu{angleMod(directionalInfluence - M_PI - heading)};
    double newMu;
    if (abs(positiveMu) < abs(negativeMu)) {
        newMu = positiveMu;
    };
    if (abs(negativeMu) < abs(positiveMu)) {
        newMu = negativeMu;
    };

    // This will bias the change in heading to align w/ the environment:
    angleDelta = angleMod(angleDelta + newMu);
    heading = angleMod(heading + angleDelta);

    // Calculating new position:
    double stepSize{weibullDistribution(generatorWeibull)};
    x += stepSize * cos(heading);
    y += stepSize * sin(heading);
}

// Private functions for cell behaviour:
double CellAgent::sampleVonMises() {
    // See Efficient Simulation of the von Mises Distribution - Best & Fisher 1979
    bool angleSampled{false};
    double sampledVonMises;
    while (!angleSampled) {
        // Sample from our distributions:
        double U1{uniformDistribution(generatorU1)};
        double U2{uniformDistribution(generatorU2)};
        double B{bernoulliDistribution(generatorB)};

        // Calculate derived values:
        double z{cos(M_PI * U1)};
        double f{(1 + vm_r * z) / (vm_r + z)};
        double c{kappa * (vm_r - f)};

        // Determine whether we accept the result:
        bool simpleCondition{(c*(2 - c) - U2) > 0};
        if (simpleCondition) {
            angleSampled = true;
            int angleSign{(2*int(B)) - 1};
            sampledVonMises = angleSign * acos(f);
            break;
        }

        bool logCondition{(log(c/U2) + 1 - c) > 0};
        if (logCondition) {
            angleSampled = true;
            int angleSign{(2*int(B)) - 1};
            sampledVonMises = angleSign * acos(f);
        }
    }
    return sampledVonMises;
}

double CellAgent::angleMod(double angle) {
    return fmod(angle + M_PI, 2*M_PI) - M_PI;;
}