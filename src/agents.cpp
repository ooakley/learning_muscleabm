#include "agents.h"

#include <random>
#include <cmath>
#include <cassert>
#include <limits>
#include <iostream>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

// Constructor:
CellAgent::CellAgent(
    double startX, double startY, double startHeading,
    unsigned int setCellSeed, int setCellID,
    double setKappa, double setWbLambda, double setWbK
    )
    : x{startX}
    , y{startY}
    , heading{startHeading}
    , directionalInfluence{M_PI}
    , directionalIntensity{0}
    , cellSeed{setCellSeed}
    , cellID{setCellID}
    , kappa{setKappa}
{
    // Initialising randomness:
    seedGenerator = std::mt19937(cellSeed);
    seedDistribution = std::uniform_int_distribution<unsigned int>(0, UINT32_MAX);

    // Initialising influence selector:
    generatorInfluence = std::mt19937(seedDistribution(seedGenerator));

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
    weibullDistribution = std::weibull_distribution<double>(setWbK, setWbLambda);
}

// Public Definitions:

// Getters:
double CellAgent::getX() {
    return x; 
}

double CellAgent::getY() {
    return y; 
}

std::tuple<double, double> CellAgent::getPosition() {
    return std::tuple<double, double>{x, y};
}

double CellAgent::getID() {
    return cellID;
}

double CellAgent::getHeading() {
    return heading;
}

// Setters:
void CellAgent::setDirectionalInfluence
(
    double newDirectionalInfluence, double newDirectionalIntensity
)
{
    directionalInfluence = newDirectionalInfluence;
    directionalIntensity = newDirectionalIntensity;
}

void CellAgent::setPosition(std::tuple<double, double> newPosition) {
    x = std::get<0>(newPosition);
    y = std::get<1>(newPosition);
}

// Simulation code:
void CellAgent::takeRandomStep() {
    // Sampling 0-centred change in heading;
    double angleDelta{sampleVonMises()};
    assert((angleDelta >= -M_PI) & (angleDelta < M_PI));

    // Checking for whether direction is influenced:
    double randomThreshold{uniformDistribution(generatorInfluence)};
    assert((randomThreshold >= 0) & (randomThreshold < 1));

    double newMu{0};
    if (randomThreshold < directionalIntensity) {
        // Adding environmental effects:
        newMu = directionalInfluence;
    }

    // This will bias the change in heading to align w/ the environment:
    angleDelta = angleDelta + newMu;
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
        int B{int(bernoulliDistribution(generatorB))};

        // Calculate derived values:
        double z{cos(M_PI * U1)};
        double f{(1 + vm_r * z) / (vm_r + z)};
        double c{kappa * (vm_r - f)};

        // Determine whether we accept the result:
        bool simpleCondition{(c*(2 - c) - U2) > 0};
        if (simpleCondition) {
            angleSampled = true;
            int angleSign{(2*B) - 1};
            sampledVonMises = angleSign * acos(f);
            break;
        }

        bool logCondition{(log(c/U2) + 1 - c) > 0};
        if (logCondition) {
            angleSampled = true;
            int angleSign{(2*B) - 1};
            sampledVonMises = angleSign * acos(f);
        }
    }
    return sampledVonMises;
}

double CellAgent::angleMod(double angle) {
    if (angle < -M_PI) {angle += 2*M_PI;};
    if (angle >= M_PI) {angle -= 2*M_PI;};
    return angle;
}