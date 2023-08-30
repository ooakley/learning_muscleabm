#include "agents.h"

#include <random>
#include <cmath>
#include <cassert>
#include <limits>
#include <iostream>

#include <boost/numeric/ublas/matrix.hpp>
namespace boostMatrix = boost::numeric::ublas;
typedef std::vector<std::tuple<int, int>> vectorOfTuples;
vectorOfTuples HEADING_CONTACT_MAPPING{
    std::tuple<int, int>{1, 0},
    std::tuple<int, int>{0, 0},
    std::tuple<int, int>{0, 1},
    std::tuple<int, int>{0, 2},
    std::tuple<int, int>{1, 2},
    std::tuple<int, int>{2, 2},
    std::tuple<int, int>{2, 1},
    std::tuple<int, int>{2, 0},
    std::tuple<int, int>{1, 0}
};

// Constructor:
CellAgent::CellAgent(
    double startX, double startY, double startHeading,
    unsigned int setCellSeed, int setCellID,
    double setWbLambda, double setAlpha, double setBeta, double setInhibition
    )
    : x{startX}
    , y{startY}
    , heading{startHeading}
    , instantaneousSpeed{0}
    , directionalInfluence{M_PI}
    , directionalIntensity{0}
    , cellContactState{boostMatrix::zero_matrix<bool>(3, 3)}
    , cellSeed{setCellSeed}
    , cellID{setCellID}
    , lambdaWeibull{setWbLambda}
    , inhibitionRate{setInhibition}
    , alphaForVonMisesXC{setAlpha}
    , betaForWeibullXC{setBeta}
{
    // Initialising randomness:
    seedGenerator = std::mt19937(cellSeed);
    seedDistribution = std::uniform_int_distribution<unsigned int>(0, UINT32_MAX);

    // Initialising influence selector:
    generatorInfluence = std::mt19937(seedDistribution(seedGenerator));

    // Initialising von Mises distribution:
    generatorU1 = std::mt19937(seedDistribution(seedGenerator));
    generatorU2 = std::mt19937(seedDistribution(seedGenerator));
    generatorB = std::mt19937(seedDistribution(seedGenerator));

    uniformDistribution = std::uniform_real_distribution<double>(0, 1);
    bernoulliDistribution = std::bernoulli_distribution();

    // Initialising Weibull distribution:
    generatorWeibull = std::mt19937(seedDistribution(seedGenerator));

    // Generators and distribution for contact inhibition calculations:
    generatorAngleUniform = std::mt19937(seedDistribution(seedGenerator));
    generatorCornerCorrection = std::mt19937(seedDistribution(seedGenerator));
    generatorForInhibitionRate = std::mt19937(seedDistribution(seedGenerator));
    angleUniformDistribution = std::uniform_real_distribution<double>(-M_PI, M_PI);
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

double CellAgent::getInhibitionRate() {
    return inhibitionRate;
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

void CellAgent::setContactStatus(boostMatrix::matrix<bool> stateToSet) {
    cellContactState = stateToSet;
}

// Simulation code:
void CellAgent::takeRandomStep() {
    // Calculating new values for angle distribution based on current speed:


    // Sampling 0-centred change in heading;
    double angleDelta;
    if (instantaneousSpeed == 0) {
        angleDelta = angleUniformDistribution(generatorAngleUniform);
        assert((angleDelta >= -M_PI) & (angleDelta < M_PI));
    } else {
        double kappa{alphaForVonMisesXC*pow(instantaneousSpeed, 2) + 0.1};
        std::cout << "Kappa: " << kappa << "\n";
        angleDelta = sampleVonMises(kappa);
        assert((angleDelta >= -M_PI) & (angleDelta < M_PI));
    }


    // Checking for whether direction is influenced:
    double randomThreshold{uniformDistribution(generatorInfluence)};
    assert((randomThreshold >= 0) & (randomThreshold < 1));

    double newMu{0};
    if (randomThreshold < directionalIntensity) {
        // Adding environmental effects:
        newMu = directionalInfluence;
    }

    // This will bias the change in heading to align with the environment:
    angleDelta = angleDelta + newMu;
    heading = angleMod(heading + angleDelta);

    // Check for collisions & break if collisions occur:
    // This checks for which cell in the Moore neighbourhood we need to query:
    int stateIndex{int(floor((heading + M_PI + (M_PI/8)) / (M_PI / 4)))};
    int iContact{std::get<0>(HEADING_CONTACT_MAPPING[stateIndex])};
    int jContact{std::get<1>(HEADING_CONTACT_MAPPING[stateIndex])};

    // We then query that cell in the Moore neighbourhood to see if there's something there:
    if (cellContactState(iContact, jContact)) {
        // Check if the cell is a corner (need to correct for radial distance here):
        bool isCorner{(stateIndex == 1 || stateIndex == 3 || stateIndex == 5 || stateIndex == 7)};
        if (isCorner) {
            const double correctionFactor{0.5 + ((9*M_PI)/32) - ((9*sqrt(2))/16)};
            if (uniformDistribution(generatorCornerCorrection) < correctionFactor) {
                // Now can run collision logic:
                bool thereIsContactInhibition{
                    uniformDistribution(generatorForInhibitionRate) < inhibitionRate
                };
                if (thereIsContactInhibition) {
                    heading = angleUniformDistribution(generatorAngleUniform);
                    instantaneousSpeed = 0;
                    return;
                }
            }
        } else {
            bool thereIsContactInhibition{
                uniformDistribution(generatorForInhibitionRate) < inhibitionRate
            };
            if (thereIsContactInhibition) {
                heading = angleUniformDistribution(generatorAngleUniform);
                instantaneousSpeed = 0;
                return;
            }
        }
    }

    // Defining distribution and sampling step size:
    std::weibull_distribution<double> weibullDistribution;
    double kWeibull{betaForWeibullXC*exp(-abs(angleDelta)) + 1};
    std::cout << "Change in angle: " << angleDelta << "\n";
    std::cout << "Calculated weibull: " << kWeibull << "\n";

    weibullDistribution = std::weibull_distribution<double>(kWeibull, lambdaWeibull);
    instantaneousSpeed = weibullDistribution(generatorWeibull);

    // Calculating new position:
    x += instantaneousSpeed * cos(heading);
    y += instantaneousSpeed * sin(heading);
}

// Private functions for cell behaviour:
double CellAgent::sampleVonMises(double kappa) {
    // See Efficient Simulation of the von Mises Distribution - Best & Fisher 1979
    // Calculating distribution parameters:
    double vm_tau = 1 + sqrt(1 + (4 * pow(kappa, 2)));
    double vm_rho = (vm_tau - sqrt(2 * vm_tau)) / (2 * kappa);
    double vm_r = (1 + pow(vm_rho, 2)) / (2 * vm_rho);

    // Performing sampling:
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
    while (angle < -M_PI) {angle += 2*M_PI;};
    while (angle >= M_PI) {angle -= 2*M_PI;};
    return angle;
}