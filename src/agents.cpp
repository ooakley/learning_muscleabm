#include "agents.h"

#include <algorithm>
#include <complex>
#include <random>
#include <cmath>
#include <cassert>
#include <limits>
#include <iostream>
#include <utility>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/math/distributions/normal.hpp>

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
    unsigned int setCellSeed, int setCellID, int setCellType,
    double setPoissonLambda, double setKappa, double setMatrixKappa,
    double setHomotypicInhibition, double setHeterotypicInhibition,
    double setPolarityPersistence, double setPolarityTurningCoupling,
    double setFlowScaling, double setFlowPolarityCoupling,
    double setCollisionRepolarisation, double setRepolarisationRate,
    double setPolarityNoiseSigma
    )
    : thereIsMatrixInteraction{true}
    , x{startX}
    , y{startY}
    , polarityX{1e-5 * cos(startHeading)}
    , polarityY{1e-5 * sin(startHeading)}
    , polarityPersistence{setPolarityPersistence}
    , polarityTurningCoupling{setPolarityTurningCoupling}
    , flowPolarityCoupling{setFlowPolarityCoupling}
    , flowScaling{setFlowScaling}
    , collisionRepolarisation{setCollisionRepolarisation}
    , repolarisationRate{setRepolarisationRate}
    , polarityNoiseSigma{setPolarityNoiseSigma}
    , movementDirection{0}
    , actinFlow{0}
    , directionalInfluence{0}
    , directionalIntensity{0}
    , localECMDensity{0}
    , directionalShift{0}
    , sampledAngle{0}
    , kappa{setKappa}
    , matrixKappa{setMatrixKappa}
    , localMatrixHeading{boostMatrix::zero_matrix<double>(3, 3)}
    , localMatrixPresence{boostMatrix::zero_matrix<double>(3, 3)}
    , cellType0ContactState{boostMatrix::zero_matrix<bool>(3, 3)}
    , cellType1ContactState{boostMatrix::zero_matrix<bool>(3, 3)}
    , localCellHeadingState{boostMatrix::zero_matrix<double>(3, 3)}
    , cellSeed{setCellSeed}
    , cellID{setCellID}
    , poissonLambda{setPoissonLambda}
    , homotypicInhibitionRate{setHomotypicInhibition}
    , heterotypicInhibitionRate{setHeterotypicInhibition}
    , cellType{setCellType}
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

    generatorProtrusion = std::mt19937(seedDistribution(seedGenerator));

    // Initialising Levy distribution:
    generatorLevy = std::mt19937(seedDistribution(seedGenerator));
    boost::math::normal_distribution<> normalDistribution(0.0, 1.0);

    // Generators and distribution for contact inhibition calculations:
    generatorAngleUniform = std::mt19937(seedDistribution(seedGenerator));
    generatorCornerCorrection = std::mt19937(seedDistribution(seedGenerator));
    generatorForInhibitionRate = std::mt19937(seedDistribution(seedGenerator));
    angleUniformDistribution = std::uniform_real_distribution<double>(-M_PI, M_PI);

    // Generator for finding random angle after loss of polarisation:
    generatorRandomRepolarisation = std::mt19937(seedDistribution(seedGenerator));

    // Generator for polarity noise:
    generatorPolarityNoiseX = std::mt19937(seedDistribution(seedGenerator));
    generatorPolarityNoiseY = std::mt19937(seedDistribution(seedGenerator)); 
    polarityNoiseDistribution = \
        std::normal_distribution<double>(0, polarityNoiseSigma);

}
// Public Definitions:

// Getters:
// Getters for values that shouldn't change:
double CellAgent::getID() const {return cellID;}
double CellAgent::getHomotypicInhibitionRate() const {return homotypicInhibitionRate;}
double CellAgent::getCellType() const {return cellType;}

// Persistent variable getters: (these variables represent aspects of the cell's "memory")
double CellAgent::getX() const {return x;}
double CellAgent::getY() const {return y;}
std::tuple<double, double> CellAgent::getPosition() const {return std::tuple<double, double>{x, y};}
double CellAgent::getPolarity() const {return findPolarityDirection();}
double CellAgent::getPolarityExtent() const {return findPolarityExtent();}
std::vector<double> CellAgent::getAttachmentHistory() const {return attachmentHistory;}

// Instantaneous variable getters: (these variables are updated each timestep,
// and represent the cell's percepts)
double CellAgent::getMovementDirection() const {return movementDirection;}
double CellAgent::getActinFlow() const {return actinFlow;}
double CellAgent::getDirectionalInfluence() const {return directionalInfluence;}
double CellAgent::getDirectionalIntensity() const {return directionalIntensity;}
double CellAgent::getDirectionalShift() const {return directionalShift;}
double CellAgent::getSampledAngle() const {return sampledAngle;}


// Setters:
void CellAgent::setPosition(std::tuple<double, double> newPosition) {
    x = std::get<0>(newPosition);
    y = std::get<1>(newPosition);
}

void CellAgent::setContactStatus(const boostMatrix::matrix<bool>& stateToSet, int cellType) {
    if (cellType == 0) {
        cellType0ContactState = stateToSet;
    } else {
        cellType1ContactState = stateToSet;
    }
}

void CellAgent::setLocalCellHeadingState(const boostMatrix::matrix<double>& stateToSet) {
    localCellHeadingState = stateToSet;
}

void CellAgent::setLocalMatrixHeading(const boostMatrix::matrix<double>& stateToSet) {
    localMatrixHeading = stateToSet;
}

void CellAgent::setLocalMatrixPresence(const boostMatrix::matrix<double>& stateToSet) {
    localMatrixPresence = stateToSet;
}

void CellAgent::setDirectionalInfluence(double setDirectionalInfluence) {
    directionalInfluence = setDirectionalInfluence;
};

void CellAgent::setDirectionalIntensity(double setDirectiontalIntensity) {
    directionalIntensity = setDirectiontalIntensity;
};

void CellAgent::setLocalECMDensity(double setLocalDensity) {
    localECMDensity = setLocalDensity;
};


// Simulation code:
void CellAgent::takeRandomStep() {
    assert(directionalIntensity <= 1);

    // localECMDensity = 0;

    // Determine protrusion - polarity centered or ECM centered:
    double thresholdValue{findPolarityExtent() / (findPolarityExtent() + localECMDensity)};
    assert(thresholdValue <= 1 && thresholdValue >= 0);
    double randomDeterminant{uniformDistribution(generatorInfluence)};

    if (randomDeterminant < thresholdValue) {
        // Calculate matrix direction selection parameters using MM kinetics
        // - we're assuming directionality saturates at some value of kappa.
        double currentPolarity{findPolarityExtent()};
        double effectivePolarity{(polarityTurningCoupling * currentPolarity) / (kappa + currentPolarity)};
        double angleDelta = sampleVonMises(effectivePolarity);
        movementDirection = angleMod(findPolarityDirection() + angleDelta);
    } else {
        // Calculate matrix direction selection parameters using MM kinetics
        // - we're assuming directionality saturates at some value of kappa.
        double effectiveKappa{(matrixKappa * directionalIntensity) / (0.2 + directionalIntensity)};
        double angleDelta = sampleVonMises(effectiveKappa);
        movementDirection = angleMod(
            findPolarityDirection() + directionalInfluence + angleDelta
        );
    }

    // Determining collisions:
    std::pair<bool, double> collision{checkForCollisions()};
    if (collision.first) {
        // Calculating polarity change parameters due to collision:
        double effectiveRepolarisationMagnitude{
            collisionRepolarisation * std::abs(std::sin(collision.second))
        };
        double effectiveRepolarisationRate{
            repolarisationRate * std::abs(std::sin(collision.second))
        };
        double repolarisationAbsoluteDirection{
            angleMod(findPolarityDirection() + collision.second)
        };

        // Calculating repolarisation vector:
        double polarityChangeX{
            effectiveRepolarisationMagnitude*cos(repolarisationAbsoluteDirection)
        };
        double polarityChangeY{
            effectiveRepolarisationMagnitude*sin(repolarisationAbsoluteDirection)
        };

        // Updating polarity:
        polarityX = (1-effectiveRepolarisationRate)*polarityX +
            effectiveRepolarisationRate*polarityChangeX;
        polarityY = (1-effectiveRepolarisationRate)*polarityY +
            effectiveRepolarisationRate*polarityChangeY;
    }

    // Calculating actin flow in protrusion, via MM kinetics:
    double effectivePolarity{(poissonLambda * findPolarityExtent()) / (0.1 + findPolarityExtent())};
    std::poisson_distribution<int> protrusionDistribution(effectivePolarity);
    const int protrusionCount{protrusionDistribution(generatorProtrusion)};

    actinFlow = protrusionCount;

    // // Update position:
    // * std::pow(localECMDensity, 2)
    // * std::pow(localECMDensity, 2)
    double dx{actinFlow * cos(movementDirection) * flowScaling};
    double dy{actinFlow * sin(movementDirection) * flowScaling};
    x = x + dx;
    y = y + dy;

    // // Update polarity based on movement direction and actin flow:
    // *localECMDensity
    double polarityChangeExtent{std::tanh(actinFlow*flowPolarityCoupling)};
    double polarityChangeX{polarityChangeExtent*cos(movementDirection)};
    double polarityChangeY{polarityChangeExtent*sin(movementDirection)};

    // Trying out extent-defined persistence:
    // double testPersistence{findPolarityExtent()};
    double newPolarityX{polarityPersistence*polarityX + (1-polarityPersistence)*polarityChangeX};
    double newPolarityY{polarityPersistence*polarityY + (1-polarityPersistence)*polarityChangeY};

    // Add white noise component to polarity:
    double polarityNoiseX{polarityNoiseDistribution(generatorPolarityNoiseX)};
    double polarityNoiseY{polarityNoiseDistribution(generatorPolarityNoiseY)};
    newPolarityX = newPolarityX + polarityNoiseX;
    newPolarityY = newPolarityY + polarityNoiseY;

    // Clamping polarity components to [-1, 1] (while preserving direction):
    double newPolarityExtent{sqrt(pow(newPolarityX, 2) + pow(newPolarityY, 2))};
    if (newPolarityExtent > 1) {
        newPolarityExtent = 1;
    }
    assert(newPolarityExtent <= 1);
    const double newPolarityDirection{std::atan2(newPolarityY, newPolarityX)};
    polarityX = newPolarityExtent * cos(newPolarityDirection);
    polarityY = newPolarityExtent * sin(newPolarityDirection);

    // Check for valid low polarisation values:
    safeZeroPolarisation();
}


// Private functions for cell behaviour:
std::pair<bool, double> CellAgent::checkForCollisions() {
    // This checks for which cell in the Moore neighbourhood we need to query:
    int stateIndex{int(floor((movementDirection + M_PI + (M_PI/8)) / (M_PI / 4)))};
    int iContact{std::get<0>(HEADING_CONTACT_MAPPING[stateIndex])};
    int jContact{std::get<1>(HEADING_CONTACT_MAPPING[stateIndex])};

    // Are any cells nearby?
    bool type0Contact = cellType0ContactState(iContact, jContact);
    bool type1Contact = cellType1ContactState(iContact, jContact);
    if ((!type0Contact) && (!type1Contact)) {
        return std::make_pair(false, 0);
    }

    // Calculate corner collision correction:
    bool isCorner{(stateIndex == 1 || stateIndex == 3 || stateIndex == 5 || stateIndex == 7)};
    const double cornerCorrectionFactor{0.5 + ((9*M_PI)/32) - ((9*sqrt(2))/16)};

    if (isCorner) {
        bool skipCollision{uniformDistribution(generatorCornerCorrection) > cornerCorrectionFactor};
        if (skipCollision) {return std::make_pair(false, 0);}
    }

    // Calculate correction for relative cell angles:
    double otherCellHeading{localCellHeadingState(iContact, jContact)};
    double angularDistance{calculateMinimumAngularDistance(findPolarityDirection(), otherCellHeading)};
    double angularCorrectionFactor{cos(angularDistance)};

    // Calculate homotypic / heterotypic interactions:
    std::array<bool, 2> cellTypeContacts{type0Contact, type1Contact};
    if (cellTypeContacts[cellType]) {
        // Do homotypic interaction if present:
        double effectiveHomotypicInhibition{(homotypicInhibitionRate*angularCorrectionFactor)};
        if (uniformDistribution(generatorForInhibitionRate) < effectiveHomotypicInhibition) {
            double headingDistance{calculateAngularDistance(
                findPolarityDirection(), otherCellHeading
            )};
            return std::make_pair(true, headingDistance);
        }
    } else if (cellTypeContacts[std::abs(cellType - 1)]) {
        // Do heterotypic interaction if no homotypic interaction:
        double effectiveHeterotypicInhibition{
            (heterotypicInhibitionRate*angularCorrectionFactor)
        };
        if (uniformDistribution(generatorForInhibitionRate) < effectiveHeterotypicInhibition) {
            double headingDistance{calculateAngularDistance(
                findPolarityDirection(), otherCellHeading
            )};
            return std::make_pair(true, headingDistance);
        }
    }

    // If no collision is ultimately calculated, return no collision:
    return std::make_pair(false, 0);
}


double CellAgent::sampleVonMises(double kappa) {
    // Von Mises sampling quickly overloaded if kappa is very low - can set a minimum and take a
    // uniform distribution instead:
    if (kappa < 1e-3) {
        return angleUniformDistribution(generatorAngleUniform);
    }
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
            continue;
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


double CellAgent::sampleLevyDistribution(double mu, double c) {
    double U{uniformDistribution(generatorLevy)};
    std::cout << "Uniform sample: " << U << std::endl;
    std::cout << "Percentile: " << 1 - U/2 << std::endl;
    double ppf{boost::math::quantile(normalDistribution, 1 - U)};
    std::cout << "PPF: " << ppf << std::endl;
    double levyStepSize{mu + (c / (std::pow(ppf, 2)))};
    return levyStepSize;
}


double CellAgent::angleMod(double angle) const {
    while (angle < -M_PI) {angle += 2*M_PI;};
    while (angle >= M_PI) {angle -= 2*M_PI;};
    return angle;
}


double CellAgent::calculateAngularDistance(double headingA, double headingB) const {
    // Calculating change in theta:
    double deltaHeading{headingA - headingB};
    while (deltaHeading <= -M_PI) {deltaHeading += 2*M_PI;}
    while (deltaHeading > M_PI) {deltaHeading -= 2*M_PI;}
    return deltaHeading;
}


double CellAgent::calculateMinimumAngularDistance(double headingA, double headingB) const {
    // Calculating change in theta:
    double deltaHeading{headingA - headingB};
    while (deltaHeading <= -M_PI) {deltaHeading += 2*M_PI;}
    while (deltaHeading > M_PI) {deltaHeading -= 2*M_PI;}

    double flippedHeading;
    if (deltaHeading < 0) {
        flippedHeading = M_PI + deltaHeading;
    } else {
        flippedHeading = -(M_PI - deltaHeading);
    }

    // Selecting smallest change in theta and ensuring correct range:
    if (std::abs(deltaHeading) < std::abs(flippedHeading)) {
        assert((std::abs(deltaHeading) <= M_PI/2));
        return deltaHeading;
    } else {
        assert((std::abs(flippedHeading) <= M_PI/2));
        return flippedHeading;
    };
}


double CellAgent::findPolarityDirection() const {
    return std::atan2(polarityY, polarityX);
};


double CellAgent::findPolarityExtent() const {
    double polarityExtent{
        std::sqrt(
            std::pow(polarityX, 2) +
            std::pow(polarityY, 2)
        )
    };
    assert(polarityExtent <= 1.0);
    return polarityExtent;
};


void CellAgent::safeZeroPolarisation() {
    if ((polarityY == 0) & (polarityX == 0)) {
        double newAngle{angleUniformDistribution(generatorRandomRepolarisation)};
        polarityX = std::cos(newAngle) * 1e-5;
        polarityY = std::sin(newAngle) * 1e-5;
    }
};


// double CellAgent::getAverageAttachmentHeading() {
//     double sineMean{0};
//     double cosineMean{0};
//     for (auto & heading : attachmentHistory) {
//         sineMean += std::sin(heading);
//         cosineMean += std::cos(heading);
//     }

//     sineMean /= 5;
//     cosineMean /= 5;

//     assert(std::abs(sineMean) <= 1);
//     assert(std::abs(cosineMean) <= 1);
//     double angleAverage{std::atan2(sineMean, cosineMean)};
//     return angleAverage;
// };