#include "agents.h"

#include <algorithm>
#include <complex>
#include <random>
#include <cmath>
#include <cassert>
#include <limits>
#include <iostream>
#include <utility>
#include <ranges>

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
    // Defined behaviour parameters:
    bool setMatrixInteraction, unsigned int setCellSeed, int setCellID,

    // Cell motility and polarisation dynamics:
    double setHalfSatCellAngularConcentration, double setMaxCellAngularConcentration,
    double setHalfSatMeanActinFlow, double setMaxMeanActinFlow,
    double setFlowScaling, double setPolarityPersistence,
    double setActinPolarityRedistributionRate,
    double setPolarityNoiseSigma,

    // Matrix sensation parameters:
    double setHalfSatMatrixAngularConcentration, double setMaxMatrixAngularConcentration,

    // Collision parameters:
    int setCellType, double setHomotypicInhibitionRate, double setHeterotypicInhibitionRate,
    double setCollisionRepolarisation, double setCollisionRepolarisationRate,
    double setCellBodyRadius, double setMaxCellExtension, double setInhibitionStrength,

    // Randomised initial state parameters:
    double startX, double startY, double startHeading
    )
    // Model infrastructure:
    : thereIsMatrixInteraction{setMatrixInteraction}
    , cellSeed{setCellSeed}
    , cellID{setCellID}

    // Polarisation and movement parameters:
    , halfSatCellAngularConcentration{setHalfSatCellAngularConcentration}
    , maxCellAngularConcentration{setMaxCellAngularConcentration}
    , halfSatMeanActinFlow{setHalfSatMeanActinFlow}
    , maxMeanActinFlow{setMaxMeanActinFlow}
    , flowScaling{setFlowScaling}
    , polarityPersistence{setPolarityPersistence}
    , actinPolarityRedistributionRate{setActinPolarityRedistributionRate}
    , polarityNoiseSigma{setPolarityNoiseSigma}

    // Matrix sensation parameters:
    , halfSatMatrixAngularConcentration{setHalfSatMatrixAngularConcentration}
    , maxMatrixAngularConcentration{setMaxMatrixAngularConcentration}

    // Collision parameters:
    , cellType{setCellType}
    , homotypicInhibitionRate{setHomotypicInhibitionRate}
    , heterotypicInhibitionRate{setHeterotypicInhibitionRate}
    , collisionRepolarisation{setCollisionRepolarisation}
    , cellBodyRadius{setCellBodyRadius}
    , maxCellExtension{setMaxCellExtension}
    , inhibitionStrength{setInhibitionStrength}

    // State parameters:
    , x{startX}
    , y{startY}
    // , polarityX{1e-5 * cos(startHeading)}
    // , polarityY{1e-5 * sin(startHeading)}
    , movementDirection{0}
    , actinFlow{0}
    , directionalInfluence{0}
    , directionalIntensity{0}
    , localECMDensity{0}
    , directionalShift{0}
    , sampledAngle{0}
    , localMatrixHeading{boostMatrix::zero_matrix<double>(3, 3)}
    , localMatrixPresence{boostMatrix::zero_matrix<double>(3, 3)}
    , cellType0ContactState{boostMatrix::zero_matrix<bool>(3, 3)}
    , cellType1ContactState{boostMatrix::zero_matrix<bool>(3, 3)}
    , localCellHeadingState{boostMatrix::zero_matrix<double>(3, 3)}
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
double CellAgent::getPolarity() {return findPolarityDirection();}
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

// void CellAgent::setContactStatus(const boostMatrix::matrix<bool>& stateToSet, int cellType) {
//     if (cellType == 0) {
//         cellType0ContactState = stateToSet;
//     } else {
//         cellType1ContactState = stateToSet;
//     }
// }

// void CellAgent::setLocalCellHeadingState(const boostMatrix::matrix<double>& stateToSet) {
//     localCellHeadingState = stateToSet;
// }

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

void CellAgent::setLocalCellList(std::vector<std::shared_ptr<CellAgent>> setLocalAgents) {
    localAgents = setLocalAgents;
}

// Simulation code:
void CellAgent::takeRandomStep() {
    assert(directionalIntensity <= 1);

    // localECMDensity = 0;

    // Determine protrusion - polarity centered or ECM centered:
    double polarityExtent{findPolarityExtent()};
    double thresholdValue{polarityExtent / (polarityExtent + localECMDensity)};
    assert(thresholdValue <= 1 && thresholdValue >= 0);
    double randomDeterminant{uniformDistribution(generatorInfluence)};

    if (randomDeterminant < thresholdValue) {
        // Calculate matrix direction selection parameters using MM kinetics
        // - we're assuming directionality saturates at some degree of polarisation.
        double currentPolarity{findPolarityExtent()};
        double cellAngularConcentration{
            (maxCellAngularConcentration * currentPolarity) /
            (halfSatCellAngularConcentration + currentPolarity)
        };
        double angleDelta = sampleVonMises(cellAngularConcentration);
        movementDirection = angleMod(findPolarityDirection() + angleDelta);
    } else {
        // Calculate matrix direction selection parameters using MM kinetics
        // - we're assuming directionality saturates at some value of matrix
        // directional homogeneity.
        double matrixAngularConcentration{
            (maxMatrixAngularConcentration * directionalIntensity) /
            (halfSatMatrixAngularConcentration + directionalIntensity)
        };
        double angleDelta = sampleVonMises(matrixAngularConcentration);
        movementDirection = angleMod(
            findPolarityDirection() + directionalInfluence + angleDelta
        );
    }

    // Calculating actin flow in protrusion, via MM kinetics:
    double meanActinFlow{
        (maxMeanActinFlow * findPolarityExtent()) /
        (halfSatMeanActinFlow + findPolarityExtent())
    };
    std::poisson_distribution<int> protrusionDistribution(meanActinFlow + 0.1);
    const int actinFlow{protrusionDistribution(generatorProtrusion)};

    // Update flow history:
    double actinX{actinFlow * cos(movementDirection)};
    double actinY{actinFlow * sin(movementDirection)};
    double polarityNoiseX{polarityNoiseDistribution(generatorPolarityNoiseX)};
    double polarityNoiseY{polarityNoiseDistribution(generatorPolarityNoiseY)};
    addToPolarisationHistory(actinX + polarityNoiseX, actinY + polarityNoiseY);

    // double cellBodyRadius{20};
    // double maxCellExtension{300};
    // double inhibitionStrength{0.5};

    // Calculating effect of CIL:
    for (auto& localAgent: localAgents) {
        // Useful points:
        double actingX{getX()};
        double actingY{getY()};

        // double actingExtension{getPolarityExtent() * maxCellExtension};
        // double actingExtension{
        //     (maxCellExtension * getPolarityExtent()) /
        //     (0.1 + getPolarityExtent())
        // };
        double actingExtension{getPolarityExtent() * maxCellExtension};

        // Take modulo of position in case interaction is across the periodic boundary:
        double localX{localAgent->getX()};
        double localY{localAgent->getY()};

        // ---> Determining correct modulus for X:
        if (actingX - localX > (2048 / 2)) {
            localX += 2048;
        }
        if (actingX - localX < -(2048 / 2)) {
            localX -= 2048;
        }

        // ---> Determining correct modulus for Y:
        if (actingY - localY > (2048 / 2)) {
            localY += 2048;
        }
        if (actingY - localY < -(2048 / 2)) {
            localY -= 2048;
        }

        double localExtension{localAgent->getPolarityExtent() * maxCellExtension};

        // Distance and differences:
        double xDifference{localX - actingX};
        double yDifference{localY - actingY};
        double cellDistance{std::sqrt(std::pow(xDifference, 2) + std::pow(yDifference, 2))};
        double directionToLocalCell{std::atan2(yDifference, xDifference)};

        // Calculating direct collision between cell bodies:
        if (cellDistance < cellBodyRadius*2) {
            for(unsigned index = 0; index < xPolarityHistory.size(); ++index) {
                // Calculating projection of protrusion onto collision direction:
                double xProtrusion{xPolarityHistory[index]};
                double yProtrusion{yPolarityHistory[index]};
                double protrusionAngle{std::atan2(yProtrusion, xProtrusion)};

                double normCollisionX{std::cos(directionToLocalCell)};
                double normCollisionY{std::sin(directionToLocalCell)};

                double protrusionCollisionDotProduct{
                    xProtrusion*normCollisionX + yProtrusion*normCollisionY
                };

                // No effect if actin flow is less than perpendicular:
                if (protrusionCollisionDotProduct < 0) {
                    continue;
                }

                // Updating polarity based on contact inhibition with cell body:
                xPolarityHistory[index] -= 
                    normCollisionX * protrusionCollisionDotProduct * inhibitionStrength;
                yPolarityHistory[index] -=
                    normCollisionY * protrusionCollisionDotProduct * inhibitionStrength;
            }
            continue;
        }

        // Calculating vector line equation for acting cell:
        std::vector<double> actingPolarityPointA = {actingX, actingY, 1};
        std::vector<double> actingPolarityPointB = {
            actingX + std::cos(getPolarity()), actingY + std::sin(getPolarity()), 1
        };
        std::vector<double> actingPolarityLine{crossProduct(actingPolarityPointA, actingPolarityPointB)};

        // Calculating vector line equation for local cell:
        std::vector<double> localPolarityPointA = {localX, localY, 1};
        std::vector<double> localPolarityPointB = {
            localX + std::cos(localAgent->getPolarity()), localY + std::sin(localAgent->getPolarity()), 1
        };
        std::vector<double> localPolarityLine{crossProduct(localPolarityPointA, localPolarityPointB)};

        // Getting distance from local cell centre to this line:
        // See https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#A_vector_projection_proof:
        double a{actingPolarityLine[0]};
        double b{actingPolarityLine[1]};
        double c{actingPolarityLine[2]};
        double distanceFromActingExtensionToLocalCell{
            std::abs(a*localX + b*localY + c) /
            std::sqrt(std::pow(a, 2) + std::pow(b, 2))
        };
        double distanceAlongExtension{
            std::sqrt(std::pow(cellDistance, 2) - std::pow(distanceFromActingExtensionToLocalCell, 2))
        };

        // Setting effect of cell extension to cell body contact:
        bool withinCellBody{distanceFromActingExtensionToLocalCell < cellBodyRadius};
        bool withinExtension{distanceAlongExtension <= actingExtension};

        if (withinCellBody & withinExtension) {
            // Selecting tangent that is closest to current shape direction:
            double localTangent;
            double localTangentA{localAgent->getPolarity() + M_PI/2};
            double localTangentB{localAgent->getPolarity() - M_PI/2};
            if (localTangentA - getPolarity() < localTangentB - getPolarity()) {
                localTangent = localTangentA;
            } else {
                localTangent = localTangentB;
            }

            for(unsigned index = 0; index < xPolarityHistory.size(); ++index) {
                // Calculating projection of protrusion onto collision direction:
                double xProtrusion{xPolarityHistory[index]};
                double yProtrusion{yPolarityHistory[index]};
                double protrusionAngle{std::atan2(yProtrusion, xProtrusion)};
                if (std::cos(protrusionAngle - localTangent) <= 0) {
                    continue;
                }

                // Calculate collision direction-of-effect as tangent of local cell's shape direction:
                double normCollisionX{std::cos(localTangent)};
                double normCollisionY{std::sin(localTangent)};
                double protrusionCollisionDotProduct{
                    xProtrusion*normCollisionX + yProtrusion*normCollisionY
                };

                assert(protrusionCollisionDotProduct >= 0);

                // Updating polarity based on contact inhibition with cell body:
                xPolarityHistory[index] -= 
                    normCollisionX * protrusionCollisionDotProduct * inhibitionStrength;
                yPolarityHistory[index] -=
                    normCollisionY * protrusionCollisionDotProduct * inhibitionStrength;
            }
            
            // If there is extension -> cell body contact, extension-extension contact is guaranteed,
            // and we don't want to double count the effect of one cell.
            continue;
        }

        // Calculations for cell extension to cell extension contact:
        // Calculating point of intersection:
        std::vector<double> intersectionPoint{crossProduct(actingPolarityLine, localPolarityLine)};
        double intersectionX{intersectionPoint[0]};
        double intersectionY{intersectionPoint[1]};

        // Determining distances from point of intersection:
        double distanceFromActing{std::sqrt(
            std::pow(intersectionX - actingX, 2) + std::pow(intersectionY - actingY, 2)
        )};
        double distanceFromLocal{std::sqrt(
            std::pow(intersectionX - localX, 2) + std::pow(intersectionY - localY, 2)
        )};

        // Determining intersection:
        bool withinActing{distanceFromActing < actingExtension};
        bool withinLocal{distanceFromLocal < localExtension};

        // Updating actin flows based on repulsion from cell extensions:
        if (withinActing & withinLocal) {
            // Selecting local cell directional tangent that is closest to current shape direction:
            double localTangent;
            double localTangentA{localAgent->getPolarity() + M_PI/2};
            double localTangentB{localAgent->getPolarity() - M_PI/2};
            if (localTangentA - getPolarity() < localTangentB - getPolarity()) {
                localTangent = localTangentA;
            } else {
                localTangent = localTangentB;
            }

            for(unsigned index = 0; index < xPolarityHistory.size(); ++index) {
                // Calculating projection of protrusion onto collision direction:
                double xProtrusion{xPolarityHistory[index]};
                double yProtrusion{yPolarityHistory[index]};
                double protrusionAngle{std::atan2(yProtrusion, xProtrusion)};
                if (std::cos(protrusionAngle - localTangent) <= 0) {
                    continue;
                }

                // Calculate collision direction-of-effect as tangent of local cell's shape direction:
                double normCollisionX{std::cos(localTangent)};
                double normCollisionY{std::sin(localTangent)};

                double protrusionCollisionDotProduct{
                    xProtrusion*normCollisionX + yProtrusion*normCollisionY
                };
                assert(protrusionCollisionDotProduct >= 0);

                // Updating polarity based on contact inhibition with cell body:
                xPolarityHistory[index] -= 
                    normCollisionX * protrusionCollisionDotProduct * inhibitionStrength;
                yPolarityHistory[index] -=
                    normCollisionY * protrusionCollisionDotProduct * inhibitionStrength;
            }
        }
    }

    // Update position:
    double totalFlow{findTotalFlow()};
    double polarityDirection{findPolarityDirection()};
    double dx{std::cos(polarityDirection) * totalFlow};
    double dy{std::sin(polarityDirection) * totalFlow};
    addToMovementHistory(std::cos(polarityDirection), std::sin(polarityDirection));
    x = x + dx;
    y = y + dy;

    // // // Update polarity based on movement direction and actin flow:
    // double polarityChangeExtent{std::tanh(actinFlow*actinPolarityRedistributionRate)};
    // double polarityChangeX{polarityChangeExtent*cos(movementDirection)};
    // double polarityChangeY{polarityChangeExtent*sin(movementDirection)};

    // double newPolarityX{polarityPersistence*polarityX + (1-polarityPersistence)*polarityChangeX};
    // double newPolarityY{polarityPersistence*polarityY + (1-polarityPersistence)*polarityChangeY};

    // // Add white noise component to polarity:
    // double polarityNoiseX{polarityNoiseDistribution(generatorPolarityNoiseX)};
    // double polarityNoiseY{polarityNoiseDistribution(generatorPolarityNoiseY)};
    // newPolarityX = newPolarityX + polarityNoiseX;
    // newPolarityY = newPolarityY + polarityNoiseY;

    // // Clamping polarity components to [-1, 1] (while preserving direction):
    // double newPolarityExtent{sqrt(pow(newPolarityX, 2) + pow(newPolarityY, 2))};
    // if (newPolarityExtent > 1) {
    //     newPolarityExtent = 1;
    // }
    // assert(newPolarityExtent <= 1);
    // const double newPolarityDirection{std::atan2(newPolarityY, newPolarityX)};
    // polarityX = newPolarityExtent * cos(newPolarityDirection);
    // polarityY = newPolarityExtent * sin(newPolarityDirection);

    // Check for valid low polarisation values:
    // safeZeroPolarisation();
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


double CellAgent::findPolarityDirection() {
    // Averaging over history:
    double xFlowSum{0}, yFlowSum{0};
    for (double xFlow : xPolarityHistory) {
        xFlowSum += xFlow;
    }
    for (double yFlow : yPolarityHistory) {
        yFlowSum += yFlow;
    }

    // std::cout << xFlowSum << " " << yFlowSum << std::endl;

    // Ensuring no directional bias from atan if zero polarisation:
    double polarityDirection;
    if ((yFlowSum == 0) & (xFlowSum == 0)) {
        polarityDirection = angleUniformDistribution(generatorRandomRepolarisation);
    } else {
        polarityDirection = std::atan2(yFlowSum, xFlowSum);
    }

    return polarityDirection;
};


double CellAgent::findPolarityExtent() const {
    // Averaging over history:
    double xFlowSum{0}, yFlowSum{0};
    for (double xFlow : xPolarityHistory) {
        xFlowSum += xFlow;
    }
    for (double yFlow : yPolarityHistory) {
        yFlowSum += yFlow;
    }

    // Scaling flow to polarisation:
    xFlowSum *= actinPolarityRedistributionRate;
    yFlowSum *= actinPolarityRedistributionRate;

    // Taking extent:
    double rawPolarityExtent{std::sqrt(std::pow(xFlowSum, 2) + std::pow(yFlowSum, 2))};
    double polarityExtent{std::tanh(rawPolarityExtent)};
    assert(polarityExtent <= 1.0 & polarityExtent >=0);
    return polarityExtent + 1e-5;
};

void CellAgent::addToPolarisationHistory(double actinFlowX, double actinFlowY) {
    xPolarityHistory.push_back(actinFlowX);
    yPolarityHistory.push_back(actinFlowY);
    if (xPolarityHistory.size() > 25) {
        xPolarityHistory.pop_front();
        yPolarityHistory.pop_front();
    }
};

double CellAgent::findTotalFlow() {
    // Averaging over history:
    double xFlowSum{0}, yFlowSum{0};
    for (double xFlow : xPolarityHistory) {
        xFlowSum += xFlow;
    }
    for (double yFlow : yPolarityHistory) {
        yFlowSum += yFlow;
    }

    // Scaling flow to movement:
    xFlowSum *= flowScaling;
    yFlowSum *= flowScaling;

    // Taking extent:
    return std::sqrt(std::pow(xFlowSum, 2) + std::pow(yFlowSum, 2));
}

void CellAgent::addToMovementHistory(double movementX, double movementY) {
    xMovementHistory.push_back(movementX);
    yMovementHistory.push_back(movementY);
    if (xMovementHistory.size() > 25) {
        xMovementHistory.pop_front();
        yMovementHistory.pop_front();
    }
}

double CellAgent::findDirectionalConcentration() {
    // If there is no movement, there is no concentration of directions:
    if (xMovementHistory.size() == 0) {
        return 0;
    }

    // Averaging over history:
    double xMovementSum{0}, yMovementSum{0};
    for (double xMovement : xMovementHistory) {
        xMovementSum += xMovement;
    }
    for (double yMovement : yMovementHistory) {
        yMovementSum += yMovement;
    }

    // Finding total path length:
    double directionalConcentration{std::sqrt(std::pow(xMovementSum, 2) + std::pow(yMovementSum, 2))};
    return directionalConcentration / xMovementHistory.size();
}

double CellAgent::findShapeDirection() {
    // Averaging over history:
    double xMovementSum{0}, yMovementSum{0};
    for (double xMovement : xMovementHistory) {
        xMovementSum += xMovement;
    }
    for (double yMovement : yMovementHistory) {
        yMovementSum += yMovement;
    }

    // Finding total path length:
    double shapeDirection{std::atan2(yMovementSum, xMovementSum)};
    return shapeDirection;
}

std::vector<double> CellAgent::crossProduct(std::vector<double> const a, std::vector<double> const b) {
    // Basic cross product calculation:
    std::vector<double> resultVector(3);  
    resultVector[0] = a[1]*b[2] - a[2]*b[1];
    resultVector[1] = a[2]*b[0] - a[0]*b[2];
    resultVector[2] = a[0]*b[1] - a[1]*b[0];
    return resultVector;
}