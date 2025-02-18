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
    , polarityX{1e-5 * cos(startHeading)}
    , polarityY{1e-5 * sin(startHeading)}
    , polarityDirection{startHeading}
    , polarityMagnitude{1e-5}
    , flowDirection{startHeading}
    , flowMagnitude{1e-5}
    , scaledFlowMagnitude{1e-5}
    , eccentricityConstant{0.9}
    , movementDirection{0}
    // , actinFlow{0}
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
    positionalNoiseDistribution = std::normal_distribution<double>(0, 1);

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
double CellAgent::getPolarityDirection() const {return polarityDirection;}
double CellAgent::getPolarityMagnitude() const {return polarityMagnitude;}

// Instantaneous variable getters: (these variables are updated each timestep,
// and represent the cell's percepts)
double CellAgent::getMovementDirection() const {return movementDirection;}
double CellAgent::getActinFlowMagnitude() const {return flowMagnitude;}
double CellAgent::getActinFlowDirection() const {return flowDirection;}
double CellAgent::getScaledActinFlowMagnitude() const {return scaledFlowMagnitude;}

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

    // Determine protrusion - polarity centered or ECM centered:
    double thresholdValue{polarityMagnitude / (polarityMagnitude + localECMDensity)};
    assert(thresholdValue <= 1 && thresholdValue >= 0);
    double randomDeterminant{uniformDistribution(generatorInfluence)};
    double angleDelta{0};
    if (randomDeterminant < thresholdValue) {
        // Calculate polarity direction selection parameters using MM kinetics
        // - we're assuming directionality saturates at some degree of polarisation.
        double cellAngularConcentration{
            (maxCellAngularConcentration * polarityMagnitude) /
            (halfSatCellAngularConcentration + polarityMagnitude)
        };
        angleDelta = sampleVonMises(cellAngularConcentration);
        movementDirection = angleMod(polarityDirection + angleDelta);
    } else {
        // Calculate matrix direction selection parameters using MM kinetics
        // - we're assuming directionality saturates at some value of matrix
        // directional homogeneity.
        double matrixAngularConcentration{
            (maxMatrixAngularConcentration * directionalIntensity) /
            (halfSatMatrixAngularConcentration + directionalIntensity)
        };
        angleDelta = sampleVonMises(matrixAngularConcentration) + directionalInfluence;
        movementDirection = angleMod(polarityDirection + angleDelta);
    }

    // Calculating actin flow in protrusion, via MM kinetics:
    double meanActinFlow{
        (maxMeanActinFlow * polarityMagnitude) /
        (halfSatMeanActinFlow + polarityMagnitude)
    };
    std::poisson_distribution<int> protrusionDistribution(std::max(meanActinFlow, 1.0));
    const double actinFlow{static_cast<double>(protrusionDistribution(generatorProtrusion))};

    // Update flow history:
    double actinX{actinFlow * cos(movementDirection)};
    double actinY{actinFlow * sin(movementDirection)};
    double polarityNoiseX{polarityNoiseDistribution(generatorPolarityNoiseX)};
    double polarityNoiseY{polarityNoiseDistribution(generatorPolarityNoiseY)};

    // Seeing which protrusions die off this timestep:
    ageActinHistory();

    // Adding new protrusion if there's space:
    addToActinHistory(actinX, actinY);

    // Accumulators for CIL-dependent polarity reorientation:
    double polarityChangeCilX{0.};
    double polarityChangeCilY{0.};

    // Variables of interest - TODO:
    // double eccentricityRate{2};
    double solidity{10};
    double polarityRate{0.0005};
    double redistRatio{0.4};
    double advecRatio{0.6};

    double rhoaRedistributionRate{redistRatio*polarityRate};
    double advectionRate{advecRatio*polarityRate};
    double polarityDiffusion{1*polarityRate};

    // *** *** Sampling from area of collisions *** ***
    // Useful points:
    double actingCellX{getX()};
    double actingCellY{getY()};

    // Sampling from random radius in cell area:
    // angleDelta
    double samplePointDirection{angleUniformDistribution(generatorAngleUniform)};
    // uniformDistribution(generatorPolarityNoiseX)
    double samplePointRadius{1 + (positionalNoiseDistribution(generatorCornerCorrection)/3)};
    double effectiveRadius{samplePointRadius*cellBodyRadius};
    double samplePointX{effectiveRadius * std::cos(samplePointDirection)};
    double samplePointY{effectiveRadius * std::sin(samplePointDirection)};

    // // Scaling origin points by eccentricity:
    // double cappedActingEccentricity{1 - std::exp(-eccentricityRate*polarityMagnitude)};
    double cappedActingEccentricity{eccentricityConstant};
    double minorAxisScaling{std::sqrt(1 - std::pow(cappedActingEccentricity, 2))};
    double majorAxisScaling{std::pow(1 - std::pow(cappedActingEccentricity, 2), -0.25)};
    double actingFrameX{samplePointX*minorAxisScaling};
    double actingFrameY{samplePointY*majorAxisScaling};

    // Transforming points from acting frame to global frame:
    double rotatingBasisAngle{polarityDirection - M_PI_2};
    double transformedX{
        (actingFrameX*std::cos(rotatingBasisAngle)) -
        (actingFrameY*std::sin(rotatingBasisAngle))
    };
    double transformedY{
        (actingFrameX*std::sin(rotatingBasisAngle)) +
        (actingFrameY*std::cos(rotatingBasisAngle))
    };

    // Finding total distance from cell:
    double sampleGlobalX{actingCellX + transformedX};
    double sampleGlobalY{actingCellY + transformedY};
    double sampleDistanceFromActing{std::sqrt(
        std::pow(transformedX, 2) +
        std::pow(transformedY, 2)
    )};
    double angleFromActingToSample{std::atan2(transformedY, transformedX)};

    // Closest agent to query:
    std::vector<double> distancesToProtrusion{};
    for (auto& localAgent: localAgents) {
        // Take modulo of position in case interaction is across the periodic boundary:
        double localCellX{localAgent->getX()};
        double localCellY{localAgent->getY()};

        // ---> Determining correct modulus for X:
        if (actingCellX - localCellX > (2048 / 2)) {
            localCellX += 2048;
        }
        else if (actingCellX - localCellX < -(2048 / 2)) {
            localCellX -= 2048;
        }

        // ---> Determining correct modulus for Y:
        if (actingCellY - localCellY > (2048 / 2)) {
            localCellY += 2048;
        }
        else if (actingCellY - localCellY < -(2048 / 2)) {
            localCellY -= 2048;
        }

        // Getting total distance to local cell:
        double localFrameX{sampleGlobalX - localCellX};
        double localFrameY{sampleGlobalY - localCellY};

        // Getting vector components in frame of reference of local cell:
        double localBasisAngle{M_PI_2 - localAgent->getPolarityDirection()};
        double localEllipseX{
            (localFrameX*std::cos(localBasisAngle)) -
            (localFrameY*std::sin(localBasisAngle))
        };
        double localEllipseY{
            (localFrameX*std::sin(localBasisAngle)) +
            (localFrameY*std::cos(localBasisAngle))
        };

        // Getting local scaling:
        // double cappedLocalEccentricity{
        //     1 - std::exp(-eccentricityRate*localAgent->getPolarityMagnitude())
        // };
        double cappedLocalEccentricity{eccentricityConstant};
        double localMinorAxisScaling{std::sqrt(1 - std::pow(cappedLocalEccentricity, 2))};
        double localMajorAxisScaling{std::pow(1 - std::pow(cappedLocalEccentricity, 2), -0.25)};
        double sigmaX{localMinorAxisScaling * (cellBodyRadius)};
        double sigmaY{localMajorAxisScaling * (cellBodyRadius)};

        // Scaling by ellipse axes:
        double scaledLocalX{localEllipseX / sigmaX};
        double scaledLocalY{localEllipseY / sigmaY};
        double scaledDistance{std::sqrt(
            std::pow(scaledLocalX, 2) +
            std::pow(scaledLocalY, 2)
        )};

        // Adding to total distances vector:
        distancesToProtrusion.emplace_back(scaledDistance);
    }

    // Finding minimum distance, ignoring collision if closest to actingCell:
    auto minimumIterator{std::min_element(distancesToProtrusion.begin(), distancesToProtrusion.end())};
    double minimumDistance{0};
    if (minimumIterator == distancesToProtrusion.end()) {
        // If no cells in local area, we have to prevent possible dereferencing of end iterator:
        minimumDistance = 2048;
    } else {
        minimumDistance = *minimumIterator;
    }

    // If sampled point impinges on other cell area, calculate likelihood of collision
    if (minimumDistance < samplePointRadius) {
        // Getting cell index:
        int cellIndex{static_cast<int>(
            std::distance(
                distancesToProtrusion.begin(),
                std::min_element(distancesToProtrusion.begin(), distancesToProtrusion.end())
            )
        )};

        // Take modulo of position in case interaction is across the periodic boundary:
        double localCellX{localAgents[cellIndex]->getX()};
        double localCellY{localAgents[cellIndex]->getY()};

        // ---> Determining correct modulus for X:
        if (actingCellX - localCellX > (2048 / 2)) {
            localCellX += 2048;
        }
        else if (actingCellX - localCellX < -(2048 / 2)) {
            localCellX -= 2048;
        }

        // ---> Determining correct modulus for Y:
        if (actingCellY - localCellY > (2048 / 2)) {
            localCellY += 2048;
        }
        else if (actingCellY - localCellY < -(2048 / 2)) {
            localCellY -= 2048;
        }

        // Getting angle to nucleus:
        double xDifference{actingCellX - localCellX};
        double yDifference{actingCellY - localCellY};
        double angleLocalToActing{std::atan2(yDifference, xDifference)};

        // Finding probability of collision:
        double exponentTerm{std::pow(std::pow(minimumDistance, 2)/2, solidity)};
        double finalProbability{std::exp(-exponentTerm)};

        // (1/std::sqrt(2*M_PI))*
        // if (finalProbability < 0.1) {
        //     continue;
        // } 

        // Determining collision:
        double randomDeterminant{uniformDistribution(generatorInfluence)};
        if (randomDeterminant < finalProbability) {
            for (auto& actinFlow : actinHistory) {
                // Find axis along which to reduce actin flow:
                double actinFlowDirection{std::atan2(actinFlow[1], actinFlow[0])};
                double componentOfFlowOntoSample{
                    std::cos(actinFlowDirection - angleFromActingToSample)
                };
                if (componentOfFlowOntoSample <= 0) {
                    continue;
                }

                // Reduce actin flow in protrusion:
                double actinFlowExtent{std::sqrt(
                    std::pow(actinFlow[0], 2) +
                    std::pow(actinFlow[1], 2)
                )};
                double reductionInFlow{inhibitionStrength * actinFlowExtent * componentOfFlowOntoSample};
                double dxActinFlow{-std::cos(angleFromActingToSample) * reductionInFlow};
                double dyActinFlow{-std::sin(angleFromActingToSample) * reductionInFlow};
                actinFlow[0] += dxActinFlow;
                actinFlow[1] += dyActinFlow;
            }

            // Getting directional effect of protrusion:
            // double normCollisionX{-std::cos(angleFromActingToSample)};
            // double normCollisionY{-std::sin(angleFromActingToSample)};
            double componentOfTensionOntoPolarity{std::cos(polarityDirection - angleLocalToActing)};
            // componentOfTensionOntoPolarity <= 0
            if (componentOfTensionOntoPolarity <= 0) {
                // Getting directional effect:
                double normCollisionX{std::cos(angleLocalToActing)};
                double normCollisionY{std::sin(angleLocalToActing)};

                // Simulate effect of CIL on RhoA redistribution:
                polarityChangeCilX += normCollisionX * rhoaRedistributionRate * std::abs(componentOfTensionOntoPolarity);
                polarityChangeCilY += normCollisionY * rhoaRedistributionRate * std::abs(componentOfTensionOntoPolarity);
            }
        }
    }

    // Update current flow-derived values:
    flowDirection = findTotalActinFlowDirection();
    flowMagnitude = findTotalActinFlowMagnitude();
    scaledFlowMagnitude = std::tanh(flowMagnitude);

    // Update polarity based on current actin flows:
    double rawPolarityMagnitude{std::sqrt(std::pow(polarityX, 2) + std::pow(polarityY, 2))};
    double rawMagnitudeSquared{std::pow(rawPolarityMagnitude, 2)};
    std::vector<double> flowComponents{findTotalActinFlowComponents()};

    double xDiffusiveComponent{polarityDiffusion*rawMagnitudeSquared*std::cos(polarityDirection)};
    double yDiffusiveComponent{polarityDiffusion*rawMagnitudeSquared*std::sin(polarityDirection)};
    double xAdvectionComponent{advectionRate*flowComponents[0] + polarityChangeCilX};
    double yAdvectionComponent{advectionRate*flowComponents[1] + polarityChangeCilY};

    // assert(polarityChangeCilX == 0);
    // assert(polarityChangeCilY == 0);

    double dPolarityX{xAdvectionComponent - xDiffusiveComponent};
    double dPolarityY{yAdvectionComponent - yDiffusiveComponent};

    polarityX += dPolarityX;
    polarityY += dPolarityY;

    if (std::isnan(polarityX)) {
        std::cout << "dPolarityX: " << dPolarityX << std::endl;
        assert(!std::isnan(polarityX));
    }
    if (std::isnan(polarityY)) {
        std::cout << "dPolarityY: " << dPolarityY << std::endl;
        assert(!std::isnan(polarityY));
    }

    // Updating derived values:
    polarityDirection = std::atan2(polarityY, polarityX);
    polarityMagnitude = std::tanh(std::sqrt(std::pow(polarityX, 2) + std::pow(polarityY, 2)));
    if (polarityMagnitude == 0) {
        polarityMagnitude += 1e-5;
        polarityDirection = angleUniformDistribution(generatorAngleUniform);
    }

    // Update position:
    double cellDisplacement{findCellMovementMagnitude()};
    double dx{std::cos(flowDirection) * cellDisplacement};
    double dy{std::sin(flowDirection) * cellDisplacement};
    addToMovementHistory(std::cos(flowDirection), std::sin(flowDirection));
    x += dx;
    y += dy;

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

std::vector<double> CellAgent::sampleAttachmentPoint() {
    // Getting current position:
    double actingCellX{getX()};
    double actingCellY{getY()};

    // Sampling from random radius in cell area:
    double angleDelta = angleUniformDistribution(generatorAngleUniform);
    // angleDelta
    // M_PI_2 + (angleDelta / 2)
    double samplePointDirection{angleDelta};
    // uniformDistribution(generatorPolarityNoiseX)
    double samplePointRadius{1 + (positionalNoiseDistribution(generatorCornerCorrection)/3)};
    double effectiveRadius{samplePointRadius*cellBodyRadius};
    double samplePointX{effectiveRadius * std::cos(samplePointDirection)};
    double samplePointY{effectiveRadius * std::sin(samplePointDirection)};

    // // Scaling origin points by eccentricity:
    // double cappedActingEccentricity{1 - std::exp(-eccentricityRate*polarityMagnitude)};
    double cappedActingEccentricity{eccentricityConstant};
    double minorAxisScaling{std::sqrt(1 - std::pow(cappedActingEccentricity, 2))};
    double majorAxisScaling{std::pow(1 - std::pow(cappedActingEccentricity, 2), -0.25)};
    double actingFrameX{samplePointX*minorAxisScaling};
    double actingFrameY{samplePointY*majorAxisScaling};

    // Transforming points from acting frame to global frame:
    double rotatingBasisAngle{polarityDirection - M_PI_2};
    double transformedX{
        (actingFrameX*std::cos(rotatingBasisAngle)) -
        (actingFrameY*std::sin(rotatingBasisAngle))
    };
    double transformedY{
        (actingFrameX*std::sin(rotatingBasisAngle)) +
        (actingFrameY*std::cos(rotatingBasisAngle))
    };

    // Finding total distance from cell:
    double sampleGlobalX{actingCellX + transformedX};
    double sampleGlobalY{actingCellY + transformedY};

    return {sampleGlobalX, sampleGlobalY};
};

// // Private functions for cell behaviour:
// std::pair<bool, double> CellAgent::checkForCollisions() {
//     // This checks for which cell in the Moore neighbourhood we need to query:
//     int stateIndex{int(floor((movementDirection + M_PI + (M_PI/8)) / (M_PI / 4)))};
//     int iContact{std::get<0>(HEADING_CONTACT_MAPPING[stateIndex])};
//     int jContact{std::get<1>(HEADING_CONTACT_MAPPING[stateIndex])};

//     // Are any cells nearby?
//     bool type0Contact = cellType0ContactState(iContact, jContact);
//     bool type1Contact = cellType1ContactState(iContact, jContact);
//     if ((!type0Contact) && (!type1Contact)) {
//         return std::make_pair(false, 0);
//     }

//     // Calculate corner collision correction:
//     bool isCorner{(stateIndex == 1 || stateIndex == 3 || stateIndex == 5 || stateIndex == 7)};
//     const double cornerCorrectionFactor{0.5 + ((9*M_PI)/32) - ((9*sqrt(2))/16)};

//     if (isCorner) {
//         bool skipCollision{uniformDistribution(generatorCornerCorrection) > cornerCorrectionFactor};
//         if (skipCollision) {return std::make_pair(false, 0);}
//     }

//     // Calculate correction for relative cell angles:
//     double otherCellHeading{localCellHeadingState(iContact, jContact)};
//     double angularDistance{calculateMinimumAngularDistance(findPolarityDirection(), otherCellHeading)};
//     double angularCorrectionFactor{cos(angularDistance)};

//     // Calculate homotypic / heterotypic interactions:
//     std::array<bool, 2> cellTypeContacts{type0Contact, type1Contact};
//     if (cellTypeContacts[cellType]) {
//         // Do homotypic interaction if present:
//         double effectiveHomotypicInhibition{(homotypicInhibitionRate*angularCorrectionFactor)};
//         if (uniformDistribution(generatorForInhibitionRate) < effectiveHomotypicInhibition) {
//             double headingDistance{calculateAngularDistance(
//                 findPolarityDirection(), otherCellHeading
//             )};
//             return std::make_pair(true, headingDistance);
//         }
//     } else if (cellTypeContacts[std::abs(cellType - 1)]) {
//         // Do heterotypic interaction if no homotypic interaction:
//         double effectiveHeterotypicInhibition{
//             (heterotypicInhibitionRate*angularCorrectionFactor)
//         };
//         if (uniformDistribution(generatorForInhibitionRate) < effectiveHeterotypicInhibition) {
//             double headingDistance{calculateAngularDistance(
//                 findPolarityDirection(), otherCellHeading
//             )};
//             return std::make_pair(true, headingDistance);
//         }
//     }

//     // If no collision is ultimately calculated, return no collision:
//     return std::make_pair(false, 0);
// }


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

double CellAgent::findTotalActinFlowDirection() const {
    // Averaging over history:
    double xFlowSum{0}, yFlowSum{0};
    for (const auto & actinFlow : actinHistory) {
        xFlowSum += actinFlow[0];
        yFlowSum += actinFlow[1];
    }

    double flowDirection{std::atan2(yFlowSum, xFlowSum)};
    return flowDirection;
};

double CellAgent::findTotalActinFlowMagnitude() const {
    // Averaging over history:
    double xFlowSum{0}, yFlowSum{0};
    for (const auto & actinFlow : actinHistory) {
        xFlowSum += actinFlow[0];
        yFlowSum += actinFlow[1];
    }

    // Taking extent:
    double actinExtent{std::sqrt(std::pow(xFlowSum, 2) + std::pow(yFlowSum, 2))};
    return actinExtent;
};

std::vector<double> CellAgent::findTotalActinFlowComponents() const {
    // Averaging over history:
    std::vector<double> flowComponents{0., 0.};
    for (const auto & actinFlow : actinHistory) {
        flowComponents[0] += actinFlow[0];
        flowComponents[1] += actinFlow[1];
    }
    return flowComponents;
};

void CellAgent::ageActinHistory() {
    // Get iterator through history:
    std::list<std::vector<double>>::iterator historyIterator{actinHistory.begin()};

    // Looping through history, deleting actin flows that age:
    while (historyIterator != actinHistory.end()) {
        double uniformNumber{uniformDistribution(generatorInfluence)};
        if (uniformNumber < 0.1) {
            // Erasing element and updating iterator to position after deleted element:
            historyIterator = actinHistory.erase(historyIterator);
            continue;
        }
        // Advancing iterator:
        historyIterator++;
    }
}

void CellAgent::addToActinHistory(double actinFlowX, double actinFlowY) {
    // Ignore if cell has reached maximum number of actin flows:
    if (actinHistory.size() >= 10) {
        return;
    }

    // Add to vector otherwise:
    std::vector<double> actinFlow{actinFlowX, actinFlowY};
    actinHistory.emplace_back(actinFlow);
};

double CellAgent::findCellMovementMagnitude() {
    // Averaging over history:
    double xFlowSum{0}, yFlowSum{0};
    for (const auto & actinFlow : actinHistory) {
        xFlowSum += actinFlow[0];
        yFlowSum += actinFlow[1];
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

// Getting origin point of protrusion, need to do some rejection sampling for super-Gaussians:
// double cellAngularConcentration{
//     (maxCellAngularConcentration * polarityMagnitude) /
//     (halfSatCellAngularConcentration + polarityMagnitude)
// };
// double angleDelta = sampleVonMises(cellAngularConcentration);

// double originMinor{positionalNoiseDistribution(generatorPolarityNoiseX)*cellBodyRadius};
// double originMajor{positionalNoiseDistribution(generatorPolarityNoiseX)*cellBodyRadius};