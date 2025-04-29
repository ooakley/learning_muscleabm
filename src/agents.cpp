#include "agents.h"

#include <algorithm>
#include <complex>
#include <random>
#include <numbers>
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

    // Movement parameters:
    double setHalfSatCellAngularConcentration, double setMaxCellAngularConcentration,
    double setHalfSatMeanActinFlow, double setMaxMeanActinFlow,
    double setFlowScaling,

    // Polarisation system parameters:
    double setPolarityDiffusionRate,
    double setActinAdvectionRate,
    double setContactAdvectionRate,

    // Matrix sensation parameters:
    double setHalfSatMatrixAngularConcentration,
    double setMaxMatrixAngularConcentration,

    // Collision parameters:
    double setCellBodyRadius,
    double setAspectRatio,
    double setSharpness,
    double setInhibitionStrength,

    // Randomised initial state parameters:
    double startX, double startY, double startHeading,

    // Binary simulation parameters:
    bool setActinMagnitudeIsFixed,
    bool setActinDirectionIsFixed,
    bool setThereIsExtensionRepulsion,
    bool setCollisionsAreDeterministic,
    bool setMatrixAlignmentIsDeterministic
    )
    // Model infrastructure:
    : thereIsMatrixInteraction{setMatrixInteraction}
    , cellSeed{setCellSeed}
    , cellID{setCellID}

    // Movement parameters:
    , halfSatCellAngularConcentration{setHalfSatCellAngularConcentration}
    , maxCellAngularConcentration{setMaxCellAngularConcentration}
    , halfSatMeanActinFlow{setHalfSatMeanActinFlow}
    , maxMeanActinFlow{setMaxMeanActinFlow}
    , flowScaling{setFlowScaling}

    // Polarisation system parameters:
    , polarityDiffusionRate{setPolarityDiffusionRate*0.01}
    , actinAdvectionRate{setActinAdvectionRate*0.01}
    , contactAdvectionRate{setContactAdvectionRate*0.01}

    // Matrix sensation parameters:
    , halfSatMatrixAngularConcentration{setHalfSatMatrixAngularConcentration}
    , maxMatrixAngularConcentration{setMaxMatrixAngularConcentration}

    // Collision parameters:
    , cellBodyRadius{setCellBodyRadius}
    , cellAspectRatio{setAspectRatio}
    , contactDistributionSharpness{setSharpness}
    , inhibitionStrength{setInhibitionStrength}
    , majorAxisScaling{std::sqrt(setAspectRatio)}
    , minorAxisScaling{std::sqrt(1/setAspectRatio)}

    // Binary simulation parameters:
    , actinMagnitudeIsFixed{setActinMagnitudeIsFixed}
    , actinDirectionIsFixed{setActinDirectionIsFixed}
    , thereIsExtensionRepulsion{setThereIsExtensionRepulsion}
    , collisionsAreDeterministic{setCollisionsAreDeterministic}
    , matrixAlignmentIsDeterministic{setMatrixAlignmentIsDeterministic}

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
    , shapeDirection{startHeading}
    , movementDirection{0}
    , directionalInfluence{0}
    , directionalIntensity{0}
    , localECMDensity{0}
    , directionalShift{0}
    , sampledAngle{0}
    , polarityChangeCilX{0}
    , polarityChangeCilY{0}
{
    // Initialising randomness:
    seedGenerator = std::mt19937(cellSeed);
    seedDistribution = std::uniform_int_distribution<unsigned int>(0, UINT32_MAX);

    // Initialising influence selector:
    generatorInfluence = std::mt19937(seedDistribution(seedGenerator));

    // General Distributions:
    uniformDistribution = std::uniform_real_distribution<double>(0, 1);
    angleUniformDistribution = std::uniform_real_distribution<double>(-M_PI, M_PI);
    bernoulliDistribution = std::bernoulli_distribution();
    standardNormalDistribution = std::normal_distribution<double>(0, 1);

    // Generators for von Mises sampling:
    generatorU1 = std::mt19937(seedDistribution(seedGenerator));
    generatorU2 = std::mt19937(seedDistribution(seedGenerator));
    generatorB = std::mt19937(seedDistribution(seedGenerator));

    // Generators for collision shape sampling:
    generatorCollisionRadiusSampling = std::mt19937(seedDistribution(seedGenerator));
    generatorCollisionAngleSampling = std::mt19937(seedDistribution(seedGenerator));

    // Generators for matrix shape sampling:
    generatorMatrixRadiusSampling = std::mt19937(seedDistribution(seedGenerator));
    generatorMatrixAngleSampling = std::mt19937(seedDistribution(seedGenerator));

    // Generator for finding random angle after loss of polarisation:
    generatorRandomRepolarisation = std::mt19937(seedDistribution(seedGenerator));
    randomDeltaSample = std::mt19937(seedDistribution(seedGenerator));

    // Ensuring shape direction is direction-agnostic:
    shapeDirection = nematicAngleMod(shapeDirection);

    // Sampling for initial state of low-discrepancy angle distribution:
    lowDiscrepancySample = uniformDistribution(generatorCollisionAngleSampling);

    // Ensuring initial value in position history:
    addToPositionHistory(x, y);
}

// Public Definitions:

// Getters:
// Getters for values that shouldn't change:
double CellAgent::getID() const {return cellID;}

// Persistent variable getters: (these variables represent aspects of the cell's "memory")
double CellAgent::getX() const {return x;}
double CellAgent::getY() const {return y;}
std::tuple<double, double> CellAgent::getPosition() const {return std::tuple<double, double>{x, y};}
double CellAgent::getPolarityDirection() const {return polarityDirection;}
double CellAgent::getPolarityMagnitude() const {return polarityMagnitude;}
double CellAgent::getActinFlowMagnitude() const {return flowMagnitude;}
double CellAgent::getActinFlowDirection() const {return flowDirection;}
double CellAgent::getShapeDirection() const {return shapeDirection;}

// Instantaneous variable getters: (these variables are updated each timestep,
// and represent the cell's percepts)
double CellAgent::getMovementDirection() const {return movementDirection;}
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

void CellAgent::setCILPolarityChange(double changeX, double changeY) {
    polarityChangeCilX += changeX;
    polarityChangeCilY += changeY;
}

void CellAgent::setActinState(double setFlowDirection, double setFlowMagnitude) {
    flowDirection = setFlowDirection;
    flowMagnitude = setFlowMagnitude;
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
    movementDirection = determineMovementDirection();

    // Calculate actin flow in protrusion, via MM kinetics:
    double actinFlow{determineActinFlow()};

    // Update flow history:
    double actinChangeX{actinFlow * cos(movementDirection)};
    double actinChangeY{actinFlow * sin(movementDirection)};

    double newActinX{0.9*std::cos(flowDirection)*flowMagnitude + 0.1*actinChangeX};
    double newActinY{0.9*std::sin(flowDirection)*flowMagnitude + 0.1*actinChangeY};

    // Update current flow-derived values:
    // flowDirection = findTotalActinFlowDirection();
    // flowMagnitude = findTotalActinFlowMagnitude();
    // scaledFlowMagnitude = std::tanh(flowMagnitude);
    flowDirection = std::atan2(newActinY, newActinX);
    // flowDirection = 1;
    flowMagnitude = std::sqrt(std::pow(newActinX, 2) + std::pow(newActinY, 2));
    scaledFlowMagnitude = std::tanh(flowMagnitude);

    // // Seeing which protrusions die off this timestep:
    // ageActinHistory();

    // // Adding new protrusion if there's space:
    // addToActinHistory(actinX, actinY);

    // Accumulators for CIL-dependent polarity reorientation:
    // double polarityChangeCilX{0.};
    // double polarityChangeCilY{0.};

    // Run collision logic:
    // if (xPositionHistory.size() < 2) {
    //     runDeterministicCollisionLogic();
    // } else {
    //     runTrajectoryDependentCollisionLogic();
    // }

    // Determine collision with other cells:
    // for (int i; i < 3; i++) {
    //     runStochasticCollisionLogic();
    // }
    // runStochasticCollisionLogic();

    // // Determine collision with world boundaries:
    // bool xOutOfBounds{x < cellBodyRadius or x > 2048-cellBodyRadius};
    // bool yOutOfBounds{y < cellBodyRadius or y > 2048-cellBodyRadius};
    // if (xOutOfBounds or yOutOfBounds) {
    //     // Get actin flow:
    //     double xFlowComponent{std::cos(flowDirection)*flowMagnitude};
    //     double yFlowComponent{std::sin(flowDirection)*flowMagnitude};

    //     // Collide with wall:
    //     if (x < cellBodyRadius and xFlowComponent < 0) {
    //         xFlowComponent = 0;
    //         polarityX = 0;
    //         polarityChangeCilX = 0;
    //     }
    //     if (x > 2048-cellBodyRadius and xFlowComponent > 0) {
    //         xFlowComponent = 0;
    //         polarityX = 0;
    //         polarityChangeCilX = 0;
    //     }
    //     if (y < cellBodyRadius and yFlowComponent < 0) {
    //         yFlowComponent = 0;
    //         polarityY = 0;
    //         polarityChangeCilY = 0;
    //     }
    //     if (y > 2048-cellBodyRadius and yFlowComponent > 0) {
    //         yFlowComponent = 0;
    //         polarityY = 0;
    //         polarityChangeCilY = 0;
    //     }

    //     // Set acting cell actin flow to new values:
    //     flowDirection = std::atan2(yFlowComponent, xFlowComponent);
    //     flowMagnitude = std::sqrt(std::pow(xFlowComponent, 2) + std::pow(yFlowComponent, 2));
    // }

    // if (collisionsAreDeterministic) {
    //     
    // else {
    // }

    // Update polarity based on current actin flows:
    double rawPolarityMagnitude{std::sqrt(std::pow(polarityX, 2) + std::pow(polarityY, 2))};
    double rawMagnitudeSquared{std::pow(rawPolarityMagnitude, 2)};
    double flowComponentX{std::cos(flowDirection)*flowMagnitude};
    double flowComponentY{std::sin(flowDirection)*flowMagnitude};
    double xDiffusiveComponent{polarityDiffusionRate*rawMagnitudeSquared*std::cos(polarityDirection)};
    double yDiffusiveComponent{polarityDiffusionRate*rawMagnitudeSquared*std::sin(polarityDirection)};
    double xAdvectionComponent{actinAdvectionRate*flowComponentX + polarityChangeCilX};
    double yAdvectionComponent{actinAdvectionRate*flowComponentY + polarityChangeCilY};

    double dPolarityX{xAdvectionComponent - xDiffusiveComponent};
    double dPolarityY{yAdvectionComponent - yDiffusiveComponent};

    polarityX += dPolarityX;
    polarityY += dPolarityY;

    if (std::isnan(polarityX)) {
        std::cout << "rawPolarityMagnitude: " << rawPolarityMagnitude << std::endl;
        std::cout << "flowMagnitude: " << flowMagnitude << std::endl;
        std::cout << "xAdvectionComponent: " << xAdvectionComponent << std::endl;
        std::cout << "xDiffusiveComponent: " << xDiffusiveComponent << std::endl;
        std::cout << "dPolarityX: " << dPolarityX << std::endl;

        flowMagnitude = 0;
        polarityX = 0;
        polarityY = 0;
        // assert(!std::isnan(polarityX));
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
        polarityDirection = angleUniformDistribution(generatorRandomRepolarisation);
        polarityX = polarityMagnitude*std::cos(polarityDirection);
        polarityY = polarityMagnitude*std::sin(polarityDirection);
    }

    // Update position:
    double cellDisplacement{flowMagnitude*flowScaling};
    double dx{std::cos(flowDirection) * cellDisplacement};
    double dy{std::sin(flowDirection) * cellDisplacement};
    addToMovementHistory(std::cos(flowDirection), std::sin(flowDirection));
    x += dx;
    y += dy;

    // Roll position if out of bounds:
    while (x < 0) {x += 2048;}
    while (y < 0) {y += 2048;}
    x = std::fmodf(x, 2048);
    y = std::fmodf(y, 2048);
    addToPositionHistory(x, y);

    // Update shape direction:
    double shapeDelta{calculateShapeDeltaTowardsActin(shapeDirection, flowDirection)};
    // double shapeDirectionX{0.99*std::cos(shapeDirection) + 0.01*std::cos(shapeDirection + shapeDelta)};
    // double shapeDirectionY{0.99*std::sin(shapeDirection) + 0.01*std::sin(shapeDirection + shapeDelta)};
    double shapeDirectionX{0.95*std::cos(shapeDirection) + 0.05*flowMagnitude*std::cos(shapeDirection + shapeDelta)};
    double shapeDirectionY{0.95*std::sin(shapeDirection) + 0.05*flowMagnitude*std::sin(shapeDirection + shapeDelta)};
    shapeDirection = nematicAngleMod(std::atan2(shapeDirectionY, shapeDirectionX));
    // assert(shapeDirection >= 0);
    // assert(shapeDirection <= M_PI);

    // Zero out accumulators for CIL effects:
    polarityChangeCilX = polarityChangeCilX * 0.0;
    polarityChangeCilY = polarityChangeCilY * 0.0;
}


double CellAgent::determineMovementDirection() {
    // Instantiate eventual return value:
    double calculatedMovementDirection{0};
    // Calculate direction in simplest case of deterministic matrix alignment:
    if (matrixAlignmentIsDeterministic) {
        // Get random directional noise if present:
        double angleDelta{0};   
        if (!actinDirectionIsFixed) {
            // Get relevant parameters for distribution:
            double cellAngularConcentration{
                (maxCellAngularConcentration * polarityMagnitude) /
                (halfSatCellAngularConcentration + polarityMagnitude)
            };
            double matrixAngularConcentration{
                (maxMatrixAngularConcentration * directionalIntensity) /
                (halfSatMatrixAngularConcentration + directionalIntensity)
            };

            // Average across both:
            double averagedConcentration{(cellAngularConcentration + matrixAngularConcentration) / 2};
            angleDelta = sampleVonMises(averagedConcentration);
        }

        // Get averaged direction:
        calculatedMovementDirection = angleMod(polarityDirection + (directionalInfluence/2) + angleDelta);
    } else {
        // Get random directional noise if present:
        double angleDelta{0};

        // Calculate relative sampling rates of matrix and cell directions:
        double thresholdValue{polarityMagnitude / (polarityMagnitude + localECMDensity)};
        assert(thresholdValue <= 1 && thresholdValue >= 0);
        double randomDeterminant{uniformDistribution(generatorInfluence)};
        if (randomDeterminant < thresholdValue) {
            // Calculate polarity direction selection parameters using MM kinetics
            // - we're assuming directionality saturates at some degree of polarisation.
            if (!actinDirectionIsFixed) {
                double cellAngularConcentration{
                    (maxCellAngularConcentration * polarityMagnitude) /
                    (halfSatCellAngularConcentration + polarityMagnitude)
                };
                angleDelta = sampleVonMises(cellAngularConcentration);
            }
            calculatedMovementDirection = angleMod(polarityDirection + angleDelta);
        } else {
            // Calculate matrix direction selection parameters using MM kinetics
            // - we're assuming directionality saturates at some value of matrix
            // directional homogeneity.
            if (!actinDirectionIsFixed) {
                double matrixAngularConcentration{
                    (maxMatrixAngularConcentration * directionalIntensity) /
                    (halfSatMatrixAngularConcentration + directionalIntensity)
                };
                angleDelta = sampleVonMises(matrixAngularConcentration);
            }
            calculatedMovementDirection = angleMod(polarityDirection + directionalInfluence + angleDelta);
        }
    }
    return calculatedMovementDirection;
}


double CellAgent::determineActinFlow() {
    // Instantiate eventual return value:
    double calculatedActinFlow{0};
    
    // Calculated mean flow based on polarisation magnitude:
    double meanActinFlow{
        (maxMeanActinFlow * polarityMagnitude) /
        (halfSatMeanActinFlow + polarityMagnitude)
    };
    double clippedMeanActinFlow{std::max(meanActinFlow, 1.0)};

    // Return deterministic value if fixed:
    if (actinMagnitudeIsFixed) {
        return clippedMeanActinFlow;
    }
    // Return sampled value from poisson distribution otherwise:
    else {
        std::poisson_distribution<int> protrusionDistribution(clippedMeanActinFlow);
        return static_cast<double>(protrusionDistribution(generatorProtrusion));
    }
}


void CellAgent::runTrajectoryDependentCollisionLogic() {
    // Useful points:
    double actingCellX{getX()};
    double actingCellY{getY()};

    // Sampling from random radius in cell area:
    lowDiscrepancySample += (std::numbers::phi_v<double> - 1);
    lowDiscrepancySample = std::fmod(lowDiscrepancySample, 1);
    double divergence{(lowDiscrepancySample * M_PI) - M_PI_2};
    double samplePointDirection{flowDirection + divergence};

    // Getting point in frame:
    double actingFrameX{std::cos(samplePointDirection) * cellBodyRadius};
    double actingFrameY{std::sin(samplePointDirection) * cellBodyRadius};
    
    // Transforming points from acting frame to global frame:
    double globalFrameX{actingCellX + actingFrameX};
    double globalFrameY{actingCellY + actingFrameY};
    double angleFromActingToSample{std::atan2(actingFrameY, actingFrameX)};

    // Loop through local agents and determine collisions:
    for (auto& localAgent: localAgents) {
        const auto& [startX, startY, endX, endY] = localAgent->sampleTrajectoryStadium();

        // Find point most relevant for shift across periodic boundary:
        double distanceToStart{std::sqrt(
            std::pow(startX - globalFrameX, 2) +
            std::pow(startY - globalFrameY, 2)
        )};
        double distanceToEnd{std::sqrt(
            std::pow(endX - globalFrameX, 2) +
            std::pow(endY - globalFrameY, 2)
        )};

        // Determine whether any or all of the trajectory points will take the modulus:
        double correctedStartX{0};
        double correctedStartY{0};
        double correctedEndX{0};
        double correctedEndY{0};

        // Take modulus of all:
        if (distanceToStart > 2048/2 and distanceToEnd > 2048/2) {
            correctedStartX = takePeriodicModulus(startX, globalFrameX);
            correctedStartY = takePeriodicModulus(startY, globalFrameY);
            correctedEndX = takePeriodicModulus(endX, globalFrameX);
            correctedEndY = takePeriodicModulus(endY, globalFrameY);
        }

        // Take modulus of one:
        else {
            if (distanceToStart < distanceToEnd) {
                // Take modulus relative to start:
                correctedStartX = startX;
                correctedStartY = startY;
                correctedEndX = takePeriodicModulus(endX, startX);
                correctedEndY = takePeriodicModulus(endY, startY);
            } else {
                // Take modulus relative to end:
                correctedStartX =  takePeriodicModulus(startX, endX);
                correctedStartY = takePeriodicModulus(startY, endY);
                correctedEndX = endX;
                correctedEndY = endY;
            }
        }

        // Determine whether collision occurs:
        bool collisionDetected{
            isPositionInStadium(
                globalFrameX, globalFrameY, correctedStartX, correctedStartY, correctedEndX, correctedEndY
            )
        };
        if (collisionDetected) {
            // Take modulo of position in case interaction is across the periodic boundary:
            double localCellX{takePeriodicModulus(localAgent->getX(), globalFrameX)};
            double localCellY{takePeriodicModulus(localAgent->getY(), globalFrameY)};

            // Get angle to local cell:
            double actingToLocalX{localCellX - actingCellX};
            double actingToLocalY{localCellY - actingCellY};
            double angleActingToLocal{std::atan2(actingToLocalY, actingToLocalX)};

            // Exert reduction in actin flow for acting cell:
            double angleOfRestitution{angleActingToLocal - M_PI};
            double componentOfActingFlowOntoCollision{
                std::cos(flowDirection - angleOfRestitution)
            };
            if (componentOfActingFlowOntoCollision < 0) {
                // Calculate change in actin flow:
                double reductionInFlow{
                    inhibitionStrength * flowMagnitude * std::abs(componentOfActingFlowOntoCollision)
                };
                double dxActinFlow{std::cos(angleOfRestitution) * reductionInFlow};
                double dyActinFlow{std::sin(angleOfRestitution) * reductionInFlow};

                // Update actin flow:
                double xFlowComponent{std::cos(flowDirection)*flowMagnitude};
                double yFlowComponent{std::sin(flowDirection)*flowMagnitude};
                xFlowComponent += dxActinFlow;
                yFlowComponent += dyActinFlow;

                // Set acting cell actin flow to new values:
                flowDirection = std::atan2(yFlowComponent, xFlowComponent);
                flowMagnitude = std::sqrt(std::pow(xFlowComponent, 2) + std::pow(yFlowComponent, 2));
            }

            // Exert reduction in actin flow for acting cell:
            double localFlowDirection{localAgent->getActinFlowDirection()};
            double localFlowMagnitude{localAgent->getActinFlowMagnitude()};
            double componentOfLocalFlowOntoCollision{
                std::cos(localFlowDirection - angleActingToLocal)
            };
            if (componentOfLocalFlowOntoCollision <= 0) {
                // Calculate change in actin flow:
                double reductionInFlow{inhibitionStrength * localFlowMagnitude * std::abs(componentOfLocalFlowOntoCollision)};
                double dxActinFlow{std::cos(angleActingToLocal) * reductionInFlow};
                double dyActinFlow{std::sin(angleActingToLocal) * reductionInFlow};

                // Update actin flow:
                double xFlowComponent{std::cos(localFlowDirection)*localFlowMagnitude};
                double yFlowComponent{std::sin(localFlowDirection)*localFlowMagnitude};
                xFlowComponent += dxActinFlow;
                yFlowComponent += dyActinFlow;
                double updatedLocalFlowDirection{std::atan2(yFlowComponent, xFlowComponent)};
                double updatedLocalFlowMagnitude{std::sqrt(std::pow(xFlowComponent, 2) + std::pow(yFlowComponent, 2))};

                // Set local cell actin flow to new values:
                localAgent->setActinState(updatedLocalFlowDirection, updatedLocalFlowMagnitude);
            }

            // Calculate CIL effect for acting cell:
            double actingRepulsionX{std::cos(angleOfRestitution)};
            double actingRepulsionY{std::sin(angleOfRestitution)};
            polarityChangeCilX += actingRepulsionX * contactAdvectionRate;
            polarityChangeCilY += actingRepulsionY * contactAdvectionRate;

            // Calculate CIL effect for local cell:
            double localRepulsionX{std::cos(angleActingToLocal)};
            double localRepulsionY{std::sin(angleActingToLocal)};
            double localCILUpdateX{localRepulsionX * contactAdvectionRate};
            double localCILUpdateY{localRepulsionY * contactAdvectionRate};
            localAgent->setCILPolarityChange(localCILUpdateX, localCILUpdateY);
        }
    }
}


void CellAgent::runStochasticCollisionLogic() {
    // Useful points:
    double actingCellX{getX()};
    double actingCellY{getY()};

    double actinDirectionActingFrame{flowDirection - shapeDirection};

    // Sampling from random radius in cell area:
    lowDiscrepancySample += (std::numbers::phi_v<double> - 1);
    lowDiscrepancySample = std::fmod(lowDiscrepancySample, 1);
    double divergence{(lowDiscrepancySample * M_PI) - M_PI_2};
    double samplePointDirection{actinDirectionActingFrame + divergence};

    // Derive relevant ellipse variables:
    // Getting R from polar form of ellipse equation:
    double eccentricity{std::sqrt(1 - ((std::pow(minorAxisScaling, 2))/(std::pow(majorAxisScaling, 2))))};
    double radialDenominator{std::sqrt(1 - std::pow(eccentricity*std::cos(samplePointDirection), 2))};
    double radialCoordinate{minorAxisScaling / (radialDenominator)};
    double unitEllipseX{radialCoordinate * std::cos(samplePointDirection)};
    double unitEllipseY{radialCoordinate * std::sin(samplePointDirection)};

    // // Scaling origin points by eccentricity:
    // Scaling axes to maintain constant cell area:
    double actingFrameX{unitEllipseX * cellBodyRadius * majorAxisScaling};
    double actingFrameY{unitEllipseY * cellBodyRadius * minorAxisScaling};

    // Transforming points from acting frame to global frame:
    double rotatingBasisAngle{shapeDirection};
    double transformedX{
        (actingFrameX*std::cos(rotatingBasisAngle)) -
        (actingFrameY*std::sin(rotatingBasisAngle))
    };
    double transformedY{
        (actingFrameX*std::sin(rotatingBasisAngle)) +
        (actingFrameY*std::cos(rotatingBasisAngle))
    };
    double angleFromActingToSample{std::atan2(transformedY, transformedX)};

    // Find sample point in global coordinates:
    double sampleGlobalX{actingCellX + transformedX};
    double sampleGlobalY{actingCellY + transformedY};
    double sampleDistanceFromActing{std::sqrt(
        std::pow(unitEllipseX, 2) +
        std::pow(unitEllipseY, 2)
    )};

    // Find closest agent to query:
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
        double localBasisAngle{-localAgent->getShapeDirection()};
        double localEllipseX{
            (localFrameX*std::cos(localBasisAngle)) -
            (localFrameY*std::sin(localBasisAngle))
        };
        double localEllipseY{
            (localFrameX*std::sin(localBasisAngle)) +
            (localFrameY*std::cos(localBasisAngle))
        };

        // Scale components to unit circle:
        double unitEllipseFrameX{localEllipseX/cellBodyRadius};
        double unitEllipseFrameY{localEllipseY/cellBodyRadius};
        double unitCircleX{std::copysign(
            std::abs(unitEllipseFrameX/majorAxisScaling),
            unitEllipseFrameX
        )};
        double unitCircleY{std::copysign(
            std::abs(unitEllipseFrameY/minorAxisScaling),
            unitEllipseFrameY
        )};
        double unitCircleDistance{std::sqrt(
            std::pow(unitCircleX, 2) +
            std::pow(unitCircleY, 2)
        )};

        // If point lies inside another cell; indicate by negative distance:
        if (unitCircleDistance < 1) {
            distancesToProtrusion.emplace_back(unitCircleDistance - 1);
            continue;
        }

        // If point lies outside another cell, we need to estimate euclidean distance to boundary:
        double boundaryAngle{std::atan2(unitCircleY, unitCircleX)};
        double boundaryCircleX{std::cos(boundaryAngle)};
        double boundaryCircleY{std::sin(boundaryAngle)};
        double pointOnEllipseX{std::copysign(
            std::abs(boundaryCircleX),
            boundaryCircleX
        )};
        double pointOnEllipseY{std::copysign(
            std::abs(boundaryCircleY),
            boundaryCircleY
        )};
        double distanceToBoundaryX{unitEllipseFrameX-pointOnEllipseX};
        double distanceToBoundaryY{unitEllipseFrameY-pointOnEllipseY};
        double sampleDistance{std::sqrt(
            std::pow(distanceToBoundaryX, 2) +
            std::pow(distanceToBoundaryY, 2)
        )};

        // Adding to total distances vector:
        distancesToProtrusion.emplace_back(sampleDistance);
    }

    // Finding minimum distance:
    auto minimumIterator{std::min_element(distancesToProtrusion.begin(), distancesToProtrusion.end())};
    double minimumDistance{0};
    if (minimumIterator == distancesToProtrusion.end()) {
        // If no cells in local area, we have to prevent possible dereferencing of end iterator:
        minimumDistance = 2048;
    } else {
        minimumDistance = *minimumIterator;
    }

    // We ignore the collision if the sample point is closest to the currently acting cell:
    // minimumDistance < sampleDistanceFromActing*2
    if (minimumDistance < 2048) {
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

        // Get angle from local cell to sampled position:
        double localFrameX{sampleGlobalX - localCellX};
        double localFrameY{sampleGlobalY - localCellY};
        double angleFromLocalToSample{std::atan2(localFrameY, localFrameX)};

        // Get angle to nucleus:
        double xDifference{actingCellX - localCellX};
        double yDifference{actingCellY - localCellY};
        double angleLocalToActing{std::atan2(yDifference, xDifference)};
        double nuclearDistance{std::sqrt(
            std::pow(xDifference, 2) + 
            std::pow(yDifference, 2)
        )};

        // Find intersection of shape directions:
        double actingShapeDirection{shapeDirection};
        double localShapeDirection{localAgents[cellIndex]->getShapeDirection()};

        // Get acting line in projective coordinates:
        std::vector<double> actingStart{actingCellX, actingCellY, 1};
        std::vector<double> actingEnd{
            actingCellX + std::cos(actingShapeDirection),
            actingCellY + std::sin(actingShapeDirection),
            1
        };
        std::vector<double> actingLine{crossProduct(actingStart, actingEnd)};

        // Get local line in projective coordinates:
        std::vector<double> localStart{localCellX, localCellY, 1};
        std::vector<double> localEnd{
            localCellX + std::cos(localShapeDirection),
            localCellY + std::sin(localShapeDirection),
            1
        };
        std::vector<double> localLine{crossProduct(localStart, localEnd)};

        // Get point of intersection:
        std::vector<double> projectiveIntersection{crossProduct(actingLine, localLine)};

        // Find probability of collision:
        double finalProbability{0};
        if (minimumDistance < 0) {
            finalProbability = 1;
        } else {
            // double exponentTerm{std::pow(std::pow(minimumDistance, 2)/2, contactDistributionSharpness)};
            finalProbability = std::exp(-minimumDistance*contactDistributionSharpness);
        }

        // Determine if collision happens - the shape directions of the two cells cannot be parallel:
        double randomDeterminant{uniformDistribution(generatorInfluence)};
        if ((randomDeterminant < finalProbability) and (projectiveIntersection[2] != 0)) {
            // Get point of intersection:
            double intersectionX{projectiveIntersection[0]/projectiveIntersection[2]};
            double intersectionY{projectiveIntersection[1]/projectiveIntersection[2]};

            // Get acting shape direction pointing towards intersection:
            double collisionActingX{intersectionX - actingCellX};
            double collisionActingY{intersectionY - actingCellY};
            double collisionActingDirection{std::atan2(collisionActingY, collisionActingX)};
            double collisionActingDistance{std::sqrt(
                std::pow(collisionActingX, 2) +
                std::pow(collisionActingY, 2)
            )};

            // Get local shape direction pointing towards intersection:
            double collisionLocalX{intersectionX - localCellX};
            double collisionLocalY{intersectionY - localCellY};
            double collisionLocalDirection{std::atan2(collisionLocalY, collisionLocalX)};
            double collisionLocalDistance{std::sqrt(
                std::pow(collisionLocalX, 2) +
                std::pow(collisionLocalY, 2)
            )};

            // Find axis along which to reduce actin flow for acting cell:
            double actingEffectX{std::cos(collisionLocalDirection) - std::cos(collisionActingDirection)};
            double actingEffectY{std::sin(collisionLocalDirection) - std::sin(collisionActingDirection)};
            double actingExtensionEffectDirection{std::atan2(actingEffectY, actingEffectX)};
            double actingBodyEffectDirection{angleLocalToActing};
            double overallActingEffectDirection{std::atan2(
                std::sin(actingExtensionEffectDirection) + std::sin(actingBodyEffectDirection),
                std::cos(actingExtensionEffectDirection) + std::cos(actingBodyEffectDirection)
            )};
            double componentOfActingFlowOntoSample{
                std::cos(flowDirection - overallActingEffectDirection)
            };
            if (componentOfActingFlowOntoSample < 0) {
                // Calculate change in actin flow:
                double reductionInFlow{inhibitionStrength * flowMagnitude * std::abs(componentOfActingFlowOntoSample)};
                double dxActinFlow{std::cos(overallActingEffectDirection) * reductionInFlow};
                double dyActinFlow{std::sin(overallActingEffectDirection) * reductionInFlow};

                // Update actin flow:
                double xFlowComponent{std::cos(flowDirection)*flowMagnitude};
                double yFlowComponent{std::sin(flowDirection)*flowMagnitude};
                xFlowComponent += dxActinFlow;
                yFlowComponent += dyActinFlow;

                // Set acting cell actin flow to new values:
                flowDirection = std::atan2(yFlowComponent, xFlowComponent);
                flowMagnitude = std::sqrt(std::pow(xFlowComponent, 2) + std::pow(yFlowComponent, 2));

                // Flat reduction in magnitude:
                // flowMagnitude *= inhibitionStrength;
            }

            // Find axis along which to reduce actin flow for local cell:
            double localFlowDirection{localAgents[cellIndex]->getActinFlowDirection()};
            double localFlowMagnitude{localAgents[cellIndex]->getActinFlowMagnitude()};
            double localExtensionEffectDirection{actingExtensionEffectDirection - M_PI};
            double localBodyEffectDirection{actingBodyEffectDirection - M_PI};
            double overallLocalEffectDirection{std::atan2(
                std::sin(localExtensionEffectDirection) + std::sin(localBodyEffectDirection),
                std::cos(localExtensionEffectDirection) + std::cos(localBodyEffectDirection)
            )};
            double componentOfLocalFlowOntoSample{
                std::cos(localFlowDirection - overallLocalEffectDirection)
            };
            if (componentOfLocalFlowOntoSample <= 0) {
                // Calculate change in actin flow:
                double reductionInFlow{inhibitionStrength * localFlowMagnitude * std::abs(componentOfLocalFlowOntoSample)};
                double dxActinFlow{std::cos(overallLocalEffectDirection) * reductionInFlow};
                double dyActinFlow{std::sin(overallLocalEffectDirection) * reductionInFlow};

                // Update actin flow:
                double xFlowComponent{std::cos(localFlowDirection)*localFlowMagnitude};
                double yFlowComponent{std::sin(localFlowDirection)*localFlowMagnitude};
                xFlowComponent += dxActinFlow;
                yFlowComponent += dyActinFlow;
                double updatedLocalFlowDirection{std::atan2(yFlowComponent, xFlowComponent)};
                double updatedLocalFlowMagnitude{std::sqrt(std::pow(xFlowComponent, 2) + std::pow(yFlowComponent, 2))};
                // updatedLocalFlowMagnitude *= inhibitionStrength;

                // Set local cell actin flow to new values:
                localAgents[cellIndex]->setActinState(updatedLocalFlowDirection, updatedLocalFlowMagnitude);
            }

            // Calculating scaling of body repulsion:
            double bodyScaling{std::min((2*cellBodyRadius)/nuclearDistance, 100.0)};

            // Simulate effect of CIL on RhoA redistribution for acting cell:
            double actingExtensionScaling{std::min((cellBodyRadius*majorAxisScaling)/collisionActingDistance, 100.0)};
            double actingBodyRepulsionX{std::cos(angleLocalToActing)*bodyScaling};
            double actingBodyRepulsionY{std::sin(angleLocalToActing)*bodyScaling};
            double actingExtensionRepulsionX{
                std::cos(actingExtensionEffectDirection)*actingExtensionScaling
            };
            double actingExtensionRepulsionY{
                std::sin(actingExtensionEffectDirection)*actingExtensionScaling
            };
            polarityChangeCilX += (actingExtensionRepulsionX + actingBodyRepulsionX) * contactAdvectionRate;
            polarityChangeCilY += (actingExtensionRepulsionY + actingBodyRepulsionY) * contactAdvectionRate;

            // Simulate effect of CIL on RhoA redistribution for local cell:
            double localExtensionScaling{std::min((cellBodyRadius*majorAxisScaling)/collisionLocalDistance, 100.0)};
            double localBodyRepulsionX{std::cos(angleLocalToActing - M_PI)*bodyScaling};
            double localBodyRepulsionY{std::sin(angleLocalToActing - M_PI)*bodyScaling};
            double localExtensionRepulsionX{
                std::cos(localExtensionEffectDirection)*localExtensionScaling
            };
            double localExtensionRepulsionY{
                std::sin(localExtensionEffectDirection)*localExtensionScaling
            };
            double localCILx{(localExtensionRepulsionX + localBodyRepulsionX) * contactAdvectionRate};
            double localCILy{(localExtensionRepulsionY + localBodyRepulsionY) * contactAdvectionRate};
            localAgents[cellIndex]->setCILPolarityChange(localCILx, localCILy);
        }
    }
};


void CellAgent::runDeterministicCollisionLogic() {
    // Define acting cell position vars for safety (don't want to accidentally change x/y):
    double actingCellX{getX()};
    double actingCellY{getY()};

    // Loop through all local agents:
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

        // Getting total displacement and heading to local cell:
        double actingToLocalX{localCellX - actingCellX};
        double actingToLocalY{localCellY - actingCellY};
        double angleActingToLocal{std::atan2(actingToLocalY, actingToLocalX)};
        double displacementActingToLocal{std::sqrt(
            std::pow(actingToLocalX, 2) + 
            std::pow(actingToLocalY, 2)
        )};

        // Collide if within radius:
        if (displacementActingToLocal < (2 * cellBodyRadius)) {
            // Exert reduction in actin flow:
            double angleOfRestitution{angleActingToLocal - M_PI};
            double componentOfActingFlowOntoCollision{
                std::cos(flowDirection - angleOfRestitution)
            };
            if (componentOfActingFlowOntoCollision < 0) {
                // Calculate change in actin flow:
                double reductionInFlow{
                    inhibitionStrength * flowMagnitude * std::abs(componentOfActingFlowOntoCollision)
                };
                double dxActinFlow{std::cos(angleOfRestitution) * reductionInFlow};
                double dyActinFlow{std::sin(angleOfRestitution) * reductionInFlow};

                // Update actin flow:
                double xFlowComponent{std::cos(flowDirection)*flowMagnitude};
                double yFlowComponent{std::sin(flowDirection)*flowMagnitude};
                xFlowComponent += dxActinFlow;
                yFlowComponent += dyActinFlow;

                // Set acting cell actin flow to new values:
                flowDirection = std::atan2(yFlowComponent, xFlowComponent);
                flowMagnitude = std::sqrt(std::pow(xFlowComponent, 2) + std::pow(yFlowComponent, 2));
            }

            // Calculate CIL effect:
            // double bodyScaling{(2*cellBodyRadius) - displacementActingToLocal};
            double actingBodyRepulsionX{std::cos(angleOfRestitution)};
            double actingBodyRepulsionY{std::sin(angleOfRestitution)};
            polarityChangeCilX += actingBodyRepulsionX * contactAdvectionRate;
            polarityChangeCilY += actingBodyRepulsionY * contactAdvectionRate;
        }

    }
}


std::vector<double> CellAgent::sampleAttachmentPoint() {
    // Getting current position:
    double actingCellX{getX()};
    double actingCellY{getY()};

    // Sampling from random radius in cell area:
    double divergence{(lowDiscrepancySample * M_PI) - M_PI_2};
    double samplePointDirection{flowDirection + divergence};

    // Getting point in frame:
    double actingFrameX{std::cos(samplePointDirection) * cellBodyRadius};
    double actingFrameY{std::sin(samplePointDirection) * cellBodyRadius};
    
    // Transforming points from acting frame to global frame:
    double globalFrameX{actingCellX + actingFrameX};
    double globalFrameY{actingCellY + actingFrameY};

    return {globalFrameX, globalFrameY};
};


std::tuple<double, double, double, double> CellAgent::sampleTrajectoryStadium() {
    // Weight sampling to more recent points in trajectory:
    int currentLength{static_cast<int>(xPositionHistory.size())};
    std::vector<double> probabilityWeights(currentLength);
    // std::iota(probabilityWeights.begin(), probabilityWeights.end(), 1);
    std::fill(probabilityWeights.begin(), probabilityWeights.end(), 1);

    // Sample trajectory indices:
    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    std::discrete_distribution<int> indexDistribution(
        probabilityWeights.begin(), probabilityWeights.end()
    );
    int indexA{indexDistribution(generator)};
    int indexB{indexDistribution(generator)};

    // Return positions at these points:
    return {
        xPositionHistory[indexA], yPositionHistory[indexA],
        xPositionHistory[indexB], yPositionHistory[indexB]
    };
}


bool CellAgent::isPositionInStadium(
    double samplePointX, double samplePointY,
    double startX, double startY,
    double endX, double endY
) {
    // Get intermediate calculations:
    double xStartToSample{samplePointX - startX};
    double yStartToSample{samplePointY - startY};
    double xStartToEnd{endX - startX};
    double yStartToEnd{endY - startY};

    // Get scaled dot product:
    double scaledDotProduct{xStartToSample*xStartToEnd + yStartToSample*yStartToEnd};
    scaledDotProduct /= std::pow(xStartToEnd, 2) + std::pow(yStartToEnd, 2);

    // Determine closest point on segment:
    double closestPointX{0};
    double closestPointY{0};
    if (scaledDotProduct < 0) {
        closestPointX = startX;
        closestPointY = startY;
    } else if (scaledDotProduct > 1) {
        closestPointX = endX;
        closestPointY = endY;
    } else {
        closestPointX = startX + scaledDotProduct*xStartToEnd;
        closestPointY = startY + scaledDotProduct*yStartToEnd;
    }

    // Find distance to closest point:
    double minimumDistance{std::sqrt(
        std::pow(samplePointX - closestPointX, 2) +
        std::pow(samplePointY - closestPointY, 2)
    )};

    return minimumDistance < cellBodyRadius;
}


// // Private functions for cell behaviour:
double CellAgent::sampleVonMises(double kappa) {
    // Von Mises sampling quickly overloaded if kappa is very low - can set a minimum and take a
    // uniform distribution instead:
    if (kappa < 1e-3) {
        return angleUniformDistribution(randomDeltaSample);
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


double CellAgent::angleMod(double angle) const {
    while (angle < -M_PI) {angle += 2*M_PI;};
    while (angle >= M_PI) {angle -= 2*M_PI;};
    return angle;
}


double CellAgent::nematicAngleMod(double angle) const {
    while (angle < 0) {angle += M_PI;};
    while (angle >= M_PI) {angle -= M_PI;};
    return angle;
}

double CellAgent::takePeriodicModulus(double queryPosition, double localPosition) {
    // Find and apply relevant modulus:
    double modulusPosition{queryPosition};
    if (localPosition - queryPosition > (2048 / 2)) {
        modulusPosition += 2048;
    }
    else if (localPosition - queryPosition < -(2048 / 2)) {
        modulusPosition -= 2048;
    }

    return modulusPosition;
};

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

double CellAgent::calculateShapeDeltaTowardsActin(double shapeHeading, double actinHeading) const {
    // Ensuring input values are in the correct range:
    assert((shapeHeading >= 0) & (shapeHeading <= M_PI));
    assert((actinHeading >= -M_PI) & (actinHeading <= M_PI));

    // Calculating change in theta (Shape is direction agnostic so we have to reverse it):
    double deltaHeading{actinHeading - shapeHeading};
    while (deltaHeading <= -M_PI) {deltaHeading += M_PI;}

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
    }
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
    // Average over history:
    double xFlowSum{0}, yFlowSum{0};
    for (const auto & actinFlow : actinHistory) {
        xFlowSum += actinFlow[0];
        yFlowSum += actinFlow[1];
    }

    // Scale flow to movement:
    xFlowSum *= flowScaling;
    yFlowSum *= flowScaling;

    // Take extent:
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

void CellAgent::addToPositionHistory(double positionX, double positionY) {
    xPositionHistory.push_back(positionX);
    yPositionHistory.push_back(positionY);
    if (yPositionHistory.size() > 100) {
        xPositionHistory.pop_front();
        yPositionHistory.pop_front();
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

// double CellAgent::findShapeDirection() {
//     // Averaging over history:
//     double xMovementSum{0}, yMovementSum{0};
//     for (double xMovement : xMovementHistory) {
//         xMovementSum += xMovement;
//     }
//     for (double yMovement : yMovementHistory) {
//         yMovementSum += yMovement;
//     }

//     // Finding total path length:
//     double shapeDirection{std::atan2(yMovementSum, xMovementSum)};
//     return shapeDirection;
// }

std::vector<double> CellAgent::crossProduct(std::vector<double> const a, std::vector<double> const b) {
    // Basic cross product calculation:
    std::vector<double> resultVector(3);  
    resultVector[0] = a[1]*b[2] - a[2]*b[1];
    resultVector[1] = a[2]*b[0] - a[0]*b[2];
    resultVector[2] = a[0]*b[1] - a[1]*b[0];
    return resultVector;
}