#include "agents.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

// Constructor:
CellAgent::CellAgent(
    // Defined behaviour parameters:
    unsigned int setCellSeed, int setCellID,
    double setWorldSize,
    double setdt,

    // Movement parameters:
    double setCueDiffusionRate,
    double setCueKa,
    double setFluctuationAmplitude,
    double setFluctuationTimescale,
    double setActinAdvectionRate,
    double setCollisionAdvectionRate,
    double setMaximumSteadyStateActinFlow,

    // Collision parameters:
    double setCellBodyRadius,
    double setCollisionFlowReductionRate,
    double setAdhesionReductionRate,

    // Shape parameters:
    double setCellStiffness,
    double setSurfaceStickiness,

    // Matrix parameters:
    double setMatrixCoupling,

    // Randomised initial state parameters:
    double startX, double startY, double startHeading
    )
    // Model infrastructure:
    : cellID{setCellID}
    , worldSize{setWorldSize}
    , dt{setdt}

    // Movement parameters:
    , cueDiffusionRate{setCueDiffusionRate}
    , cueKa{setCueKa}
    , fluctuationAmplitude{setFluctuationAmplitude}
    , fluctuationTimescale{setFluctuationTimescale}
    , actinAdvectionRate{setActinAdvectionRate}
    , collisionAdvectionRate{setCollisionAdvectionRate}
    , maximumSteadyStateActinFlow{setMaximumSteadyStateActinFlow}

    // Collision parameters:
    , cellBodyRadius{setCellBodyRadius}
    , collisionFlowReductionRate{setCollisionFlowReductionRate}
    , adhesionReductionRate{setAdhesionReductionRate}

    // Shape parameters:
    , cellStiffness{setCellStiffness}
    , surfaceStickiness{setSurfaceStickiness}
    , adhesionStiffness{333.3} // 1000 nN/pix
    , adhesionFragility{1000} // 1000 nN

    // Matrix parameters:
    , matrixCoupling{setMatrixCoupling}

    // State parameters:
    , x{startX}
    , y{startY}
    , stadiumX{startX}
    , stadiumY{startY}
    , polarityDirection{startHeading}
    , polarityMagnitude{1e-5}
    , flowDirection{startHeading}
    , flowMagnitude{1e-5}
    , shapeDirection{startHeading}
    , adhesionFraction{1e-3}
    , effectiveRadius{setCellBodyRadius}
    , polarityChangeCilX{0}
    , polarityChangeCilY{0}
    , movementDirection{0}
    , directionalShift{0}
    , sampledAngle{0}
    , directionalInfluence{0}
    , directionalIntensity{0}

    // History variables:
    , collisionsThisTimepoint{0}
    , finalCILEffectX{0}
    , finalCILEffectY{0}
{
    // Seed each generator with a successive draw from a generator seeded by the cell seed:
    std::mt19937 seedGenerator(setCellSeed);
    std::uniform_int_distribution<unsigned int> seedDistribution(0, UINT32_MAX);
    actinFlowGenerator = std::mt19937(seedDistribution(seedGenerator));
    attachmentPointGenerator = std::mt19937(seedDistribution(seedGenerator));

    // General Distributions:
    uniformDistribution = std::uniform_real_distribution<double>(0, 1);
    angleUniformDistribution = std::uniform_real_distribution<double>(-M_PI, M_PI);
    standardNormalDistribution = std::normal_distribution<double>(0, 1);

    // Ensuring shape direction is direction-agnostic:
    shapeDirection = nematicAngleMod(shapeDirection);
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
double CellAgent::getDirectionalInfluence() const {return directionalInfluence;}
double CellAgent::getDirectionalIntensity() const {return directionalIntensity;}
double CellAgent::getDirectionalShift() const {return directionalShift;}
double CellAgent::getSampledAngle() const {return sampledAngle;}

int CellAgent::getCollisionNumber() const {return collisionsThisTimepoint;}
double CellAgent::getTotalCILEffectX() const {return finalCILEffectX;}
double CellAgent::getTotalCILEffectY() const {return finalCILEffectY;}
double CellAgent::getStadiumX() const {return stadiumX;}
double CellAgent::getStadiumY() const {return stadiumY;}
double CellAgent::getEffectiveRadius() const {return effectiveRadius;}

// Setters:
void CellAgent::setPosition(std::tuple<double, double> newPosition) {
    x = std::get<0>(newPosition);
    y = std::get<1>(newPosition);
}

void CellAgent::setCILPolarityChange(double changeX, double changeY) {
    polarityChangeCilX += changeX;
    polarityChangeCilY += changeY;
}

void CellAgent::setDirectionalInfluence(double setDirectionalInfluence) {
    directionalInfluence = setDirectionalInfluence;
};

void CellAgent::setDirectionalIntensity(double setDirectiontalIntensity) {
    directionalIntensity = setDirectiontalIntensity;
};

void CellAgent::setLocalCellList(const std::vector<CellAgent*>& setLocalAgents) {
    localAgents = setLocalAgents;
}


// Simulation code:
void CellAgent::takeRandomStep() {
    assert(directionalIntensity <= 1);

    // Collide with adjacent cells:
    collisionsThisTimepoint = 0;

    // Determine advection derived from actin flow:
    double actinAdvectionX{flowMagnitude*std::cos(flowDirection)};
    double actinAdvectionY{flowMagnitude*std::sin(flowDirection)};

    // Sum sources of advection:
    double totalAdvectionX{
        actinAdvectionRate*actinAdvectionX +
        collisionAdvectionRate*polarityChangeCilX
    };
    double totalAdvectionY{
        actinAdvectionRate*actinAdvectionY +
        collisionAdvectionRate*polarityChangeCilY
    };

    double totalAdvectionMagnitude{std::sqrt(
        std::pow(totalAdvectionX, 2) +
        std::pow(totalAdvectionY, 2)
    )};
    double totalAdvectionDirection{std::atan2(totalAdvectionY, totalAdvectionX)};

    // Zero out CIL effects:
    polarityChangeCilX = 0.0;
    polarityChangeCilY = 0.0;

    // Get solution to cue concentration profile at cell front and cell back:
    double scaledAdvectionMagnitude{totalAdvectionMagnitude / (2 * cellBodyRadius)};
    double exponentialTerm{std::exp(-scaledAdvectionMagnitude / cueDiffusionRate)};

    // Without advection the profile is flat, so the front and back are equally active (the limit
    // of the expressions below as advection goes to 0, where they would divide zero by zero):
    double effectiveActinPolarisation{0};
    if (exponentialTerm < 1) {
        double cueConcentrationFront{
            scaledAdvectionMagnitude /
            (cueDiffusionRate*(1 - exponentialTerm))
        };
        double cueConcentrationBack{
            scaledAdvectionMagnitude*exponentialTerm /
            (cueDiffusionRate*(1 - exponentialTerm))
        };

        // Run concentration through Hill equation and find front/back activity differential:
        double cueActivityFront{cueConcentrationFront / (cueKa + cueConcentrationFront)};
        double cueActivityBack{cueConcentrationBack / (cueKa + cueConcentrationBack)};
        effectiveActinPolarisation = cueActivityFront - cueActivityBack;
    }

    // Correct for small advections:
    if (effectiveActinPolarisation < 0) {
        effectiveActinPolarisation = 0;
    }

    // Prevent singularites in calculating updates to actin stochastic differential equation - 
    // because we've converted to polar coordinates, there are several places where we divide
    // by flow magnitude.
    if (flowMagnitude < 1e-2) {
        flowMagnitude = 1e-2;
    }

    // Calculate actin update:
    double gamma{1/fluctuationTimescale};
    double steadyState{maximumSteadyStateActinFlow*effectiveActinPolarisation};

    // We have to apply some interesting stochastic calculus for the matrix coupling to be
    // coherent:
    double effectiveCoupling{matrixCoupling*directionalIntensity};
    double cosDelta{std::cos(directionalInfluence)};
    double effectiveAmplitude{fluctuationAmplitude * std::exp(-effectiveCoupling*cosDelta*cosDelta)};

    // Sample update using drift (deterministic) and diffusion (random) terms:
    double magnitudeUpdateDrift{
        gamma*(steadyState - flowMagnitude) + (effectiveAmplitude/(2*flowMagnitude))
    };
    double magnitudeUpdateDiffusion{
        std::sqrt(fluctuationAmplitude)*standardNormalDistribution(actinFlowGenerator)
    };

    double angleUpdateDrift{
        gamma*std::sin(calculateAngularDistance(totalAdvectionDirection, flowDirection))
    };
    double angleUpdateDiffusion{
        (standardNormalDistribution(actinFlowGenerator) * std::sqrt(effectiveAmplitude)) / flowMagnitude
    };

    // Apply update - in stochastic differential equations, randomly sampled terms are scaled by the
    // square root of dt.
    flowMagnitude += (magnitudeUpdateDrift * dt) + (magnitudeUpdateDiffusion*std::sqrt(dt));
    flowDirection += (angleUpdateDrift * dt) + (angleUpdateDiffusion*std::sqrt(dt));

    // The update can overshoot the magnitude below zero. The flow vector is then the same as one
    // of magnitude |r| along the opposite heading, so reflect to that: collisions decide whether
    // the cell is moving towards a neighbour from its heading, assuming a positive magnitude.
    if (flowMagnitude < 0) {
        flowMagnitude = -flowMagnitude;
        flowDirection += M_PI;
    }
    flowDirection = angleMod(flowDirection);

    // Update flow direction and magnitude based on collisions:
    runTrajectoryDependentCollisionLogic();

    // Update position:
    double cellDisplacement{flowMagnitude};
    double dx{std::cos(flowDirection) * cellDisplacement};
    double dy{std::sin(flowDirection) * cellDisplacement};
    x += dx;
    y += dy;

    // Roll position if out of bounds:
    if (x < 0) {
        double remainder{std::fmod(-x, worldSize)};
        x = worldSize - remainder;
    }
    if (y < 0) {
        double remainder{std::fmod(-y, worldSize)};
        y = worldSize - remainder;
    }
    x = std::fmod(x, worldSize);
    y = std::fmod(y, worldSize);

    // Update stadium positions and run relevant ODEs:
    runStickSlipLogic();
}


void CellAgent::runStickSlipLogic() {
    // Calculate extension of cell back:
    double xCellSpan{x - stadiumX};
    if (xCellSpan < -worldSize / 2) {xCellSpan += worldSize;};
    if (xCellSpan > worldSize / 2) {xCellSpan -= worldSize;};
    double yCellSpan{y - stadiumY};
    if (yCellSpan < -worldSize / 2) {yCellSpan += worldSize;};
    if (yCellSpan > worldSize / 2) {yCellSpan -= worldSize;};

    double stretchDistance{std::sqrt(
        std::pow(xCellSpan, 2) + 
        std::pow(yCellSpan, 2)
    )};
    double cellExtension{stretchDistance};
    if (std::isnan(cellExtension)) {
        std::cout << "Initial cell extension calculation from position has failed." << std::endl;
        std::cout << "stretchDistance: " << stretchDistance << std::endl;
        std::cout << "cellBodyRadius: " << cellBodyRadius << std::endl;
        assert(!std::isnan(cellExtension));
    }

    // Calculate change in adhesions with Newton-Raphson:
    double timeScaling{1e-4 * 60}; // In min^-1
    int subdivisionCount{1};
    bool nrConverged{false};

    // Adjust for small adhesions:
    adhesionFraction = std::max(adhesionFraction, 1e-3);

    // Set up next values:
    double nextAdhesion;
    double nextExtension;

    // Calculate asymptotic behaviour (i.e. if the expoential goes crazy),
    // assume that the cell is going fully snap back to its rest length 
    // in the next minute & don't bother calculating the flailing NR iterations:
    double u{cellStiffness/adhesionFragility};
    double exponent{u * (cellExtension / adhesionFraction)};
    if (exponent > 100) {
        nextAdhesion = 1e-3;
        nextExtension = 1e-3;
    } else {
        while (!nrConverged) {
            // Initialise to current values:
            nextAdhesion = adhesionFraction;
            nextExtension = cellExtension;

            // Calculate step size:
            double stepSize{(timeScaling * dt) / subdivisionCount};

            // Iterate through subdivided steps:
            for (int i = 0; i < subdivisionCount; i++) {
                // Calculate adhesions:
                auto [updateAdhesion, updateExtension, convergenceFailure] = implicitNextState(stepSize, nextAdhesion, nextExtension);
                if (convergenceFailure) {
                    goto nextSubdivision;
                }

                // If no estimation failure (so far), update next substep:
                nextAdhesion = updateAdhesion;
                nextExtension = updateExtension;
            }

            // If we've reached the end of the loop without failure, we've converged:
            nrConverged = true;

            // We can skip to the next subdivision of the current timestep if the NR fails:
            nextSubdivision:;
            if (subdivisionCount >= 8192 and not nrConverged) {
                nextAdhesion = 1e-3;
                nextExtension = 1e-3;
                nrConverged = true;
            }
            subdivisionCount *= 4;
        }
    }

    // Update adhesions:
    adhesionFraction = nextAdhesion;

    // Update cell extension:
    double nextSpan{nextExtension};
    if (nextSpan > stretchDistance) {
        // We change nothing - no extensive force, only retraction.
    } else {
        stadiumX = x - (xCellSpan / stretchDistance) * nextSpan;
        stadiumY = y - (yCellSpan / stretchDistance) * nextSpan;
    }

    // Roll position if out of bounds:
    if (stadiumX < 0) {
        double remainder{std::fmod(-stadiumX, worldSize)};
        stadiumX = worldSize - remainder;
    }
    if (stadiumY < 0) {
        double remainder{std::fmod(-stadiumY, worldSize)};
        stadiumY = worldSize - remainder;
    }
    stadiumX = std::fmod(stadiumX, worldSize);
    stadiumY = std::fmod(stadiumY, worldSize);

    if (std::isnan(stadiumX) or std::isnan(stadiumY)) {
        std::cout << "Error in NR iteration." << std::endl;
        stadiumX = x;
        stadiumY = y;
    }
}


std::tuple<double, double, bool> CellAgent::implicitNextState(double stepSize, double A0, double X0) {
    /*
    Adhesion fraction is represented by A and the cell extension by X (for clarity).
    We need to find the Jacobian matrix under our implicit functions F_A and F_X, for which we want
    to find the root - which we define as follows:
    F_A = step*dA/dt(A_next, X_next) - A_next + A_current
    F_X = step*dX/dt(A_next, X_next) - X_next + X_current

    Where the Jacobian is:
    J = (dF_A/dA, dF_A/dX)
        (dF_X/dA, dF_X/dX)

    solving for:
    dx = J^-1 . [F_A, F_X] gives the convergence update for the implicit equation.

    (Useful worked examples in https://utkstair.org/clausius/docs/mse301/pdf/intronumericalrecipes_v01_chapter05_rootfindsys.pdf)

    Finding the Jacobian requires to find all partial derivatives of the functions dA/dt and dX/dT:

    dA/dt = r(1 - A) - A.exp((k.X)/(f.A))
    dX/dt = -(k.X)/(g.A).exp((k.X)/(f.A))

    let k/f = u, and k/g = v

    d/dA[dA/dt] = exp(u.(X/A)).(u.(X/A) - 1) - r
    d/dX[dA/dt] = -u.exp(u.(X/A))
    d/dA[dX/dt] = v.exp(u.(X/A)).(X/A^2 + (u.X^2)/A^3)
    d/dX[dX/dt] = -exp(u.(X/A)).(v/A + u.v.(X/A^2))

    let phi = exp(u.(X/A)) for ease of calculation:

    d/dA[dA/dt] = phi.(u.(X/A) - 1) - r
    d/dX[dA/dt] = -u.phi
    d/dA[dX/dt] = v.phi.(X/A^2 + (u.X^2)/A^3)
    d/dX[dX/dt] = -phi.(v/A + u.v.(X/A^2))

    (To future readers - you can check these by hand, the
    calculations aren't too hard & the process is instructive)

    Now we can define the NR Jacobian matrix as follows (where A & X are now the current guess):
    dF_A/dA = step*d/dA[dA/dt] - 1
    dF_A/dX = step*d/dX[dA/dt]
    dF_X/dA = step*d/dA[dX/dt]
    dF_X/dX = step*d/dX[dX/dt] - 1
    */
    double A{A0};
    double X{X0};
 
    double residualNorm{1};
    int iterationCount{0};
 
    const double u{cellStiffness/adhesionFragility};
    const double v{cellStiffness/adhesionStiffness};
    const double r{surfaceStickiness};
 
    while (residualNorm > 1e-4) {
        // Break if we exceed iteration count:
        if (iterationCount > 100) {
            return {0, 0, true};
        }
 
        // Shared terms: s = X/A, phi = exp(u.s)
        const double inverseA{1.0 / A};
        const double s{X * inverseA};
        const double us{u * s};
        const double stepPhi{stepSize * std::exp(us)};
 
        // F_A = step*(r(1 - A) - A.phi) - A + A0
        // F_X = step*(-v.s.phi) - X + X0
        const double F_A{stepSize*r*(1 - A) - stepPhi*A - A + A0};
        const double F_X{-stepPhi*v*s - X + X0};
 
        // J11 = step*(phi.(u.s - 1) - r) - 1
        // J12 = step*(-u.phi)
        // J21 = step*v.phi.(X/A^2 + u.X^2/A^3) = step*v.phi.(s/A).(1 + u.s)
        // J22 = step*(-phi.(v/A + u.v.X/A^2)) - 1 = -step*v.phi.(1/A).(1 + u.s) - 1
        const double J11{stepPhi*(us - 1) - stepSize*r - 1};
        const double J12{-stepPhi*u};
        const double sharedTerm{stepPhi * v * inverseA * (1 + us)};
        const double J21{sharedTerm * s};
        const double J22{-sharedTerm - 1};
 
        // Solve the 2x2 linear system (invert Jacobian):
        const double det{J11 * J22 - J12 * J21};
        if (std::abs(det) < 1e-14 || !std::isfinite(det)) {
            return {0, 0, true};
        }
        const double deltaA{(-F_A * J22 + F_X * J12) / det};
        const double deltaX{(-F_X * J11 + F_A * J21) / det};
 
        // We damp the step if it shoots the adhesion fraction below zero:
        double dampScale{1.0};
        while ((A + dampScale * deltaA) <= 0.0 && dampScale > 1e-3) {
            dampScale *= 0.5;
        }
 
        A += dampScale * deltaA;
        X += dampScale * deltaX;
 
        if (!std::isfinite(A) || !std::isfinite(X)) {
            return {0, 0, true};
        }
 
        residualNorm = std::sqrt(deltaA * deltaA + deltaX * deltaX);
        iterationCount += 1;
    }
 
    return {A, X, false};
}


void CellAgent::runTrajectoryDependentCollisionLogic() {
    // Get cell centre:
    const double globalFrameX{getX()};
    const double globalFrameY{getY()};
 
    // Actin flow as (unit direction, signed magnitude); filled on the first collision:
    double flowUnitX{0.0};
    double flowUnitY{0.0};
    double currentFlowMagnitude{flowMagnitude};
    bool flowVectorReady{false};
    bool flowChanged{false};
    const double inverseBodyRadius{1.0 / cellBodyRadius};
 
    // Loop through local agents and determine collisions:
    for (auto& localAgent: localAgents) {
        const auto& [startX, startY, endX, endY] = localAgent->sampleTrajectoryStadium();
        double localEffectiveRadius = localAgent->getEffectiveRadius();
 
        // Take the images of the local cell's centre and stadium point nearest this cell, then
        // the image of the stadium point nearest the centre, so that the segment cannot wrap
        // around the world when the local cell is about half the world away:
        double correctedStartX{takePeriodicModulus(startX, globalFrameX)};
        double correctedStartY{takePeriodicModulus(startY, globalFrameY)};
        double correctedEndX{takePeriodicModulus(takePeriodicModulus(endX, globalFrameX), correctedStartX)};
        double correctedEndY{takePeriodicModulus(takePeriodicModulus(endY, globalFrameY), correctedStartY)};
 
        // Determine whether collision occurs:
        const auto [collisionDetected, closestX, closestY, minimumDistance, clampedDotProduct] = isPositionInStadium(
                globalFrameX, globalFrameY,
                correctedStartX, correctedStartY,
                correctedEndX, correctedEndY,
                localEffectiveRadius
        );
 
        if (collisionDetected) {
            // Record collision:
            collisionsThisTimepoint += 1;
 
            if (!flowVectorReady) {
                flowUnitX = std::cos(flowDirection);
                flowUnitY = std::sin(flowDirection);
                flowVectorReady = true;
            }
 
            // Unit vector from acting cell to closest point on the local cell
            // (equal to cos/sin of the old angleActingToLocal; atan2(0, 0) == 0 gives (1, 0)):
            const double actingToLocalX{takePeriodicModulus(closestX, globalFrameX) - globalFrameX};
            const double actingToLocalY{takePeriodicModulus(closestY, globalFrameY) - globalFrameY};
            const double separation{std::sqrt(actingToLocalX*actingToLocalX + actingToLocalY*actingToLocalY)};
            double towardsLocalX{1.0};
            double towardsLocalY{0.0};
            if (separation > 0) {
                towardsLocalX = actingToLocalX / separation;
                towardsLocalY = actingToLocalY / separation;
            }
 
            // Get degree of overlap: (centralAngle - sin(centralAngle)) / pi, with
            // sin(centralAngle) = 2.sin(centralAngle/2).cos(centralAngle/2)
            const double cosHalfAngle{0.5 * minimumDistance * inverseBodyRadius};
            const double sinHalfAngle{std::sqrt(std::max(0.0, 1 - cosHalfAngle*cosHalfAngle))};
            double overlapRatio{(2*std::acos(cosHalfAngle) - 2*sinHalfAngle*cosHalfAngle) / M_PI};
            overlapRatio = std::clamp(overlapRatio, 0.0, 1.0);
 
            // Exert reduction in actin flow for acting cell:
            // cos(flowDirection - (angleActingToLocal - pi)) = -(flowUnit . towardsLocal)
            const double componentOfActingFlowOntoCollision{
                -(flowUnitX*towardsLocalX + flowUnitY*towardsLocalY)
            };
            if (componentOfActingFlowOntoCollision < 0) {
                // Calculate change in actin flow:
                const double reductionInFlow{
                    dt * collisionFlowReductionRate * overlapRatio * std::abs(componentOfActingFlowOntoCollision)
                };
                const double cappedReductionInFlow{
                    std::min(reductionInFlow, currentFlowMagnitude * std::abs(componentOfActingFlowOntoCollision))
                };
 
                // Update actin flow (the restitution direction is -towardsLocal):
                const double xFlowComponent{flowUnitX*currentFlowMagnitude - towardsLocalX*cappedReductionInFlow};
                const double yFlowComponent{flowUnitY*currentFlowMagnitude - towardsLocalY*cappedReductionInFlow};
 
                if (std::isnan(xFlowComponent) or std::isnan(yFlowComponent)) {
                    std::cout << "--- --- --- ---" << std::endl;
                    std::cout << "clampedDotProduct: " << clampedDotProduct << std::endl;
                    std::cout << "minimumDistance: " << minimumDistance << std::endl;
                    std::cout << "overlapRatio: " << overlapRatio << std::endl;
                    std::cout << "xFlowComponent " << xFlowComponent << std::endl;
                    std::cout << "yFlowComponent " << yFlowComponent << std::endl;
                }
 
                // Set acting cell actin flow to new values:
                currentFlowMagnitude = std::sqrt(xFlowComponent*xFlowComponent + yFlowComponent*yFlowComponent);
                if (currentFlowMagnitude > 0) {
                    flowUnitX = xFlowComponent / currentFlowMagnitude;
                    flowUnitY = yFlowComponent / currentFlowMagnitude;
                } else if (currentFlowMagnitude == 0) {
                    flowUnitX = 1.0; // atan2(0, 0) == 0 in the original
                    flowUnitY = 0.0;
                } else {
                    flowUnitX = xFlowComponent; // NaN: propagate as before
                    flowUnitY = yFlowComponent;
                }
                flowChanged = true;
            }
 
            // Calculate CIL effect:
            polarityChangeCilX -= towardsLocalX;
            polarityChangeCilY -= towardsLocalY;
            // --> Simulate effect of CIL on RhoA redistribution for local cell:
            localAgent->setCILPolarityChange(towardsLocalX, towardsLocalY);
        }
 
        // Determine whether adhesion is affected (adhesion collision, or ac):
        double acCorrectedStartX{takePeriodicModulus(startX, stadiumX)};
        double acCorrectedStartY{takePeriodicModulus(startY, stadiumY)};
        double acCorrectedEndX{takePeriodicModulus(takePeriodicModulus(endX, stadiumX), acCorrectedStartX)};
        double acCorrectedEndY{takePeriodicModulus(takePeriodicModulus(endY, stadiumY), acCorrectedStartY)};
        const auto [acDetected, acClosestX, acClosestY, acMinimumDistance, acClampedDotProduct] = isPositionInStadium(
            stadiumX, stadiumY,
            acCorrectedStartX, acCorrectedStartY,
            acCorrectedEndX, acCorrectedEndY,
            localEffectiveRadius
        );
 
        if (acDetected) {
            adhesionFraction -= dt * adhesionFraction * adhesionReductionRate;
        }
    }
 
    // Convert the flow back to polar form once:
    if (flowChanged) {
        flowDirection = std::atan2(flowUnitY, flowUnitX);
        flowMagnitude = currentFlowMagnitude;
    }
}


std::array<double, 2> CellAgent::sampleAttachmentPoint() {
    // Getting current position:
    double actingCellX{getX()};
    double actingCellY{getY()};

    // Sampling from random radius in cell area:
    double samplePointDirection{angleUniformDistribution(attachmentPointGenerator)};
    double samplePointRadius{std::sqrt(uniformDistribution(attachmentPointGenerator)) * cellBodyRadius};

    // Getting point in frame:
    double actingFrameX{std::cos(samplePointDirection) * samplePointRadius};
    double actingFrameY{std::sin(samplePointDirection) * samplePointRadius};

    // Transforming points from acting frame to global frame:
    double globalFrameX{actingCellX + actingFrameX};
    double globalFrameY{actingCellY + actingFrameY};

    return {globalFrameX, globalFrameY};
};


std::tuple<double, double, double, double> CellAgent::sampleTrajectoryStadium() {
    // Return stadium:
    return {
        x, y,
        stadiumX, stadiumY
    };
}


std::tuple<bool, double, double, double, double> CellAgent::isPositionInStadium(
    double samplePointX, double samplePointY,
    double startX, double startY,
    double endX, double endY,
    double localEffectiveRadius
) {
    // Get intermediate calculations:
    double xStartToSample{samplePointX - startX};
    double yStartToSample{samplePointY - startY};
    double xStartToEnd{endX - startX};
    double yStartToEnd{endY - startY};

    // Get scaled dot product, treating a zero-length segment (a cell whose rear is at its centre)
    // as its start point, instead of dividing by zero:
    double scaledDotProduct{xStartToSample*xStartToEnd + yStartToSample*yStartToEnd};
    const double segmentLengthSquared{std::pow(xStartToEnd, 2) + std::pow(yStartToEnd, 2)};
    if (segmentLengthSquared > 0) {
        scaledDotProduct /= segmentLengthSquared;
    } else {
        scaledDotProduct = 0;
    }
    double clampedDotProduct{std::clamp(scaledDotProduct, 0.0, 1.0)};

    // Determine closest point on segment:
    double closestPointX{0};
    double closestPointY{0};
    double minimumDistance{0};
    bool isColliding{false};

    if (scaledDotProduct < 0) {
        // Colliding with cell body:
        closestPointX = startX;
        closestPointY = startY;

        // Get minimum distance:
        minimumDistance = std::sqrt(
            std::pow(samplePointX - closestPointX, 2) +
            std::pow(samplePointY - closestPointY, 2)
        );

        // Collision distance is two cell radii:
        isColliding = minimumDistance < (effectiveRadius + localEffectiveRadius);
    } else if (scaledDotProduct > 1) {
        // Colliding with final point of extension:
        closestPointX = endX;
        closestPointY = endY;

        // Get minimum distance:
        minimumDistance = std::sqrt(
            std::pow(samplePointX - closestPointX, 2) +
            std::pow(samplePointY - closestPointY, 2)
        );

        // Collision distance is two cell radii:
        isColliding = minimumDistance < (effectiveRadius + localEffectiveRadius);
    } else {
        // Colliding with central part of extension:
        closestPointX = startX + scaledDotProduct*xStartToEnd;
        closestPointY = startY + scaledDotProduct*yStartToEnd;

        // Get minimum distance:
        minimumDistance = std::sqrt(
            std::pow(samplePointX - closestPointX, 2) +
            std::pow(samplePointY - closestPointY, 2)
        );

        // Collision distance is two cell radii:
        isColliding = minimumDistance < (effectiveRadius + localEffectiveRadius);
    }

    return {isColliding, closestPointX, closestPointY, minimumDistance, clampedDotProduct};
}


// Utility functions:
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
    if (localPosition - queryPosition > (worldSize / 2)) {
        modulusPosition += worldSize;
    }
    else if (localPosition - queryPosition < -(worldSize / 2)) {
        modulusPosition -= worldSize;
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
