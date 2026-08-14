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
    unsigned int setCellSeed, int setCellID,
    double setdt,

    // Movement parameters:
    double setCueDiffusionRate,
    double setCueKa,
    double setFluctuationAmplitude,
    double setFluctuationTimescale,
    double setActinAdvectionRate,
    double setMatrixAdvectionRate,
    double setCollisionAdvectionRate,
    double setMaximumSteadyStateActinFlow,

    // Collision parameters:
    double setCellBodyRadius,
    double setAspectRatio,
    double setCollisionFlowReductionRate,

    // Shape parameters:
    double setCellStiffness,
    double setSurfaceStickiness,

    // Randomised initial state parameters:
    double startX, double startY, double startHeading
    )
    // Model infrastructure:
    : cellSeed{setCellSeed}
    , cellID{setCellID}
    , dt{setdt}

    // Movement parameters:
    , cueDiffusionRate{setCueDiffusionRate}
    , cueKa{setCueKa}
    , fluctuationAmplitude{setFluctuationAmplitude}
    , fluctuationTimescale{setFluctuationTimescale}
    , actinAdvectionRate{setActinAdvectionRate}
    , matrixAdvectionRate{setMatrixAdvectionRate}
    , collisionAdvectionRate{setCollisionAdvectionRate}
    , maximumSteadyStateActinFlow{setMaximumSteadyStateActinFlow}

    // Collision parameters:
    , cellBodyRadius{setCellBodyRadius}
    , cellAspectRatio{setAspectRatio}
    , collisionFlowReductionRate{setCollisionFlowReductionRate}
    , majorAxisScaling{std::sqrt(setAspectRatio)}
    , minorAxisScaling{std::sqrt(1/setAspectRatio)}

    // Shape parameters:
    , cellStiffness{setCellStiffness}
    , surfaceStickiness{setSurfaceStickiness}
    , adhesionStiffness{1000}
    , adhesionFragility{250}

    // State parameters:
    , x{startX}
    , y{startY}
    , stadiumX{startX}
    , stadiumY{startY}
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
    , adhesionFraction{1e-3}

    // History variables:
    , collisionsThisTimepoint{0}
    , finalCILEffectX{0}
    , finalCILEffectY{0}
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

    // Generator for deposition-movement history samping:
    generatorMovementSampling = std::mt19937(seedDistribution(seedGenerator));

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

int CellAgent::getCollisionNumber() const {return collisionsThisTimepoint;}
double CellAgent::getTotalCILEffectX() const {return finalCILEffectX;}
double CellAgent::getTotalCILEffectY() const {return finalCILEffectY;}
double CellAgent::getStadiumX() const {return stadiumX;}
double CellAgent::getStadiumY() const {return stadiumY;}

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

    // Collide with adjacent cells:
    collisionsThisTimepoint = 0;
    // runDeterministicCollisionLogic();
    // runCircularStochasticCollisionLogic();
    // runTrajectoryDependentCollisionLogic();

    // Determine advection derived from actin flow:
    double actinAdvectionX{flowMagnitude*std::cos(flowDirection)};
    double actinAdvectionY{flowMagnitude*std::sin(flowDirection)};

    // Determine advection derived from matrix:
    double matrixAdvectionX{directionalIntensity*std::cos(flowDirection + directionalInfluence)};
    double matrixAdvectionY{directionalIntensity*std::sin(flowDirection + directionalInfluence)};

    // Sum sources of advection:
    double totalAdvectionX{
        actinAdvectionRate*actinAdvectionX +
        matrixAdvectionRate*matrixAdvectionX +
        collisionAdvectionRate*polarityChangeCilX
    };
    double totalAdvectionY{
        actinAdvectionRate*actinAdvectionY +
        matrixAdvectionRate*matrixAdvectionY +
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
    // double exponentialTerm{std::exp(-totalAdvectionMagnitude / cueDiffusionRate)};
    // double cueConcentrationFront{
    //     totalAdvectionMagnitude / 
    //     (cueDiffusionRate*(1 - exponentialTerm))
    // };
    // double cueConcentrationBack{
    //     totalAdvectionMagnitude*exponentialTerm / 
    //     (cueDiffusionRate*(1 - exponentialTerm))
    // };
    double scaledAdvectionMagnitude{totalAdvectionMagnitude / (2 * cellBodyRadius)};
    double exponentialTerm{std::exp(-scaledAdvectionMagnitude / cueDiffusionRate)};
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
    double effectiveActinPolarisation{cueActivityFront - cueActivityBack};

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

    // Sample update using drift (deterministic) and diffusion (random) terms:
    double magnitudeUpdateDrift{
        gamma*(steadyState - flowMagnitude) + (fluctuationAmplitude/(2*flowMagnitude))
    };
    double magnitudeUpdateDiffusion{
        std::sqrt(fluctuationAmplitude)*standardNormalDistribution(generatorInfluence)
    };
    double angleUpdateDrift{
        gamma*std::sin(calculateAngularDistance(totalAdvectionDirection, flowDirection))
    };
    double angleUpdateDiffusion{
        (standardNormalDistribution(generatorProtrusion) * std::sqrt(fluctuationAmplitude)) / flowMagnitude
    };

    // if (angleUpdateDrift >= calculateAngularDistance(totalAdvectionDirection, flowDirection)) {
    //     std::cout << "Angle update OOB..." << std::endl;
    //     angleUpdateDrift = calculateAngularDistance(totalAdvectionDirection, flowDirection);
    // }

    // Apply update - in stochastic differential equations, randomly sampled terms are scaled by the
    // square root of dt.
    flowMagnitude += (magnitudeUpdateDrift * dt) + (magnitudeUpdateDiffusion*std::sqrt(dt));
    flowDirection += (angleUpdateDrift * dt) + (angleUpdateDiffusion*std::sqrt(dt));
    flowDirection = angleMod(flowDirection);

    // Update flow direction and magnitude based on collisions:
    // runAlternativeTrajectoryDependentCollisionLogic();
    runTrajectoryDependentCollisionLogic();

    // Update position:
    double cellDisplacement{flowMagnitude};
    double dx{std::cos(flowDirection) * cellDisplacement};
    double dy{std::sin(flowDirection) * cellDisplacement};
    x += dx;
    y += dy;

    // Roll position if out of bounds:
    if (x < 0) {
        double remainder{std::fmod(-x, 2048)};
        x = 2048. - remainder;
    }
    if (y < 0) {
        double remainder{std::fmod(-y, 2048)};
        y = 2048. - remainder;
    }
    x = std::fmod(x, 2048);
    y = std::fmod(y, 2048);

    // Update stadium positions and run relevant ODEs:
    runStickSlipLogic();
    // stadiumX = x;
    // stadiumY = y;

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
}


void CellAgent::runStickSlipLogic() {
    // Calculate extension of cell back:
    double xCellSpan{x - stadiumX};
    if (xCellSpan < -1024) {xCellSpan += 2048;};
    if (xCellSpan > 1024) {xCellSpan -= 2048;};
    double yCellSpan{y - stadiumY};
    if (yCellSpan < -1024) {yCellSpan += 2048;};
    if (yCellSpan > 1024) {yCellSpan -= 2048;};

    double stretchDistance{std::sqrt(
        std::pow(xCellSpan, 2) + 
        std::pow(yCellSpan, 2)
    )};
    // double cellExtension{std::max(stretchDistance - cellBodyRadius, 0.0) / cellBodyRadius};
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

    // Ensure our paper calculations make sense:
    // checkJacobianFD(timeScaling * dt, adhesionFraction, cellExtension);

    double nextAdhesion;
    double nextExtension;
    while (!nrConverged) {
        // Initialise to current values:
        nextAdhesion = adhesionFraction;
        nextExtension = cellExtension;

        // Calculate step size:
        double stepSize{(timeScaling * dt) / subdivisionCount};

        // Iterate through subdivided steps:
        bool convergenceFailure{false};
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
        // if (subdivisionCount > 1) {
        //     std::cout << ">> Convergence with adaptive subdivion of: " << subdivisionCount << std::endl;
        // }
        nrConverged = true;

        // We can skip to the next subdivision of the current timestep if the NR fails:
        nextSubdivision:;
        if (subdivisionCount >= 8192 and not nrConverged) {
            std::cout << ">> Convergence not reached with adaptive subdivision of: " << subdivisionCount << std::endl;
            std::cout << "adhesionFraction: " << adhesionFraction << std::endl;
            std::cout << "cellExtension: " << cellExtension << std::endl;

            // nextAdhesion = adhesionFraction;
            // nextExtension = cellExtension * 0.9;
            nextAdhesion = std::max(adhesionFraction, 1e-3);
            nextExtension = cellExtension * 0.5;
            nrConverged = true;
        }
        subdivisionCount *= 4;
    }

    // Update adhesions:
    adhesionFraction = nextAdhesion;

    // Update cell extension:
    // double nextSpan{(nextExtension  * cellBodyRadius) + cellBodyRadius};
    double nextSpan{nextExtension};
    if (nextSpan > stretchDistance) {
        // We change nothing - no extensive force, only retraction.
    } else {
        stadiumX = x - (xCellSpan / stretchDistance) * nextSpan;
        stadiumY = y - (yCellSpan / stretchDistance) * nextSpan;
    }

    // Roll position if out of bounds:
    if (stadiumX < 0) {
        double remainder{std::fmod(-stadiumX, 2048)};
        stadiumX = 2048. - remainder;
    }
    if (stadiumY < 0) {
        double remainder{std::fmod(-stadiumY, 2048)};
        stadiumY = 2048. - remainder;
    }
    stadiumX = std::fmod(stadiumX, 2048);
    stadiumY = std::fmod(stadiumY, 2048);
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

    double u{cellStiffness/adhesionFragility};
    double v{cellStiffness/adhesionStiffness};
    double r{surfaceStickiness};

    while (residualNorm > 1e-4) {
        // Break if we exceed iteration count:
        if (iterationCount > 50) {
            return {0, 0, true};
        }

        // Calculate phi:
        double phi{std::exp(u * (X/A))};

        // Get F_A and F_X:
        // --> dA/dt = r(1 - A) - A.exp((k.X)/(f.A))
        double Adot{r*(1 - A) - A*phi};
        double F_A{stepSize*Adot - A + A0};
        // --> dX/dt = -(k.X)/(g.A).exp((k.X)/(f.A))
        double Xdot{-v*(X/A)*phi};
        double F_X{stepSize*Xdot - X + X0};

        // Get dF_A/dA, or J_11:
        // d/dA[dA/dt] = phi.(u.(X/A) - 1) - r
        double AdotByA{phi*(u * (X/A) - 1) - r};
        double J11{stepSize*AdotByA - 1};

        // Get dF_A/dX, or J_12:
        // d/dX[dA/dt] = -u.phi
        double AdotByX{-u*phi};
        double J12{stepSize*AdotByX};

        // Get dF_X/dA, or J_21:
        // d/dA[dX/dt] = v.phi.(X/A^2 + (u.X^2)/A^3)
        double XdotByA{v * phi * ((X / std::pow(A, 2)) + (u*std::pow(X, 2) / std::pow(A, 3)))};
        double J21{stepSize*XdotByA};

        // Get dF_X/dX, or J_22:
        // d/dX[dX/dt] = -phi.(v/A + u.v.(X/A^2))
        double XdotByX{-phi * (v / A + (u * v * (X / std::pow(A, 2))))};
        double J22{stepSize*XdotByX - 1};

        // Solve the 2x2 linear system (invert Jacobian):
        double det{J11 * J22 - J12 * J21};

        // -- Return failure of convergence if Jacobian is singular:
        if (std::abs(det) < 1e-14 || !std::isfinite(det)) {
            return {0, 0, true};
        }

        // -- Get updates:
        double deltaA{(-F_A * J22 + F_X * J12) / det};
        double deltaX{(-F_X * J11 + F_A * J21) / det};

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


void CellAgent::checkJacobianFD(double stepSize, double A, double X) {
    double u{cellStiffness/adhesionFragility};
    double v{cellStiffness/adhesionStiffness};
    double r{surfaceStickiness};

    // A0/X0 don't affect the Jacobian (they only shift F_A/F_X by a constant),
    // so we can use A, X themselves as the "current" values for this check.
    double A0{A};
    double X0{X};

    // --- Residual evaluator, matching implicitNextState's F_A/F_X exactly ---
    auto residuals = [&](double Aval, double Xval) -> std::pair<double, double> {
        double phi{std::exp(u * (Xval/Aval))};
        double Adot{r*(1 - Aval) - Aval*phi};
        double F_A{stepSize*Adot - Aval + A0};
        double Xdot{-v*(Xval/Aval)*phi};
        double F_X{stepSize*Xdot - Xval + X0};
        return {F_A, F_X};
    };

    // --- Analytic Jacobian 
    double phi{std::exp(u * (X/A))};
    double AdotByA{phi*(u * (X/A) - 1) - r};
    double J11_analytic{stepSize*AdotByA - 1};
    double AdotByX{-u*phi};
    double J12_analytic{stepSize*AdotByX};
    double XdotByA{v * phi * ((X / std::pow(A, 2)) + (u * std::pow(X, 2) / std::pow(A, 3)))};
    double J21_analytic{stepSize*XdotByA};
    double XdotByX{-phi * (v / A + (u * v * (X / std::pow(A, 2))))};
    double J22_analytic{stepSize*XdotByX - 1};

    // --- Numerical Jacobian via central differences ---
    double h{1e-6 * std::max(1.0, std::abs(A))};   // step for perturbing A
    double hx{1e-6 * std::max(1.0, std::abs(X))};  // step for perturbing X

    auto [F_A_Aplus, F_X_Aplus] = residuals(A + h, X);
    auto [F_A_Aminus, F_X_Aminus] = residuals(A - h, X);
    double J11_numeric{(F_A_Aplus - F_A_Aminus) / (2*h)};
    double J21_numeric{(F_X_Aplus - F_X_Aminus) / (2*h)};

    auto [F_A_Xplus, F_X_Xplus] = residuals(A, X + hx);
    auto [F_A_Xminus, F_X_Xminus] = residuals(A, X - hx);
    double J12_numeric{(F_A_Xplus - F_A_Xminus) / (2*hx)};
    double J22_numeric{(F_X_Xplus - F_X_Xminus) / (2*hx)};

    // --- Report ---
    auto relError = [](double analytic, double numeric) {
        double denom{std::max(1e-12, std::abs(analytic))};
        return std::abs(analytic - numeric) / denom;
    };

    std::cout << "Jacobian cross-check at A=" << A << ", X=" << X << ":\n";
    std::cout << "J11: analytic=" << J11_analytic << " numeric=" << J11_numeric
               << " relErr=" << relError(J11_analytic, J11_numeric) << "\n";
    std::cout << "J12: analytic=" << J12_analytic << " numeric=" << J12_numeric
               << " relErr=" << relError(J12_analytic, J12_numeric) << "\n";
    std::cout << "J21: analytic=" << J21_analytic << " numeric=" << J21_numeric
               << " relErr=" << relError(J21_analytic, J21_numeric) << "\n";
    std::cout << "J22: analytic=" << J22_analytic << " numeric=" << J22_numeric
               << " relErr=" << relError(J22_analytic, J22_numeric) << "\n";

    for (double err : {relError(J11_analytic, J11_numeric), relError(J12_analytic, J12_numeric),
                        relError(J21_analytic, J21_numeric), relError(J22_analytic, J22_numeric)}) {
        if (err > 1e-4) {
            std::cout << ">> WARNING: Jacobian mismatch exceeds tolerance!\n";
            break;
        }
    }
}


std::tuple<double, bool> CellAgent::implicitNextAdhesion(
    double stepSize, double adhesionFraction, double cellExtension
) {
    // -- Get relevant exponential terms:
    double exponentialScale{(cellStiffness * cellExtension) / adhesionFragility};
    double popRate{std::exp(exponentialScale / adhesionFraction)};

    // -- Check if we have guaranteed overflow:
    if (HUGE_VAL == popRate) {
        std::cout << "n too small..." << std::endl;
        std::cout << "exponentialScale: " << exponentialScale << std::endl;
        std::cout << "popRate: " << popRate << std::endl;
        assert(false);
    }

    // -- Get initial guess from forward Euler (or not, if broken):
    double forwardAdhesionUpdate{
        (surfaceStickiness * (1 - adhesionFraction)) - (adhesionFraction * popRate)
    };
    double nextAdhesion{adhesionFraction + (stepSize * forwardAdhesionUpdate)};
    if (std::isnan(nextAdhesion)) {
        nextAdhesion = adhesionFraction;
    }

    // -- NR Iteration to get backward integration of adhesion occupancy:
    double epsilon{1};
    int iterationCount{0};
    while (std::abs(epsilon) > 1e-4) {
        // Break if we exceed max iterations:
        if (iterationCount > 50) {
            std::cout << "Hello!" << std::endl;
            return {1, true};
        };
    
        // Get NR numerator:
        double nrPopRate{std::exp(exponentialScale / nextAdhesion)};
        double nrAdhesionUpdate{
            (surfaceStickiness * (1 - nextAdhesion)) - (nextAdhesion * nrPopRate)
        };
        double nrNumerator{nextAdhesion - adhesionFraction - (stepSize * nrAdhesionUpdate)};

        // Get NR denominator:
        double nrDenominator{
            1 - (stepSize * (
                ((exponentialScale * nrPopRate) / nextAdhesion)
                - nrPopRate
                - surfaceStickiness
            ))
        };

        // Update estimate of root:
        epsilon = nrNumerator / nrDenominator;
        nextAdhesion -= nrNumerator / nrDenominator;

        // Break if we run into divide by zero errors:
        if (std::isnan(nextAdhesion)) {
            return {1, true};
        }
    
        // Update iteration count:
        iterationCount += 1;
    }

    // Return functional next adhesion value if iteration successful:
    return {nextAdhesion, false};
}

std::tuple<double, bool> CellAgent::implicitNextExtension(
    double stepSize, double adhesionFraction, double cellExtension
) {
    // Calculate change in back position:
    // -- Get constants:
    double dAlpha{cellStiffness / (adhesionFraction * adhesionStiffness)};
    double dBeta{cellStiffness / (adhesionFraction * adhesionFragility)};

    // -- Check if we have guaranteed overflow:
    if (HUGE_VAL == std::exp(dBeta * cellExtension)) {
        std::cout << "n too small..." << std::endl;
        std::cout << "dAlpha: " << dAlpha << std::endl;
        std::cout << "dBeta: " << dBeta << std::endl;
        assert(false);
    }

    // -- Get initial guess with forward Euler:
    double forwardExtensionUpdate{dAlpha * cellExtension * std::exp(dBeta * cellExtension)};
    double nextExtension{
        cellExtension - (stepSize * forwardExtensionUpdate)
    };

    // -- If initial guess is broken, use current value:
    if (std::isnan(nextExtension)) {
        nextExtension = cellExtension;
    }

    // -- Iterate with NR to get implicit Euler estimation of change in extension:
    double epsilon{1};
    double iterationCount{0};
    while (std::abs(epsilon) > 1e-4) {
        // Break if we exceed max iterations:
        if (iterationCount > 50) {
            // std::cout << "Extension NR exceeded max iterations..." << std::endl;
            // std::cout << "epsilon: " << epsilon << std::endl;
            return {0, true};
        };

        // Get NR extension numerator:
        double nrExtensionUpdate{
            dAlpha * nextExtension * std::exp(dBeta * nextExtension)
        };
        double nrNumerator{nextExtension - cellExtension + (stepSize * nrExtensionUpdate)};

        // Get NR extension denominator:
        double nrDenominator{
            1 + (stepSize * (
                (dAlpha*std::exp(dBeta * nextExtension))
                + ((dAlpha * dBeta * nextExtension) * std::exp(dBeta * nextExtension))
            ))
        };

        // Update estimate of root:
        epsilon = nrNumerator / nrDenominator;
        nextExtension -= nrNumerator / nrDenominator;

        // Break if we run into divide by zero errors:
        if (std::isnan(nextExtension)) {
            std::cout << "Extension NR estimation failed..." << std::endl;
            std::cout << "epsilon: " << epsilon << std::endl;
            std::cout << "nrNumerator: " << nrNumerator << std::endl;
            std::cout << "nrDenominator: " << nrDenominator << std::endl;
            return {0, true};
        }

        // Update iteration count:
        iterationCount += 1;
    }

    if (std::isnan(nextExtension)) {
        std::cout << "Extension ODE broken..." << std::endl;
        std::cout << "nextExtension: " << nextExtension << std::endl;
        assert(!std::isnan(nextExtension));
    }

    return {nextExtension, false};
}

void CellAgent::runTrajectoryDependentCollisionLogic() {
    // Get cell centre:
    double globalFrameX{getX()};
    double globalFrameY{getY()};

    // Loop through local agents and determine collisions:
    for (auto& localAgent: localAgents) {
        const auto& [startX, startY, endX, endY] = localAgent->sampleTrajectoryStadium();

        // // Find point most relevant for shift across periodic boundary:
        // double distanceToStart{std::sqrt(
        //     std::pow(startX - globalFrameX, 2) +
        //     std::pow(startY - globalFrameY, 2)
        // )};
        // double distanceToEnd{std::sqrt(
        //     std::pow(endX - globalFrameX, 2) +
        //     std::pow(endY - globalFrameY, 2)
        // )};

        // Determine whether any or all of the trajectory points will take the modulus:
        double correctedStartX{takePeriodicModulus(startX, globalFrameX)};
        double correctedStartY{takePeriodicModulus(startY, globalFrameY)};
        double correctedEndX{takePeriodicModulus(endX, globalFrameX)};
        double correctedEndY{takePeriodicModulus(endY, globalFrameY)};

        // // Take modulus of all:
        // if (distanceToStart > 2048/2 and distanceToEnd > 2048/2) {
        //     correctedStartX = takePeriodicModulus(startX, globalFrameX);
        //     correctedStartY = takePeriodicModulus(startY, globalFrameY);
        //     correctedEndX = takePeriodicModulus(endX, globalFrameX);
        //     correctedEndY = takePeriodicModulus(endY, globalFrameY);
        // }

        // // Take modulus of one:
        // else {
        //     if (distanceToStart < distanceToEnd) {
        //         // Take modulus relative to start:
        //         correctedStartX = startX;
        //         correctedStartY = startY;
        //         correctedEndX = takePeriodicModulus(endX, startX);
        //         correctedEndY = takePeriodicModulus(endY, startY);
        //     } else {
        //         // Take modulus relative to end:
        //         correctedStartX =  takePeriodicModulus(startX, endX);
        //         correctedStartY = takePeriodicModulus(startY, endY);
        //         correctedEndX = endX;
        //         correctedEndY = endY;
        //     }
        // }

        // Determine whether collision occurs:
        const auto [collisionDetected, closestX, closestY, minimumDistance, clampedDotProduct] = isPositionInStadium(
                globalFrameX, globalFrameY, correctedStartX, correctedStartY, correctedEndX, correctedEndY
        );

        if (collisionDetected) {
            // Record collision:
            collisionsThisTimepoint += 1;

            // Take modulo of position in case interaction is across the periodic boundary:
            double localCellX{takePeriodicModulus(closestX, globalFrameX)};
            double localCellY{takePeriodicModulus(closestY, globalFrameY)};

            // Get angle to local cell:
            double actingToLocalX{localCellX - globalFrameX};
            double actingToLocalY{localCellY - globalFrameY};
            double angleActingToLocal{std::atan2(actingToLocalY, actingToLocalX)};

            // Get degree of overlap:
            double halfDistance{minimumDistance / 2};
            double centralAngle{2*std::acos(halfDistance/cellBodyRadius)};
            double overlapArea{std::pow(cellBodyRadius, 2)*(centralAngle - std::sin(centralAngle))};
            double overlapRatio{overlapArea / (0.5*M_PI*std::pow(cellBodyRadius, 2))};
            overlapRatio = std::clamp(overlapRatio, 0.0, 1.0);

            // Exert reduction in actin flow for acting cell:
            double angleOfRestitution{angleActingToLocal - M_PI};
            double componentOfActingFlowOntoCollision{
                std::cos(flowDirection - angleOfRestitution)
            };
            if (componentOfActingFlowOntoCollision < 0) {
                // Calculate change in actin flow:
                double reductionInFlow{
                    dt * collisionFlowReductionRate * overlapRatio
                };
                double cappedReductionInFlow{std::min(reductionInFlow, flowMagnitude * std::abs(componentOfActingFlowOntoCollision))};

                double dxActinFlow{std::cos(angleOfRestitution) * cappedReductionInFlow};
                double dyActinFlow{std::sin(angleOfRestitution) * cappedReductionInFlow};

                // Update actin flow:
                double xFlowComponent{std::cos(flowDirection) * flowMagnitude};
                double yFlowComponent{std::sin(flowDirection) * flowMagnitude};
                xFlowComponent += dxActinFlow;
                yFlowComponent += dyActinFlow;

                if (std::isnan(xFlowComponent) or std::isnan(yFlowComponent)) {
                    std::cout << "--- --- --- ---" << std::endl;
                    std::cout << "clampedDotProduct: " << clampedDotProduct << std::endl;
                    std::cout << "minimumDistance: " << minimumDistance << std::endl;
                    std::cout << "scaledDistance: " << minimumDistance/(1 + clampedDotProduct) << std::endl;
                    std::cout << "halfDistance: " << halfDistance << std::endl;
                    std::cout << "overlapArea: " << overlapArea << std::endl;
                    std::cout << "overlapRatio: " << overlapRatio << std::endl;
                    std::cout << "xFlowComponent " << xFlowComponent << std::endl;
                    std::cout << "yFlowComponent " << xFlowComponent << std::endl;
                }

                // Set acting cell actin flow to new values:
                flowDirection = std::atan2(yFlowComponent, xFlowComponent);
                flowMagnitude = std::sqrt(std::pow(xFlowComponent, 2) + std::pow(yFlowComponent, 2));
            }

            // // Exert reduction in actin flow for local cell:
            // double localFlowDirection{localAgent->getActinFlowDirection()};
            // double localFlowMagnitude{localAgent->getActinFlowMagnitude()};
            // double componentOfLocalFlowOntoCollision{
            //     std::cos(localFlowDirection - angleActingToLocal)
            // };
            // if (componentOfLocalFlowOntoCollision <= 0) {
            //     // Calculate change in actin flow:
            //     double reductionInFlow{
            //         dt * collisionFlowReductionRate * localFlowMagnitude * std::abs(componentOfLocalFlowOntoCollision) * overlapRatio
            //     };
            //     double cappedReductionInFlow{std::min(reductionInFlow, localFlowMagnitude * std::abs(componentOfLocalFlowOntoCollision))};
            //     double dxActinFlow{std::cos(angleActingToLocal) * cappedReductionInFlow};
            //     double dyActinFlow{std::sin(angleActingToLocal) * cappedReductionInFlow};

            //     // Update actin flow:
            //     double xFlowComponent{std::cos(localFlowDirection)*localFlowMagnitude};
            //     double yFlowComponent{std::sin(localFlowDirection)*localFlowMagnitude};
            //     xFlowComponent += dxActinFlow;
            //     yFlowComponent += dyActinFlow;
            //     double updatedLocalFlowDirection{std::atan2(yFlowComponent, xFlowComponent)};
            //     double updatedLocalFlowMagnitude{std::sqrt(std::pow(xFlowComponent, 2) + std::pow(yFlowComponent, 2))};

            //     // Set local cell actin flow to new values:
            //     localAgent->setActinState(updatedLocalFlowDirection, updatedLocalFlowMagnitude);
            // }

            // Calculate CIL effect:
            double actingRepulsionX{std::cos(angleActingToLocal)};
            double actingRepulsionY{std::sin(angleActingToLocal)};
            polarityChangeCilX -= actingRepulsionX;
            polarityChangeCilY -= actingRepulsionY;
            // --> Simulate effect of CIL on RhoA redistribution for local cell:
            localAgent->setCILPolarityChange(actingRepulsionX, actingRepulsionY);
        }

        // Determine whether adhesion is affected (adhesion collision, or ac):
        double acCorrectedStartX{takePeriodicModulus(startX, stadiumX)};
        double acCorrectedStartY{takePeriodicModulus(startY, stadiumY)};
        double acCorrectedEndX{takePeriodicModulus(endX, stadiumX)};
        double acCorrectedEndY{takePeriodicModulus(endY, stadiumY)};
        const auto [acDetected, acClosestX, acClosestY, acMinimumDistance, acClampedDotProduct] = isPositionInStadium(
            stadiumX, stadiumY, acCorrectedStartX, acCorrectedStartY, acCorrectedEndX, acCorrectedEndY
        );

        if (acDetected) {
            // Get degree of overlap:
            double halfDistance{acMinimumDistance / 2};
            double centralAngle{2*std::acos(halfDistance/cellBodyRadius)};
            double overlapArea{std::pow(cellBodyRadius, 2)*(centralAngle - std::sin(centralAngle))};
            double overlapRatio{overlapArea / (M_PI*std::pow(cellBodyRadius, 2))};
            // double adhesionDecayRate{0.1};
            // double adhesionDecay{dt * overlapRatio * adhesionDecayRate};
            // adhesionDecay = std::min(adhesionDecay, adhesionFraction);
            overlapRatio = std::clamp(overlapRatio, 0.0, 1.0);
            double overlapUpdate{adhesionFraction * overlapRatio};
            adhesionFraction -= dt * overlapUpdate;
        }
    }
}


void CellAgent::runAlternativeTrajectoryDependentCollisionLogic() {
    // Get cell centre:
    double globalFrameX{getX()};
    double globalFrameY{getY()};

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
        const auto [collisionDetected, closestX, closestY, minimumDistance, clampedDotProduct] = isPositionInStadium(
                globalFrameX, globalFrameY, correctedStartX, correctedStartY, correctedEndX, correctedEndY
        );

        if (collisionDetected) {
            // Record collision:
            collisionsThisTimepoint += 1;

            // Take modulo of position in case interaction is across the periodic boundary:
            double localCellX{takePeriodicModulus(closestX, globalFrameX)};
            double localCellY{takePeriodicModulus(closestY, globalFrameY)};

            // Get angle to local cell:
            double actingToLocalX{localCellX - globalFrameX};
            double actingToLocalY{localCellY - globalFrameY};
            double angleActingToLocal{std::atan2(actingToLocalY, actingToLocalX)};

            // Calculate CIL effect:
            double actingRepulsionX{std::cos(angleActingToLocal)};
            double actingRepulsionY{std::sin(angleActingToLocal)};
            polarityChangeCilX -= actingRepulsionX;
            polarityChangeCilY -= actingRepulsionY;
            // --> Simulate effect of CIL on RhoA redistribution for local cell:
            localAgent->setCILPolarityChange(actingRepulsionX, actingRepulsionY);

            // Get degree of overlap:
            double halfDistance{minimumDistance / 2};
            double centralAngle{2*std::acos(halfDistance/cellBodyRadius)};
            double baseOverlapArea{std::pow(cellBodyRadius, 2)*(centralAngle - std::sin(centralAngle))};

            // Calculate amount of overlap in cell front:
            double collisionActinAngularDistance{angleMod(flowDirection - angleActingToLocal)};
            // No collision if direction of motion places actin front outside of collision zone:
            if (std::abs(collisionActinAngularDistance) >= (centralAngle / 2) + M_PI_2) {
                continue;
            }
            double frontAdjustment{(std::cos(collisionActinAngularDistance) + 1) / 2};

            // Get adjusted flow reduction factor:
            double overlapRatio{baseOverlapArea / (0.5*M_PI*std::pow(cellBodyRadius, 2))};
            overlapRatio *= frontAdjustment;
            overlapRatio = std::clamp(overlapRatio, 0.0, 1.0);

            double reductionInFlow{dt * collisionFlowReductionRate * overlapRatio};
            double cappedReductionInFlow{std::min(reductionInFlow, flowMagnitude)};
            flowMagnitude -= cappedReductionInFlow;
        }

        // Determine whether adhesion is affected (adhesion collision, or ac):
        double acCorrectedStartX{takePeriodicModulus(startX, stadiumX)};
        double acCorrectedStartY{takePeriodicModulus(startY, stadiumY)};
        double acCorrectedEndX{takePeriodicModulus(endX, stadiumX)};
        double acCorrectedEndY{takePeriodicModulus(endY, stadiumY)};
        const auto [acDetected, acClosestX, acClosestY, acMinimumDistance, acClampedDotProduct] = isPositionInStadium(
            stadiumX, stadiumY, acCorrectedStartX, acCorrectedStartY, acCorrectedEndX, acCorrectedEndY
        );

        if (acDetected) {
            // Get degree of overlap:
            double halfDistance{acMinimumDistance / 2};
            double centralAngle{2*std::acos(halfDistance/cellBodyRadius)};
            double overlapArea{std::pow(cellBodyRadius, 2)*(centralAngle - std::sin(centralAngle))};
            double overlapRatio{overlapArea / (M_PI*std::pow(cellBodyRadius, 2))};
            overlapRatio = std::clamp(overlapRatio, 0.0, 0.9);
            adhesionFraction *= 1 - overlapRatio;
        }
    }
}


void CellAgent::runCircularStochasticCollisionLogic() {
    // Useful points:
    double actingCellX{getX()};
    double actingCellY{getY()};

    // Sampling from random point along cell perimeter:
    lowDiscrepancySample += (std::numbers::phi_v<double> - 1);
    lowDiscrepancySample = std::fmod(lowDiscrepancySample, 1);
    double divergence{(lowDiscrepancySample * M_PI) - M_PI_2};
    double samplePointDirection{flowDirection + divergence};

    // Find sample point in global coordinates:
    double sampleGlobalX{actingCellX + std::cos(samplePointDirection)*cellBodyRadius};
    double sampleGlobalY{actingCellY + std::sin(samplePointDirection)*cellBodyRadius};

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
        double sampleToLocalX{localCellX - sampleGlobalX};
        double sampleToLocalY{localCellY - sampleGlobalY};
        double angleSampleToLocal{std::atan2(sampleToLocalY, sampleToLocalX)};
        double displacementSampleToLocal{std::sqrt(
            std::pow(sampleToLocalX, 2) + 
            std::pow(sampleToLocalY, 2)
        )};
        bool withinRadius{displacementSampleToLocal < cellBodyRadius};

        if (withinRadius) {
            // Record collision:
            collisionsThisTimepoint += 1;

            // Get difference between cell centers:
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
                    dt * collisionFlowReductionRate * flowMagnitude * std::abs(componentOfActingFlowOntoCollision)
                };
                double cappedReductionInFlow{std::min(reductionInFlow, flowMagnitude * std::abs(componentOfActingFlowOntoCollision))};
                double dxActinFlow{std::cos(angleOfRestitution) * cappedReductionInFlow};
                double dyActinFlow{std::sin(angleOfRestitution) * cappedReductionInFlow};

                // Update actin flow:
                double xFlowComponent{std::cos(flowDirection)*flowMagnitude};
                double yFlowComponent{std::sin(flowDirection)*flowMagnitude};
                xFlowComponent += dxActinFlow;
                yFlowComponent += dyActinFlow;

                // Set acting cell actin flow to new values:
                flowDirection = std::atan2(yFlowComponent, xFlowComponent);
                flowMagnitude = std::sqrt(std::pow(xFlowComponent, 2) + std::pow(yFlowComponent, 2));
            }

            // Exert reduction in actin flow for local cell:
            double localFlowDirection{localAgent->getActinFlowDirection()};
            double localFlowMagnitude{localAgent->getActinFlowMagnitude()};
            double componentOfLocalFlowOntoSample{
                std::cos(localFlowDirection - angleActingToLocal)
            };
            if (componentOfLocalFlowOntoSample <= 0) {
                // Calculate change in actin flow:
                double reductionInFlow{
                    dt * collisionFlowReductionRate * localFlowMagnitude * std::abs(componentOfLocalFlowOntoSample)
                };
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

            // Calculate CIL effect:
            double actingRepulsionX{std::cos(angleActingToLocal)};
            double actingRepulsionY{std::sin(angleActingToLocal)};
            polarityChangeCilX -= actingRepulsionX;
            polarityChangeCilY -= actingRepulsionY;
            // --> Simulate effect of CIL on RhoA redistribution for local cell:
            localAgent->setCILPolarityChange(actingRepulsionX, actingRepulsionY);
        }
    }
};


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
            finalProbability = 1;
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
                double reductionInFlow{collisionFlowReductionRate * flowMagnitude * std::abs(componentOfActingFlowOntoSample)};
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
                double reductionInFlow{collisionFlowReductionRate * localFlowMagnitude * std::abs(componentOfLocalFlowOntoSample)};
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
            polarityChangeCilX += (actingExtensionRepulsionX + actingBodyRepulsionX);
            polarityChangeCilY += (actingExtensionRepulsionY + actingBodyRepulsionY);

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
            double localCILx{(localExtensionRepulsionX + localBodyRepulsionX)};
            double localCILy{(localExtensionRepulsionY + localBodyRepulsionY)};
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

        bool withinRadius{displacementActingToLocal < (2 * cellBodyRadius)};
        bool withinFront{std::abs(angleMod(angleActingToLocal-flowDirection)) < M_PI_2};
        //  and withinFront
        // Collide if within radius:
        if (withinRadius) {
            // Record collision:
            collisionsThisTimepoint += 1;

            // Get area of overlap:
            double sagitta{cellBodyRadius - (displacementActingToLocal/2)};
            double centralAngle{2*std::acos((cellBodyRadius - sagitta)/cellBodyRadius)};
            double overlapArea{std::pow(cellBodyRadius, 2)*(centralAngle - std::sin(centralAngle))};
            double overlapRatio{overlapArea / (0.5*M_PI*std::pow(cellBodyRadius, 2))};

            // Exert reduction in actin flow for acting cell:
            double angleOfRestitution{angleActingToLocal - M_PI};
            double componentOfActingFlowOntoCollision{
                std::cos(flowDirection - angleOfRestitution)
            };
            if (componentOfActingFlowOntoCollision < 0) {
                // Calculate change in actin flow:
                double reductionInFlow{
                    dt * collisionFlowReductionRate * flowMagnitude * std::abs(componentOfActingFlowOntoCollision) * overlapRatio
                };
                double cappedReductionInFlow{std::min(reductionInFlow, flowMagnitude * std::abs(componentOfActingFlowOntoCollision))};
                double dxActinFlow{std::cos(angleOfRestitution) * cappedReductionInFlow};
                double dyActinFlow{std::sin(angleOfRestitution) * cappedReductionInFlow};

                // Update actin flow:
                double xFlowComponent{std::cos(flowDirection)*flowMagnitude};
                double yFlowComponent{std::sin(flowDirection)*flowMagnitude};
                xFlowComponent += dxActinFlow;
                yFlowComponent += dyActinFlow;

                // Set acting cell actin flow to new values:
                flowDirection = std::atan2(yFlowComponent, xFlowComponent);
                flowMagnitude = std::sqrt(std::pow(xFlowComponent, 2) + std::pow(yFlowComponent, 2));
            }

            // Exert reduction in actin flow for local cell:
            double localFlowDirection{localAgent->getActinFlowDirection()};
            double localFlowMagnitude{localAgent->getActinFlowMagnitude()};
            double componentOfLocalFlowOntoSample{
                std::cos(localFlowDirection - angleActingToLocal)
            };
            if (componentOfLocalFlowOntoSample <= 0) {
                // Calculate change in actin flow:
                double reductionInFlow{
                    dt * collisionFlowReductionRate * localFlowMagnitude * std::abs(componentOfLocalFlowOntoSample) * overlapRatio
                };
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

            // Calculate CIL effect:
            // double bodyScaling{(2*cellBodyRadius) - displacementActingToLocal};
            double actingRepulsionX{std::cos(angleActingToLocal)};
            double actingRepulsionY{std::sin(angleActingToLocal)};
            polarityChangeCilX -= actingRepulsionX;
            polarityChangeCilY -= actingRepulsionY;

            // Simulate effect of CIL on RhoA redistribution for local cell:
            double localBodyRepulsionX{std::cos(angleActingToLocal)};
            double localBodyRepulsionY{std::sin(angleActingToLocal)};
            localAgent->setCILPolarityChange(actingRepulsionX, actingRepulsionY);
        }
    }
}


std::vector<double> CellAgent::sampleAttachmentPoint() {
    // Getting current position:
    double actingCellX{getX()};
    double actingCellY{getY()};

    // Sampling from random radius in cell area:
    // double divergence{(lowDiscrepancySample * M_PI) - M_PI_2};
    // double samplePointDirection{flowDirection + divergence};
    double samplePointDirection{angleUniformDistribution(generatorMatrixRadiusSampling)};
    double samplePointRadius{uniformDistribution(generatorU1) * cellBodyRadius};

    // Getting point in frame:
    double actingFrameX{std::cos(samplePointDirection) * samplePointRadius};
    double actingFrameY{std::sin(samplePointDirection) * samplePointRadius};

    // Transforming points from acting frame to global frame:
    double globalFrameX{actingCellX + actingFrameX};
    double globalFrameY{actingCellY + actingFrameY};

    return {globalFrameX, globalFrameY};
};


std::tuple<int, double, double> CellAgent::samplePositionHistory() {
    // Weight sampling to more recent points in trajectory:
    int currentLength{static_cast<int>(xPositionHistory.size())};
    // std::vector<double> probabilityWeights(currentLength);
    // std::iota(probabilityWeights.begin(), probabilityWeights.end(), 1);
    // std::fill(probabilityWeights.begin(), probabilityWeights.end(), 1);

    // Sample trajectory indices:
    std::uniform_int_distribution<> indexDistribution(0, currentLength-1);
    int sampledIndex{indexDistribution(generatorInfluence)};

    // Return positions at these points:
    return {
        sampledIndex, xPositionHistory[sampledIndex], yPositionHistory[sampledIndex]
    };
}


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
    double clampedDotProduct{std::clamp(scaledDotProduct, 0.0, 1.0)};

    // Determine closest point on segment:
    double closestPointX{0};
    double closestPointY{0};
    double minimumDistance{0};
    bool isColliding{false};
    bool endCollision{false};

    // Check stadium length:
    double stadiumLength{std::sqrt(std::pow(xStartToEnd, 2) + std::pow(yStartToEnd, 2))};
    if (stadiumLength < cellBodyRadius) {
        // Colliding with cell body:
        closestPointX = startX;
        closestPointY = startY;

        // Get minimum distance:
        minimumDistance = std::sqrt(
            std::pow(samplePointX - closestPointX, 2) +
            std::pow(samplePointY - closestPointY, 2)
        );

        // Collision distance is two cell radii:
        isColliding = minimumDistance < (cellBodyRadius * 2);
        clampedDotProduct = 0;
        return {isColliding, closestPointX, closestPointY, minimumDistance, clampedDotProduct};
    }

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
        isColliding = minimumDistance < (cellBodyRadius * 2);
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
        isColliding = minimumDistance < (cellBodyRadius * 2);
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
        isColliding = minimumDistance < (cellBodyRadius * 2);
    }

    // Find distance to closest point:
    // double minimumDistance{std::sqrt(
    //     std::pow(samplePointX - closestPointX, 2) +
    //     std::pow(samplePointY - closestPointY, 2)
    // )};
    // bool isColliding{minimumDistance < (cellBodyRadius * 2)};
    return {isColliding, closestPointX, closestPointY, minimumDistance, clampedDotProduct};
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


void CellAgent::addToMovementHistory(double movementX, double movementY) {
    xMovementHistory.push_back(movementX);
    yMovementHistory.push_back(movementY);
    if (xMovementHistory.size() > 25) {
        xMovementHistory.pop_front();
        yMovementHistory.pop_front();
    }
}

double CellAgent::sampleMovementHistory() {
    // Return random direction if we have no movement history:
    int sampleSize{static_cast<int>(xMovementHistory.size())};
    if (sampleSize == 0) {
        return angleUniformDistribution(generatorMovementSampling);
    }

    // Return 1 density if fibers present:
    std::uniform_int_distribution<> indexDistribution(0, sampleSize-1);
    int sampledIndex{indexDistribution(generatorMovementSampling)};
    double xComponent{xMovementHistory[sampledIndex]};
    double yComponent{yMovementHistory[sampledIndex]};
    return std::atan2(yComponent, xComponent);
};

void CellAgent::addToPositionHistory(double positionX, double positionY) {
    xPositionHistory.push_back(positionX);
    yPositionHistory.push_back(positionY);
    if (yPositionHistory.size() > 1440) {
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