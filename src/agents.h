#pragma once
#include <memory>
#include <random>
#include <tuple>
#include <vector>

class CellAgent {
public:
    // Constructor and intialisation:
    CellAgent(
        // Defined behaviour parameters:
        unsigned int setCellSeed, int setCellID,
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
    );

    // Actual simulations that the cell runs:
    std::vector<double> sampleAttachmentPoint();
    std::tuple<double, double, double, double> sampleTrajectoryStadium();

    // Getters:
    // Getters for values that shouldn't change:
    double getID() const;

    // Persistent variable getters: (these variables represent aspects of the cell's "memory")
    double getX() const;
    double getY() const;
    std::tuple<double, double> getPosition() const;
    double getPolarityDirection() const;
    double getPolarityMagnitude() const;
    double getActinFlowDirection() const;
    double getActinFlowMagnitude() const;
    double getShapeDirection() const;

    // Instantaneous variable getters: (these variables are updated each timestep,
    // and represent the cell's percepts/actions)
    double getMovementDirection() const;
    double getDirectionalInfluence() const;
    double getDirectionalIntensity() const;
    double getDirectionalShift() const;
    double getSampledAngle() const;
    int getCollisionNumber() const;
    double getTotalCILEffectX() const;
    double getTotalCILEffectY() const;
    double getStadiumX() const;
    double getStadiumY() const;
    double getEffectiveRadius() const;

    // Setters:
    // Setters for simulation (moving cells around etc.):
    void setPosition(std::tuple<double, double> newPosition);
    void setCILPolarityChange(double changeX, double changeY);

    // Setters for simulating cell perception (e.g. updating cell percepts):
    void setDirectionalInfluence(double setDirectionalInfluence);
    void setDirectionalIntensity(double setDirectiontalIntensity);
    void setLocalCellList(std::vector<std::shared_ptr<CellAgent>> setLocalAgents);

    // Simulation code:
    void takeRandomStep();

private:
    // Model infrastructure:
    int cellID;
    double dt;

    // Movement parameters:
    double cueDiffusionRate;
    double cueKa;
    double fluctuationAmplitude;
    double fluctuationTimescale;
    double actinAdvectionRate;
    double collisionAdvectionRate;
    double maximumSteadyStateActinFlow;

    // Collision parameters:
    double cellBodyRadius;
    double collisionFlowReductionRate;
    double adhesionReductionRate;

    // Shape parameters:
    double cellStiffness;
    double surfaceStickiness;
    double adhesionStiffness;
    double adhesionFragility;

    // Matrix parameters:
    double matrixCoupling;

    // State variables:
    double x;
    double y;
    double stadiumX;
    double stadiumY;
    double polarityDirection;
    double polarityMagnitude;
    double flowDirection;
    double flowMagnitude;
    double shapeDirection;
    double adhesionFraction;
    double effectiveRadius;

    // Contact inhibition state variables:
    double polarityChangeCilX;
    double polarityChangeCilY;

    // Properties calculated each timestep:
    double movementDirection;
    double directionalShift; // -pi <= theta < pi
    double sampledAngle;

    // Matrix percept state variables:
    double directionalInfluence; // -pi <= theta < pi
    double directionalIntensity; // 0 <= I < 1
    std::vector<std::shared_ptr<CellAgent>> localAgents;

    // History variables for analysis:
    int collisionsThisTimepoint;
    double finalCILEffectX;
    double finalCILEffectY;

    // Random number generators, one per random process, each seeded from the cell seed:
    std::mt19937 actinFlowGenerator; // Noise in actin flow magnitude and direction.
    std::mt19937 attachmentPointGenerator; // Positions of matrix attachment points.

    // General distributions:
    std::uniform_real_distribution<double> uniformDistribution;
    std::uniform_real_distribution<double> angleUniformDistribution;
    std::normal_distribution<double> standardNormalDistribution;

    // Simulation subfunctions:
    void runTrajectoryDependentCollisionLogic();
    std::tuple<bool, double, double, double, double> isPositionInStadium(
        double samplePointX, double samplePointY,
        double startX, double startY,
        double endX, double endY,
        double localEffectiveRadius
    );

    void runStickSlipLogic();
    std::tuple<double, double, bool> implicitNextState(
        double stepSize, double A0, double X0
    );

    // Utility functions:
    double angleMod(double angle) const;
    double nematicAngleMod(double angle) const;
    double calculateAngularDistance(double headingA, double headingB) const;
    double takePeriodicModulus(double queryPosition, double localPosition);
};
