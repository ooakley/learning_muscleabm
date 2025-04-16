#pragma once
#include <random>
#include <deque>
#include <utility>
#include <list>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/math/distributions/normal.hpp>
namespace boostMatrix = boost::numeric::ublas;

class CellAgent {
public:
    // Constructor and intialisation:
    CellAgent(
        // Defined behaviour parameters:
        bool setMatrixInteraction, unsigned int setCellSeed, int setCellID,

        // Movement parameters:
        double setHalfSatCellAngularConcentration,
        double setMaxCellAngularConcentration,
        double setHalfSatMeanActinFlow,
        double setMaxMeanActinFlow,
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
    double getScaledActinFlowMagnitude() const;
    double getDirectionalInfluence() const;
    double getDirectionalIntensity() const;
    double getDirectionalShift() const;
    double getSampledAngle() const;

    // Setters:
    // Setters for simulation (moving cells around etc.):
    void setPosition(std::tuple<double, double> newPosition);
    void setCILPolarityChange(double changeX, double changeY);
    void setActinState(double setFlowDirection, double setFlowMagnitude);

    // Setters for simulating cell perception (e.g. updating cell percepts):
    void setDirectionalInfluence(double setDirectionalInfluence);
    void setDirectionalIntensity(double setDirectiontalIntensity);
    void setLocalECMDensity(double setLocalDensity);
    void setLocalCellList(std::vector<std::shared_ptr<CellAgent>> setLocalAgents);

    // Simulation code:
    void takeRandomStep();

private:
    // Randomness and seeding:
    unsigned int cellSeed;
    int cellID;
    std::mt19937 seedGenerator;
    std::uniform_int_distribution<unsigned int> seedDistribution;

    // Whether to simulate matrix interactions:
    bool thereIsMatrixInteraction;

    // Dynamic cell state data:
    double x;
    double y;
    std::list<std::vector<double>> actinHistory;
    std::deque<double> xMovementHistory;
    std::deque<double> yMovementHistory;
    std::deque<double> xPositionHistory;
    std::deque<double> yPositionHistory;
    double polarityX;
    double polarityY;
    double polarityDirection;
    double polarityMagnitude;
    double flowDirection;
    double flowMagnitude;
    double scaledFlowMagnitude;
    double shapeDirection;
    // double eccentricityConstant;

    // Movement parameters:
    double halfSatCellAngularConcentration;
    double maxCellAngularConcentration;
    double halfSatMeanActinFlow;
    double maxMeanActinFlow;
    double flowScaling;

    // Polarisation system parameters:
    double polarityDiffusionRate;
    double actinAdvectionRate;
    double contactAdvectionRate;

    // Contact inhibition parameters:
    double lowDiscrepancySample;
    double cellBodyRadius;
    double cellAspectRatio;
    double majorAxisScaling;
    double minorAxisScaling;
    double contactDistributionSharpness;
    double inhibitionStrength;
    double polarityChangeCilX;
    double polarityChangeCilY;

    // Matrix sensation properties:
    double halfSatMatrixAngularConcentration;
    double maxMatrixAngularConcentration;

    // Binary simulation parameters:
    bool actinMagnitudeIsFixed;
    bool actinDirectionIsFixed;
    bool thereIsExtensionRepulsion;
    bool collisionsAreDeterministic;
    bool matrixAlignmentIsDeterministic;

    // Properties calculated each timestep:
    double movementDirection;
    double directionalShift; // -pi <= theta < pi
    double sampledAngle;

    // Percepts:
    double directionalInfluence; // -pi <= theta < pi
    double directionalIntensity; // 0 <= I < 1
    double localECMDensity; // 0 <= D < 1
    std::vector<std::shared_ptr<CellAgent>> localAgents;

    // Poisson sampling for step size:
    std::mt19937 generatorProtrusion;
    double poissonLambda;

    // General distributions:
    std::uniform_real_distribution<double> uniformDistribution;
    std::uniform_real_distribution<double> angleUniformDistribution;
    std::bernoulli_distribution bernoulliDistribution;
    std::normal_distribution<double> standardNormalDistribution;

    // We need to use a special distribution (von Mises) to sample from a random
    // direction over a circle - unfortunately not included in std library:

    // --> Member variables for von Mises sampling for directional step size:
    std::mt19937 generatorU1, generatorU2, generatorB;

    // --> Member functions for von Mises sampling:
    double sampleVonMises(double kappa);

    // Generators for collision shape sampling:
    std::mt19937 generatorCollisionRadiusSampling;
    std::mt19937 generatorCollisionAngleSampling;

    // Generators for matrix shape sampling:
    std::mt19937 generatorMatrixRadiusSampling;
    std::mt19937 generatorMatrixAngleSampling;

    // Generator for selecting for environmental influence:
    std::mt19937 generatorInfluence;

    // Generator for finding random angle after loss of movement polarisation:
    std::mt19937 generatorRandomRepolarisation;
    std::mt19937 randomDeltaSample;

    // Simulation subfunctions:
    double determineMovementDirection();
    double determineActinFlow();
    void runTrajectoryDependentCollisionLogic();
    void runStochasticCollisionLogic();
    void runDeterministicCollisionLogic();
    bool isPositionInStadium(
        double samplePointX, double samplePointY,
        double startX, double startY,
        double endX, double endY
    );

    // Effectively a utility function for calculating the modulus of angles:
    double angleMod(double angle) const;
    double nematicAngleMod(double angle) const;
    double calculateMinimumAngularDistance(double headingA, double headingB) const;
    double calculateShapeDeltaTowardsActin(double shapeHeading, double actinHeading) const;
    double calculateAngularDistance(double headingA, double headingB) const;
    double findTotalActinFlowDirection() const;
    double findTotalActinFlowMagnitude() const;
    std::vector<double> findTotalActinFlowComponents() const;

    void addToActinHistory(double actinFlowX, double actinFlowY);
    void addToPositionHistory(double positionX, double positionY);
    void ageActinHistory();
    double findCellMovementMagnitude();

    void addToMovementHistory(double movementX, double movementY);
    double findDirectionalConcentration();
    double takePeriodicModulus(double queryPosition, double localPosition);
    // double findShapeDirection();

    std::vector<double> crossProduct(
        std::vector<double> const a, std::vector<double> const b
    );

    // Function to prevent 0 polarisation:
    // void safeZeroPolarisation();
};