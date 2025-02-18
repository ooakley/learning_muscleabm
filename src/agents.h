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
    );

    // Actual simulations that the cell runs:
    std::vector<double> sampleAttachmentPoint();

    // Getters:
    // Getters for values that shouldn't change:
    double getID() const;
    double getHomotypicInhibitionRate() const;
    double getCellType() const;

    // Persistent variable getters: (these variables represent aspects of the cell's "memory")
    double getX() const;
    double getY() const;
    std::tuple<double, double> getPosition() const;
    double getPolarityDirection() const;
    double getPolarityMagnitude() const;
    // std::vector<double> getAttachmentHistory() const;

    // Instantaneous variable getters: (these variables are updated each timestep,
    // and represent the cell's percepts/actions)
    double getMovementDirection() const;
    double getActinFlowDirection() const;
    double getActinFlowMagnitude() const;
    double getScaledActinFlowMagnitude() const;
    double getDirectionalInfluence() const;
    double getDirectionalIntensity() const;
    double getDirectionalShift() const;
    double getSampledAngle() const;

    // Setters:
    // Setters for simulation (moving cells around etc.):
    void setPosition(std::tuple<double, double> newPosition);
    // void setLocalCellHeadingState(const boostMatrix::matrix<double>& stateToSet);

    // Setters for simulating cell perception (e.g. updating cell percepts):
    void setLocalMatrixHeading(const boostMatrix::matrix<double>& matrixToSet);
    void setLocalMatrixPresence(const boostMatrix::matrix<double>& matrixToSet);
    // void setContactStatus(const boostMatrix::matrix<bool>& stateToSet, int cellType);

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
    double polarityX;
    double polarityY;
    double polarityDirection;
    double polarityMagnitude;
    double flowDirection;
    double flowMagnitude;
    double scaledFlowMagnitude;
    double eccentricityConstant;

    // Polarisation and movement parameters:
    double halfSatCellAngularConcentration;
    double maxCellAngularConcentration;
    double halfSatMeanActinFlow;
    double maxMeanActinFlow;
    double flowScaling;
    double actinPolarityRedistributionRate;
    double polarityPersistence;

    // Contact inhibition parameters:
    double cellBodyRadius;
    double maxCellExtension;
    double inhibitionStrength;

    // Matrix sensation properties:
    double halfSatMatrixAngularConcentration;
    double maxMatrixAngularConcentration;

    // Constant cell properties:
    double collisionRepolarisation;
    double collisionRepolarisationRate;

    // Infrastructure for additive polarisation noise:
    double polarityNoiseSigma;
    std::mt19937 generatorPolarityNoiseX;
    std::mt19937 generatorPolarityNoiseY;
    std::normal_distribution<double> polarityNoiseDistribution;
    std::normal_distribution<double> positionalNoiseDistribution;

    // Properties calculated each timestep:
    double movementDirection;
    double directionalInfluence; // -pi <= theta < pi
    double directionalIntensity; // 0 <= I < 1
    double localECMDensity; // 0 <= D < 1
    double directionalShift; // -pi <= theta < pi
    double sampledAngle;

    // Percepts:
    boostMatrix::matrix<double> localMatrixHeading;
    boostMatrix::matrix<double> localMatrixPresence;
    std::vector<std::shared_ptr<CellAgent>> localAgents;

    boostMatrix::matrix<bool> cellType0ContactState;
    boostMatrix::matrix<bool> cellType1ContactState;
    boostMatrix::matrix<double> localCellHeadingState;

    // Poisson sampling for step size:
    std::mt19937 generatorProtrusion;
    double poissonLambda;

    // Levy sampling for step size:
    std::mt19937 generatorLevy;
    boost::math::normal_distribution<> normalDistribution;

    // We need to use a special distribution (von Mises) to sample from a random
    // direction over a circle - unfortunately not included in std

    // Member variables for von Mises sampling for directional step size:
    std::mt19937 generatorU1, generatorU2, generatorB;
    std::uniform_real_distribution<double> uniformDistribution;
    std::bernoulli_distribution bernoulliDistribution;

    // Member variables for contact inhibition calculations:
    std::mt19937 generatorAngleUniform;
    std::mt19937 generatorCornerCorrection;
    std::mt19937 generatorForInhibitionRate;
    std::uniform_real_distribution<double> angleUniformDistribution;
    double homotypicInhibitionRate;
    double heterotypicInhibitionRate;
    int cellType;

    // Generator for selecting for environmental influence:
    std::mt19937 generatorInfluence;

    // Generator for finding random angle after loss of movement polarisation:
    std::mt19937 generatorRandomRepolarisation;

    // Collision detection behaviour:
    // std::pair<bool, double> checkForCollisions();

    // Member functions for von Mises sampling:
    double sampleVonMises(double kappa);

    // Member functions for Levy sampling:
    double sampleLevyDistribution(double mu, double c);

    // Effectively a utility function for calculating the modulus of angles:
    double angleMod(double angle) const;
    double calculateMinimumAngularDistance(double headingA, double headingB) const;
    double calculateAngularDistance(double headingA, double headingB) const;
    double findTotalActinFlowDirection() const;
    double findTotalActinFlowMagnitude() const;
    std::vector<double> findTotalActinFlowComponents() const;

    void addToActinHistory(double actinFlowX, double actinFlowY);
    void ageActinHistory();
    double findCellMovementMagnitude();

    void addToMovementHistory(double movementX, double movementY);
    double findDirectionalConcentration();
    double findShapeDirection();

    std::vector<double> crossProduct(
        std::vector<double> const a, std::vector<double> const b
    );

    // Function to prevent 0 polarisation:
    // void safeZeroPolarisation();
};