#pragma once
#include <random>
#include <boost/numeric/ublas/matrix.hpp>
#include <utility>
namespace boostMatrix = boost::numeric::ublas;

class CellAgent {
public:
    // Constructor and intialisation:
    CellAgent(
        double startX, double startY, double startHeading,
        unsigned int setCellSeed, int setCellID, int setCellType,
        double setWbK, double setKappa,
        double setHomotypicInhibition, double setHeterotypicInhibiton,
        double setPolarityPersistence, double setPolarityTurningCoupling,
        double setFlowScaling, double setFlowPolarityCoupling,
        double setCollisionRepolarisation, double setRepolarisationRate
    );

    // Getters:
    // Getters for values that shouldn't change:
    double getID();
    double getHomotypicInhibitionRate();
    double getCellType();

    // Persistent variable getters: (these variables represent aspects of the cell's "memory")
    double getX();
    double getY();
    std::tuple<double, double> getPosition();
    double getPolarity();
    double getPolarityExtent();
    std::vector<double> getAttachmentHistory();

    // Instantaneous variable getters: (these variables are updated each timestep,
    // and represent the cell's percepts/actions)
    double getMovementDirection();
    double getActinFlow();
    double getDirectionalInfluence();
    double getDirectionalIntensity();
    double getAverageAttachmentHeading();
    double getDirectionalShift();
    double getSampledAngle();


    // Setters:
    // Setters for simulation (moving cells around etc.):
    void setPosition(std::tuple<double, double> newPosition);
    void setLocalCellHeadingState(boostMatrix::matrix<double> stateToSet);

    // Setters for simulating cell perception (e.g. updating cell percepts):
    void setLocalMatrixHeading(boostMatrix::matrix<double> matrixToSet);
    void setLocalMatrixPresence(boostMatrix::matrix<bool> matrixToSet);
    void setContactStatus(boostMatrix::matrix<bool> stateToSet, int cellType);

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

    // Position and physical characteristics:
    double x;
    double y;
    double polarityX;
    double polarityY;
    std::vector<double> attachmentHistory;
    double kappa;
    double polarityPersistence;
    double polarityTurningCoupling;
    double flowPolarityCoupling;
    double flowScaling;
    double collisionRepolarisation;
    double repolarisationRate;

    // Properties calculated each timestep:
    double movementDirection;
    double actinFlow;
    double directionalInfluence; // -pi <= theta < pi
    double directionalIntensity; // 0 <= I < 1
    double directionalShift; // -pi <= theta < pi
    double sampledAngle;

    // Percepts:
    boostMatrix::matrix<double> localMatrixHeading;
    boostMatrix::matrix<bool> localMatrixPresence;

    boostMatrix::matrix<bool> cellType0ContactState;
    boostMatrix::matrix<bool> cellType1ContactState;
    boostMatrix::matrix<double> localCellHeadingState;

    // Weibull sampling for step size:
    std::mt19937 generatorWeibull;
    double kWeibull;

    std::tuple<double, double> getAverageDeltaHeading();
    double calculateCellDeltaTowardsECM(double ecmHeading, double cellHeading);

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
    std::pair<bool, double> checkForCollisions();

    // Member functions for von Mises sampling:
    double sampleVonMises(double kappa);

    // Effectively a utility function for calculating the modulus of angles:
    double angleMod(double angle);
    double calculateMinimumAngularDistance(double headingA, double headingB);
    double calculateAngularDistance(double headingA, double headingB);
    double findPolarityDirection();
    double findPolarityExtent();
};