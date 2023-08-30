#pragma once
#include <random>
#include <boost/numeric/ublas/matrix.hpp>
namespace boostMatrix = boost::numeric::ublas;

class CellAgent {
public:
    // Constructor and intialisation:
    CellAgent(
        double startX, double startY, double startHeading,
        unsigned int setCellSeed, int setCellID,
        double setWbLambda, double setAlpha, double setBeta, double setInhibition
    );

    // Getters:
    double getX();
    double getY();
    std::tuple<double, double> getPosition();
    double getID();
    double getHeading();
    double getInhibitionRate();

    // Setters:
    void setDirectionalInfluence(double newDirectionalInfluence, double newDirectionalIntensity);
    void setPosition(std::tuple<double, double> newPosition);
    void setContactStatus(boostMatrix::matrix<bool> stateToSet);

    // Simulation code:
    void takeRandomStep();

private:
    // Randomness and seeding:
    unsigned int cellSeed;
    int cellID;
    std::mt19937 seedGenerator;
    std::uniform_int_distribution<unsigned int> seedDistribution;

    // Position and physical characteristics:
    double x;
    double y;
    double heading;
    double instantaneousSpeed;
    double directionalInfluence; // -pi <= theta < pi
    double directionalIntensity; // 0 <= I < 1
    boostMatrix::matrix<bool> cellContactState;

    // Speed-persistence relationship constants:
    double alphaForVonMisesXC;
    double betaForWeibullXC;

    // Weibull sampling for step size:
    std::mt19937 generatorWeibull;
    double lambdaWeibull;

    // We need to use a special distribution (von Mises) to sample from a random
    // direction over a circle - unfortunately not included in std

    // Member variables for von Mises sampling:
    std::mt19937 generatorU1, generatorU2, generatorB;
    std::uniform_real_distribution<double> uniformDistribution;
    std::bernoulli_distribution bernoulliDistribution;

    // Member variables for contact inhibition calculations:
    std::mt19937 generatorAngleUniform;
    std::mt19937 generatorCornerCorrection;
    std::mt19937 generatorForInhibitionRate;
    std::uniform_real_distribution<double> angleUniformDistribution;
    double inhibitionRate;

    // Generator for selecting for environmental influence:
    std::mt19937 generatorInfluence;

    // Member functions for von Mises sampling:
    double sampleVonMises(double kappa);

    // Effectively a utility function for calculating the modulus of angles:
    double angleMod(double angle);

};