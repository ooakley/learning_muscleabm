#pragma once
#include <random>

class CellAgent {
public:
    // Constructor and intialisation:
    CellAgent(
        double startX, double startY, double startHeading,
        unsigned int setCellSeed, int setCellID,
        double setKappa, double setWbLambda, double setWbK
    );

    // Getters:
    double getX();
    double getY();
    std::tuple<double, double> getPosition();
    double getID();
    double getHeading();

    // Setters:
    void setDirectionalInfluence(double newDirectionalInfluence, double newDirectionalIntensity);
    void setPosition(std::tuple<double, double> newPosition);

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
    double directionalInfluence; // -pi <= theta < pi
    double directionalIntensity; // 0 <= I < 1

    // Weibull sampling for step size:
    std::mt19937 generatorWeibull;
    std::weibull_distribution<double> weibullDistribution;

    // We need to use a special distribution (von Mises) to sample from a random
    // direction over a circle - unfortunately not included in std

    // Member variables for von Mises sampling:
    double kappa, vm_tau, vm_rho, vm_r;
    std::mt19937 generatorU1, generatorU2, generatorB;
    std::uniform_real_distribution<double> uniformDistribution;
    std::bernoulli_distribution bernoulliDistribution;

    // Generator for selecting for environmental influence:
    std::mt19937 generatorInfluence;

    // Member functions for von Mises sampling:
    double sampleVonMises();

    // Effectively a utility function for calculating the modulus of angles:
    double angleMod(double angle);

};