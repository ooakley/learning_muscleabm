#pragma once
#include <random>

class CellAgent {
public:
    // Constructor and intialisation:
    CellAgent(
        double startX, double startY, double startHeading,
        int setCellSeed,
        double setKappa, double setWbLambda, double setWbK
    );

    // Getters:
    double getX();
    double getY();
    double getID();
    double getHeading();

    // Setters:
    void setDirectionalInfluence(double newDirectionalInfluence);

    // Simulation code:
    void takeRandomStep();

private:
    // Randomness and seeding:
    int cellSeed;
    std::mt19937 seedGenerator;
    std::uniform_int_distribution<unsigned int> seedDistribution;

    // Position and physical characteristics:
    double x;
    double y;
    double heading;
    double directionalInfluence; // 0 <= theta < pi

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

    // Member functions for von Mises sampling:
    double sampleVonMises();

    // Effectively a utility function for calculating the modulus of angles:
    double angleMod(double angle);

};