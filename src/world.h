#pragma once
#include <random>
#include "agents.h"
#include "ecm.h"
#include "collision.h"

// Structure to store necessary parameters for the simulation:
struct CellParameters {
    // Cell movement parameters:
    double dt,
    cueDiffusionRate,
    cueKa,
    fluctuationAmplitude,
    fluctuationTimescale,
    actinAdvectionRate,
    matrixAdvectionRate,
    collisionAdvectionRate,
    maximumSteadyStateActinFlow,

    // Collision parameters:
    cellBodyRadius,
    aspectRatio,
    collisionFlowReductionRate,

    // Shape parameters:
    stretchFactor,
    slipFactor;
};

class World {
public:
    // Constructor and intialisation:
    World
    (
        int setWorldSeed,
        double setWorldSideLength,
        int setECMElementCount,
        int setNumberOfCells,
        bool setThereIsMatrixInteraction,
        double setMatrixTurnoverRate,
        double setMatrixAdditionRate,
        CellParameters setCellParameters
    );

    // Getters:
    void writePositionsToCSV(std::ofstream& csvFile);
    void writeMatrixToCSV(std::ofstream& matrixFile);

    // Setters:

    // Public simulation functions:
    void runSimulationStep();

private:
    // Private member variables:
    // World characteristics:
    double worldSideLength;
    int simulationTime;

    // ECM Information:
    int countECMElement;
    double lengthECMElement;
    bool thereIsMatrixInteraction;

    // Complex objects from our libraries:
    std::vector<std::shared_ptr<CellAgent>> cellAgentVector;
    ECMField ecmField;
    CellParameters cellParameters;
    CollisionCellList collisionCellList;

    // Cell population characteristics:
    int numberOfCells;

    // Variables for initialising generators:
    int worldSeed;
    std::mt19937 seedGenerator;
    std::uniform_int_distribution<unsigned int> seedDistribution;

    // Generators for shuffling, cell positioning and cell rng seeding:
    std::mt19937 shuffleGenerator;
    std::mt19937 cellSeedGenerator;

    std::mt19937 xPositionGenerator;
    std::mt19937 yPositionGenerator;
    std::mt19937 headingGenerator;
    std::mt19937 contactInhibitionGenerator;

    std::mt19937 kernelSamplingGenerator;

    // Distributions for seeding position and initial heading:
    std::uniform_real_distribution<double> positionDistribution;
    std::uniform_real_distribution<double> headingDistribution;
    std::uniform_real_distribution<double> contactInhibitionDistribution;

    // Private member functions:
    // Initialisation Functions:
    void initialiseCellVector();
    std::shared_ptr<CellAgent> initialiseCell(int setCellID);

    // Simulation functions:
    void runCellStep(std::shared_ptr<CellAgent> actingCell);
    void depositAtAttachment(
        std::vector<double> attachmentPoint,
        double heading, double polarity, double weighting
    );

    // Calculating percepts for cells:
    std::tuple<double, double> getPerceptAtAttachment(
        std::vector<double> attachmentPoint, double cellPolarity
    );
    double calculateCellDeltaTowardsECM(double ecmHeading, double cellHeading);

    // World utility functions:
    std::array<int, 2> getECMIndexFromLocation(std::tuple<double, double> position);

    // Basic utility functions:
    int sign(double value);
    std::tuple<double, double> rollPosition(std::tuple<double, double> position);
    std::tuple<int, int> rollIndex(int i, int j);
};