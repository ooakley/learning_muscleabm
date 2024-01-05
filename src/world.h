#pragma once
#include <random>
#include "agents.h"
#include "ecm.h"

// Structure to store necessary parameters for the simulation:
struct CellParameters {
    float wbK, kappa, homotypicInhibition, heterotypicInhibition,
    polarityPersistence, polarityTurningCoupling, flowScaling, flowPolarityCoupling,
    collisionRepolarisation, repolarisationRate, polarityNoiseSigma;
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
        float setCellTypeProportions,
        float setMatrixPersistence,
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

    // Complex objects from our libraries:
    std::vector<CellAgent> cellAgentVector;
    ECMField ecmField;
    CellParameters cellParameters;

    // Cell population characteristics:
    int numberOfCells;
    float cellTypeProportions;

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

    // Distributions for seeding position and initial heading:
    std::uniform_real_distribution<double> positionDistribution;
    std::uniform_real_distribution<double> headingDistribution;
    std::uniform_real_distribution<double> contactInhibitionDistribution;

    // Private member functions:
    // Initialisation Functions:
    void initialiseCellVector();
    CellAgent initialiseCell(int setCellID);

    // Simulation functions:
    void runCellStep(CellAgent& actingCell);
    void setMovementOnMatrix(
        std::tuple<double, double> cellStart,
        std::tuple<double, double> cellFinish,
        double cellHeading
    );

    // World utility functions:
    std::array<int, 2> getIndexFromLocation(std::tuple<double, double> position);

    // Basic utility functions:
    int sign(double value);
    std::tuple<double, double> rollPosition(std::tuple<double, double> position);
};