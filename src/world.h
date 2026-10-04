#pragma once
#include <array>
#include <fstream>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

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
    collisionAdvectionRate,
    maximumSteadyStateActinFlow,

    // Collision parameters:
    cellBodyRadius,
    collisionFlowReductionRate,
    adhesionReductionRate,

    // Shape parameters:
    cellStiffness,
    surfaceStickiness,

    // Matrix parameters:
    matrixCoupling;
};

class World {
public:
    // Constructor and intialisation:
    World
    (
        unsigned int setWorldSeed,
        double setWorldSideLength,
        int setECMElementCount,
        int setNumberOfCells,
        double setMatrixSampleRate,
        double setPatternSigma,
        int setPatternFibreCount,
        CellParameters setCellParameters
    );

    // Getters:
    void writePositionsToCSV(std::ofstream& csvFile);
    void writeVerbosePositionsToCSV(std::ofstream& csvFile);
    void writeMatrixToCSV(std::ofstream& matrixFile);
    void writeSummarisedMatrixToCSV(std::ofstream& matrixFile);

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
    double matrixSampleRate;
    double patternSigma;
    int patternFibreCount;

    // Complex objects from our libraries:
    std::vector<std::shared_ptr<CellAgent>> cellAgentVector;
    ECMField ecmField;
    CellParameters cellParameters;
    CollisionCellList collisionCellList;

    // Cell population characteristics:
    int numberOfCells;

    // Random number generators, one per random process, each seeded from the world seed:
    std::mt19937 cellInitialisationGenerator; // Initial position, heading and seed of each cell.
    std::mt19937 cellOrderGenerator; // Order in which cells act in each timestep.
    std::mt19937 attachmentCountGenerator; // Number of matrix attachment points of each cell.

    // Distributions for initialising cells:
    std::uniform_real_distribution<double> positionDistribution;
    std::uniform_real_distribution<double> headingDistribution;
    std::uniform_int_distribution<unsigned int> cellSeedDistribution;

    // Private member functions:
    // Initialisation Functions:
    void initialiseCellVector();
    std::shared_ptr<CellAgent> initialiseCell(int setCellID);

    // Simulation functions:
    void runCellStep(std::shared_ptr<CellAgent> actingCell);

    // Calculating percepts for cells:
    double calculateCellDeltaTowardsECM(double ecmHeading, double cellHeading);

    // World utility functions:
    std::array<int, 2> getECMIndexFromLocation(std::tuple<double, double> position);

    // Basic utility functions:
    std::tuple<double, double> rollPosition(std::tuple<double, double> position);
    std::tuple<int, int> rollIndex(int i, int j);
};