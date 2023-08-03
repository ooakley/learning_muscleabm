#pragma once
#include <random>
#include "agents.h"
#include "ecm.h"

class World {
// This is a long type name we shorten for readability:
public:
    // Constructor and intialisation:
    World
    (
        int setWorldSeed,
        double setWorldSideLength,
        int setECMElementCount,
        int setNumberOfCells
    );

    // Getters:
    void writePositionsToCSV(std::ofstream& csvFile);

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

    // Cell population characteristics:
    int numberOfCells;

    // Variables for initialising generators:
    int worldSeed;
    std::mt19937 seedGenerator;
    std::uniform_int_distribution<unsigned int> seedDistribution;

    // Generators for shuffling, cell positioning and cell rng seeding:
    std::mt19937 shuffleGenerator;
    std::mt19937 xPositionGenerator;
    std::mt19937 yPositionGenerator;
    std::mt19937 headingGenerator;
    std::mt19937 cellSeedGenerator;

    // Distributions for seeding position and initial heading:
    std::uniform_real_distribution<double> positionDistribution;
    std::uniform_real_distribution<double> headingDistribution;

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