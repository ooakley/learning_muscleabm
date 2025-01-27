#pragma once
#include <random>
#include "agents.h"
#include "ecm.h"
#include "collision.h"

// Structure to store necessary parameters for the simulation:
struct CellParameters {
    double halfSatCellAngularConcentration, maxCellAngularConcentration,
    halfSatMeanActinFlow, maxMeanActinFlow, flowScaling,
    polarityPersistence, actinPolarityRedistributionRate, polarityNoiseSigma,
    halfSatMatrixAngularConcentration, maxMatrixAngularConcentration,
    homotypicInhibitionRate, heterotypicInhibitionRate,
    collisionRepolarisation, collisionRepolarisationRate,
    cellBodyRadius, maxCellExtension, inhibitionStrength;
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
        double setCellTypeProportions,
        bool setThereIsMatrixInteraction,
        double setMatrixTurnoverRate,
        double setMatrixAdditionRate,
        double setCellDepositionSigma,
        double setCellSensationSigma,
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
    boostMatrix::matrix<double> cellSensationKernel;
    boostMatrix::matrix<double> cellDepositionKernel;
    bool thereIsMatrixInteraction;

    // Complex objects from our libraries:
    std::vector<std::shared_ptr<CellAgent>> cellAgentVector;
    ECMField ecmField;
    CellParameters cellParameters;
    CollisionCellList collisionCellList;

    // Cell population characteristics:
    int numberOfCells;
    int attachmentNumber;
    double cellTypeProportions;
    double cellDepositionSigma;
    double cellSensationSigma;

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
    void setMovementOnMatrix(
        std::tuple<double, double> cellStart,
        std::tuple<double, double> cellFinish,
        double cellHeading, double cellPolarity
    );
    void depositAtAttachments(
        std::vector<std::tuple<int, int>> attachmentIndices,
        double heading, double polarity, double weighting
    );

    // Calculating percepts for cells:
    std::tuple<double, double, double> getAverageDeltaHeading(CellAgent queryCell);
    std::tuple<double, double, double, std::vector<std::tuple<int, int>>>
        sampleAttachmentHeadings(std::shared_ptr<CellAgent> queryCell);
    double calculateCellDeltaTowardsECM(double ecmHeading, double cellHeading);

    // World utility functions:
    std::array<int, 2> getECMIndexFromLocation(std::tuple<double, double> position);
    boostMatrix::matrix<double> generateGaussianKernel(double sigma);

    // Basic utility functions:
    int sign(double value);
    std::tuple<double, double> rollPosition(std::tuple<double, double> position);
    std::tuple<int, int> rollIndex(int i, int j);
};