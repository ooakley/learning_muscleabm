// Behavioural tests of cell movement: wrapping at the periodic boundary, the deterministic actin
// flow dynamics, randomness, and the state a cell reports.
#include <cmath>
#include <cstdlib>
#include <vector>

#include "gtest/gtest.h"
#include "test_access.h"
#include "test_helpers.h"

using Access = CellAgentTestAccess;

// --- The periodic boundary ---

TEST(MovementWrapping, CrossingEachEdgeKeepsTheStepLength) {
    struct Crossing {double x, y, heading;};
    for (const Crossing& crossing : {
        Crossing{2047.5, 500, 0}, Crossing{0.5, 500, M_PI - 1e-12},
        Crossing{500, 2047.5, M_PI / 2}, Crossing{500, 0.5, -M_PI / 2}
    }) {
        CellSpec spec;
        spec.x = crossing.x;
        spec.y = crossing.y;
        spec.heading = crossing.heading;
        auto cell{makeCell(spec)};
        Access::setFlow(*cell, crossing.heading, 1.0);
        cell->takeRandomStep();

        EXPECT_GE(cell->getX(), 0);
        EXPECT_LT(cell->getX(), WORLD_SIZE);
        EXPECT_GE(cell->getY(), 0);
        EXPECT_LT(cell->getY(), WORLD_SIZE);
        const double stepLength{std::hypot(
            minimalImage(crossing.x, cell->getX()), minimalImage(crossing.y, cell->getY())
        )};
        EXPECT_NEAR(stepLength, cell->getActinFlowMagnitude(), 1e-9) << "heading " << crossing.heading;
    }
}

TEST(MovementWrapping, WorldRollsPositionsIntoTheWorld) {
    auto world{makeWorld({})};
    for (const auto& [x, y] : {
        std::pair{2048.0, 0.0}, std::pair{-1e-9, 5.0}, std::pair{-2048.0, 4096.5}, std::pair{1e-300, 2047.999}
    }) {
        const auto [rolledX, rolledY] = WorldTestAccess::rollPosition(*world, x, y);
        EXPECT_GE(rolledX, 0);
        EXPECT_LT(rolledX, WORLD_SIZE);
        EXPECT_GE(rolledY, 0);
        EXPECT_LT(rolledY, WORLD_SIZE);
        EXPECT_NEAR(std::abs(minimalImage(x, rolledX)), 0, 1e-3);
        EXPECT_NEAR(std::abs(minimalImage(y, rolledY)), 0, 1e-3);
    }
}

TEST(MovementWrapping, WorldKeepsPositionsAtDoublePrecision) {
    // Stepping a cell through the World must leave it exactly where stepping it on its own does:
    // rolling the position with fmodf rounded it to float precision every step.
    WorldSpec spec;
    spec.cell.fluctuationAmplitude = 1e-3;
    auto world{makeWorld(spec)};
    CellAgent& cell{*WorldTestAccess::cells(*world)[0]};
    WorldTestAccess::placeCell(*world, cell, 1234.56789123, 987.654321987, 1234.0, 987.0);
    for (int step = 0; step < 20; ++step) {
        CellAgent alone{cell};
        alone.setLocalCellList({});
        alone.takeRandomStep();
        WorldTestAccess::runCellStep(*world, cell);
        ASSERT_EQ(cell.getX(), alone.getX()) << "step " << step;
        ASSERT_EQ(cell.getY(), alone.getY()) << "step " << step;
    }
}

TEST(MovementWrapping, SmallerWorldsWrapAtTheirOwnSize) {
    // Cells and their rears must stay in the world, and wrap at its size, not at a fixed 2048 px:
    // the rear only retracts, so a cell's extension grows by at most its step length each step,
    // and a wrapping error shows up as a sudden jump in extension.
    const double worldSize{1024};
    WorldSpec spec;
    spec.worldSize = worldSize;
    spec.ecmElementCount = 32;
    spec.numberOfCells = 40;
    spec.matrixSampleRate = 5;
    spec.cell.cellBodyRadius = 20;
    spec.cell.fluctuationAmplitude = 1e-3;
    auto world{makeWorld(spec)};
    auto extension = [worldSize](const CellAgent& cell) {
        return std::hypot(
            minimalImage(cell.getStadiumX(), cell.getX(), worldSize),
            minimalImage(cell.getStadiumY(), cell.getY(), worldSize)
        );
    };
    std::vector<double> previousExtensions(spec.numberOfCells, 0);
    for (int step = 0; step < 1000; ++step) {
        world->runSimulationStep();
        for (CellAgent* cell : WorldTestAccess::cells(*world)) {
            for (double coordinate : {cell->getX(), cell->getY(), cell->getStadiumX(), cell->getStadiumY()}) {
                ASSERT_GE(coordinate, 0) << "step " << step;
                ASSERT_LT(coordinate, worldSize) << "step " << step;
            }
            const int id{static_cast<int>(cell->getID())};
            ASSERT_LE(extension(*cell), previousExtensions[id] + std::abs(cell->getActinFlowMagnitude()) + 1e-6)
                << "step " << step << ", cell " << id;
            previousExtensions[id] = extension(*cell);
        }
    }
}

// --- Deterministic actin flow ---

namespace {

// Front-back difference in cue activity for a given advection, as in the model:
double polarisation(double advection, const CellSpec& spec) {
    const double scaled{advection / (2 * spec.cellBodyRadius)};
    const double exponential{std::exp(-scaled / spec.cueDiffusionRate)};
    const double front{scaled / (spec.cueDiffusionRate * (1 - exponential))};
    const double back{front * exponential};
    return std::max(front / (spec.cueKa + front) - back / (spec.cueKa + back), 0.0);
}

}  // namespace

TEST(MovementFlow, WithoutNoiseDirectionIsConstant) {
    CellSpec spec;
    spec.heading = 0.7;
    auto cell{makeCell(spec)};
    for (int step = 0; step < 500; ++step) {cell->takeRandomStep();}
    EXPECT_NEAR(cell->getActinFlowDirection(), 0.7, 1e-9);
}

TEST(MovementFlow, WithoutNoiseSpeedConvergesToTheSteadyState) {
    // The steady state solves v = M.P(a.v), where P is the cue polarisation:
    CellSpec spec;
    auto steadyStateResidual = [&spec](double v) {
        return spec.maximumSteadyStateActinFlow * polarisation(spec.actinAdvectionRate * v, spec) - v;
    };
    double low{0.5};
    double high{1.5};
    ASSERT_GT(steadyStateResidual(low), 0);
    ASSERT_LT(steadyStateResidual(high), 0);
    for (int i = 0; i < 100; ++i) {
        const double middle{(low + high) / 2};
        (steadyStateResidual(middle) > 0 ? low : high) = middle;
    }

    auto cell{makeCell(spec)};
    for (int step = 0; step < 2000; ++step) {cell->takeRandomStep();}
    EXPECT_NEAR(cell->getActinFlowMagnitude(), low, 1e-6);
}

TEST(MovementFlow, WithNoSteadyStateFlowDecaysButStaysFinite) {
    CellSpec spec;
    spec.maximumSteadyStateActinFlow = 0;
    auto cell{makeCell(spec)};
    Access::setFlow(*cell, 0, 2.0);
    for (int step = 0; step < 500; ++step) {
        cell->takeRandomStep();
        ASSERT_TRUE(std::isfinite(cell->getActinFlowMagnitude()));
        ASSERT_GE(cell->getActinFlowMagnitude(), 0);
    }
    // Flow is held at the 1e-2 floor before each update, and decays by a factor (1 - 1/tau):
    EXPECT_NEAR(cell->getActinFlowMagnitude(), 1e-2 * (1 - 1 / spec.fluctuationTimescale), 1e-12);
}

TEST(MovementFlow, ZeroAdvectionKeepsTheCellFinite) {
    // With no actin advection and no contact inhibition, the cue profile is flat. Computing its
    // polarisation divided zero by zero, making the cell's flow and position NaN, which then
    // tripped an assert in stick-slip. Run in a subprocess, so that an abort fails only this test:
    EXPECT_EXIT({
        CellSpec spec;
        spec.actinAdvectionRate = 0;
        spec.fluctuationAmplitude = 1e-3;
        auto cell{makeCell(spec)};
        bool finite{true};
        for (int step = 0; step < 100; ++step) {
            cell->takeRandomStep();
            finite = finite && std::isfinite(cell->getX()) && std::isfinite(cell->getY())
                && std::isfinite(cell->getActinFlowMagnitude()) && std::isfinite(cell->getStadiumX());
        }
        std::exit(finite ? 0 : 1);
    }, ::testing::ExitedWithCode(0), "");
}

// --- Randomness ---

class SeededCells : public ::testing::Test {
protected:
    void SetUp() override {
        CellSpec spec;
        spec.fluctuationAmplitude = 1e-3;
        cell = makeCell(spec);
        cellSameSeed = makeCell(spec);
        spec.seed = 1;
        cellDifferentSeed = makeCell(spec);
    }

    std::unique_ptr<CellAgent> cell;
    std::unique_ptr<CellAgent> cellSameSeed;
    std::unique_ptr<CellAgent> cellDifferentSeed;
};

TEST_F(SeededCells, CellsMoveEveryStep) {
    for (int step = 0; step < 100; ++step) {
        const double previousX{cell->getX()};
        const double previousY{cell->getY()};
        cell->takeRandomStep();
        EXPECT_NE(previousX, cell->getX());
        EXPECT_NE(previousY, cell->getY());
    }
}

TEST_F(SeededCells, SameSeedGivesTheSameTrajectory) {
    for (int step = 0; step < 100; ++step) {
        cell->takeRandomStep();
        cellSameSeed->takeRandomStep();
        EXPECT_EQ(cell->getX(), cellSameSeed->getX());
        EXPECT_EQ(cell->getY(), cellSameSeed->getY());
    }
}

TEST_F(SeededCells, DifferentSeedsGiveDifferentTrajectories) {
    for (int step = 0; step < 100; ++step) {
        cell->takeRandomStep();
        cellDifferentSeed->takeRandomStep();
        EXPECT_NE(cell->getX(), cellDifferentSeed->getX());
        EXPECT_NE(cell->getY(), cellDifferentSeed->getY());
    }
}

TEST(MovementRandomness, DifferentSeedsDrawIndependentNoise) {
    // Without collisions or matrix, changes in flow direction are driven by noise alone, so if
    // cells with different seeds draw independent noise, they turn the same way about half the time:
    CellSpec spec;
    spec.fluctuationAmplitude = 1e-4;
    spec.fluctuationTimescale = 20;
    auto cellA{makeCell(spec)};
    spec.seed = 1;
    auto cellB{makeCell(spec)};
    const int steps{1000};
    int sameTurns{0};
    for (int step = 0; step < steps; ++step) {
        const double previousA{cellA->getActinFlowDirection()};
        const double previousB{cellB->getActinFlowDirection()};
        cellA->takeRandomStep();
        cellB->takeRandomStep();
        const double turnA{std::remainder(cellA->getActinFlowDirection() - previousA, 2 * M_PI)};
        const double turnB{std::remainder(cellB->getActinFlowDirection() - previousB, 2 * M_PI)};
        sameTurns += (turnA > 0) == (turnB > 0);
    }
    EXPECT_NEAR(static_cast<double>(sameTurns) / steps, 0.5, 0.1);
}

TEST(MovementRandomness, WorldsWithTheSameSeedAreIdentical) {
    WorldSpec spec;
    spec.numberOfCells = 50;
    spec.matrixSampleRate = 5;
    spec.cell.fluctuationAmplitude = 1e-3;
    auto worldA{makeWorld(spec)};
    auto worldB{makeWorld(spec)};
    for (int step = 0; step < 100; ++step) {
        worldA->runSimulationStep();
        worldB->runSimulationStep();
    }
    const auto cellsA{WorldTestAccess::cells(*worldA)};
    const auto cellsB{WorldTestAccess::cells(*worldB)};
    for (std::size_t i = 0; i < cellsA.size(); ++i) {
        EXPECT_EQ(cellsA[i]->getX(), cellsB[i]->getX());
        EXPECT_EQ(cellsA[i]->getStadiumY(), cellsB[i]->getStadiumY());
    }
}

// --- Reported state ---

TEST(MovementState, EveryReportedValueIsInitialisedAndFinite) {
    WorldSpec spec;
    spec.numberOfCells = 20;
    spec.matrixSampleRate = 5;
    auto world{makeWorld(spec)};
    for (int step = 0; step <= 50; ++step) {
        for (CellAgent* cell : WorldTestAccess::cells(*world)) {
            for (double value : {
                cell->getX(), cell->getY(), cell->getStadiumX(), cell->getStadiumY(),
                cell->getPolarityDirection(), cell->getPolarityMagnitude(),
                cell->getActinFlowDirection(), cell->getActinFlowMagnitude(), cell->getShapeDirection(),
                cell->getMovementDirection(), cell->getDirectionalInfluence(),
                cell->getDirectionalIntensity(), cell->getDirectionalShift(), cell->getSampledAngle(),
                cell->getTotalCILEffectX(), cell->getTotalCILEffectY(), cell->getEffectiveRadius()
            }) {
                ASSERT_TRUE(std::isfinite(value)) << "step " << step << ", cell " << cell->getID();
            }
        }
        world->runSimulationStep();
    }
}

TEST(MovementState, VerboseOutputFieldsThatAreNeverUpdated) {
    // These verbose-output fields are never updated by the model, so they stay at their initial
    // values. If one starts being computed, update this test (and the analyses that read them).
    CellSpec spec;
    spec.heading = 0.4;
    spec.fluctuationAmplitude = 1e-3;
    auto cell{makeCell(spec)};
    for (int step = 0; step < 50; ++step) {cell->takeRandomStep();}
    EXPECT_EQ(cell->getMovementDirection(), 0);
    EXPECT_EQ(cell->getDirectionalShift(), 0);
    EXPECT_EQ(cell->getSampledAngle(), 0);
    EXPECT_EQ(cell->getTotalCILEffectX(), 0);
    EXPECT_EQ(cell->getTotalCILEffectY(), 0);
    EXPECT_EQ(cell->getPolarityDirection(), 0.4);
    EXPECT_EQ(cell->getPolarityMagnitude(), 1e-5);
}
