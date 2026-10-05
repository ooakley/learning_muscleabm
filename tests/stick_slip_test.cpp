// Behavioural tests of stick-slip retraction of the cell rear (its stadium point): the rear only
// retracts, snaps back under large stretch, wraps correctly at the periodic boundary, and the
// implicit solver for the adhesion and extension equations is accurate.
#include <cmath>
#include <random>

#include "gtest/gtest.h"
#include "test_access.h"
#include "test_helpers.h"

using Access = CellAgentTestAccess;

namespace {

double extension(const CellAgent& cell) {
    return std::hypot(
        minimalImage(cell.getStadiumX(), cell.getX()), minimalImage(cell.getStadiumY(), cell.getY())
    );
}

std::unique_ptr<CellAgent> stretchedCell(double x, double y, double stadiumX, double stadiumY,
                                         double adhesion, CellSpec spec = {}) {
    spec.x = x;
    spec.y = y;
    auto cell{makeCell(spec)};
    Access::setStadium(*cell, stadiumX, stadiumY);
    Access::setAdhesionFraction(*cell, adhesion);
    return cell;
}

// The right-hand side of the stick-slip equations for adhesion A and extension X:
std::pair<double, double> stickSlipRates(const CellAgent& cell, double A, double X) {
    const double u{Access::stickSlipU(cell)};
    const double v{Access::stickSlipV(cell)};
    const double r{Access::stickSlipR(cell)};
    const double phi{std::exp(u * X / A)};
    return {r * (1 - A) - A * phi, -v * (X / A) * phi};
}

}  // namespace

TEST(StickSlip, RearOnlyRetracts) {
    std::mt19937 generator(1);
    std::uniform_real_distribution<double> uniform(0, 1);
    for (int trial = 0; trial < 500; ++trial) {
        CellSpec spec;
        spec.cellStiffness = 0.001 + uniform(generator);
        spec.surfaceStickiness = 0.01 + 50 * uniform(generator);
        const double theta{2 * M_PI * uniform(generator)};
        const double length{500 * uniform(generator)};
        auto cell{stretchedCell(
            1000, 1000, 1000 - length * std::cos(theta), 1000 - length * std::sin(theta),
            1e-3 + uniform(generator), spec
        )};
        const double stadiumXBefore{cell->getStadiumX()};
        const double stadiumYBefore{cell->getStadiumY()};

        Access::runStickSlip(*cell);
        EXPECT_LE(extension(*cell), length + 1e-9);
        // The rear moves along the segment, towards the centre:
        const double cross{
            (stadiumXBefore - 1000) * (cell->getStadiumY() - 1000) -
            (stadiumYBefore - 1000) * (cell->getStadiumX() - 1000)
        };
        EXPECT_NEAR(cross, 0, 1e-6 * (1 + length * length));
    }
}

TEST(StickSlip, LargeStretchSnapsTheRearBack) {
    // u.X/A = (1 / 1000) * 200 / 1e-3 = 200 > 100, so the rear snaps back to ~1e-3 px:
    CellSpec spec;
    spec.cellStiffness = 1;
    auto cell{stretchedCell(1000, 1000, 800, 1000, 1e-3, spec)};
    Access::runStickSlip(*cell);
    EXPECT_NEAR(extension(*cell), 1e-3, 1e-9);
    EXPECT_NEAR(Access::adhesionFraction(*cell), 1e-3, 1e-12);
}

TEST(StickSlip, SnapBackNeverLengthensAShorterCell) {
    // Snapping back sets the extension to 1e-3 px, which must not push out the rear of a cell
    // already shorter than that (here u.X/A = 1010 * 1e-4 / 1e-3 = 101 > 100):
    CellSpec spec;
    spec.cellStiffness = 1.01e6;
    auto cell{stretchedCell(1000, 1000, 1000 - 1e-4, 1000, 1e-3, spec)};
    Access::runStickSlip(*cell);
    EXPECT_LE(extension(*cell), 1e-4 + 1e-12);
}

TEST(StickSlip, RearWrapsAcrossThePeriodicBoundary) {
    // A centre at x = 2 with its rear at x = 2040 is 10 px long through the boundary, on both axes:
    for (const bool alongY : {false, true}) {
        auto cell{alongY ? stretchedCell(500, 2, 500, 2040, 0.5) : stretchedCell(2, 500, 2040, 500, 0.5)};
        Access::runStickSlip(*cell);
        for (double coordinate : {cell->getStadiumX(), cell->getStadiumY()}) {
            EXPECT_GE(coordinate, 0);
            EXPECT_LT(coordinate, WORLD_SIZE);
        }
        EXPECT_LE(extension(*cell), 10 + 1e-9);
        // The rear stays on the far side of the boundary, behind the centre:
        const double behind{alongY ? minimalImage(cell->getStadiumY(), 2) : minimalImage(cell->getStadiumX(), 2)};
        EXPECT_GE(behind, 0);
    }
}

TEST(StickSlip, ZeroExtensionStaysAtTheCentre) {
    auto cell{stretchedCell(700, 300, 700, 300, 0.5)};
    Access::runStickSlip(*cell);
    EXPECT_EQ(cell->getStadiumX(), 700);
    EXPECT_EQ(cell->getStadiumY(), 300);
}

TEST(StickSlip, NearTheSnapThresholdTheStateStaysFinite) {
    // u.X/A between 90 and 100, just below the snap-back threshold, where the solver struggles:
    for (double length = 900; length <= 1000; length += 5) {
        auto cell{stretchedCell(1500, 1000, 1500 - length, 1000, 1e-3)};
        Access::runStickSlip(*cell);
        EXPECT_TRUE(std::isfinite(cell->getStadiumX()));
        EXPECT_TRUE(std::isfinite(Access::adhesionFraction(*cell)));
        EXPECT_GE(cell->getStadiumX(), 0);
        EXPECT_LT(cell->getStadiumX(), WORLD_SIZE);
        EXPECT_LE(extension(*cell), length + 1e-9);
    }
}

TEST(StickSlip, OverManyStepsTheRearTrailsTheFront) {
    // Each step the front moves by the flow magnitude and the rear only retracts, so extension
    // grows by at most the step length:
    CellSpec spec;
    spec.fluctuationAmplitude = 1e-3;
    spec.cellStiffness = 0.01;
    spec.surfaceStickiness = 30;
    auto cell{makeCell(spec)};
    for (int step = 0; step < 2000; ++step) {
        const double before{extension(*cell)};
        cell->takeRandomStep();
        ASSERT_LE(extension(*cell), before + std::abs(cell->getActinFlowMagnitude()) + 1e-9)
            << "step " << step;
    }
}

// --- The implicit solver ---

TEST(StickSlipSolver, ConvergedStepsSatisfyTheImplicitEquations) {
    // A converged backward Euler step from (A0, X0) satisfies
    // A = A0 + h.dA/dt(A, X) and X = X0 + h.dX/dt(A, X):
    CellSpec spec;
    spec.cellStiffness = 5;
    auto cell{makeCell(spec)};
    int converged{0};
    for (double A0 : {0.01, 0.1, 0.5, 0.9}) {
        for (double X0 : {0.5, 5.0, 20.0}) {
            for (double h : {1e-4, 1e-3, 6e-3}) {
                const auto [A, X, failed] = Access::implicitNextState(*cell, h, A0, X0);
                if (failed) {continue;}
                converged += 1;
                const auto [rateA, rateX] = stickSlipRates(*cell, A, X);
                EXPECT_NEAR(A, A0 + h * rateA, 1e-6) << "A0 " << A0 << ", X0 " << X0 << ", h " << h;
                EXPECT_NEAR(X, X0 + h * rateX, 1e-6 * (1 + X0)) << "A0 " << A0 << ", X0 " << X0 << ", h " << h;
            }
        }
    }
    EXPECT_GT(converged, 30);
}

TEST(StickSlipSolver, RepeatedStepsConvergeToTheTrueSolution) {
    // Backward Euler is first order, so halving the step should about halve the error against an
    // accurate (RK4) solution:
    CellSpec spec;
    spec.cellStiffness = 50;
    auto cell{makeCell(spec)};
    const double A0{0.5};
    const double X0{20};
    const double duration{0.2};

    double referenceA{A0};
    double referenceX{X0};
    const int referenceSteps{20000};
    const double dt{duration / referenceSteps};
    for (int i = 0; i < referenceSteps; ++i) {
        const auto [a1, x1] = stickSlipRates(*cell, referenceA, referenceX);
        const auto [a2, x2] = stickSlipRates(*cell, referenceA + dt / 2 * a1, referenceX + dt / 2 * x1);
        const auto [a3, x3] = stickSlipRates(*cell, referenceA + dt / 2 * a2, referenceX + dt / 2 * x2);
        const auto [a4, x4] = stickSlipRates(*cell, referenceA + dt * a3, referenceX + dt * x3);
        referenceA += dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4);
        referenceX += dt / 6 * (x1 + 2 * x2 + 2 * x3 + x4);
    }

    double previousError{0};
    for (int steps : {16, 32, 64, 128}) {
        double A{A0};
        double X{X0};
        for (int i = 0; i < steps; ++i) {
            const auto [nextA, nextX, failed] = Access::implicitNextState(*cell, duration / steps, A, X);
            ASSERT_FALSE(failed) << steps << " steps";
            A = nextA;
            X = nextX;
        }
        const double error{std::hypot(A - referenceA, X - referenceX)};
        if (previousError > 0) {
            EXPECT_NEAR(previousError / error, 2, 0.3) << steps << " steps";
        }
        previousError = error;
    }
}
