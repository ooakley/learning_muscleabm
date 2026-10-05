// Behavioural tests of the matrix: where cells sample and deposit fibres, what they sense from
// them, the background pattern, and how fibres couple to a cell's actin flow.
#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "ecm.h"
#include "test_access.h"
#include "test_helpers.h"

using Access = CellAgentTestAccess;

namespace {

double nematic(double angle) {
    while (angle < 0) {angle += M_PI;}
    while (angle >= M_PI) {angle -= M_PI;}
    return angle;
}

// Total number of fibres in a world's matrix:
int countFibres(World& world) {
    ECMField& ecm{WorldTestAccess::ecm(world)};
    const int elements{WorldTestAccess::ecmElementCount(world)};
    int total{0};
    for (int i = 0; i < elements; ++i) {
        for (int j = 0; j < elements; ++j) {total += static_cast<int>(ecm.getFibreDeque(i, j).size());}
    }
    return total;
}

// Fills every site of a world's matrix with the given fibre headings:
void fillMatrix(World& world, const std::vector<double>& headings) {
    ECMField& ecm{WorldTestAccess::ecm(world)};
    const int elements{WorldTestAccess::ecmElementCount(world)};
    for (int i = 0; i < elements; ++i) {
        for (int j = 0; j < elements; ++j) {
            for (double heading : headings) {ecm.addToFibreMatrix(i, j, heading);}
        }
    }
}

}  // namespace

// --- Sampling and deposition ---

TEST(MatrixDeposition, AttachmentPointsAreUniformOverTheCellBody) {
    CellSpec spec;
    spec.cellBodyRadius = 40;
    auto cell{makeCell(spec)};
    const int sampleCount{40000};
    int innerCount{0};
    double sumCos{0};
    for (int i = 0; i < sampleCount; ++i) {
        const auto [x, y] = cell->sampleAttachmentPoint();
        const double r{std::hypot(x - spec.x, y - spec.y)};
        ASSERT_LE(r, spec.cellBodyRadius);
        innerCount += r < spec.cellBodyRadius / 2;
        sumCos += std::cos(std::atan2(y - spec.y, x - spec.x));
    }
    // A quarter of the area lies within half the radius:
    EXPECT_NEAR(static_cast<double>(innerCount) / sampleCount, 0.25, 0.01);
    EXPECT_NEAR(sumCos / sampleCount, 0, 0.01);
}

TEST(MatrixDeposition, FibresFollowFlowAndWrapAroundTheWorld) {
    // A cell in the corner deposits on all four corners of the matrix:
    WorldSpec spec;
    spec.matrixSampleRate = 200;
    spec.cell.cellBodyRadius = 50;
    spec.cell.fluctuationAmplitude = 1e-2;  // So that the flow direction changes during the step.
    auto world{makeWorld(spec)};
    CellAgent& cell{*WorldTestAccess::cells(*world)[0]};
    WorldTestAccess::placeCell(*world, cell, 5, 5, 4, 5);
    Access::setFlow(cell, 2.5, 1.0);

    WorldTestAccess::runCellStep(*world, cell);
    ASSERT_GT(std::abs(cell.getActinFlowDirection() - 2.5), 1e-3);
    const float expectedHeading{static_cast<float>(nematic(cell.getActinFlowDirection()))};
    ECMField& ecm{WorldTestAccess::ecm(*world)};
    const int elements{WorldTestAccess::ecmElementCount(*world)};
    const double elementLength{WorldTestAccess::ecmElementLength(*world)};
    int deposited{0};
    bool wrappedRow{false};
    bool wrappedColumn{false};
    for (int i = 0; i < elements; ++i) {
        for (int j = 0; j < elements; ++j) {
            const auto& fibres{ecm.getFibreDeque(i, j)};
            if (fibres.empty()) {continue;}
            deposited += static_cast<int>(fibres.size());
            for (float heading : fibres) {EXPECT_EQ(heading, expectedHeading);}
            // Every site lies within the cell body of the starting point (5, 5):
            const double siteX{(j + 0.5) * elementLength};
            const double siteY{(i + 0.5) * elementLength};
            EXPECT_LT(std::abs(minimalImage(5, siteX)), 50 + elementLength);
            EXPECT_LT(std::abs(minimalImage(5, siteY)), 50 + elementLength);
            wrappedRow |= i == elements - 1;
            wrappedColumn |= j == elements - 1;
        }
    }
    EXPECT_GT(deposited, 100);
    EXPECT_TRUE(wrappedRow);
    EXPECT_TRUE(wrappedColumn);
}

TEST(MatrixDeposition, NoSamplingWithZeroRate) {
    WorldSpec spec;
    spec.matrixSampleRate = 0;
    auto world{makeWorld(spec)};
    fillMatrix(*world, {0.3});
    CellAgent& cell{*WorldTestAccess::cells(*world)[0]};
    const int fibresBefore{countFibres(*world)};
    for (int step = 0; step < 20; ++step) {WorldTestAccess::runCellStep(*world, cell);}
    EXPECT_EQ(countFibres(*world), fibresBefore);
    EXPECT_EQ(cell.getDirectionalIntensity(), 0);
    EXPECT_EQ(cell.getDirectionalInfluence(), 0);
}

// --- Sensing ---

TEST(MatrixSensing, AlignedFibresGiveFullIntensityAndTheirAngle) {
    // Fibres at 0.3 rad seen by a cell heading at 0, and fibres at 2.9 rad, which is
    // 2.9 - pi = -0.24 rad from the cell as a nematic direction:
    for (const auto& [fibreHeading, expectedInfluence] : {std::pair{0.3, 0.3}, std::pair{2.9, 2.9 - M_PI}}) {
        WorldSpec spec;
        spec.matrixSampleRate = 20;
        auto world{makeWorld(spec)};
        fillMatrix(*world, {fibreHeading});
        CellAgent& cell{*WorldTestAccess::cells(*world)[0]};
        Access::setFlow(cell, 0, 1.0);
        WorldTestAccess::runCellStep(*world, cell);
        EXPECT_EQ(cell.getDirectionalIntensity(), 1.0 - 1e-4);  // Clamped below 1.
        EXPECT_NEAR(cell.getDirectionalInfluence(), expectedInfluence, 1e-6);
    }
}

TEST(MatrixSensing, IsotropicFibresGiveLowIntensity) {
    WorldSpec spec;
    spec.matrixSampleRate = 400;
    auto world{makeWorld(spec)};
    std::vector<double> headings;
    for (int k = 0; k < 64; ++k) {headings.push_back(M_PI * k / 64);}
    fillMatrix(*world, headings);
    CellAgent& cell{*WorldTestAccess::cells(*world)[0]};
    WorldTestAccess::runCellStep(*world, cell);
    EXPECT_LT(cell.getDirectionalIntensity(), 0.15);
}

TEST(MatrixSensing, EmptyMatrixLeavesPerceptsAtZero) {
    WorldSpec spec;
    spec.matrixSampleRate = 20;
    auto world{makeWorld(spec)};
    CellAgent& cell{*WorldTestAccess::cells(*world)[0]};
    cell.setDirectionalIntensity(0.5);
    cell.setDirectionalInfluence(0.5);
    WorldTestAccess::runCellStep(*world, cell);
    EXPECT_EQ(cell.getDirectionalIntensity(), 0);
    EXPECT_EQ(cell.getDirectionalInfluence(), 0);
}

// --- Background pattern ---

TEST(MatrixPattern, FibreCountAndSpread) {
    // Doubled nematic headings drawn from N(0, sigma) have a mean resultant length of
    // exp(-2 sigma^2):
    const double sigma{0.2};
    WorldSpec spec;
    spec.patternSigma = sigma;
    spec.patternFibreCount = 30;
    spec.ecmElementCount = 16;
    auto world{makeWorld(spec)};
    ECMField& ecm{WorldTestAccess::ecm(*world)};
    double cosSum{0};
    double sinSum{0};
    int count{0};
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            const auto& fibres{ecm.getFibreDeque(i, j)};
            ASSERT_EQ(fibres.size(), 30u);
            for (float heading : fibres) {
                ASSERT_GE(heading, 0);
                ASSERT_LT(heading, M_PI);
                cosSum += std::cos(2 * heading);
                sinSum += std::sin(2 * heading);
                count += 1;
            }
        }
    }
    EXPECT_NEAR(std::hypot(cosSum, sinSum) / count, std::exp(-2 * sigma * sigma), 0.01);
    EXPECT_NEAR(sinSum / count, 0, 0.02);  // Centred on heading 0.
}

// --- Coupling of fibres to actin flow ---

namespace {

// Mean squared change in flow direction per step, scaled by flow magnitude, which for noise
// alone is the effective fluctuation amplitude:
double scaledAngularVariance(double coupling, double intensity, double influence) {
    CellSpec spec;
    spec.fluctuationAmplitude = 1e-4;
    spec.matrixCoupling = coupling;
    spec.seed = 3;
    auto cell{makeCell(spec)};
    Access::setFlow(*cell, 0, 1.0);
    cell->setDirectionalIntensity(intensity);
    cell->setDirectionalInfluence(influence);
    double sumSquares{0};
    const int steps{4000};
    for (int step = 0; step < steps; ++step) {
        const double before{cell->getActinFlowDirection()};
        const double magnitude{std::max(cell->getActinFlowMagnitude(), 1e-2)};
        cell->takeRandomStep();
        const double change{std::remainder(cell->getActinFlowDirection() - before, 2 * M_PI)};
        sumSquares += std::pow(change * magnitude, 2);
    }
    return sumSquares / steps;
}

}  // namespace

TEST(MatrixCoupling, AlignedFibresSuppressAngularNoise) {
    const double amplitude{1e-4};
    EXPECT_NEAR(scaledAngularVariance(0, 0.9, 0) / amplitude, 1, 0.1);
    EXPECT_NEAR(scaledAngularVariance(5, 0.9, 0) / amplitude, std::exp(-5 * 0.9), 0.003);
    // Perpendicular fibres (cos^2 = 0) do not suppress it:
    EXPECT_NEAR(scaledAngularVariance(5, 0.9, M_PI / 2) / amplitude, 1, 0.1);
}
