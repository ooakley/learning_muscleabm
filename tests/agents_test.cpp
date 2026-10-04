#include "gtest/gtest.h"
#include "agents.h"

#include <cmath>

class CellAgentTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Basic cell agent:
        cellAgent = new CellAgent(
            0, 1, 1,
            1, 1, 1, 1, 1, 1, 1,
            1, 1, 1,
            1, 1,
            1,
            0, 0, 0
        );

        // Cell agent with the same random seed:
        cellAgentAlt = new CellAgent(
            0, 1, 1,
            1, 1, 1, 1, 1, 1, 1,
            1, 1, 1,
            1, 1,
            1,
            0, 0, 0
        );

        // Cell agent with a different random seed:
        cellAgentDifferent = new CellAgent(
            1, 1, 1,
            1, 1, 1, 1, 1, 1, 1,
            1, 1, 1,
            1, 1,
            1,
            0, 0, 0
        );
    }

    void TearDown() override {
        delete cellAgent;
        delete cellAgentAlt;
        delete cellAgentDifferent;
    }

    CellAgent* cellAgent;
    CellAgent* cellAgentAlt;
    CellAgent* cellAgentDifferent;
};

TEST_F(CellAgentTest, RandomWalkAtZero) {
    double previousX{0};
    double previousY{0};
    // Take random steps multiple times
    for (int i = 0; i < 100; i++) {
        cellAgent->takeRandomStep();

        // Check that the position has changed:
        EXPECT_NE(previousX, cellAgent->getX());
        EXPECT_NE(previousY, cellAgent->getY());

        // Update the start position:
        previousX = cellAgent->getX();
        previousY = cellAgent->getY();
    }
}

TEST_F(CellAgentTest, ReproducibleRandomWalk) {
    // Take random steps multiple times
    for (int i = 0; i < 100; i++) {
        cellAgent->takeRandomStep();
        cellAgentAlt->takeRandomStep();

        // Check both agents have taken exactly the same step:
        EXPECT_DOUBLE_EQ(cellAgent->getX(), cellAgentAlt->getX());
        EXPECT_DOUBLE_EQ(cellAgent->getY(), cellAgentAlt->getY());
    }
}

TEST_F(CellAgentTest, SeededRandomWalk) {
    // Take random steps multiple times
    for (int i = 0; i < 100; i++) {
        cellAgent->takeRandomStep();
        cellAgentDifferent->takeRandomStep();

        // Check both agents have taken different steps:
        EXPECT_NE(cellAgent->getX(), cellAgentDifferent->getX());
        EXPECT_NE(cellAgent->getY(), cellAgentDifferent->getY());
    }
}
TEST(CellAgentSeeding, IndependentActinFlowNoise) {
    // Without collisions or matrix, changes in actin flow direction are driven by noise alone,
    // so if cells with different seeds draw independent noise, they turn the same way about
    // half the time:
    CellAgent cellA(
        0, 1, 1,
        0.01, 1, 1e-4, 20, 1, 0, 1,
        50, 0, 0,
        0.1, 10,
        0,
        0, 0, 0
    );
    CellAgent cellB(
        1, 2, 1,
        0.01, 1, 1e-4, 20, 1, 0, 1,
        50, 0, 0,
        0.1, 10,
        0,
        0, 0, 0
    );

    int stepCount{1000};
    int sameTurnCount{0};
    for (int i = 0; i < stepCount; i++) {
        double previousA{cellA.getActinFlowDirection()};
        double previousB{cellB.getActinFlowDirection()};
        cellA.takeRandomStep();
        cellB.takeRandomStep();
        double turnA{std::remainder(cellA.getActinFlowDirection() - previousA, 2*M_PI)};
        double turnB{std::remainder(cellB.getActinFlowDirection() - previousB, 2*M_PI)};
        if ((turnA > 0) == (turnB > 0)) {
            sameTurnCount += 1;
        }
    }
    EXPECT_NEAR(static_cast<double>(sameTurnCount) / stepCount, 0.5, 0.1);
}
