#include "gtest/gtest.h"
#include "agents.h"

class CellAgentTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Basic cell agent:
        cellAgent = new CellAgent(
            0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1,
            1, 1,
            0, 0, 0
        );

        // Cell agent with the same random seed:
        cellAgentAlt = new CellAgent(
            0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1,
            1, 1,
            0, 0, 0
        );

        // Cell agent with a different random seed:
        cellAgentDifferent = new CellAgent(
            1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1,
            1, 1,
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

        // // Update the start position:
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