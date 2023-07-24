#include "gtest/gtest.h"
#include "CellAgent.h"

class CellAgentTest : public ::testing::Test {
protected:
  void SetUp() override {
    cellAgent = new CellAgent(0, 0);
    cellAgent->setPosition(0, 0);
    cellAgent->initialiseRng(0);
    cellAgent->initialiseNormalDistribution(3);
  }

  void TearDown() override {
    delete cellAgent;
  }

  CellAgent* cellAgent;
};

TEST_F(CellAgentTest, RandomWalkAtZero) {
    float previousX{0};
    float previousY{0};
    // Take random steps multiple times
    for (int i = 0; i < 100; i++) {
        cellAgent->takeRandomStep();

        // Check that the position has changed:
        EXPECT_NE(previousX, cellAgent->getX());
        EXPECT_NE(previousY, cellAgent->getY());

        // // Update the start position:
        // previousX = cellAgent->getX();
        // previousY = cellAgent->getY();
    }
}