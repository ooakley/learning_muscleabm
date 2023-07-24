#include "gtest/gtest.h"

// Running all tests and including main:
int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}