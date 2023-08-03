#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include "world.h"

int main(int argc, char** argv) {

    std::ofstream csvFile;
    csvFile.open("positions.csv");

    World mainWorld{World(0, 1000, 1000, 1000)};
    for (int i = 0; i < 1000; ++i) {
        mainWorld.runSimulationStep();
        mainWorld.writePositionsToCSV(csvFile);
    }

    return 0;
}
