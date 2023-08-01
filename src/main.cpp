#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include "agents.h"

const int width = 100;
const int height = 100;

int main(int argc, char** argv) {
    std::vector<CellAgent> agentVector;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> xdis(0, 100);
    std::uniform_real_distribution<double> ydis(0, 100);
    std::uniform_real_distribution<double> angledis(-M_PI, M_PI);

    std::ofstream csvfile;
    csvfile.open("positions.csv");

    for(int i=0; i<100; i++) {
        agentVector.push_back(
            CellAgent(xdis(gen), ydis(gen), angledis(gen), 1, 10, 1.5, i)
        ); 
    }

    for(int t=0; t<10000; t++) {
        for(auto& agent : agentVector) {
            agent.takeRandomStep();
            csvfile << t << ",";
            csvfile << agent.getID() << "," << agent.getX() << "," << agent.getY(); 
            csvfile << "\n";
        }
    }

    return 0;
}
