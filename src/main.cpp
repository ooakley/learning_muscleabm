#include <random>
#include <vector>
#include <fstream>

const int width = 100;
const int height = 100;

class Agent {
    public:
        double x;
        double y;

        Agent(double initX, double initY) : x(initX), y(initY) {}

        void step() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> dis(0, 1.0);

            double dx = dis(gen);
            double dy = dis(gen);

            x = std::fmod(x + dx, width);
            y = std::fmod(y + dy, height);
        }
};

int main(int argc, char** argv) {
    std::vector<Agent> agents;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> xdis(0, 100);
    std::uniform_real_distribution<> ydis(0, 100);


    for(int i=0; i<100; i++) {
        agents.push_back(Agent(xdis(gen), ydis(gen))); 
    }

    for(int t=0; t<10000; t++) {
        for(auto& agent : agents) {
            agent.step();
        }
    }

    return 0;
}
