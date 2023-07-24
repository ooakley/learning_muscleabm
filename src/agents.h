#include <random>

class CellAgent {
public:
    // Constructor and initialisers:
    CellAgent(float startX, float startY);
    void setPosition(float newX, float newY);
    void initialiseRng(int seed);
    void initialiseNormalDistribution(float std);

    // Getters and setters:
    float getX();
    float getY();

    // Simulation code:
    void takeRandomStep();

private:
    float x;
    float y;
    std::mt19937 rng;
    std::normal_distribution<float> normDist;
};