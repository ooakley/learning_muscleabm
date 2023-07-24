#include "CellAgent.h"

#include <random>

// Constructor:
CellAgent::CellAgent(float startX, float startY){
    x = startX;
    y = startY;
}

// Public Definitions:

// Constructor and initialisers:
void CellAgent::setPosition(float newX, float newY) {
  x = newX;
  y = newY;
}

void CellAgent::initialiseRng(int seed) {
    rng = std::mt19937(seed);
}

void CellAgent::initialiseNormalDistribution(float std) {
    normDist = std::normal_distribution<float>(0, std);
}

// Getters and setters:
float CellAgent::getX() {
  return x; 
}

float CellAgent::getY() {
  return y; 
}

// Simulation code:
void CellAgent::takeRandomStep() {
  float stepX = normDist(rng);
  float stepY = normDist(rng);

  x += stepX;
  y += stepY;
}