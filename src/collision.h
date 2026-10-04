#pragma once
#include "agents.h"

#include <array>
#include <vector>

using AgentPointer = CellAgent*;

// A grid over the (periodic) world, for finding the cells that an acting cell could collide with.
// Each cell is registered in every grid element that its capsule - the segment from its centre
// to its stadium point, widened by the interaction radius - overlaps. Any cell whose capsule
// contains a point is then registered in that point's grid element, so the cells an acting cell
// could collide with are those registered at its centre and at its stadium point.
class CollisionCellList {
public:
    // Constructor and intialisation:
    CollisionCellList() = default;
    CollisionCellList(double setFieldSize, double setInteractionRadius, int setNumberOfCells);

    // Setter functions:
    void addToCollisionMatrix(AgentPointer agentPointer);
    void removeFromCollisionMatrix(AgentPointer agentPointer);

    // Finds the cells that could collide with the acting cell, sorted by ID, so that collisions are
    // resolved in an order that does not depend on the grid:
    void getLocalAgents(const CellAgent& actingAgent, std::vector<AgentPointer>& localAgents) const;

private:
    double fieldSize;
    double interactionRadius;
    int collisionElements;
    double lengthCollisionElement;

    // Registered cells of each grid element, indexed by row * collisionElements + column:
    std::vector<std::vector<AgentPointer>> collisionMatrix;

    // Grid elements each cell is registered in, as {first row, last row, first column, last
    // column}, indexed by cell ID:
    std::vector<std::array<int, 4>> registeredRanges;

    // Utility functions:
    int rollOverIndex(int index) const;
    int getIndexFromPosition(double position) const;
    double takePeriodicModulus(double queryPosition, double localPosition) const;
    std::array<int, 2> getAxisRange(double start, double end) const;
};
