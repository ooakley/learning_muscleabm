#pragma once
#include "agents.h"

#include <array>
#include <unordered_map>
#include <vector>

using AgentPointer = CellAgent*;
using GridUnit = std::unordered_map<int, AgentPointer>;
using CollisionRow = std::vector<GridUnit>;
using CollisionMatrix = std::vector<CollisionRow>;

class CollisionCellList {
public:
    // Constructor and intialisation:
    CollisionCellList
    (
        int setCollisionElements,
        double fieldSize
    );
    
    // The cell map:
    CollisionMatrix collisionMatrix;
    int collisionElements;
    double lengthCollisionElement;

    // Setter functions:
    void addToCollisionMatrix(double x, double y, AgentPointer agentPointer);
    void removeFromCollisionMatrix(double x, double y, AgentPointer agentPointer);
    void getLocalAgents(double x, double y, std::vector<AgentPointer>& localAgents);

    // Utility functions:
    int rollOverIndex(int index) const;
    std::array<int, 2> getIndexFromLocation(double positionX, double positionY);
};