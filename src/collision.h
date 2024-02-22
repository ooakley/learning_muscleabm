#pragma once
#include "agents.h"

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

    CollisionMatrix collisionMatrix;
    void addToCollisionMatrix(int i, int j, AgentPointer agentPointer);
    void removeFromCollisionMatrix(int i, int j, AgentPointer agentPointer);
};