#pragma once
#include "agents.h"

using AgentPointer = std::shared_ptr<CellAgent>;
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
    std::vector<AgentPointer> getLocalAgents(double x, double y);

    // Utility functions:
    int rollOverIndex(int index) const;
    std::vector<int> getIndexFromLocation(double positionX, double positionY);
};