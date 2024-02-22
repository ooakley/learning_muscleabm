#include "collision.h"
#include <unordered_map>

using AgentPointer = CellAgent*;
using GridUnit = std::unordered_map<int, AgentPointer>;
using CollisionRow = std::vector<GridUnit>;
using CollisionMatrix = std::vector<CollisionRow>;

CollisionCellList::CollisionCellList(
    int setCollisionElements, double fieldSize
    )
{
    // Initialise the collision matrix:
    CollisionMatrix collisionMatrix{};
    for (int i = 0; i < setCollisionElements; ++i) {
        CollisionRow rowConstruct{};
        for (int j = 0; j < setCollisionElements; ++j) {
            rowConstruct.push_back(GridUnit());
        }
        collisionMatrix.push_back(rowConstruct);
    }
}

void CollisionCellList::addToCollisionMatrix(int i, int j, AgentPointer agentPointer) {
    std::pair<int, AgentPointer> valuePair{agentPointer->getID(), agentPointer};
    collisionMatrix[i][j].insert(valuePair);
}

void CollisionCellList::removeFromCollisionMatrix(int i, int j, AgentPointer agentPointer) {
    collisionMatrix[i][j].erase(agentPointer->getID());
}
