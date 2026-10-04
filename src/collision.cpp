#include "collision.h"

#include <cmath>
#include <unordered_map>

CollisionCellList::CollisionCellList(
    int setCollisionElements, double fieldSize
)
    : collisionElements{setCollisionElements}
    , lengthCollisionElement{fieldSize / setCollisionElements}
{
    // Initialise the collision matrix:
    for (int i = 0; i < collisionElements; ++i) {
        CollisionRow rowConstruct{};
        for (int j = 0; j < collisionElements; ++j) {
            GridUnit emptyUnit;
            rowConstruct.push_back(emptyUnit);
        }
        collisionMatrix.push_back(rowConstruct);
    }
}

void CollisionCellList::addToCollisionMatrix(double x, double y, AgentPointer agentPointer) {
    const auto indices{getIndexFromLocation(x, y)};
    collisionMatrix[indices[0]][indices[1]].insert({agentPointer->getID(), agentPointer});
}

void CollisionCellList::removeFromCollisionMatrix(double x, double y, AgentPointer agentPointer) {
    const auto indices{getIndexFromLocation(x, y)};
    collisionMatrix[indices[0]][indices[1]].erase(agentPointer->getID());
}

void CollisionCellList::getLocalAgents(double x, double y, std::vector<AgentPointer>& localAgents) {
    localAgents.clear();
    const auto indices{getIndexFromLocation(x, y)};

    // Loop through neighbourhood:
    for (int k = -1; k < 2; ++k) {
        for (int l = -1; l < 2; ++l) {
            int safeRow{rollOverIndex(indices[0] + k)};
            int safeCol{rollOverIndex(indices[1] + l)};
            for (auto& [agentID, pointer]: collisionMatrix[safeRow][safeCol]) {
                localAgents.emplace_back(pointer);
            }
        }
    }
}

int CollisionCellList::rollOverIndex(int index) const {
    while (index < 0) {
        index = index + collisionElements;
    }
    return index % collisionElements;
}

std::array<int, 2> CollisionCellList::getIndexFromLocation(double positionX, double positionY) {
    // Get indices:
    int xIndex{int(std::floor(positionX / lengthCollisionElement))};
    int yIndex{int(std::floor(positionY / lengthCollisionElement))};

    // Note that the y index goes first here because of how we index matrices:
    return {yIndex, xIndex};
}