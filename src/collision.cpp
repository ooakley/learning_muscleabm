#include "collision.h"

#include <algorithm>
#include <cmath>

CollisionCellList::CollisionCellList(
    double setFieldSize, double setInteractionRadius, int setNumberOfCells
)
    : fieldSize{setFieldSize}
    , interactionRadius{setInteractionRadius}
    , collisionElements{std::max(1, static_cast<int>(setFieldSize / setInteractionRadius))}
    , lengthCollisionElement{setFieldSize / collisionElements}
    , collisionMatrix(collisionElements * collisionElements)
    , registeredRanges(setNumberOfCells)
{
}

void CollisionCellList::addToCollisionMatrix(AgentPointer agentPointer) {
    // Find the cell's segment, taking the image of the stadium point nearest the cell centre:
    const double startX{agentPointer->getX()};
    const double startY{agentPointer->getY()};
    const double endX{takePeriodicModulus(agentPointer->getStadiumX(), startX)};
    const double endY{takePeriodicModulus(agentPointer->getStadiumY(), startY)};

    // Register the cell in every grid element its capsule overlaps:
    const auto [firstRow, lastRow] = getAxisRange(startY, endY);
    const auto [firstColumn, lastColumn] = getAxisRange(startX, endX);
    for (int row = firstRow; row <= lastRow; ++row) {
        const int rowOffset{rollOverIndex(row) * collisionElements};
        for (int column = firstColumn; column <= lastColumn; ++column) {
            collisionMatrix[rowOffset + rollOverIndex(column)].push_back(agentPointer);
        }
    }
    registeredRanges[static_cast<int>(agentPointer->getID())] = {firstRow, lastRow, firstColumn, lastColumn};
}

void CollisionCellList::removeFromCollisionMatrix(AgentPointer agentPointer) {
    const auto [firstRow, lastRow, firstColumn, lastColumn] =
        registeredRanges[static_cast<int>(agentPointer->getID())];
    for (int row = firstRow; row <= lastRow; ++row) {
        const int rowOffset{rollOverIndex(row) * collisionElements};
        for (int column = firstColumn; column <= lastColumn; ++column) {
            std::vector<AgentPointer>& gridUnit{collisionMatrix[rowOffset + rollOverIndex(column)]};
            auto agentIterator{std::find(gridUnit.begin(), gridUnit.end(), agentPointer)};
            *agentIterator = gridUnit.back();
            gridUnit.pop_back();
        }
    }
}

void CollisionCellList::getLocalAgents(
    const CellAgent& actingAgent, std::vector<AgentPointer>& localAgents
) const {
    localAgents.clear();

    // Collect the cells registered at the acting cell's centre and stadium point:
    const std::array<std::array<double, 2>, 2> queryPoints{{
        {actingAgent.getX(), actingAgent.getY()},
        {actingAgent.getStadiumX(), actingAgent.getStadiumY()}
    }};
    for (const auto& [queryX, queryY] : queryPoints) {
        const int row{getIndexFromPosition(queryY)};
        const int column{getIndexFromPosition(queryX)};
        const std::vector<AgentPointer>& gridUnit{collisionMatrix[row * collisionElements + column]};
        localAgents.insert(localAgents.end(), gridUnit.begin(), gridUnit.end());
    }

    // Sort by ID and remove cells found at both points:
    std::sort(
        localAgents.begin(), localAgents.end(),
        [](AgentPointer a, AgentPointer b) {return a->getID() < b->getID();}
    );
    localAgents.erase(std::unique(localAgents.begin(), localAgents.end()), localAgents.end());
}

int CollisionCellList::rollOverIndex(int index) const {
    return ((index % collisionElements) + collisionElements) % collisionElements;
}

int CollisionCellList::getIndexFromPosition(double position) const {
    return rollOverIndex(static_cast<int>(std::floor(position / lengthCollisionElement)));
}

double CollisionCellList::takePeriodicModulus(double queryPosition, double localPosition) const {
    // Find the image of the query position nearest the local position:
    double modulusPosition{queryPosition};
    if (localPosition - queryPosition > (fieldSize / 2)) {
        modulusPosition += fieldSize;
    }
    else if (localPosition - queryPosition < -(fieldSize / 2)) {
        modulusPosition -= fieldSize;
    }
    return modulusPosition;
}

std::array<int, 2> CollisionCellList::getAxisRange(double start, double end) const {
    // Grid elements covered along one axis by a segment widened by the interaction radius (with a
    // small margin for rounding), before rolling over the periodic boundary:
    const double margin{1e-3};
    const double lower{std::min(start, end) - interactionRadius - margin};
    const double upper{std::max(start, end) + interactionRadius + margin};
    const int first{static_cast<int>(std::floor(lower / lengthCollisionElement))};
    const int last{static_cast<int>(std::floor(upper / lengthCollisionElement))};

    // Cover the whole axis once if the capsule spans it:
    if (last - first + 1 >= collisionElements) {
        return {0, collisionElements - 1};
    }
    return {first, last};
}
