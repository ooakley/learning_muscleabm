#pragma once
// Access to the internals of CellAgent and World, so that behavioural tests can set up exact
// scenarios (a cell's stadium point, adhesion, flow) and call the steps of the model directly.
#include <tuple>
#include <vector>

#include "agents.h"
#include "world.h"

class CellAgentTestAccess {
public:
    // State:
    static void setStadium(CellAgent& cell, double stadiumX, double stadiumY) {
        cell.stadiumX = stadiumX;
        cell.stadiumY = stadiumY;
    }
    static void setFlow(CellAgent& cell, double direction, double magnitude) {
        cell.flowDirection = direction;
        cell.flowMagnitude = magnitude;
    }
    static double adhesionFraction(const CellAgent& cell) {return cell.adhesionFraction;}
    static void setAdhesionFraction(CellAgent& cell, double adhesionFraction) {
        cell.adhesionFraction = adhesionFraction;
    }
    static double polarityChangeX(const CellAgent& cell) {return cell.polarityChangeCilX;}
    static double polarityChangeY(const CellAgent& cell) {return cell.polarityChangeCilY;}
    static void clearPolarityChange(CellAgent& cell) {
        cell.polarityChangeCilX = 0;
        cell.polarityChangeCilY = 0;
    }

    // Parameters of the stick-slip equations, dA/dt = r(1 - A) - A.exp(u.X/A) and
    // dX/dt = -v.(X/A).exp(u.X/A):
    static double stickSlipU(const CellAgent& cell) {return cell.cellStiffness / cell.adhesionFragility;}
    static double stickSlipV(const CellAgent& cell) {return cell.cellStiffness / cell.adhesionStiffness;}
    static double stickSlipR(const CellAgent& cell) {return cell.surfaceStickiness;}

    // Steps of the model:
    static void runCollisions(CellAgent& cell, const std::vector<CellAgent*>& localAgents) {
        cell.setLocalCellList(localAgents);
        cell.runTrajectoryDependentCollisionLogic();
    }
    static void runStickSlip(CellAgent& cell) {cell.runStickSlipLogic();}
    static std::tuple<double, double, bool> implicitNextState(
        CellAgent& cell, double stepSize, double adhesion, double extension
    ) {
        return cell.implicitNextState(stepSize, adhesion, extension);
    }
    static std::tuple<bool, double, double, double, double> isPositionInStadium(
        CellAgent& cell, double sampleX, double sampleY,
        double startX, double startY, double endX, double endY, double localRadius
    ) {
        return cell.isPositionInStadium(sampleX, sampleY, startX, startY, endX, endY, localRadius);
    }
    static double takePeriodicModulus(CellAgent& cell, double queryPosition, double localPosition) {
        return cell.takePeriodicModulus(queryPosition, localPosition);
    }
};

class WorldTestAccess {
public:
    static std::vector<CellAgent*> cells(World& world) {
        std::vector<CellAgent*> cellPointers;
        for (auto& cell : world.cellAgentVector) {cellPointers.push_back(cell.get());}
        return cellPointers;
    }
    static ECMField& ecm(World& world) {return world.ecmField;}
    static int ecmElementCount(const World& world) {return world.countECMElement;}
    static double ecmElementLength(const World& world) {return world.lengthECMElement;}

    // Moves a cell, keeping the collision grid up to date:
    static void placeCell(
        World& world, CellAgent& cell, double x, double y, double stadiumX, double stadiumY
    ) {
        world.collisionCellList.removeFromCollisionMatrix(&cell);
        cell.setPosition({x, y});
        CellAgentTestAccess::setStadium(cell, stadiumX, stadiumY);
        world.collisionCellList.addToCollisionMatrix(&cell);
    }

    static void runCellStep(World& world, CellAgent& cell) {world.runCellStep(cell);}
    static std::tuple<double, double> rollPosition(World& world, double x, double y) {
        return world.rollPosition({x, y});
    }
    static double cellDeltaTowardsECM(World& world, double ecmHeading, double cellHeading) {
        return world.calculateCellDeltaTowardsECM(ecmHeading, cellHeading);
    }
};
