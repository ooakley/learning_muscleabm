// Behavioural tests of collisions between cells: detection, the effect on actin flow and contact
// inhibition, the periodic boundary, and the collision grid.
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "collision.h"
#include "test_access.h"
#include "test_helpers.h"

using Access = CellAgentTestAccess;

namespace {

// A cell with a short segment pointing back along -x, so that it is not zero-length:
std::unique_ptr<CellAgent> makeCellAt(double x, double y, int id, CellSpec spec = {}) {
    spec.x = x;
    spec.y = y;
    spec.id = id;
    auto cell{makeCell(spec)};
    Access::setStadium(*cell, x - 1, y);
    return cell;
}

}  // namespace

TEST(Collision, CellsCloserThanTwoRadiiCollide) {
    auto acting{makeCellAt(500, 500, 0)};
    auto local{makeCellAt(560, 500, 1)};  // Segment from x = 560 back to 559.
    Access::runCollisions(*acting, {local.get()});
    EXPECT_EQ(acting->getCollisionNumber(), 1);
}

TEST(Collision, ThresholdIsSumOfRadii) {
    // The local segment's nearest point is its stadium point at x = 559, so collisions happen
    // below a centre-to-segment distance of 2R = 100:
    for (const auto& [offset, expected] : {std::pair{99.999, 1}, std::pair{100.001, 0}}) {
        auto acting{makeCellAt(500, 500, 0)};
        auto local{makeCellAt(500 + offset + 1, 500, 1)};
        Access::runCollisions(*acting, {local.get()});
        EXPECT_EQ(acting->getCollisionNumber(), expected) << "distance " << offset;
    }
}

TEST(Collision, CellsWithZeroLengthSegmentsCollide) {
    // Every cell starts with its rear at its centre, so its segment has zero length:
    CellSpec spec;
    spec.x = 500;
    auto acting{makeCell(spec)};
    spec.x = 560;
    spec.id = 1;
    auto local{makeCell(spec)};
    ASSERT_EQ(local->getStadiumX(), local->getX());

    Access::runCollisions(*acting, {local.get()});
    EXPECT_EQ(acting->getCollisionNumber(), 1);
    // A zero-length segment is a point: the distance to it is the distance to the centre.
    const auto [colliding, closestX, closestY, distance, dotProduct] = Access::isPositionInStadium(
        *acting, 500, 500, 560, 500, 560, 500, local->getEffectiveRadius()
    );
    EXPECT_TRUE(colliding);
    EXPECT_EQ(closestX, 560);
    EXPECT_EQ(distance, 60);
}

TEST(Collision, DetectedAcrossPeriodicBoundary) {
    // Cells at x = 10 and x = 2040 are 18 px apart through the boundary:
    auto acting{makeCellAt(10, 500, 0)};
    auto local{makeCellAt(2040, 500, 1)};
    Access::runCollisions(*acting, {local.get()});
    EXPECT_EQ(acting->getCollisionNumber(), 1);

    // Contact inhibition pushes each cell away from the other, through the boundary: the acting
    // cell towards +x, the local cell towards -x.
    EXPECT_GT(Access::polarityChangeX(*acting), 0.99);
    EXPECT_LT(Access::polarityChangeX(*local), -0.99);
}

TEST(Collision, NoPhantomCollisionAcrossHalfTheWorld) {
    // Regression: the ends of a neighbour about 1024 px away were wrapped separately, giving a
    // segment that spanned the world and passed by the acting cell. These positions come from a
    // simulation where that happened.
    for (const bool alongY : {false, true}) {
        auto place = [alongY](double a, double b) {return alongY ? std::pair{b, a} : std::pair{a, b};};
        const auto [actingX, actingY] = place(1162.16, 642.848);
        const auto [localX, localY] = place(138.103, 579.097);
        const auto [localStadiumX, localStadiumY] = place(138.194, 579.139);
        auto acting{makeCellAt(actingX, actingY, 0)};
        auto local{makeCellAt(localX, localY, 1)};
        Access::setStadium(*local, localStadiumX, localStadiumY);
        Access::setStadium(*acting, actingX, actingY + 0.1);
        const double adhesionBefore{Access::adhesionFraction(*acting)};

        Access::runCollisions(*acting, {local.get()});
        EXPECT_EQ(acting->getCollisionNumber(), 0) << (alongY ? "along y" : "along x");
        EXPECT_EQ(Access::adhesionFraction(*acting), adhesionBefore) << (alongY ? "along y" : "along x");
    }
}

TEST(Collision, FlowTowardsNeighbourIsReducedButNeverReversed) {
    CellSpec spec;
    spec.collisionFlowReductionRate = 100;  // Far more than enough to stop the cell.
    auto acting{makeCellAt(500, 500, 0, spec)};
    auto local{makeCellAt(560, 500, 1)};
    Access::setFlow(*acting, 0, 1.0);  // Heading straight at the neighbour.

    Access::runCollisions(*acting, {local.get()});
    const double flowX{std::cos(acting->getActinFlowDirection()) * acting->getActinFlowMagnitude()};
    EXPECT_GE(flowX, -1e-12);
    EXPECT_LT(acting->getActinFlowMagnitude(), 1.0);
}

TEST(Collision, FlowAwayFromNeighbourIsUnchanged) {
    auto acting{makeCellAt(500, 500, 0)};
    auto local{makeCellAt(560, 500, 1)};
    Access::setFlow(*acting, M_PI - 1e-9, 0.7);  // Heading away from the neighbour.

    Access::runCollisions(*acting, {local.get()});
    EXPECT_EQ(acting->getCollisionNumber(), 1);
    EXPECT_DOUBLE_EQ(acting->getActinFlowMagnitude(), 0.7);
    EXPECT_DOUBLE_EQ(acting->getActinFlowDirection(), M_PI - 1e-9);
}

TEST(Collision, ContactInhibitionIsEqualAndOpposite) {
    auto acting{makeCellAt(500, 500, 0)};
    auto local{makeCellAt(530, 540, 1)};
    Access::runCollisions(*acting, {local.get()});
    ASSERT_EQ(acting->getCollisionNumber(), 1);
    EXPECT_DOUBLE_EQ(Access::polarityChangeX(*acting), -Access::polarityChangeX(*local));
    EXPECT_DOUBLE_EQ(Access::polarityChangeY(*acting), -Access::polarityChangeY(*local));
    EXPECT_NEAR(std::hypot(Access::polarityChangeX(*acting), Access::polarityChangeY(*acting)), 1, 1e-12);
}

TEST(Collision, AdhesionCollisionsCannotLeaveNegativeAdhesion) {
    // dt.adhesionReductionRate > 1 would make adhesion negative on a single collision:
    CellSpec spec;
    spec.adhesionReductionRate = 1.5;
    auto acting{makeCellAt(500, 500, 0, spec)};
    std::vector<std::unique_ptr<CellAgent>> neighbours;
    std::vector<CellAgent*> neighbourPointers;
    for (int i = 0; i < 3; ++i) {
        neighbours.push_back(makeCellAt(500 + 20*(i + 1), 500, i + 1));
        neighbourPointers.push_back(neighbours.back().get());
    }
    acting->setLocalCellList(neighbourPointers);
    acting->takeRandomStep();
    EXPECT_GE(Access::adhesionFraction(*acting), 0);
}

// --- The collision grid ---

namespace {

struct Population {
    std::vector<std::unique_ptr<CellAgent>> cells;
    std::vector<CellAgent*> pointers() const {
        std::vector<CellAgent*> result;
        for (const auto& cell : cells) {result.push_back(cell.get());}
        return result;
    }
};

// Cells at random positions, with segments up to maxExtension long in random directions:
Population randomPopulation(int count, double radius, double maxExtension, unsigned int seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> position(0, WORLD_SIZE);
    std::uniform_real_distribution<double> angle(-M_PI, M_PI);
    std::uniform_real_distribution<double> extension(0, maxExtension);
    Population population;
    for (int id = 0; id < count; ++id) {
        CellSpec spec;
        spec.id = id;
        spec.cellBodyRadius = radius;
        spec.x = position(generator);
        spec.y = position(generator);
        auto cell{makeCell(spec)};
        const double theta{angle(generator)};
        const double length{extension(generator) + 1e-3};
        Access::setStadium(
            *cell,
            std::fmod(spec.x - length*std::cos(theta) + WORLD_SIZE, WORLD_SIZE),
            std::fmod(spec.y - length*std::sin(theta) + WORLD_SIZE, WORLD_SIZE)
        );
        population.cells.push_back(std::move(cell));
    }
    return population;
}

void expectGridFindsEveryCollider(const Population& population, double radius) {
    CollisionCellList grid(WORLD_SIZE, 2*radius, static_cast<int>(population.cells.size()));
    for (CellAgent* cell : population.pointers()) {grid.addToCollisionMatrix(cell);}

    std::vector<CellAgent*> candidates;
    for (CellAgent* acting : population.pointers()) {
        grid.removeFromCollisionMatrix(acting);
        grid.getLocalAgents(*acting, candidates);

        // Sorted by ID, without duplicates, and never the acting cell:
        EXPECT_TRUE(std::is_sorted(candidates.begin(), candidates.end(),
            [](const CellAgent* a, const CellAgent* b) {return a->getID() < b->getID();}));
        EXPECT_EQ(std::adjacent_find(candidates.begin(), candidates.end()), candidates.end());
        EXPECT_EQ(std::find(candidates.begin(), candidates.end(), acting), candidates.end());

        // A superset of the cells it could collide with:
        for (CellAgent* local : population.pointers()) {
            if (local == acting || !couldCollide(*acting, *local)) {continue;}
            EXPECT_NE(std::find(candidates.begin(), candidates.end(), local), candidates.end())
                << "cell " << local->getID() << " missing for cell " << acting->getID();
        }
        grid.addToCollisionMatrix(acting);
    }
}

}  // namespace

TEST(CollisionGrid, FindsEveryPossibleCollider) {
    // Short and long segments (up to about 1000 px), including many crossing the boundary:
    for (const double maxExtension : {1.0, 200.0, 1000.0}) {
        for (const double radius : {12.0, 60.0}) {
            SCOPED_TRACE("max extension " + std::to_string(maxExtension) + ", radius " + std::to_string(radius));
            expectGridFindsEveryCollider(randomPopulation(150, radius, maxExtension, 7), radius);
        }
    }
}

TEST(CollisionGrid, InteractionRadiusLargerThanWorld) {
    // A single grid element, holding every cell:
    expectGridFindsEveryCollider(randomPopulation(20, 1500, 50, 3), 1500);
}

TEST(CollisionGrid, RemovedCellsAreNotFound) {
    Population population{randomPopulation(50, 30, 300, 11)};
    CollisionCellList grid(WORLD_SIZE, 60, 50);
    for (CellAgent* cell : population.pointers()) {grid.addToCollisionMatrix(cell);}

    // Remove a cell, move it elsewhere, and probe both places:
    CellAgent& moved{*population.cells[0]};
    const double oldX{moved.getX()};
    const double oldY{moved.getY()};
    grid.removeFromCollisionMatrix(&moved);
    CellSpec probeSpec;
    probeSpec.id = 99;
    probeSpec.cellBodyRadius = 30;
    probeSpec.x = oldX;
    probeSpec.y = oldY;
    auto probe{makeCell(probeSpec)};
    std::vector<CellAgent*> candidates;
    grid.getLocalAgents(*probe, candidates);
    EXPECT_EQ(std::find(candidates.begin(), candidates.end(), &moved), candidates.end());

    moved.setPosition({std::fmod(oldX + 1000, WORLD_SIZE), oldY});
    Access::setStadium(moved, moved.getX(), moved.getY());
    grid.addToCollisionMatrix(&moved);
    grid.getLocalAgents(*probe, candidates);
    EXPECT_EQ(std::find(candidates.begin(), candidates.end(), &moved), candidates.end());
    probe->setPosition({moved.getX(), moved.getY()});
    Access::setStadium(*probe, moved.getX(), moved.getY());
    grid.getLocalAgents(*probe, candidates);
    EXPECT_NE(std::find(candidates.begin(), candidates.end(), &moved), candidates.end());
}

TEST(CollisionGrid, CollisionOutcomeMatchesCheckingEveryCell) {
    // For every cell, resolving collisions against the grid's candidates gives exactly the same
    // result as resolving them against every other cell (both in ID order):
    const double radius{40};
    Population population{randomPopulation(120, radius, 400, 5)};
    CollisionCellList grid(WORLD_SIZE, 2*radius, 120);
    for (CellAgent* cell : population.pointers()) {grid.addToCollisionMatrix(cell);}

    std::vector<CellAgent*> candidates;
    for (CellAgent* acting : population.pointers()) {
        grid.removeFromCollisionMatrix(acting);
        grid.getLocalAgents(*acting, candidates);
        std::vector<CellAgent*> everyone;
        for (CellAgent* local : population.pointers()) {if (local != acting) {everyone.push_back(local);}}

        CellAgent viaGrid{*acting};
        CellAgent viaEveryone{*acting};
        Access::runCollisions(viaGrid, candidates);
        Access::runCollisions(viaEveryone, everyone);
        EXPECT_EQ(viaGrid.getCollisionNumber(), viaEveryone.getCollisionNumber());
        EXPECT_EQ(viaGrid.getActinFlowDirection(), viaEveryone.getActinFlowDirection());
        EXPECT_EQ(viaGrid.getActinFlowMagnitude(), viaEveryone.getActinFlowMagnitude());
        EXPECT_EQ(Access::adhesionFraction(viaGrid), Access::adhesionFraction(viaEveryone));
        grid.addToCollisionMatrix(acting);
    }
}
