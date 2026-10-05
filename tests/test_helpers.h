#pragma once
// Builders and reference calculations shared by the behavioural tests.
#include <algorithm>
#include <cmath>
#include <memory>

#include "agents.h"
#include "test_access.h"
#include "world.h"

constexpr double WORLD_SIZE{2048};

// Parameters of a single cell, with deterministic defaults (no noise in actin flow):
struct CellSpec {
    unsigned int seed{0};
    int id{0};
    double dt{1};
    double cueDiffusionRate{0.01};
    double cueKa{1};
    double fluctuationAmplitude{0};
    double fluctuationTimescale{20};
    double actinAdvectionRate{10};
    double collisionAdvectionRate{0};
    double maximumSteadyStateActinFlow{1.5};
    double cellBodyRadius{50};
    double collisionFlowReductionRate{0.2};
    double adhesionReductionRate{0.5};
    double cellStiffness{0.1};
    double surfaceStickiness{10};
    double matrixCoupling{0};
    double x{500};
    double y{500};
    double heading{0};
};

inline std::unique_ptr<CellAgent> makeCell(const CellSpec& spec) {
    return std::make_unique<CellAgent>(
        spec.seed, spec.id, spec.dt,
        spec.cueDiffusionRate, spec.cueKa, spec.fluctuationAmplitude, spec.fluctuationTimescale,
        spec.actinAdvectionRate, spec.collisionAdvectionRate, spec.maximumSteadyStateActinFlow,
        spec.cellBodyRadius, spec.collisionFlowReductionRate, spec.adhesionReductionRate,
        spec.cellStiffness, spec.surfaceStickiness,
        spec.matrixCoupling,
        spec.x, spec.y, spec.heading
    );
}

inline CellParameters cellParametersFrom(const CellSpec& spec) {
    CellParameters parameters;
    parameters.dt = spec.dt;
    parameters.cueDiffusionRate = spec.cueDiffusionRate;
    parameters.cueKa = spec.cueKa;
    parameters.fluctuationAmplitude = spec.fluctuationAmplitude;
    parameters.fluctuationTimescale = spec.fluctuationTimescale;
    parameters.actinAdvectionRate = spec.actinAdvectionRate;
    parameters.collisionAdvectionRate = spec.collisionAdvectionRate;
    parameters.maximumSteadyStateActinFlow = spec.maximumSteadyStateActinFlow;
    parameters.cellBodyRadius = spec.cellBodyRadius;
    parameters.collisionFlowReductionRate = spec.collisionFlowReductionRate;
    parameters.adhesionReductionRate = spec.adhesionReductionRate;
    parameters.cellStiffness = spec.cellStiffness;
    parameters.surfaceStickiness = spec.surfaceStickiness;
    parameters.matrixCoupling = spec.matrixCoupling;
    return parameters;
}

// Parameters of a world:
struct WorldSpec {
    unsigned int seed{1};
    double worldSize{WORLD_SIZE};
    int ecmElementCount{64};
    int numberOfCells{1};
    double matrixSampleRate{0};
    double patternSigma{0};
    int patternFibreCount{0};
    CellSpec cell{};
};

inline std::unique_ptr<World> makeWorld(const WorldSpec& spec) {
    return std::make_unique<World>(
        spec.seed, spec.worldSize, spec.ecmElementCount, spec.numberOfCells,
        spec.matrixSampleRate, spec.patternSigma, spec.patternFibreCount,
        cellParametersFrom(spec.cell)
    );
}

// Displacement from a to b, taking the nearest periodic image:
inline double minimalImage(double a, double b, double worldSize = WORLD_SIZE) {
    return std::remainder(b - a, worldSize);
}

// Distance from point p to the segment from s to e, where the segment's start is taken at its
// image nearest p and its end at the image nearest its start, as in the collision model:
inline double periodicPointToSegmentDistance(
    double px, double py, double sx, double sy, double ex, double ey,
    double worldSize = WORLD_SIZE
) {
    const double startX{px + minimalImage(px, sx, worldSize)};
    const double startY{py + minimalImage(py, sy, worldSize)};
    const double segmentX{minimalImage(sx, ex, worldSize)};
    const double segmentY{minimalImage(sy, ey, worldSize)};
    const double lengthSquared{segmentX*segmentX + segmentY*segmentY};
    double t{0};
    if (lengthSquared > 0) {
        t = std::clamp(((px - startX)*segmentX + (py - startY)*segmentY) / lengthSquared, 0.0, 1.0);
    }
    return std::hypot(px - (startX + t*segmentX), py - (startY + t*segmentY));
}

// Whether two cells could collide under the model's rules: the acting cell's centre or stadium
// point within two body radii of the local cell's segment:
inline bool couldCollide(const CellAgent& acting, const CellAgent& local, double worldSize = WORLD_SIZE) {
    const double threshold{acting.getEffectiveRadius() + local.getEffectiveRadius()};
    for (const auto& [px, py] : {
        std::array<double, 2>{acting.getX(), acting.getY()},
        std::array<double, 2>{acting.getStadiumX(), acting.getStadiumY()}
    }) {
        if (periodicPointToSegmentDistance(
            px, py, local.getX(), local.getY(), local.getStadiumX(), local.getStadiumY(), worldSize
        ) < threshold) {
            return true;
        }
    }
    return false;
}
