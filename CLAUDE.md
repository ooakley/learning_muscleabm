# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`README.md` covers setup, the history matching pipeline, the job scripts and the configs. Read it first. This file adds what takes several files to piece together. Run everything from the repository root.

## Commands

```bash
# Build the simulator and the tests (the scripts expect build/src/main):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# Offline, point GoogleTest at a local copy, e.g. /usr/src/googletest from apt's googletest:
cmake -S . -B build -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/usr/src/googletest

# C++ tests (also run by .github/workflows/tests.yml on every push and PR):
ctest --test-dir build --output-on-failure
./build/tests/main_test --gtest_filter='CollisionGrid.*'   # one suite or test

# One simulation from a config (outputs go to its outputFolder):
uv run python python_scripts/simulation/call_json_parameters.py --path_to_config configs/example_config.json

# Lint:
uvx ruff check
uvx --from shellcheck-py shellcheck bash_scripts/*.sh
```

## The simulator (`src/`)

- **Main loop.** `main.cpp` parses the options (Boost program_options) into `CellParameters`. For each `superIterationCount` it builds a `World`, seeded from a generator seeded by `jobArrayID`. It runs `timestepsToRun` steps, writing `positions_seed{NNN}.csv` and `matrix_seed{NNN}.txt`.
- **Configs.** `call_json_parameters.py` turns every key of a JSON config into a `--key value` option, so every key must be a registered option, or the simulator exits with an error. When adding or removing an option, update every config under `configs/`.
- **One step.** `World::runSimulationStep` shuffles the cells, then runs `runCellStep` for each, in order. A cell step:
  1. samples matrix attachment points (a Poisson count) and sets the cell's percepts from the fibres there (`ECMField::sampleFibreMatrix`);
  2. removes the cell from the collision grid and collects its neighbours;
  3. runs `CellAgent::takeRandomStep`: the actin-flow SDE, then collisions, then movement, then stick-slip;
  4. deposits fibres along the new flow heading;
  5. wraps the position and re-registers the cell in the grid.
- **Cell geometry.** A cell is a capsule: the segment from its centre `(x, y)` (the front) to its stadium point (the rear), widened by `cellBodyRadius`. Stick-slip (`runStickSlipLogic`, an implicit Newton solver in `implicitNextState`) only ever retracts the rear.
- **Collisions.** `CollisionCellList` registers each cell in every grid element its capsule (widened by two body radii) overlaps. A cell's candidate neighbours are then the cells registered at its centre and at its stadium point. They are sorted by ID, so results don't depend on the grid. It indexes per-cell state by `getID()`, so IDs must be `0..numberOfCells-1`.
- **Matrix.** `ECMField` holds a deque of nematic fibre headings, in [0, π), per site, capped at `ECMField::maxFibresPerSite`.
- **Randomness.** Each random process has its own generator, seeded by successive draws from a generator seeded by its owner's seed (World, ECM, cell). Adding or removing a draw changes every later seed, and so the outputs for a given `jobArrayID`.

### Invariants and pitfalls
- **Periodic boundary.** The world is periodic with side `worldSize` (2048 in every config). Wrap displacements with the nearest image. For a neighbour's segment, take the image of its start nearest the query point, then the image of its end nearest its start. Wrapping both ends independently produced phantom collisions with cells about half the world away.
- **Flow magnitude.** It must stay ≥ 0 after the SDE update: a negative magnitude is reflected to the opposite heading. The collision logic judges whether a cell is moving towards a neighbour from its heading alone.
- **Asserts.** They are live in release builds: `CMakeLists.txt` sets the release flags to `-O3` without `-DNDEBUG`. A NaN that reaches an assert aborts the run (which counts as a failed simulation), so guard the divisions that can be 0/0 (zero-length segments, zero advection).
- **Unused verbose fields.** `movementDirection`, `directionalShift`, `sampledAngle`, `finalCILEffectX/Y` and the polarity fields are never updated by the model. They exist only to fill the verbose output format.

### Output formats (read by `python_scripts/simulation/`)
- **Positions:** `frame,particle,x,y,stadium_x,stadium_y` per cell per step. With `verbosePositionOutput`, 19 columns instead (see `writeVerbosePositionsToCSV`), as read by `speed_persistence_analysis.py` and `wasserstein_distance_analysis.py`.
- **Matrix:** by default the final matrix, one line per site listing every fibre heading. With `summariseMatrixOutput`, one line of `heading,concentration,count` per site. With `writeMatrixTimeseries`, one matrix per step, as `make_example_videos.py` reads.
- **Coordinates:** the analyses halve coordinates to match the 1024 px wet lab field, so Python code uses `WORLD_SIZE = 1024`.

## C++ tests (`tests/`)

The tests are behavioural, one file per area: collisions, matrix deposition and sensing, movement, and stick-slip.
- **Internal state.** `CellAgentTestAccess` and `WorldTestAccess` (`tests/test_access.h`) are friends of `CellAgent` and `World`.
- **Builders.** Construct cells and worlds through `makeCell(CellSpec)` and `makeWorld(WorldSpec)` (`tests/test_helpers.h`), so that constructor changes are made in one place. Default specs have no noise in the actin flow.
- **Code that may abort.** Run it inside `EXPECT_EXIT`, so that an abort fails only that test.
- **Changes to the model.** Check whether a change alters results by comparing a run's outputs byte for byte with the previous build, using the same arguments. When results do change, compare summary statistics over several seeds.

## Gotchas
- `.gitignore` ignores `out*`, so a new file named e.g. `output.h` is silently left out of commits.
- `build/`, `model_experiments/`, `wetlab_data/` and `*.npy` are not tracked.
- `configs/history_matching_configs/` holds pipeline settings (read by `muscleabm/history_matching.py`), not simulation options.
