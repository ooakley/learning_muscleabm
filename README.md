# An agent-based model of collective cell patterning in muscle development.

Cells move in a realistic, polarisation driven manner, and lay down extracellular matrix as they do so. This extracellular matrix in turn influences the motion of cells that follow.

The simulator is written in C++ (`src/`). Python scripts run sweeps of it, emulate its outputs with Gaussian processes (GPs), and fit it to wet lab data by history matching. Everything is run from the repository root.

## Setup

### Simulator

The simulator needs CMake 3.24.3 or later, a C++20 compiler, and Boost 1.81 or later (`program_options` and `filesystem`). On the cluster, load them with:

```
ml load Boost/1.81.0-GCC-12.2.0 CMake/3.24.3-GCCcore-12.2.0
```

Then build it into `build/`, where the scripts expect it (`build/src/main`):

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target main
```

To run the C++ tests, build everything and run `ctest`:

```
cmake --build build && (cd build && ctest)
```

The build downloads GoogleTest from GitHub. Without internet access, point it at a local copy with `-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=<path>` when configuring.

### Python environment

Install the environment with `uv sync`. This also installs `muscleabm/`, the code shared between scripts (GP emulators, data loading, sweep generation, sensitivity analysis and history matching configs), so that the scripts can import it. Run it again after pulling changes to `pyproject.toml`.

## Running a single simulation

`call_json_parameters.py` runs the simulator with the options in a config, and writes its outputs to the config's `outputFolder`:

```
uv run python python_scripts/simulation/call_json_parameters.py --path_to_config configs/example_config.json
```

The configs in `configs/model_parameter_json/` write their outputs next to themselves.

## History matching

History matching fits the simulator to the wet lab data in waves. Each wave simulates a set of parameters, a GP emulator of each model metric is trained on every wave so far, and MCMC against the wet lab data gives a posterior, from which the next wave's parameters are drawn.

1. **Wet lab data.** Analyse the tracked trajectories into `wetlab_data/site_dataframe.csv`:

   ```
   sbatch bash_scripts/analyse_trajectories.sh
   ```

   then fit the wet lab GPs that the MCMC uses as its targets, into `wetlab_data/gp_results/`:

   ```
   uv run python plotting_scripts/test_wetlab_gp.py
   ```

2. **Wave 0.** Generate the initial Sobol' sweep from a sweep config. This creates the experiment folder, `model_experiments/{date}-{experiment_name}`, with the config at its root and the sweep in `hm0/`:

   ```
   uv run python python_scripts/search/generate_sobol_search.py \
       --experiment_config_path configs/gridsearch_configs/collisions_shape_gridsearch.json
   ```

3. **Waves.** Submit the waves as a chain of SLURM jobs, set by a history matching config (see [Configs](#configs)):

   ```
   bash bash_scripts/submit_hm_waves.sh model_experiments/{date}-{experiment_name} \
       configs/history_matching_configs/default.json
   ```

   For each wave this submits, each job waiting for the one before:

   | Step | Job script | Python script | Output, in the wave folder `hm{n}/` |
   |---|---|---|---|
   | Generate (from wave 1) | `run_python_stage.sh` | `search/generate_hm_sweep.py` | Samples of the previous wave's posterior, `run_data/` |
   | Simulate | `gridsearch.sh` | `simulation/call_json_parameters.py`, `simulation/site_analysis.py` | Simulation outputs and metrics, in `run_data/` |
   | Collate | `collate_gridsearch_data.sh` | `collation/collate_site_analyses.py` | `summary_data/` |
   | Collate waves | `run_python_stage.sh` | `collation/collate_hm.py` | Every wave so far, in the experiment's `global_dataset/` |
   | Validate (from wave 1) | `run_python_stage.sh` | `inference/validate_wave.py` | The previous wave's GPs tested on this wave, in their models folder in `hm{n-1}/` |
   | Train | `gp_regression.sh` | `emulation/gp_training.py` | GP models, in `pll_gp_models_.../` |
   | MCMC | `mcmc_fit.sh` | `inference/mcmc_gp_cov_fit.py` | Posterior chains, in `disc_cov_mcmc_results/` |

   To print the jobs without submitting them, set `DRY_RUN=true`. To restart a chain after a failure, give the wave and step to restart from; the comment at the top of `submit_hm_waves.sh` describes both.

## Scripts

Python scripts are grouped in `python_scripts/` by stage:

| Folder | Contents |
|---|---|
| `simulation/` | Running one simulation from its argument file, and analysing its outputs |
| `search/` | Generating parameter sweeps: the initial Sobol' search, and history matching waves |
| `collation/` | Collating per-simulation outputs into summary data, and across waves |
| `emulation/` | Training GP (and baseline neural network) emulators of the model metrics |
| `inference/` | MCMC against the wet lab data, and validating emulators against the next wave |
| `sensitivity/` | Hessians, Fisher information, eigenparameters and interventions |
| `wetlab/` | Analysing the wet lab trajectories |

Most are run by the job scripts in `bash_scripts/`: the history matching steps above, and these analyses, each submitted with `sbatch bash_scripts/<script> <arguments>`:

| Job script | Python script |
|---|---|
| `gp_noise_regression.sh` | `emulation/gp_noise_training.py` |
| `nn_regression.sh` | `emulation/nn_cv_training.py` |
| `hessian_estimation.sh` | `sensitivity/generate_full_rank_hessians.py` |
| `gee_estimation.sh` | `sensitivity/global_eigenparameter_estimation.py` |
| `run_embedding.sh` | `sensitivity/run_isomap_embedding.py` |

The rest are run directly with `uv run python python_scripts/<stage>/<script>.py`, and describe their arguments with `--help`. `generate_sobol_search.py` and `call_json_parameters.py` are the usual starting points.

`marimo_scripts/` holds exploratory marimo notebooks, opened with `uv run marimo edit marimo_scripts/<notebook>.py`. `plotting_scripts/` holds marimo apps that make figures, which can also be run as scripts.

### Job scripts

Each job script describes its arguments at its top.

- Logs are written to `logs/`, in a folder for each part of the codebase, named like the stage folders of `python_scripts/`: `simulation/`, `search/`, `collation/`, `emulation/` (GP and neural network training), `inference/` (MCMC and wave validation), `sensitivity/` and `wetlab/`. `run_python_stage.sh` logs to `python_stage/` when submitted by hand; `submit_hm_waves.sh` sends each of its steps to the folder of its script.
- The jobs run Python with `uv run --no-sync`, so that array tasks do not all try to update the environment at once. Run `uv sync` before submitting after the dependencies change (`submit_hm_waves.sh` does this itself).

## Configs

`configs/` holds the simulation configs: `gridsearch_configs/` for parameter sweeps (passed to `python_scripts/search/generate_sobol_search.py`), and `model_parameter_json/` for single simulations (passed to `python_scripts/simulation/call_json_parameters.py`), whose outputs are written next to them. `example_config.json` is another single simulation config.

`history_matching_configs/` holds the configs of history matching runs, passed to `bash_scripts/submit_hm_waves.sh`. Each sets the number of waves, the size of each wave after the first (split equally between the phenotypes), the MCMC temperature each wave's samples are drawn from, and the GP architecture and training settings; `default.json` holds the settings the pipeline used before, and `muscleabm/history_matching.py` describes each setting.

## Linting

Check the Python code with [ruff](https://docs.astral.sh/ruff/), whose rules are set in `pyproject.toml`, and the job scripts with [ShellCheck](https://www.shellcheck.net/):

```
uvx ruff check
uvx --from shellcheck-py shellcheck bash_scripts/*.sh
```

`uvx ruff check` checks `python_scripts/` and `muscleabm/`. The notebooks and plotting apps are checked when named, e.g. `uvx ruff check marimo_scripts`.
