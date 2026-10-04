# An agent-based model of collective cell patterning in muscle development.

Cells move in a realistic, polarisation driven manner, and lay down extracellular matrix as they do so. This extracellular matrix in turn influences the motion of cells that follow.

## Python scripts

Install the environment with `uv sync` from the repository root. This also installs `muscleabm/`, the code shared between scripts (GP emulators, data loading, sweep generation, sensitivity analysis), so that the scripts can import it. Run it again after pulling changes to `pyproject.toml`.

Scripts are run from the repository root, and are grouped in `python_scripts/` by stage:

| Folder | Contents |
|---|---|
| `simulation/` | Running one simulation from its argument file, and analysing its outputs |
| `search/` | Generating parameter sweeps: the initial Sobol' search, and history matching waves |
| `collation/` | Collating per-simulation outputs into summary data, and across waves |
| `emulation/` | Training GP (and baseline neural network) emulators of the model metrics |
| `inference/` | MCMC against the wet lab data, and validating emulators against the next wave |
| `sensitivity/` | Hessians, Fisher information, eigenparameters and interventions |
| `wetlab/` | Analysing the wet lab trajectories |

## Configs

`configs/` holds the simulation configs: `gridsearch_configs/` for parameter sweeps (passed to `python_scripts/search/generate_sobol_search.py`), and `model_parameter_json/` for single simulations (passed to `python_scripts/simulation/call_json_parameters.py`), whose outputs are written next to them. `example_config.json` is another single simulation config. Every config passes exactly the options `src/main.cpp` requires.

`history_matching_configs/` holds the configs of history matching runs, passed to `bash_scripts/submit_hm_waves.sh`. Each sets the number of waves, the size of each wave after the first (split equally between the phenotypes), the MCMC temperature each wave's samples are drawn from, and the GP architecture and training settings; `default.json` holds the settings the pipeline used before, and `muscleabm/history_matching.py` describes each setting.

## Job scripts

`bash_scripts/` holds the SLURM job scripts, submitted from the repository root with `sbatch bash_scripts/<script> <arguments>`; each describes its arguments at its top. `bash_scripts/submit_hm_waves.sh` submits the history matching pipeline as a chain of these jobs, set by a history matching config:

```
bash bash_scripts/submit_hm_waves.sh <experiment_dirpath> configs/history_matching_configs/default.json
```

- Logs are written to `logs/`, in a folder for each part of the codebase, named like the stage folders of `python_scripts/`: `simulation/`, `search/`, `collation/`, `emulation/` (GP and neural network training), `inference/` (MCMC and wave validation), `sensitivity/` and `wetlab/`. `run_python_stage.sh` logs to `python_stage/` when submitted by hand; `submit_hm_waves.sh` sends each of its steps to the folder of its script.
- The jobs run Python with `uv run --no-sync`, so that array tasks do not all try to update the environment at once. Run `uv sync` before submitting after the dependencies change (`submit_hm_waves.sh` does this itself).
