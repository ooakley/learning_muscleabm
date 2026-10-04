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

## Job scripts

`bash_scripts/` holds the SLURM job scripts, submitted from the repository root with `sbatch bash_scripts/<script> <arguments>`; each describes its arguments at its top. `bash_scripts/submit_hm_waves.sh` submits the history matching pipeline as a chain of these jobs.

- Logs are written to `logs/`.
- The jobs run Python with `uv run --no-sync`, so that array tasks do not all try to update the environment at once. Run `uv sync` before submitting after the dependencies change (`submit_hm_waves.sh` does this itself).
