"""Settings of a history matching run: reading its config, and the GP training settings.

This module only uses the standard library (and avoids features newer than Python 3.6),
as bash_scripts/submit_hm_waves.sh reads the config with the login node's python3:

    python3 -m muscleabm.history_matching <hm_config_path>

prints the settings one per line (see SHELL_SETTINGS), or an error if the config is invalid.

A history matching config is a .json file of the form:

    {
        "wave_count": 4,
        "wave_size": 32768,
        "temperature_index": 1,
        "burn_in_fraction": 0.25,
        "max_failed_fraction": 0.1,
        "simulations_per_task": 64,
        "array_task_limit": 90,
        "run_validation": true,
        "delete_run_data": true,
        "gaussian_process": {
            "training_steps": 3840,
            "inducing_point_count": 512,
            "hidden_layer_count": 2,
            "hidden_layer_neuron_count": 32,
            "learning_rate": 0.01,
            "deep_kernel_learning_rate": 0.003,
            "batch_size": 512,
            "spectral_normalisation": false
        }
    }

wave_count:        number of waves, including wave 0 (the initial Sobol' search).
wave_size:         simulations in each wave after wave 0, drawn equally from the posterior
                   of each phenotype (wave 0's size is set by the Sobol' search config).
temperature_index: temperature of the MCMC chains that each wave's samples are drawn from
                   (0 is the cold chain).
burn_in_fraction:  fraction of each MCMC chain discarded as burn-in, both when sampling
                   the posterior and when drawing the next wave from it.
max_failed_fraction: the chain stops if more than this fraction of a wave's simulations
                   failed (for any metric).
simulations_per_task: simulations run by each array task of the simulation job.
array_task_limit:  simulation array tasks allowed to run at once.
run_validation:    whether to test each wave's GP models against the simulations of the
                   next wave.
delete_run_data:   whether to delete each wave's run_data folder, to save space, once the
                   wave has passed the check of failed simulations and is in the global
                   dataset. Its simulations then cannot be rerun or recollated.
gaussian_process:  architecture and training settings of the GP emulators. They train for
                   a fixed number of optimiser steps (minibatches), so that a wave's models
                   train for as long as the first wave's, however large the dataset has grown.
                   spectral_normalisation (true or false) sets the spectral norm of every
                   linear layer of the deep kernel to 1.

Each wave's models are saved to its MODELS_DIRNAME folder, and each metric's model folder
records the settings it was trained with in training_config.json.
"""
import json
import sys

# Phenotypes whose posteriors each wave is drawn from (as in generate_hm_sweep.py):
PHENOTYPE_COUNT = 2

# Folders, inside each wave folder, that the wave's GP models and MCMC chains are saved to:
MODELS_DIRNAME = "trained_models"
MCMC_DIRNAME = "disc_cov_mcmc_results"

def boolean(value):
    """A true or false setting, given as a bool or (on the command line) as "true" or "false"."""
    if isinstance(value, bool):
        return value
    if value in ("true", "false"):
        return value == "true"
    raise ValueError("expected true or false, got {!r}".format(value))


# GP settings, with their types and defaults. The defaults are those gp_training.py uses
# when run without a config. Every setting is a positive number, apart from the booleans:
GP_SETTINGS = [
    # (name, type, default, help)
    # 3840 steps is 15 epochs of a 2^17 simulation first wave at batch size 512:
    ("training_steps", int, 3840,
     "Optimiser steps (minibatches) for the full model and each cross-validation fold, "
     "whatever the dataset size."),
    ("inducing_point_count", int, 512, "Inducing points of the sparse variational GP."),
    ("hidden_layer_count", int, 2, "Hidden layers of the deep kernel network."),
    ("hidden_layer_neuron_count", int, 32, "Neurons in each hidden layer of the deep kernel network."),
    ("learning_rate", float, 0.01,
     "Learning rate of the GP (variational parameters, inducing locations, kernel and likelihood)."),
    ("deep_kernel_learning_rate", float, 0.003, "Learning rate of the deep kernel network."),
    ("batch_size", int, 512, "Batch size for training and inference."),
    ("spectral_normalisation", boolean, False,
     "Spectrally normalise every linear layer of the deep kernel network (true or false)."),
]


class GPSettings:
    """Architecture and training settings of the GP emulators."""

    def __init__(self, **settings):
        for name, setting_type, default, _ in GP_SETTINGS:
            setattr(self, name, setting_type(settings.get(name, default)))

    def to_dict(self):
        return {name: getattr(self, name) for name, _, _, _ in GP_SETTINGS}

    def to_arguments(self):
        """Command line arguments that set these settings in gp_training.py."""
        arguments = []
        for name, _, _, _ in GP_SETTINGS:
            value = getattr(self, name)
            arguments += ["--" + name, str(value).lower() if isinstance(value, bool) else str(value)]
        return arguments

    @staticmethod
    def add_arguments(parser):
        """Add an argument for each setting to an argparse parser, defaulting to GP_SETTINGS."""
        for name, setting_type, default, help_text in GP_SETTINGS:
            parser.add_argument("--" + name, type=setting_type, default=default, help=help_text)

    @classmethod
    def from_arguments(cls, args):
        return cls(**{name: getattr(args, name) for name, _, _, _ in GP_SETTINGS})


class ConfigError(ValueError):
    pass


def _check_number(path, value, setting_type, minimum):
    """Check a config value has the given type (ints are fine for floats) and minimum."""
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if setting_type is int and not (is_number and float(value).is_integer()):
        raise ConfigError("{} must be a whole number, got {!r}.".format(path, value))
    if not is_number:
        raise ConfigError("{} must be a number, got {!r}.".format(path, value))
    if value < minimum:
        raise ConfigError("{} must be at least {}, got {!r}.".format(path, minimum, value))
    return setting_type(value)


def _check_fraction(path, value, include_zero):
    """Check a config value is a fraction: in (0, 1), or in [0, 1] if include_zero."""
    value = _check_number(path, value, float, 0)
    if include_zero and value > 1:
        raise ConfigError("{} must be between 0 and 1, got {!r}.".format(path, value))
    if not include_zero and not 0 < value < 1:
        raise ConfigError("{} must be strictly between 0 and 1, got {!r}.".format(path, value))
    return value


def _check_bool(path, value):
    if not isinstance(value, bool):
        raise ConfigError("{} must be true or false, got {!r}.".format(path, value))
    return value


def _check_keys(path, dictionary, expected_keys):
    if not isinstance(dictionary, dict):
        raise ConfigError("{} must be a JSON object.".format(path))
    missing = [key for key in expected_keys if key not in dictionary]
    unknown = sorted(set(dictionary) - set(expected_keys))
    if missing:
        raise ConfigError("{} is missing: {}.".format(path, ", ".join(missing)))
    if unknown:
        raise ConfigError("{} has unknown settings: {}.".format(path, ", ".join(unknown)))


def load_config(config_path):
    """Read and check a history matching config. Returns a dict, with gaussian_process as GPSettings."""
    try:
        with open(config_path) as config_fstream:
            config = json.load(config_fstream)
    except (OSError, ValueError) as error:
        raise ConfigError("Could not read {}: {}".format(config_path, error))

    _check_keys("The config", config, [
        "wave_count", "wave_size", "temperature_index", "burn_in_fraction", "max_failed_fraction",
        "simulations_per_task", "array_task_limit", "run_validation", "delete_run_data",
        "gaussian_process",
    ])
    checked = {
        "wave_count": _check_number("wave_count", config["wave_count"], int, 1),
        "wave_size": _check_number("wave_size", config["wave_size"], int, PHENOTYPE_COUNT),
        "temperature_index": _check_number("temperature_index", config["temperature_index"], int, 0),
        "burn_in_fraction": _check_fraction("burn_in_fraction", config["burn_in_fraction"], False),
        "max_failed_fraction": _check_fraction("max_failed_fraction", config["max_failed_fraction"], True),
        "simulations_per_task": _check_number("simulations_per_task", config["simulations_per_task"], int, 1),
        "array_task_limit": _check_number("array_task_limit", config["array_task_limit"], int, 1),
        "run_validation": _check_bool("run_validation", config["run_validation"]),
        "delete_run_data": _check_bool("delete_run_data", config["delete_run_data"]),
    }
    if checked["wave_size"] % PHENOTYPE_COUNT != 0:
        raise ConfigError(
            "wave_size must split equally between the {} phenotypes, got {}.".format(
                PHENOTYPE_COUNT, checked["wave_size"]
            )
        )

    gp_config = config["gaussian_process"]
    _check_keys("gaussian_process", gp_config, [name for name, _, _, _ in GP_SETTINGS])
    gp_settings = {}
    for name, setting_type, _, _ in GP_SETTINGS:
        if setting_type is boolean:
            gp_settings[name] = _check_bool("gaussian_process." + name, gp_config[name])
            continue
        # Every other GP setting is a positive count or rate:
        value = _check_number("gaussian_process." + name, gp_config[name], setting_type, 0)
        if value <= 0:
            raise ConfigError("gaussian_process.{} must be positive, got {!r}.".format(name, value))
        gp_settings[name] = value
    checked["gaussian_process"] = GPSettings(**gp_settings)
    return checked


# Settings printed for submit_hm_waves.sh, one per line in this order:
SHELL_SETTINGS = [
    "wave_count",
    "wave_size",
    "samples_per_phenotype",  # wave_size / PHENOTYPE_COUNT
    "temperature_index",
    "burn_in_fraction",
    "max_failed_fraction",
    "simulations_per_task",
    "array_task_limit",
    "run_validation",  # true or false
    "delete_run_data",  # true or false
    "mcmc_dirname",  # MCMC_DIRNAME
    "gp_folder_name",  # MODELS_DIRNAME
    "gp_training_arguments",  # GPSettings.to_arguments(), space separated
]


def to_lines(config):
    """The settings in SHELL_SETTINGS, one per line. None of them contains whitespace."""
    gp_settings = config["gaussian_process"]
    lines = [
        str(config["wave_count"]),
        str(config["wave_size"]),
        str(config["wave_size"] // PHENOTYPE_COUNT),
        str(config["temperature_index"]),
        str(config["burn_in_fraction"]),
        str(config["max_failed_fraction"]),
        str(config["simulations_per_task"]),
        str(config["array_task_limit"]),
        str(config["run_validation"]).lower(),
        str(config["delete_run_data"]).lower(),
        MCMC_DIRNAME,
        MODELS_DIRNAME,
        " ".join(gp_settings.to_arguments()),
    ]
    return "\n".join(lines)


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 -m muscleabm.history_matching <hm_config_path>")
    try:
        config = load_config(sys.argv[1])
    except ConfigError as error:
        sys.exit("Invalid history matching config: {}".format(error))
    print(to_lines(config))


if __name__ == "__main__":
    main()
