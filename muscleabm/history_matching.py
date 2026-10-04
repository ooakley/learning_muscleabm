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
        "gaussian_process": {
            "epochs": 15,
            "inducing_point_count": 512,
            "hidden_layer_count": 2,
            "hidden_layer_neuron_count": 32,
            "learning_rate": 0.01,
            "deep_kernel_learning_rate": 0.003,
            "batch_size": 512
        }
    }

wave_count:        number of waves, including wave 0 (the initial Sobol' search).
wave_size:         simulations in each wave after wave 0, drawn equally from the posterior
                   of each phenotype (wave 0's size is set by the Sobol' search config).
temperature_index: temperature of the MCMC chains that each wave's samples are drawn from
                   (0 is the cold chain).
gaussian_process:  architecture and training settings of the GP emulators.
"""
import json
import sys

# Phenotypes whose posteriors each wave is drawn from (as in generate_hm_sweep.py):
PHENOTYPE_COUNT = 2

# GP settings, with their types and defaults. The defaults are those gp_training.py uses
# when run without a config:
GP_SETTINGS = [
    # (name, type, default, help)
    ("epochs", int, 15, "Training epochs, for the full model and each cross-validation fold."),
    ("inducing_point_count", int, 512, "Inducing points of the sparse variational GP."),
    ("hidden_layer_count", int, 2, "Hidden layers of the deep kernel network."),
    ("hidden_layer_neuron_count", int, 32, "Neurons in each hidden layer of the deep kernel network."),
    ("learning_rate", float, 0.01,
     "Learning rate of the GP (variational parameters, inducing locations, kernel and likelihood)."),
    ("deep_kernel_learning_rate", float, 0.003, "Learning rate of the deep kernel network."),
    ("batch_size", int, 512, "Batch size for training and inference."),
]


class GPSettings:
    """Architecture and training settings of the GP emulators."""

    def __init__(self, **settings):
        for name, setting_type, default, _ in GP_SETTINGS:
            setattr(self, name, setting_type(settings.get(name, default)))

    @property
    def models_dirname(self):
        """Folder, inside a wave folder, that models trained with these settings are saved to."""
        return "pll_gp_models_e{}_ip{}_l{}_n{}".format(
            self.epochs, self.inducing_point_count, self.hidden_layer_count, self.hidden_layer_neuron_count
        )

    def to_dict(self):
        return {name: getattr(self, name) for name, _, _, _ in GP_SETTINGS}

    def to_arguments(self):
        """Command line arguments that set these settings in gp_training.py."""
        arguments = []
        for name, _, _, _ in GP_SETTINGS:
            arguments += ["--" + name, str(getattr(self, name))]
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

    _check_keys("The config", config, ["wave_count", "wave_size", "temperature_index", "gaussian_process"])
    checked = {
        "wave_count": _check_number("wave_count", config["wave_count"], int, 1),
        "wave_size": _check_number("wave_size", config["wave_size"], int, PHENOTYPE_COUNT),
        "temperature_index": _check_number("temperature_index", config["temperature_index"], int, 0),
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
        # Every GP setting is a positive count or rate:
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
    "gp_folder_name",  # GPSettings.models_dirname
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
        gp_settings.models_dirname,
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
