"""Generates a history matching wave of simulations from the MCMC posterior chains.

Takes a thinned subset of the WT and RD posterior chains of one wave, assigns each
parameter set a uniformly random cell count, and writes one simulation per parameter
set into the run_data folder of the next wave.
"""
import os
import json
import argparse

import numpy as np

from muscleabm.sampling import COUNT_PARAMETER_NAME, JSONOutputManager

# Phenotypes, in the order their samples are written, and the chain file of each:
PHENOTYPES = ["WT", "RD"]
CHAIN_FILENAMES = {"WT": "wt_mcmc_chain.npy", "RD": "rd_mcmc_chain.npy"}

# The chain has shape (steps, chains, temperatures, parameters). Samples are drawn
# from one temperature (0 is the cold chain), after discarding the burn-in:
TEMPERATURE_INDEX = 1

# Seed for the random cell counts, combined with the wave so that waves differ:
COUNT_SEED = 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dirpath", required=True,
        help="The {date}-{config} folder, containing config.json and the hm wave folders."
    )
    parser.add_argument(
        "--hm_wave_id", type=int, required=True,
        help="History matching wave whose posterior the samples are drawn from, e.g. 0 for "
             "the hm0 folder. The new wave is written to the next folder, e.g. hm1."
    )
    parser.add_argument(
        "--mcmc_dirname", default="disc_cov_mcmc_results",
        help="Folder inside the wave folder containing the WT and RD chains."
    )
    parser.add_argument(
        "--sample_count", type=int, default=512,
        help="Number of posterior parameter sets drawn from each phenotype's chain."
    )
    parser.add_argument(
        "--burn_in_fraction", type=float, default=0.25,
        help="Fraction of the chain discarded as burn-in. Use the same value as the MCMC script."
    )
    arguments = parser.parse_args()
    if not 0 < arguments.burn_in_fraction < 1:
        parser.error("--burn_in_fraction must be between 0 and 1.")
    return arguments


def thin_chain(mc_distribution, sample_count, burn_in_fraction):
    """Draw sample_count evenly spaced parameter sets from one temperature of the chain.

    mc_distribution: (steps, chains, temperatures, n_parameters - 1) unit-cube values.

    Samples are split evenly between the chains, and evenly spaced over the steps
    left after burn-in. Returns the parameter sets, (sample_count, n_parameters - 1),
    and the (step, chain) index of each within the full chain, (sample_count, 2).
    """
    if mc_distribution.ndim != 4:
        raise ValueError(
            f"Chain must have shape (steps, chains, temperatures, parameters), got {mc_distribution.shape}."
        )
    step_count, chain_count = mc_distribution.shape[0], mc_distribution.shape[1]
    burn_in_step_count = int(step_count * burn_in_fraction)
    kept_step_count = step_count - burn_in_step_count
    if sample_count > kept_step_count * chain_count:
        raise ValueError(
            f"Requested {sample_count} samples, but the chain only has "
            f"{kept_step_count * chain_count} after burn-in."
        )

    # Split the samples as evenly as possible between the chains:
    chain_indices = []
    step_indices = []
    for chain_index, chain_samples in enumerate(np.array_split(np.arange(sample_count), chain_count)):
        chain_sample_count = chain_samples.shape[0]
        chain_steps = np.linspace(
            burn_in_step_count, step_count - 1, chain_sample_count
        ).astype(int)
        chain_indices.append(np.full(chain_sample_count, chain_index))
        step_indices.append(chain_steps)
    chain_indices = np.concatenate(chain_indices)
    step_indices = np.concatenate(step_indices)

    samples = mc_distribution[step_indices, chain_indices, TEMPERATURE_INDEX, :]
    return samples, np.stack([step_indices, chain_indices], axis=1)


def write_hm_sweep(config_dictionary, wave_folderpath, posterior_parameters, rng):
    """Write one simulation per posterior parameter set into the wave's run_data folder.

    posterior_parameters: (n_samples, n_parameters - 1) unit-cube values, count column removed.

    Each parameter set is given a cell count drawn uniformly from the count range of
    the config. Returns the full unit-cube sample matrix, with the count column re-inserted.
    """
    # Locate the count parameter and its range:
    gridsearch_parameters = config_dictionary["gridsearch_parameters"]
    parameter_names = [name for name, _ in gridsearch_parameters]
    count_index = parameter_names.index(COUNT_PARAMETER_NAME)
    count_min, count_max = gridsearch_parameters[count_index][1]

    # Validate inputs:
    posterior_parameters = np.asarray(posterior_parameters, dtype=float)
    if posterior_parameters.ndim != 2 or posterior_parameters.shape[1] != len(parameter_names) - 1:
        raise ValueError(
            f"posterior_parameters must have shape (n_samples, {len(parameter_names) - 1}), "
            f"got {posterior_parameters.shape}."
        )
    if np.any(posterior_parameters < 0) or np.any(posterior_parameters > 1):
        raise ValueError("posterior_parameters must be unit-cube values in [0, 1].")

    # Assign a uniformly random cell count to each parameter set:
    sample_count = posterior_parameters.shape[0]
    exact_counts = rng.integers(int(count_min), int(count_max), size=sample_count, endpoint=True)

    # Build the full unit-cube matrix, re-inserting the count column:
    scaled_counts = (exact_counts - count_min) / (count_max - count_min)
    sample_matrix = np.insert(posterior_parameters, count_index, scaled_counts, axis=1)

    # Write the folder structure and argument files:
    print(f"Writing {sample_count} simulations...")
    output_manager = JSONOutputManager(config_dictionary, wave_folderpath)
    output_manager.generate_json_configs(sample_matrix, exact_counts)
    return sample_matrix


def main():
    # Get command line arguments:
    arguments = parse_arguments()
    experiment_folderpath = arguments.experiment_dirpath

    # The posterior of one wave generates the simulations of the next:
    source_wave_folderpath = os.path.join(experiment_folderpath, f"hm{arguments.hm_wave_id}")
    wave_folderpath = os.path.join(experiment_folderpath, f"hm{arguments.hm_wave_id + 1}")
    mcmc_folderpath = os.path.join(source_wave_folderpath, arguments.mcmc_dirname)
    print(f"Drawing samples from {mcmc_folderpath}, writing wave to {wave_folderpath}...")

    # Read in the configuration .json file of the experiment, as the chains live
    # in the unit cube defined by its parameter ranges:
    with open(os.path.join(experiment_folderpath, "config.json")) as config_fstream:
        config_dictionary = json.load(config_fstream)

    # Refuse to write over an existing wave:
    if os.path.exists(wave_folderpath):
        raise FileExistsError(f"{wave_folderpath} already exists.")

    # Draw a thinned subset of the chain for each phenotype:
    posterior_parameters = []
    sample_phenotypes = []
    sample_chain_indices = []
    for phenotype in PHENOTYPES:
        chain_filepath = os.path.join(mcmc_folderpath, CHAIN_FILENAMES[phenotype])
        print(f"Loading {phenotype} chain from {chain_filepath}...")
        mc_distribution = np.load(chain_filepath)
        samples, chain_indices = thin_chain(mc_distribution, arguments.sample_count, arguments.burn_in_fraction)

        # Repeated parameter sets are harmless, but worth knowing about:
        unique_count = np.unique(samples, axis=0).shape[0]
        print(f"Drew {samples.shape[0]} {phenotype} parameter sets, {unique_count} unique.")

        posterior_parameters.append(samples)
        sample_phenotypes += [phenotype] * samples.shape[0]
        sample_chain_indices.append(chain_indices)
    posterior_parameters = np.concatenate(posterior_parameters, axis=0)
    sample_chain_indices = np.concatenate(sample_chain_indices, axis=0)

    # Write the folder structure and argument files:
    rng = np.random.default_rng([COUNT_SEED, arguments.hm_wave_id])
    sample_matrix = write_hm_sweep(config_dictionary, wave_folderpath, posterior_parameters, rng)

    # Save the matrix, and where each of its rows came from, in the wave folder:
    np.save(os.path.join(wave_folderpath, "sample_matrix.npy"), sample_matrix)
    np.save(os.path.join(wave_folderpath, "sample_phenotypes.npy"), np.array(sample_phenotypes))
    np.save(os.path.join(wave_folderpath, "sample_chain_indices.npy"), sample_chain_indices)

    # Record how the wave was generated:
    hm_dictionary = {
        "source_hm_wave_id": arguments.hm_wave_id,
        "mcmc_dirpath": mcmc_folderpath,
        "sample_count": arguments.sample_count,
        "phenotypes": PHENOTYPES,
        "temperature_index": TEMPERATURE_INDEX,
        "burn_in_fraction": arguments.burn_in_fraction,
        "count_seed": COUNT_SEED
    }
    with open(os.path.join(wave_folderpath, "hm_config.json"), 'w') as output:
        json.dump(hm_dictionary, output, indent=4)

    return None


if __name__ == "__main__":
    main()