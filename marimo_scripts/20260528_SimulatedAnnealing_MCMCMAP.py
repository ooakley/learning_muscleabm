import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import fastkde

    import numpy as np

    import scipy.stats

    import matplotlib.pyplot as plt
    return np, plt, scipy


@app.cell
def _(np):
    mc_distribution = np.load("model_experiments/2026-05-20-collisions_shape/mcmc_results/wt_mcmc_chain.npy")
    likelihoods = np.load("model_experiments/2026-05-20-collisions_shape/mcmc_results/wt_mcmc_likelihoods.npy")
    acceptance_rate = np.load("model_experiments/2026-05-20-collisions_shape/mcmc_results/wt_mcmc_acceptance_rate.npy")
    return likelihoods, mc_distribution


@app.cell
def _(likelihoods, mc_distribution):
    # (STEP, PARTICLE, DIMENSION)
    base_temperature_samples = mc_distribution[256::4, :, 0, :]
    base_temperature_likelihoods = likelihoods[256::4, :, 0]
    flat_posterior = base_temperature_samples.reshape(-1, 11)
    flat_likelihoods = base_temperature_likelihoods.flatten()
    return (flat_posterior,)


@app.cell
def _(flat_posterior, scipy):
    kde = scipy.stats.gaussian_kde(flat_posterior[::8, :].T, bw_method=1.0)
    return (kde,)


@app.cell
def _(np):
    def get_proposal(x, rng):
        candidate_components = x + rng.normal(loc=0.0, scale=0.066, size=x.shape)
        acceptance_mask = ~np.logical_or(candidate_components < 0, candidate_components > 1)
        sampled_x = np.copy(x)
        sampled_x[acceptance_mask] = candidate_components[acceptance_mask]
        return sampled_x

    def map_simulated_annealing(kde, ensemble_size, steps, initial_temperature, cooling_rate):
        # Get initial position, and initial density:
        rng = np.random.default_rng(0)
        current_x = rng.uniform(size=(ensemble_size, 11))
        current_nld = -np.log(kde.evaluate(current_x.T))

        # Iterate through solution:
        nld_history = []
        for step_index in range(steps):
            # Print progress:
            if (step_index + 1) % 50 == 0:
                print(step_index + 1, flush=True)

            # Record density history:
            nld_history.append(np.copy(current_nld))

            # Get current temperature:
            current_temperature = initial_temperature * np.exp(-cooling_rate * step_index)

            # Create candidate solution:
            proposal_x = get_proposal(current_x, rng)

            # Get stress:
            proposal_nld = -np.log(kde.evaluate(proposal_x.T))

            # If NLD is lower, immediately accept candidate:
            base_acceptance_mask = proposal_nld < current_nld
            current_x[base_acceptance_mask] = proposal_x[base_acceptance_mask]
            current_nld[base_acceptance_mask] = proposal_nld[base_acceptance_mask]

            # Otherwise calculate acceptance probability based on temperature:
            acceptance_probability = np.exp(-(proposal_nld - current_nld) / current_temperature)
            acceptance_sample = rng.uniform(size=ensemble_size)
            temperature_acceptance_mask = acceptance_sample < acceptance_probability
            current_x[temperature_acceptance_mask] = proposal_x[temperature_acceptance_mask]
            current_nld[temperature_acceptance_mask] = proposal_nld[temperature_acceptance_mask]

        # Return iterated solutions:
        return current_x, np.stack(nld_history, axis=0)

    # candidate_map, nld_history = map_simulated_annealing(kde, 32, 10, 100, 0.003)
    return (get_proposal,)


@app.cell
def _(get_proposal, np):
    def run_ptsa_mode_finding(kde, ensemble_size, temperature_steps, steps, initial_temperature, cooling_rate):
        # Generate first candidate point & likelihood:
        rng = np.random.default_rng(0)
        base_temperatures = np.sqrt(2) ** np.arange(temperature_steps)
        tiled_temperatures = np.tile(base_temperatures, ensemble_size)
        current_x = rng.uniform(size=(ensemble_size * temperature_steps, 11))

        # Estimate current likelihood:
        current_kde = kde.evaluate(current_x.T)

        # Iterate through chain:
        chain_array = [np.copy(current_x)]
        kde_array = [np.copy(current_kde)]
        swap_acceptance = []
        for step_index in range(steps):
            # Print progress:
            if (step_index + 1) % 50 == 0:
                print(step_index + 1, flush=True)

            # Update temperatures:
            current_temperature = initial_temperature * np.exp(-cooling_rate * step_index)
            current_tiled_temperatures = current_temperature * tiled_temperatures

            # Sample new x:
            proposal_x = get_proposal(current_x, rng)
            proposal_kde = kde.evaluate(proposal_x.T)

            # If NLD is lower, immediately accept candidate:
            base_acceptance_mask = proposal_kde > current_kde
            current_x[base_acceptance_mask] = proposal_x[base_acceptance_mask]
            current_kde[base_acceptance_mask] = proposal_kde[base_acceptance_mask]

            # Otherwise calculate acceptance probability based on temperature:
            acceptance_probability = np.exp(-(current_kde - proposal_kde) / current_tiled_temperatures)
            acceptance_sample = rng.uniform(size=ensemble_size * temperature_steps)
            temperature_acceptance_mask = acceptance_sample < acceptance_probability
            current_x[temperature_acceptance_mask] = proposal_x[temperature_acceptance_mask]
            current_kde[temperature_acceptance_mask] = proposal_kde[temperature_acceptance_mask]

            # Propose swaps, reshaping into (ENSEMBLE, TEMPERATURE, DIMENSION) for convenience:
            current_x = current_x.reshape((ensemble_size, temperature_steps, 11))
            current_kde = current_kde.reshape((ensemble_size, temperature_steps))
            swap_indices = rng.choice(temperature_steps - 1, size=ensemble_size)
            swap_count = 0
            for ensemble_index in range(ensemble_size):
                # Get sampled ranks to swap:
                swap_index = swap_indices[ensemble_index]

                # Probability of swap A:
                a_current = current_kde[ensemble_index, swap_index] / (base_temperatures[swap_index] * current_temperature)
                a_proposed = current_kde[ensemble_index, swap_index] / (base_temperatures[swap_index + 1] * current_temperature)
                a_acceptance = np.exp(-(a_current - a_proposed))

                # Probability of swap B:
                b_current = current_kde[ensemble_index, swap_index + 1] / (base_temperatures[swap_index + 1] * current_temperature)
                b_proposed = current_kde[ensemble_index, swap_index + 1] / (base_temperatures[swap_index] * current_temperature)
                b_acceptance = np.exp(-(b_current - b_proposed))

                # Combined probability:
                swap_acceptance_ratio = a_acceptance * b_acceptance
                if np.isinf(b_acceptance):
                    continue

                # Do temperature swap:
                sampled_acceptance = rng.uniform()
                if sampled_acceptance < swap_acceptance_ratio:
                    # print("Sampled acceptance:", sampled_acceptance)
                    # print("Acceptance ratio:", swap_acceptance_ratio)
                    current_x[ensemble_index, [swap_index, swap_index + 1], :] = \
                        current_x[ensemble_index, [swap_index + 1, swap_index], :]
                    current_kde[ensemble_index, [swap_index, swap_index + 1]] = \
                        current_kde[ensemble_index, [swap_index + 1, swap_index]]
                    swap_count += 1

            # Record swap rate:
            swap_acceptance.append(swap_count / ensemble_size)

            # Reshape back to batch:
            current_x = current_x.reshape((-1, 11))
            current_kde = current_kde.flatten()

            # Add to chain:
            chain_array.append(np.copy(current_x))
            kde_array.append(np.copy(current_kde))

        # Return (truncated) chain:
        chain_array = np.stack(chain_array, axis=0)
        kde_array = np.stack(kde_array, axis=0)
        swap_acceptance = np.array(swap_acceptance)
        return chain_array, kde_array, swap_acceptance
    return (run_ptsa_mode_finding,)


@app.cell
def _(kde, run_ptsa_mode_finding):
    chain_array, kde_array, swap_acceptance = run_ptsa_mode_finding(kde, 16, 12, 1000, 3, 0.003)
    return chain_array, kde_array


@app.cell
def _(chain_array, plt):
    plt.plot(chain_array[:, ::8, 0])
    return


@app.cell
def _(kde_array, plt):
    plt.plot(kde_array[:, 0::8])
    return


@app.cell
def _(kde_array, np, plt):
    plt.plot(np.log(kde_array[:, 0:8]))
    return


@app.cell
def _(nld_history, np, plt):
    plt.plot(np.exp(-nld_history))
    return


@app.cell
def _(nld_history, plt):
    plt.plot(nld_history)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
