import os

import numpy as np
import pandas as pd

import statsmodels.formula.api as smf

REGRESSANDS = [
    "anni",
    "coherency_fraction",
    "mean_speed",
    "mean_mr"
]

def generate_mlm_results(site_dataframe, regressand):
    # Fit mixed linear model:
    mlm_res = smf.mixedlm(
        f"{regressand} ~ C(phenotype, Treatment(reference='CTL')) * (scaled_particle_count + I(scaled_particle_count**2))",
        site_dataframe, groups=site_dataframe["experiment"],
        re_formula="~ scaled_particle_count + I(scaled_particle_count**2)"
    ).fit(method=["powell", "lbfgs"], reml=True)
    mlm_res.save(os.path.join("wetlab_data", f"{regressand}.res"))


def main():
    # Read in & process wetlab data:
    site_dataframe = pd.read_csv("wetlab_data/site_dataframe.csv")
    particle_counts = np.array(site_dataframe["particle_count"])
    site_dataframe["scaled_particle_count"] = (particle_counts - np.mean(particle_counts)) / np.std(particle_counts)
    site_dataframe["mean_speed"] *= 60  # Get speed in µm/h.

    # Generate MLMs for each regressand:
    for regressand in REGRESSANDS:
        generate_mlm_results(site_dataframe, regressand)


if __name__ == "__main__":
    main()
