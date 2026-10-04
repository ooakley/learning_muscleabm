import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import subprocess

    import numpy as np
    import pandas as pd

    import seaborn as sns
    import matplotlib as mpl

    import matplotlib.pyplot as plt

    from datetime import datetime
    return datetime, mpl, np, os, pd, plt, sns, subprocess


@app.cell
def _(mpl):
    mpl.rcdefaults()
    return


@app.cell
def _(mpl):
    from matplotlib import rc
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = "cmr10"
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams['axes.unicode_minus'] = False
    return


@app.cell
def _(datetime, subprocess):
    # PNG 300 dpi
    # A4 dimensions: 8.27 × 11.69 inches
    # Metadata: date, script, github branch id, og experiment source
    OUT_DIRPATH = "plotting_scripts/out"
    CONTROL_PALETTE = "#1A85FF"
    RD_PALETTE = "#D41159"
    PIXEL_SIZE = 0.3469 * 2  # Pixel size in µm
    SAMPLE_EXPERIMENT_PATH = "/camp/home/eloaklo/home/shared/eloaklo/analysed_data/OEO20260313"
    SAMPLE_EXPERIMENT = "OEO20260313"
    MM_UNIT = 1/25.4  # Millimeters in inches, for matplotlib

    # Get current commit hash:
    commit_hash = subprocess.run("git rev-parse --short HEAD", shell=True, capture_output=True)
    commit_hash = commit_hash.stdout.decode("utf-8")[:-1]

    METADATA_DICTIONARY = {
        "creator": "Omar El Oakley",
        "script": "wetlab_speed.py",
        "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_experiment": SAMPLE_EXPERIMENT,
        "current_commit_hash": commit_hash
    }

    seaborn_palette = [CONTROL_PALETTE, RD_PALETTE]
    return (
        CONTROL_PALETTE,
        METADATA_DICTIONARY,
        OUT_DIRPATH,
        RD_PALETTE,
        SAMPLE_EXPERIMENT,
    )


@app.cell
def _(np, pd):
    particle_dataframe = pd.read_csv("wetlab_data/particle_dataframe.csv")
    site_dataframe = pd.read_csv("wetlab_data/site_dataframe.csv")

    # Scale relevant data:
    particle_counts = np.array(site_dataframe["particle_count"])
    site_dataframe["scaled_particle_count"] = (particle_counts - np.mean(particle_counts)) / np.std(particle_counts)
    site_dataframe["mean_speed"] *= 60
    return particle_counts, particle_dataframe, site_dataframe


@app.cell
def _(SAMPLE_EXPERIMENT, particle_dataframe, site_dataframe):
    plot_particle_df = particle_dataframe.copy(deep=True)
    plot_particle_df = plot_particle_df.loc[plot_particle_df["experiment"] == SAMPLE_EXPERIMENT]

    plot_site_df = site_dataframe.copy(deep=True)
    plot_site_df = plot_site_df.loc[plot_site_df["experiment"] == SAMPLE_EXPERIMENT]
    return plot_particle_df, plot_site_df


@app.cell
def _(plot_particle_df):
    plot_particle_df
    return


@app.cell
def _(
    CONTROL_PALETTE,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    RD_PALETTE,
    datetime,
    np,
    os,
    plot_particle_df,
    plt,
):
    def plot_subhistogram(density, ax):
        hist_kwargs = {"histtype": "step", "bins": 70, "density": True}

        # Plot high density histograms:
        density_mask = plot_particle_df["density"] == density
        density_df = plot_particle_df.loc[density_mask]
        # --- CTL:
        ctl_speeds = density_df.loc[density_df["phenotype"] == "CTL", "average_speed"] * 60
        ax.hist(ctl_speeds, color=CONTROL_PALETTE, **hist_kwargs, label="Control")
        ctl_geom_mean = np.exp(np.mean(np.log(ctl_speeds)))
        ax.vlines(ctl_geom_mean, 0, 1, color=CONTROL_PALETTE, transform=ax.get_xaxis_transform())
        # --- RD:
        rd_speeds = density_df.loc[density_df["phenotype"] == "RD", "average_speed"] * 60
        ax.hist(rd_speeds, color=RD_PALETTE, **hist_kwargs, label="RD")
        rd_geom_mean = np.exp(np.mean(np.log(rd_speeds)))
        ax.vlines(rd_geom_mean, 0, 1, color=RD_PALETTE, transform=ax.get_xaxis_transform())


    def plot_speed_histograms():
        fig, axs = plt.subplots(3, 1, figsize=(4.15, 2.85), sharex=True, sharey=True)

        font_kwargs = {
            "horizontalalignment": 'right',
            "verticalalignment": 'center',
            "fontsize": 9,
        }

        # Plot high density histograms:
        plot_subhistogram("High", axs[0])
        axs[0].text(0.99, 0.85, 'High Density', transform=axs[0].transAxes, **font_kwargs)

        # Plot medium density histograms:
        plot_subhistogram("Medium", axs[1])
        axs[1].text(0.99, 0.85, 'Medium Density', transform=axs[1].transAxes, **font_kwargs)

        # Plot low density histograms:
        plot_subhistogram("Low", axs[2])
        axs[2].text(0.99, 0.85, 'Low Density', transform=axs[2].transAxes, **font_kwargs)

        # Set xlabels:
        axs[2].set_xlabel(r"Cell mean speed ($\mu$m/h)")
        fig.text(0.01, 0.5, 'Frequency Density', va='center', rotation='vertical')

        # axs[0].set_xlim([0, 0.6])
        legend = axs[0].legend(loc="lower right", fontsize=8, markerscale=0.6)
        legend.set_in_layout(False)

        # Save figures:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "speed_hist.png"),
            dpi=300, transparent=True, bbox_inches='tight',
            metadata=METADATA_DICTIONARY
        )
        plt.show()


    plot_speed_histograms()
    return


@app.cell
def _(
    CONTROL_PALETTE,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    RD_PALETTE,
    datetime,
    os,
    plot_site_df,
    plt,
    sns,
):
    def speed_swarmplot():
        fig, ax = plt.subplots(figsize=(4, 2.9), layout="constrained")
        sns.swarmplot(
            data=plot_site_df,
            x="density", y="mean_speed",
            hue="phenotype", palette=[CONTROL_PALETTE, RD_PALETTE],
            dodge=True, ax=ax, legend=False
        )
        ax.set_xlabel("Plating density (cells/mL)")
        ax.set_xticklabels([r"$5\cdot10^3$", r"$3.3\cdot10^3$", r"$1.6\cdot10^3$"])
        ax.set_ylabel(r"Site mean speed ($\mu$m/h)")
        ax.legend(labels=['Control', 'RD'], loc="upper left", frameon=False)

        # Save figures:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(os.path.join(OUT_DIRPATH, "speed_swarm.png"), dpi=300, metadata=METADATA_DICTIONARY, transparent=True)
        plt.show()

    speed_swarmplot()
    return


@app.cell
def _(
    CONTROL_PALETTE,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    RD_PALETTE,
    datetime,
    np,
    os,
    plt,
    site_dataframe,
    sns,
):
    def speed_lmplot(experiment_df, axs):
        scatter_kws = {"s": 1}
        sns.regplot(
            data=experiment_df[experiment_df["phenotype"] == "CTL"],
            x="particle_count", y="mean_speed", order=2, ax=axs[0],
            color=CONTROL_PALETTE, label="Control", scatter_kws=scatter_kws
        )
        sns.regplot(
            data=experiment_df[experiment_df["phenotype"] == "RD"],
            x="particle_count", y="mean_speed", order=2, ax=axs[1],
            color=RD_PALETTE, label="RD", scatter_kws=scatter_kws
        )

    def generate_experiment_lmplot():
        # Set up plot:
        fig, axs = plt.subplots(1, 2, figsize=(6, 2.4), sharex=True, sharey=True, layout="constrained")

        # Plot all experiments:
        for index, experiment in enumerate(np.unique(site_dataframe["experiment"])):
            experiment_df = site_dataframe.copy(deep=True)
            experiment_df = experiment_df.loc[experiment_df["experiment"] == experiment]
            speed_lmplot(experiment_df, axs)

        # Format axes:
        axs[0].set_ylabel(r"Site mean speed ($\mu$m/h)")
        axs[0].set_xlabel("Cell density (particles/site)")
        axs[1].set_ylabel("")
        axs[1].set_xlabel("Cell density (particles/site)")

        # Label subplots:
        font_kwargs = {
            "horizontalalignment": 'right',
            "verticalalignment": 'top',
            "fontsize": 10,
        }
        axs[0].text(0.98, 0.97, 'Control Sites', transform=axs[0].transAxes, **font_kwargs)
        axs[1].text(0.98, 0.97, 'RD Sites', transform=axs[1].transAxes, **font_kwargs)

        # Save figures:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(os.path.join(OUT_DIRPATH, "speed_exp_regplot.png"), dpi=300, metadata=METADATA_DICTIONARY, transparent=True)
        plt.show()

    generate_experiment_lmplot()
    return


@app.cell
def _(np, site_dataframe):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    def generate_aic_comparison():
        # Get AIC values across 
        linear_aic = []
        quadratic_aic = []
        for index, experiment in enumerate(np.unique(site_dataframe["experiment"])):
            experiment_df = site_dataframe.copy(deep=True)
            experiment_df = experiment_df.loc[experiment_df["experiment"] == experiment]

            # Get linear fit:
            linear_formula = "mean_speed ~ C(phenotype, Treatment(reference='CTL')) * scaled_particle_count"
            linear_result = smf.ols(linear_formula, data=experiment_df).fit()
            linear_aic.append(linear_result.aic)

            # Get quadratic fit:
            quadratic_formula = \
                "mean_speed ~ C(phenotype, Treatment(reference='CTL')) * (scaled_particle_count + I(scaled_particle_count**2))"
            quadratic_result = smf.ols(quadratic_formula, data=experiment_df).fit()
            quadratic_aic.append(quadratic_result.aic)

        return linear_aic, quadratic_aic

    linear_aic, quadratic_aic = generate_aic_comparison()
    return linear_aic, quadratic_aic, sm, smf


@app.cell
def _(
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    linear_aic,
    np,
    os,
    plt,
    quadratic_aic,
):
    def plot_individual_aic():
        fig, ax = plt.subplots(figsize=(2, 3.75))

        # Plot change in AIC:
        for i in range(5):
            ax.plot([0, 1], [linear_aic[i], quadratic_aic[i]], 'ko-')

        # Plot relative likelihoods:
        for i in range(5):
            aic_difference = quadratic_aic[i] - linear_aic[i]
            relative_likelihood = np.exp((aic_difference) / 2)
            ax.text(1.5, quadratic_aic[i], f"{aic_difference:.3}", ha='center', va='center', c='k')

        ax.text(1.5, 170, r"$\Delta$AIC", ha='center', va='center', c='k')

        # Format and label axes:
        ax.set_xticks([0, 1], ["Linear", "Quadratic"])
        ax.set_xlim([-0.3, 1.85])
        ax.set_ylabel("AIC")

        # Save figure:
        plt.savefig(
            os.path.join(OUT_DIRPATH, "speed_aic.png"),
            dpi=300, metadata=METADATA_DICTIONARY, bbox_inches='tight',
            transparent=True
        )
        plt.show()

    plot_individual_aic()
    return


@app.cell
def _(np, site_dataframe, smf):
    def compare_linear_quad_mlm():
        # --- Fit mixed linear model with linear terms in particle count:
        linear_mlm_res = smf.mixedlm(
            "mean_speed ~  C(phenotype, Treatment(reference='CTL')) * scaled_particle_count",
            site_dataframe, groups=site_dataframe["experiment"],
            re_formula="~ scaled_particle_count"
        ).fit(method=["powell", "lbfgs"], reml=False)
        # --- Fit mixed linear model with quadratic terms in particle count:
        quadratic_mlm_res = smf.mixedlm(
            "mean_speed ~  C(phenotype, Treatment(reference='CTL')) * (scaled_particle_count + I(scaled_particle_count**2))",
            site_dataframe, groups=site_dataframe["experiment"],
            re_formula="~ scaled_particle_count + I(scaled_particle_count**2)"
        ).fit(method=["powell", "lbfgs"], reml=False)
        print(linear_mlm_res.summary())
        print(quadratic_mlm_res.summary())
        print(linear_mlm_res.aic)
        print(quadratic_mlm_res.aic)

        print(f"Linear: {linear_mlm_res.aic}")
        print(f"Quadratic: {quadratic_mlm_res.aic}")
        print(f"Delta AIC: {quadratic_mlm_res.aic - linear_mlm_res.aic}")

        relative_likelihood = np.exp((quadratic_mlm_res.aic - linear_mlm_res.aic) / 2)
        print(f"Relative likelihood: {relative_likelihood}")

    compare_linear_quad_mlm()
    return


@app.cell
def _(np, site_dataframe, smf):
    def compare_phenotype_mlm():
        # --- Fit mixed linear model with linear terms in particle count:
        linear_mlm_res = smf.mixedlm(
            "mean_speed ~ scaled_particle_count + I(scaled_particle_count**2)",
            site_dataframe, groups=site_dataframe["experiment"],
            re_formula="~ scaled_particle_count + I(scaled_particle_count**2)"
        ).fit(method=["powell", "lbfgs"], reml=False)
        # --- Fit mixed linear model with quadratic terms in particle count:
        quadratic_mlm_res = smf.mixedlm(
            "mean_speed ~ C(phenotype, Treatment(reference='CTL')) * (scaled_particle_count + I(scaled_particle_count**2))",
            site_dataframe, groups=site_dataframe["experiment"],
            re_formula="~ scaled_particle_count + I(scaled_particle_count**2)"
        ).fit(method=["powell", "lbfgs"], reml=False)
        print(linear_mlm_res.summary())
        print(quadratic_mlm_res.summary())
        print(f"No Phenotype: {linear_mlm_res.aic}")
        print(f"Phenotype: {quadratic_mlm_res.aic}")
        print(f"Delta AIC: {quadratic_mlm_res.aic - linear_mlm_res.aic}")

        # Get delta AIC:
        relative_likelihood = np.exp((quadratic_mlm_res.aic - linear_mlm_res.aic) / 2)
        print(f"Relative likelihood: {relative_likelihood}")

    compare_phenotype_mlm()
    return


@app.cell
def _(np, site_dataframe, sm, smf):
    def compare_mlm():
        # --- Initialise off-diagonal terms in random effects covariance matrix to zero (lose convergence otherwise):
        re_cov = np.eye(3)
        re_cov[1, 1] = 0
        free = sm.regression.mixed_linear_model.MixedLMParams.from_components(
            np.ones(6), re_cov
        )
        # --- Fit mixed linear model with linear terms in particle count:
        linear_mlm_res = smf.mixedlm(
            "mean_speed ~ C(phenotype, Treatment(reference='CTL')) * scaled_particle_count",
            site_dataframe, groups=site_dataframe["experiment"],
            re_formula="~ scaled_particle_count"
        ).fit(method=["powell", "lbfgs"], reml=False)
        # --- Fit mixed linear model with quadratic terms in particle count:
        quadratic_mlm_res = smf.mixedlm(
            "mean_speed ~ C(phenotype, Treatment(reference='CTL')) * (scaled_particle_count + I(scaled_particle_count**2))",
            site_dataframe, groups=site_dataframe["experiment"],
            re_formula="~ scaled_particle_count + I(scaled_particle_count**2)"
        ).fit(method=["powell", "lbfgs"], reml=False)

        print(linear_mlm_res.summary())
        print(quadratic_mlm_res.summary())
        print(linear_mlm_res.aic)
        print(quadratic_mlm_res.aic)

    compare_mlm()
    return


@app.cell
def _(site_dataframe, smf):
    speed_mlm_res = smf.mixedlm(
        "mean_speed ~ C(phenotype, Treatment(reference='CTL')) * (scaled_particle_count + I(scaled_particle_count**2))",
        site_dataframe, groups=site_dataframe["experiment"],
        re_formula="~ scaled_particle_count + I(scaled_particle_count**2)"
    ).fit(method=["powell", "lbfgs"], reml=True)
    print(speed_mlm_res.summary())
    return (speed_mlm_res,)


@app.cell
def _(speed_mlm_res):
    # Format latex string for inclusion of table in thesis:
    latex_string = speed_mlm_res.summary().as_latex()
    latex_string = latex_string.replace("C(phenotype, Treatment(reference='CTL'))[T.RD]", "C(Phenotype)")
    latex_string = latex_string.replace("scaled\_particle\_count", "Cell Density")
    # Properly format whitespace, per line:
    latex_string = latex_string.split("\n")
    latex_string = [' '.join(line_string.split()) for line_string in latex_string]
    # Remove group variance estimates as they are typically uninformative:

    print("\n".join(latex_string))
    return (latex_string,)


@app.cell
def _(latex_string):
    print(latex_string)
    return


@app.cell
def _(
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    np,
    os,
    plt,
    site_dataframe,
    speed_mlm_res,
):
    def plot_residuals():
        # Plot data:
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.scatter(
            site_dataframe["particle_count"],
            speed_mlm_res.resid, s=1.5, c='k'
        )

        # Format axes:
        lower_xlimit = np.min(np.array(site_dataframe["particle_count"]))
        upper_xlimit = np.max(np.array(site_dataframe["particle_count"]))
        ax.hlines(0, lower_xlimit, upper_xlimit, ls="--", color='r')
        ax.set_xlim(lower_xlimit, upper_xlimit)
        ax.set_ylabel("Standardised residuals")
        ax.set_xlabel("Cell density (particles/site)")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "speed_residuals.png"),
            dpi=300, metadata=METADATA_DICTIONARY, bbox_inches='tight',
            transparent=True
        )
        plt.show()

    plot_residuals()
    return


@app.cell
def _(METADATA_DICTIONARY, OUT_DIRPATH, np, os, plt, speed_mlm_res):
    import scipy.stats

    def plot_residual_qq_plot():
        # Calculate QQ positions:
        quantiles = np.linspace(0.00, 1.0)
        standard_normal = scipy.stats.Normal(mu=0, sigma=1)
        qq_x = standard_normal.icdf(quantiles)
        normalised_residuals = speed_mlm_res.resid / np.std(speed_mlm_res.resid)
        qq_y = np.quantile(normalised_residuals, quantiles)

        # Plot data:
        axis_limit = 2
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.scatter(qq_x, qq_y, c='k', alpha=0.5, s=20)
        ax.plot(np.linspace(-3, 3), np.linspace(-3, 3), ls="--", c="k")
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_aspect("equal")
        ax.text(-1.9, 1.9, "Residuals Q-Q Plot", ha="left", va="top")
        ax.set_xlabel("Standard normal quantiles")
        ax.set_ylabel("Standardised residuals")
        plt.savefig(
            os.path.join(OUT_DIRPATH, "speed_residual_qq_plot.png"),
            dpi=300, metadata=METADATA_DICTIONARY, bbox_inches='tight',
            transparent=True
        )
        plt.show()

    plot_residual_qq_plot()
    return


@app.cell
def _():
    CAT_VAR = "C(phenotype, Treatment(reference='CTL'))[T.RD]"
    return (CAT_VAR,)


@app.cell
def _(
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    np,
    os,
    particle_counts,
    plt,
    site_dataframe,
    speed_mlm_res,
):
    def estimate_ci_quad_reg(mlm_res, x, cond):
        # Get base parameters without phenotype interaction:
        base_intercept = mlm_res.params["Intercept"]
        base_intercept_se = mlm_res.bse["Intercept"]
        linear_coeff = mlm_res.params["scaled_particle_count"]
        linear_coeff_se = mlm_res.bse["scaled_particle_count"]
        square_coeff = mlm_res.params["I(scaled_particle_count ** 2)"]
        square_coeff_se = mlm_res.bse["I(scaled_particle_count ** 2)"]

        # Construct quadratic:
        if cond == "mle":
            a = square_coeff
            b = linear_coeff
            c = base_intercept
        if cond == "lb":
            a = square_coeff - (1.96 * square_coeff_se)
            b = linear_coeff - (1.96 * linear_coeff_se)
            c = base_intercept - (1.96 * base_intercept_se)
        if cond == "ub":
            a = square_coeff + (1.96 * square_coeff_se)
            b = linear_coeff + (1.96 * linear_coeff_se)
            c = base_intercept + (1.96 * base_intercept_se)

        # Get MLE plot:
        y = a*(x**2) + b*x + c
        return y

    def plot_random_effects_partial():
        # Calculate estimate and CI:
        lower_xlimit = np.min(np.array(site_dataframe["particle_count"]))
        upper_xlimit = np.max(np.array(site_dataframe["particle_count"]))
        x = np.linspace(lower_xlimit, upper_xlimit)
        scaled_x = (x - np.mean(particle_counts)) / np.std(particle_counts)
        lower_bound = estimate_ci_quad_reg(speed_mlm_res, scaled_x, "lb")
        upper_bound = estimate_ci_quad_reg(speed_mlm_res, scaled_x, "ub")
        regression_estimte = estimate_ci_quad_reg(speed_mlm_res, scaled_x, "mle")

        # Plot regression and CI:
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.plot(x, regression_estimte, c='k')
        ax.fill_between(x, lower_bound, upper_bound, color='k', alpha=0.2)

        # Plot residuals:
        partial_resid_x = np.array(site_dataframe["scaled_particle_count"])
        partial_resid_y = estimate_ci_quad_reg(speed_mlm_res, partial_resid_x, "mle") + speed_mlm_res.resid
        ax.scatter(site_dataframe["particle_count"], partial_resid_y, s=1, c='k', alpha=0.2, label="Partial Residuals")
        ax.set_xlim(lower_xlimit, upper_xlimit)
        ax.set_xlabel("Cell density (particles/site)")
        ax.set_ylabel(r"Site mean speed ($\mu$m/h)")

        # Save figure:
        plt.savefig(
            os.path.join(OUT_DIRPATH, "speed_random_effect_ccpr.png"),
            dpi=300, metadata=METADATA_DICTIONARY, bbox_inches='tight',
            transparent=True
        )
        plt.show()

    plot_random_effects_partial()
    return


@app.cell
def _(
    CAT_VAR,
    CONTROL_PALETTE,
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    RD_PALETTE,
    np,
    os,
    particle_counts,
    plt,
    site_dataframe,
    speed_mlm_res,
):
    def estimate_phenotype_ci_quad_reg(mlm_res, x, cond, phenotype):
        # Get base parameters without phenotype interaction:
        base_intercept = mlm_res.params["Intercept"]
        linear_coeff = mlm_res.params["scaled_particle_count"]
        square_coeff = mlm_res.params["I(scaled_particle_count ** 2)"]

        # Get phenotype interactions:
        phenotype_intercept = mlm_res.params[f"{CAT_VAR}"]
        phenotype_intercept_se = mlm_res.bse[f"{CAT_VAR}"]

        linear_interaction = mlm_res.params[f"{CAT_VAR}:scaled_particle_count"]
        linear_interaction_se = mlm_res.bse[f"{CAT_VAR}:scaled_particle_count"]

        square_interaction = mlm_res.params[f"{CAT_VAR}:I(scaled_particle_count ** 2)"]
        square_interaction_se = mlm_res.bse[f"{CAT_VAR}:I(scaled_particle_count ** 2)"]

        # Construct quadratic:
        if cond == "mle":
            a = square_coeff + (phenotype * square_interaction)
            b = linear_coeff + (phenotype * linear_interaction)
            c = base_intercept + (phenotype * phenotype_intercept)
        if cond == "lb":
            a = square_coeff + (phenotype * (square_interaction - 1.96 * square_interaction_se))
            b = linear_coeff + (phenotype * (linear_interaction - 1.96 * linear_interaction_se))
            c = base_intercept + (phenotype * (phenotype_intercept - 1.96 * phenotype_intercept_se))
        if cond == "ub":
            a = square_coeff + (phenotype * (square_interaction + 1.96 * square_interaction_se))
            b = linear_coeff + (phenotype * (linear_interaction + 1.96 * linear_interaction_se))
            c = base_intercept + (phenotype * (phenotype_intercept + 1.96 * phenotype_intercept_se))

        # Get MLE plot:
        y = a*(x**2) + b*x + c
        return y


    def plot_random_effects_phenotype():
        # Calculate estimate and CI:
        lower_xlimit = np.min(np.array(site_dataframe["particle_count"]))
        upper_xlimit = np.max(np.array(site_dataframe["particle_count"]))
        x = np.linspace(lower_xlimit, upper_xlimit)
        scaled_x = (x - np.mean(particle_counts)) / np.std(particle_counts)

        # Plot regression and CI:
        fig, ax = plt.subplots(figsize=(4, 2.5))

        # Plot control data:
        ctl_lb = estimate_phenotype_ci_quad_reg(speed_mlm_res, scaled_x, "lb", -0.5)
        ctl_ub = estimate_phenotype_ci_quad_reg(speed_mlm_res, scaled_x, "ub", -0.5)
        ctl_regression = estimate_phenotype_ci_quad_reg(speed_mlm_res, scaled_x, "mle", -0.5)
        ax.plot(x, ctl_regression, c=CONTROL_PALETTE, label="Control")
        ax.fill_between(x, ctl_lb, ctl_ub, color=CONTROL_PALETTE, alpha=0.4)

        # Plot control residuals:
        control_mask = site_dataframe["phenotype"] == "CTL"
        partial_resid_x = np.array(site_dataframe.loc[control_mask, "scaled_particle_count"])
        ctl_estimates = estimate_phenotype_ci_quad_reg(speed_mlm_res, partial_resid_x, "mle", -0.5)
        partial_resid_y = ctl_estimates + speed_mlm_res.resid[control_mask]
        ax.scatter(
            site_dataframe.loc[control_mask, "particle_count"],
            partial_resid_y, s=1, c=CONTROL_PALETTE, alpha=0.4
        )

        # Plot RD data:
        rd_lb = estimate_phenotype_ci_quad_reg(speed_mlm_res, scaled_x, "lb", 0.5)
        rd_ub = estimate_phenotype_ci_quad_reg(speed_mlm_res, scaled_x, "ub", 0.5)
        rd_regression = estimate_phenotype_ci_quad_reg(speed_mlm_res, scaled_x, "mle", 0.5)
        ax.plot(x, rd_regression, c=RD_PALETTE, label="RD")
        ax.fill_between(x, rd_lb, rd_ub, color=RD_PALETTE, alpha=0.4)

        # Plot RD residuals:
        rd_mask = site_dataframe["phenotype"] == "RD"
        partial_resid_x = np.array(site_dataframe.loc[rd_mask, "scaled_particle_count"])
        rd_estimates = estimate_phenotype_ci_quad_reg(speed_mlm_res, partial_resid_x, "mle", 0.5)
        partial_resid_y = rd_estimates + speed_mlm_res.resid[rd_mask]
        ax.scatter(
            site_dataframe.loc[rd_mask, "particle_count"],
            partial_resid_y, s=1, c=RD_PALETTE, alpha=0.4
        )

        # Format axes:
        ax.set_xlim(lower_xlimit, upper_xlimit)
        ax.set_xlabel("Cell density (particles/site)")
        ax.set_ylabel(r"Site mean speed ($\mu$m/h)")
        ax.legend()

        # Save figure:
        plt.savefig(
            os.path.join(OUT_DIRPATH, "speed_phenotype_ccpr.png"),
            dpi=300, metadata=METADATA_DICTIONARY, bbox_inches='tight',
            transparent=True
        )
        plt.show()

    plot_random_effects_phenotype()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
