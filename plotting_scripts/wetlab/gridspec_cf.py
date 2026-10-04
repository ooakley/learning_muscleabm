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

    import matplotlib.pyplot as plt

    from datetime import datetime
    return datetime, np, os, pd, plt, sns, subprocess


@app.cell
def _():
    import matplotlib as mpl

    # Font formatting:
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = "cmr10"
    mpl.rcParams['font.size'] = 9
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['axes.labelsize'] = 9

    # Tick formating:
    mpl.rcParams['xtick.major.size'] = 2
    mpl.rcParams['xtick.major.pad'] = 1.5
    mpl.rcParams['ytick.major.size'] = 2
    mpl.rcParams['ytick.major.pad'] = 1.5
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7

    # Label formatting:
    mpl.rcParams['axes.labelpad'] = 2.5

    # Layout formatting:
    mpl.rcParams['figure.constrained_layout.hspace'] = 0.04
    mpl.rcParams['figure.constrained_layout.wspace'] = 0.04
    return


@app.cell
def _(datetime, subprocess):
    # PNG 300 dpi
    # A4 dimensions: 8.27 × 11.69 inches
    # Metadata: date, script, github branch id, og experiment source
    OUT_DIRPATH = "plotting_scripts/wetlab/out"
    CONTROL_PALETTE = "#1A85FF"
    RD_PALETTE = "#D41159"
    PIXEL_SIZE = 0.3469 * 2  # Pixel size in µm
    SAMPLE_EXPERIMENT_PATH = "/camp/home/eloaklo/home/shared/eloaklo/analysed_data/OEO20260313"
    SAMPLE_EXPERIMENT = "OEO20260313"
    MM_UNIT = 1/25.4  # Millimeters in inches, for matplotlib

    TEXT_WIDTH = 135 * MM_UNIT
    TEXT_HEIGHT = 217 * MM_UNIT 

    FULL_WIDTH = 170 * MM_UNIT
    FULL_HEIGHT = TEXT_HEIGHT * 0.8

    # Get current commit hash:
    commit_hash = subprocess.run("git rev-parse --short HEAD", shell=True, capture_output=True)
    commit_hash = commit_hash.stdout.decode("utf-8")[:-1]

    METADATA_DICTIONARY = {
        "creator": "Omar El Oakley",
        "script": "wetlab_anni.py",
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
        TEXT_WIDTH,
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
    return (plot_site_df,)


@app.cell
def _(CONTROL_PALETTE, RD_PALETTE, plot_site_df, sns):
    def cf_swarmplot(ax):
        sns.swarmplot(
            data=plot_site_df,
            x="density", y="coherency_fraction",
            hue="phenotype", palette=[CONTROL_PALETTE, RD_PALETTE],
            dodge=True, ax=ax, legend=False, s=4
        )
        ax.set_xlabel("Plating density (cells/mL)")
        ax.set_xticklabels([r"$5\cdot10^3$", r"$3.3\cdot10^3$", r"$1.6\cdot10^3$"])
        ax.set_ylabel("Coherency fraction")
        ax.legend(labels=['Control', 'RD'], loc="lower right", frameon=True)
    return (cf_swarmplot,)


@app.cell
def _(CONTROL_PALETTE, RD_PALETTE, np, site_dataframe, sns):
    def cf_lmplot(experiment_df, ctl_ax, rd_ax):
        scatter_kws = {"s": 1}
        sns.regplot(
            data=experiment_df[experiment_df["phenotype"] == "CTL"],
            x="particle_count", y="coherency_fraction", order=2, ax=ctl_ax,
            color=CONTROL_PALETTE, label="Control", scatter_kws=scatter_kws
        )
        sns.regplot(
            data=experiment_df[experiment_df["phenotype"] == "RD"],
            x="particle_count", y="coherency_fraction", order=2, ax=rd_ax,
            color=RD_PALETTE, label="RD", scatter_kws=scatter_kws
        )


    def plot_experiment_lmplot(ctl_ax, rd_ax):
        # Plot all experiments:
        for index, experiment in enumerate(np.unique(site_dataframe["experiment"])):
            experiment_df = site_dataframe.copy(deep=True)
            experiment_df = experiment_df.loc[experiment_df["experiment"] == experiment]
            cf_lmplot(experiment_df, ctl_ax, rd_ax)

        # Format axes:
        ctl_ax.set_ylabel("Coherency fraction")
        ctl_ax.set_xlabel("")
        rd_ax.set_ylabel("Coherency fraction")
        rd_ax.set_xlabel("Cell density (particles/site)")

        # Force sharing of axes (gridspec makes using sharex difficult):
        rd_ax.set_xticks(ctl_ax.get_xticks())
        rd_ax.set_xlim(ctl_ax.get_xlim())
        rd_ax.set_yticks(ctl_ax.get_yticks())
        rd_ax.set_ylim(ctl_ax.get_ylim())

        # Label subplots:
        font_kwargs = {
            "horizontalalignment": 'right',
            "verticalalignment": 'top',
            "fontsize": 9,
        }
        ctl_ax.text(0.98, 0.97, 'Control Sites', transform=ctl_ax.transAxes, **font_kwargs)
        rd_ax.text(0.98, 0.97, 'RD Sites', transform=rd_ax.transAxes, **font_kwargs)
    return (plot_experiment_lmplot,)


@app.cell
def _(
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    TEXT_WIDTH,
    cf_swarmplot,
    datetime,
    os,
    plot_experiment_lmplot,
    plt,
):
    def annotate_axis(ax, label):
        ax.annotate(label, xy=(0, 1), xycoords='axes fraction',
            xytext=(-0.25, +0.25), textcoords='offset fontsize',
            va='bottom', ha='right', fontsize=10,
         )

    def cf_mosaic_a():
        # Set up gridspec:
        mosaic_specification = ".B;AB;AC;.C"
        fig = plt.figure(figsize=(TEXT_WIDTH, 4))
        axd = fig.subplot_mosaic(mosaic_specification)

        # Annotate the axes:
        annotate_axis(axd["A"], "A")
        annotate_axis(axd["B"], "B")
        annotate_axis(axd["C"], "C")

        # Plot the swarmplot:
        cf_swarmplot(axd["A"])

        # Plot the lmplot:
        plot_experiment_lmplot(axd["B"], axd["C"])

        fig.subplots_adjust(0.1, 0.1, 1 - 0.1, 1 - 0.05, hspace=0.55, wspace=0.275)

        # Save figures:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(os.path.join(OUT_DIRPATH, "cf_mosaic_A.png"), dpi=300, metadata=METADATA_DICTIONARY, transparent=True)
        plt.show()

    cf_mosaic_a()
    return (annotate_axis,)


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
            linear_formula = "coherency_fraction ~ C(phenotype, Treatment(reference='CTL')) * scaled_particle_count"
            linear_result = smf.ols(linear_formula, data=experiment_df).fit()
            linear_aic.append(linear_result.aic)

            # Get quadratic fit:
            quadratic_formula = \
                "coherency_fraction ~ C(phenotype, Treatment(reference='CTL')) * (scaled_particle_count + I(scaled_particle_count**2))"
            quadratic_result = smf.ols(quadratic_formula, data=experiment_df).fit()
            quadratic_aic.append(quadratic_result.aic)

        return linear_aic, quadratic_aic

    linear_aic, quadratic_aic = generate_aic_comparison()
    return linear_aic, quadratic_aic, smf


@app.cell
def _(linear_aic, quadratic_aic):
    def plot_individual_aic(ax):
        # Plot change in AIC:
        for i in range(5):
            ax.plot([0, 1], [linear_aic[i], quadratic_aic[i]], 'ko-', markersize=4)

        # Format and label axes:
        ax.set_xticks([0, 1], ["Linear", "Quadratic"])
        ax.set_xlim([-0.3, 1.3])
        ax.set_ylabel("AIC")

        ax.yaxis.tick_right()
    return (plot_individual_aic,)


@app.cell
def _(site_dataframe, smf):
    cf_mlm_res = smf.mixedlm(
        "coherency_fraction ~ C(phenotype, Treatment(reference='CTL')) * (scaled_particle_count + I(scaled_particle_count**2))",
        site_dataframe, groups=site_dataframe["experiment"],
        re_formula="~ scaled_particle_count + I(scaled_particle_count**2)"
    ).fit(method=["powell", "lbfgs"], reml=True)
    print(cf_mlm_res.summary())
    return (cf_mlm_res,)


@app.cell
def _(cf_mlm_res, np, site_dataframe):
    def plot_residuals(ax):
        # Plot data:
        internally_studentised_residuals = cf_mlm_res.resid / np.std(cf_mlm_res.resid)
        ax.scatter(
            site_dataframe["particle_count"],
            internally_studentised_residuals, s=1.5, c='k'
        )
        ylimit = np.max(np.abs(internally_studentised_residuals))
        ylimit *= 1.1

        # Format axes:
        lower_xlimit = np.min(np.array(site_dataframe["particle_count"]))
        upper_xlimit = np.max(np.array(site_dataframe["particle_count"]))
        ax.hlines(0, lower_xlimit, upper_xlimit, ls="--", color='r')
        ax.set_xlim(lower_xlimit, upper_xlimit)
        ax.set_ylim(-ylimit, ylimit)
        ax.set_ylabel("Studentised residuals")
        ax.set_xlabel("Cell density (particles/site)")
    return (plot_residuals,)


@app.cell
def _(cf_mlm_res, np):
    import scipy.stats

    def plot_residual_qq_plot(ax):
        # Calculate QQ positions:
        quantiles = np.linspace(0.00, 1.0)
        standard_normal = scipy.stats.Normal(mu=0, sigma=1)
        qq_x = standard_normal.icdf(quantiles)
        normalised_residuals = cf_mlm_res.resid / np.std(cf_mlm_res.resid)
        qq_y = np.quantile(normalised_residuals, quantiles)

        # Plot data:
        axis_limit = 2
        ax.scatter(qq_x, qq_y, c='k', alpha=0.5, s=20)
        ax.plot(np.linspace(-3, 3), np.linspace(-3, 3), ls="--", c="k")
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_aspect("equal")
        ax.text(-1.9, 1.9, "Residuals Q-Q Plot", ha="left", va="top")
        ax.set_xlabel("Standard normal quantiles")
        ax.set_ylabel("Standardised residuals")
    return (plot_residual_qq_plot,)


@app.cell
def _(cf_mlm_res, np, particle_counts, site_dataframe):
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

    def plot_random_effects_partial(ax):
        # Calculate estimate and CI:
        lower_xlimit = np.min(np.array(site_dataframe["particle_count"]))
        upper_xlimit = np.max(np.array(site_dataframe["particle_count"]))
        x = np.linspace(lower_xlimit, upper_xlimit)
        scaled_x = (x - np.mean(particle_counts)) / np.std(particle_counts)
        lower_bound = estimate_ci_quad_reg(cf_mlm_res, scaled_x, "lb")
        upper_bound = estimate_ci_quad_reg(cf_mlm_res, scaled_x, "ub")
        regression_estimte = estimate_ci_quad_reg(cf_mlm_res, scaled_x, "mle")

        # Plot regression and CI:
        ax.plot(x, regression_estimte, c='k')
        ax.fill_between(x, lower_bound, upper_bound, color='k', alpha=0.2)

        # Plot residuals:
        partial_resid_x = np.array(site_dataframe["scaled_particle_count"])
        partial_resid_y = estimate_ci_quad_reg(cf_mlm_res, partial_resid_x, "mle") + cf_mlm_res.resid
        ax.scatter(site_dataframe["particle_count"], partial_resid_y, s=1, c='k', alpha=0.2, label="Partial Residuals")
        ax.set_xlim(lower_xlimit, upper_xlimit)
        ax.set_xlabel("Cell density (particles/site)")
        ax.set_ylabel("Coherency fraction")
    return (plot_random_effects_partial,)


@app.cell
def _(
    CONTROL_PALETTE,
    RD_PALETTE,
    cf_mlm_res,
    np,
    particle_counts,
    site_dataframe,
):
    CAT_VAR = "C(phenotype, Treatment(reference='CTL'))[T.RD]"

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


    def plot_random_effects_phenotype(ax):
        # Calculate estimate and CI:
        lower_xlimit = np.min(np.array(site_dataframe["particle_count"]))
        upper_xlimit = np.max(np.array(site_dataframe["particle_count"]))
        x = np.linspace(lower_xlimit, upper_xlimit)
        scaled_x = (x - np.mean(particle_counts)) / np.std(particle_counts)

        CONTROL_CAT = 0.0
        RD_CAT = 1.0

        # Plot control data:
        ctl_lb = estimate_phenotype_ci_quad_reg(cf_mlm_res, scaled_x, "lb", CONTROL_CAT)
        ctl_ub = estimate_phenotype_ci_quad_reg(cf_mlm_res, scaled_x, "ub", CONTROL_CAT)
        ctl_regression = estimate_phenotype_ci_quad_reg(cf_mlm_res, scaled_x, "mle", CONTROL_CAT)
        ax.plot(x, ctl_regression, c=CONTROL_PALETTE, label="Control")
        ax.fill_between(x, ctl_lb, ctl_ub, color=CONTROL_PALETTE, alpha=0.4)

        # Plot control residuals:
        control_mask = site_dataframe["phenotype"] == "CTL"
        partial_resid_x = np.array(site_dataframe.loc[control_mask, "scaled_particle_count"])
        ctl_estimates = estimate_phenotype_ci_quad_reg(cf_mlm_res, partial_resid_x, "mle", CONTROL_CAT)
        partial_resid_y = ctl_estimates + cf_mlm_res.resid[control_mask]
        ax.scatter(
            site_dataframe.loc[control_mask, "particle_count"],
            partial_resid_y, s=1, c=CONTROL_PALETTE, alpha=0.4
        )

        # Plot RD data:
        rd_lb = estimate_phenotype_ci_quad_reg(cf_mlm_res, scaled_x, "lb", RD_CAT)
        rd_ub = estimate_phenotype_ci_quad_reg(cf_mlm_res, scaled_x, "ub", RD_CAT)
        rd_regression = estimate_phenotype_ci_quad_reg(cf_mlm_res, scaled_x, "mle", RD_CAT)
        ax.plot(x, rd_regression, c=RD_PALETTE, label="RD")
        ax.fill_between(x, rd_lb, rd_ub, color=RD_PALETTE, alpha=0.4)

        # Plot RD residuals:
        rd_mask = site_dataframe["phenotype"] == "RD"
        partial_resid_x = np.array(site_dataframe.loc[rd_mask, "scaled_particle_count"])
        rd_estimates = estimate_phenotype_ci_quad_reg(cf_mlm_res, partial_resid_x, "mle", RD_CAT)
        partial_resid_y = rd_estimates + cf_mlm_res.resid[rd_mask]
        ax.scatter(
            site_dataframe.loc[rd_mask, "particle_count"],
            partial_resid_y, s=1, c=RD_PALETTE, alpha=0.4
        )

        # Format axes:
        ax.set_xlim(lower_xlimit, upper_xlimit)
        ax.set_xlabel("Cell density (particles/site)")
        ax.set_ylabel("Coherency fraction")
        ax.legend()
    return (plot_random_effects_phenotype,)


@app.cell
def _(
    METADATA_DICTIONARY,
    OUT_DIRPATH,
    TEXT_WIDTH,
    annotate_axis,
    datetime,
    os,
    plot_individual_aic,
    plot_random_effects_partial,
    plot_random_effects_phenotype,
    plot_residual_qq_plot,
    plot_residuals,
    plt,
):
    def cf_mosaic_b():
        # Set up gridspec:
        mosaic_specification = "ACE;BDE"
        fig = plt.figure(figsize=(TEXT_WIDTH, 4))
        axd = fig.subplot_mosaic(mosaic_specification, width_ratios=[1.5, 1, 0.45])

        # Annotate the subplots:
        annotate_axis(axd["A"], "A")
        annotate_axis(axd["B"], "B")
        annotate_axis(axd["C"], "C")
        annotate_axis(axd["D"], "D")
        annotate_axis(axd["E"], "E")

        # Run the plotting logic:
        plot_random_effects_partial(axd["A"])
        plot_random_effects_phenotype(axd["B"])
        plot_residual_qq_plot(axd["C"])
        plot_residuals(axd["D"])
        plot_individual_aic(axd["E"])

        fig.subplots_adjust(0.07, 0.1, 1 - 0.07, 1 - 0.05, hspace=0.4, wspace=0.3)

        # Save figures:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(os.path.join(OUT_DIRPATH, "cf_mosaic_B.png"), dpi=300, metadata=METADATA_DICTIONARY, transparent=True)
        plt.show()

    cf_mosaic_b()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
