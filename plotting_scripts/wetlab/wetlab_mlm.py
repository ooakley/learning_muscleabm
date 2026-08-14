import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import seaborn as sns

    import matplotlib as mpl

    import matplotlib.pyplot as plt
    return mpl, np, pd, plt, sns


@app.cell
def _(mpl):
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = "cmr10"
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams['axes.unicode_minus'] = False
    return


@app.cell
def _():
    CONTROL_PALETTE = "#1A85FF"
    RD_PALETTE = "#D41159"
    return CONTROL_PALETTE, RD_PALETTE


@app.cell
def _(np, pd):
    site_dataframe = pd.read_csv("wetlab_data/site_dataframe.csv")
    site_dataframe = site_dataframe.loc[site_dataframe["particle_count"] < 500]
    particle_counts = np.array(site_dataframe["particle_count"])
    site_dataframe["scaled_particle_count"] = (particle_counts - np.mean(particle_counts)) / np.std(particle_counts)
    site_dataframe["mean_speed"] *= 60
    return particle_counts, site_dataframe


@app.cell
def _(site_dataframe, sns):
    sns.swarmplot(data=site_dataframe, x="density", y="mean_speed", hue="experiment", dodge=True)
    return


@app.cell
def _(site_dataframe, sns):
    sns.swarmplot(data=site_dataframe, x="density", y="coherency_fraction", hue="experiment", dodge=True)
    return


@app.cell
def _(site_dataframe, sns):
    sns.swarmplot(data=site_dataframe, x="density", y="interaction_fraction", hue="experiment", dodge=True)
    return


@app.cell
def _(site_dataframe, sns):
    sns.lmplot(data=site_dataframe, x="particle_count", y="mean_speed", hue="experiment")
    return


@app.cell
def _(site_dataframe, sns):
    sns.lmplot(data=site_dataframe, x="particle_count", y="mean_speed", hue="phenotype", col="experiment", order=1)
    return


@app.cell
def _(site_dataframe, sns):
    sns.lmplot(data=site_dataframe, x="particle_count", y="mean_mr", hue="phenotype", col="experiment", order=1)
    return


@app.cell
def _(site_dataframe, sns):
    sns.lmplot(data=site_dataframe, x="particle_count", y="anni", hue="phenotype", col="experiment", order=2)
    return


@app.cell
def _(site_dataframe, sns):
    sns.lmplot(data=site_dataframe, x="particle_count", y="annd", hue="phenotype", col="experiment", order=2)
    return


@app.cell
def _(site_dataframe, sns):
    sns.lmplot(data=site_dataframe, x="particle_count", y="coherency_fraction", hue="phenotype", col="experiment", order=1)
    return


@app.cell
def _(site_dataframe, sns):
    sns.lmplot(data=site_dataframe, x="particle_count", y="mean_particle_coherency", hue="phenotype", col="experiment", order=1)
    return


@app.cell
def _(site_dataframe, sns):
    sns.lmplot(data=site_dataframe, x="particle_count", y="interaction_fraction", hue="phenotype", col="experiment", order=1)
    return


@app.cell
def _():
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    return sm, smf


@app.cell
def _(site_dataframe, smf):
    # Note that R automatically includes the constitutive terms from an interaction when using asterisk notation:
    linear_lm = smf.mixedlm(
        "anni ~ C(phenotype, Treatment(reference='CTL')) * scaled_particle_count",
        site_dataframe, groups=site_dataframe["experiment"],
        re_formula="~particle_count"
    ).fit(method=["lbfgs"])

    polynomial_lm = smf.mixedlm(
        "anni ~ C(phenotype, Treatment(reference='CTL')) * (scaled_particle_count + I(scaled_particle_count**2))",
        site_dataframe, groups=site_dataframe["experiment"],
        re_formula="~scaled_particle_count + I(scaled_particle_count**2)"
    ).fit(method=["lbfgs"])
    return linear_lm, polynomial_lm


@app.cell
def _(linear_lm):
    print(linear_lm.summary())
    return


@app.cell
def _(polynomial_lm):
    print(polynomial_lm.summary())
    return


@app.cell
def _(CONTROL_PALETTE, RD_PALETTE, np, particle_counts, plt, site_dataframe):
    CAT_VAR = "C(phenotype, Treatment(reference='CTL'))[T.RD]"

    def estimate_linear_mle_regression(phenotype, x, cond, linear_lm):
        # Get base parameters without phenotype interaction:
        base_intercept = linear_lm.params["Intercept"]
        linear_coeff = linear_lm.params["scaled_particle_count"]

        # Get phenotype interactions:
        phenotype_intercept = linear_lm.params[f"{CAT_VAR}"]
        phenotype_intercept_se = linear_lm.bse[f"{CAT_VAR}"]

        linear_interaction = linear_lm.params[f"{CAT_VAR}:scaled_particle_count"]
        linear_interaction_se = linear_lm.bse[f"{CAT_VAR}:scaled_particle_count"]

        # Construct quadratic:
        if cond == "mle":
            b = linear_coeff + (phenotype * linear_interaction)
            c = base_intercept + (phenotype * phenotype_intercept)
        if cond == "lb":
            b = linear_coeff + (phenotype * (linear_interaction - 1.96 * linear_interaction_se))
            c = base_intercept + (phenotype * (phenotype_intercept - 1.96 * phenotype_intercept_se))
        if cond == "ub":
            b = linear_coeff + (phenotype * (linear_interaction + 1.96 * linear_interaction_se))
            c = base_intercept + (phenotype * (phenotype_intercept + 1.96 * phenotype_intercept_se))

        # Get MLE plot:
        y = b*x + c
        return y

    def plot_linear_predictions(model_result, ylabel):
        # Set up predictions:
        x = np.linspace(50, 450)
        scaled_x = (x - np.mean(particle_counts)) / np.std(particle_counts)

        # Set up figure:
        fig, ax = plt.subplots(layout="constrained", figsize=(4, 2.5))

        # Plot control:
        mle_ctl = estimate_linear_mle_regression(-0.5, scaled_x, "mle", model_result)
        lb_ctl = estimate_linear_mle_regression(-0.5, scaled_x, "lb", model_result)
        ub_ctl = estimate_linear_mle_regression(-0.5, scaled_x, "ub", model_result)
        ax.plot(x, mle_ctl, c=CONTROL_PALETTE, label="Control")
        ax.fill_between(x, lb_ctl, ub_ctl, color=CONTROL_PALETTE, alpha=0.2)

        # Plot RD:
        mle_rd = estimate_linear_mle_regression(0.5, scaled_x, "mle", model_result)
        lb_rd = estimate_linear_mle_regression(0.5, scaled_x, "lb", model_result)
        ub_rd = estimate_linear_mle_regression(0.5, scaled_x, "ub", model_result)
        ax.plot(x, mle_rd, c=RD_PALETTE, label="RD")
        ax.fill_between(x, lb_rd, ub_rd, color=RD_PALETTE, alpha=0.2)

        # Plot partial residual plot:
        base_intercept = model_result.params["Intercept"]
        linear_coeff = model_result.params["scaled_particle_count"]
        partial_resid_x = np.array(site_dataframe["scaled_particle_count"])
        partial_resid_y = (linear_coeff * partial_resid_x) + model_result.resid + base_intercept
        ax.scatter(site_dataframe["particle_count"], partial_resid_y, s=1, c='k', alpha=0.2, label="Partial Residuals")

        # Format plots:
        ax.set_xlim(50, 450)
        ax.set_xlabel("Particle Count")
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.show()
    return CAT_VAR, plot_linear_predictions


@app.cell
def _(plot_linear_predictions, site_dataframe, smf):
    def plot_speed():
        # Fit mixed linear model:
        linear_lm = smf.mixedlm(
            "mean_speed ~ C(phenotype, Treatment(reference='CTL')) * scaled_particle_count",
            site_dataframe, groups=site_dataframe["experiment"],
            re_formula="~scaled_particle_count"
        ).fit(method=["lbfgs"])

        # Print summary:
        print(linear_lm.summary())

        # Plot regression with CI over interaction terms:
        plot_linear_predictions(linear_lm, r"Mean Speed ($\mu$m/h)")

    plot_speed()
    return


@app.cell
def _(plot_linear_predictions, site_dataframe, smf):
    def plot_mr():
        # Fit mixed linear model:
        linear_lm = smf.mixedlm(
            "mean_mr ~ C(phenotype, Treatment(reference='CTL')) * scaled_particle_count",
            site_dataframe, groups=site_dataframe["experiment"],
            re_formula="~scaled_particle_count"
        ).fit(method=["lbfgs"])

        # Print summary:
        print(linear_lm.summary())

        # Plot regression with CI over interaction terms:
        plot_linear_predictions(linear_lm, "Meander Ratio")

    plot_mr()
    return


@app.cell
def _(plot_linear_predictions, site_dataframe, smf):
    def plot_cf():
        # Fit mixed linear model:
        linear_lm = smf.mixedlm(
            "coherency_fraction ~ C(phenotype, Treatment(reference='CTL')) * scaled_particle_count",
            site_dataframe, groups=site_dataframe["experiment"],
            re_formula="~scaled_particle_count"
        ).fit(method=["lbfgs"])

        # Print summary:
        print(linear_lm.summary())

        # Plot regression with CI over interaction terms:
        plot_linear_predictions(linear_lm, "Coherency Fraction")

    plot_cf()
    return


@app.cell
def _(
    CAT_VAR,
    CONTROL_PALETTE,
    RD_PALETTE,
    np,
    particle_counts,
    plt,
    site_dataframe,
):
    def estimate_poly_mle_regression(phenotype, x, cond, polynomial_lm):
        # Get base parameters without phenotype interaction:
        base_intercept = polynomial_lm.params["Intercept"]
        linear_coeff = polynomial_lm.params["scaled_particle_count"]
        square_coeff = polynomial_lm.params["I(scaled_particle_count ** 2)"]

        # Get phenotype interactions:
        phenotype_intercept = polynomial_lm.params[f"{CAT_VAR}"]
        phenotype_intercept_se = polynomial_lm.bse[f"{CAT_VAR}"]

        linear_interaction = polynomial_lm.params[f"{CAT_VAR}:scaled_particle_count"]
        linear_interaction_se = polynomial_lm.bse[f"{CAT_VAR}:scaled_particle_count"]

        square_interaction = polynomial_lm.params[f"{CAT_VAR}:I(scaled_particle_count ** 2)"]
        square_interaction_se = polynomial_lm.bse[f"{CAT_VAR}:I(scaled_particle_count ** 2)"]

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

    def plot_poly_predictions(model_result, ylabel):
        fig, ax = plt.subplots(figsize=(4, 2.5))
        # Set up predictions:
        x = np.linspace(50, 450)
        scaled_x = (x - np.mean(particle_counts)) / np.std(particle_counts)

        # Plot control:
        mle_ctl = estimate_poly_mle_regression(-0.5, scaled_x, "mle", model_result)
        lb_ctl = estimate_poly_mle_regression(-0.5, scaled_x, "lb", model_result)
        ub_ctl = estimate_poly_mle_regression(-0.5, scaled_x, "ub", model_result)
        ax.plot(x, mle_ctl, c=CONTROL_PALETTE, label="Control")
        ax.fill_between(x, lb_ctl, ub_ctl, color=CONTROL_PALETTE, alpha=0.2)

        # Plot RD:
        mle_rd = estimate_poly_mle_regression(0.5, scaled_x, "mle", model_result)
        lb_rd = estimate_poly_mle_regression(0.5, scaled_x, "lb", model_result)
        ub_rd = estimate_poly_mle_regression(0.5, scaled_x, "ub", model_result)
        ax.plot(x, mle_rd, c=RD_PALETTE, label="RD")
        ax.fill_between(x, lb_rd, ub_rd, color=RD_PALETTE, alpha=0.2)

        # Plot partial residual plot:
        base_intercept = model_result.params["Intercept"]
        linear_coeff = model_result.params["scaled_particle_count"]
        square_coeff = model_result.params["I(scaled_particle_count ** 2)"]

        partial_resid_x = np.array(site_dataframe["scaled_particle_count"])
        partial_resid_y = (linear_coeff * partial_resid_x) + (square_coeff * partial_resid_x**2) + model_result.resid + base_intercept
        ax.scatter(site_dataframe["particle_count"], partial_resid_y, s=1, c='k', alpha=0.2, label="Partial Residuals")

        # Format plots:
        ax.set_xlim(50, 450)
        ax.set_xlabel("Particle Count")
        ax.set_ylabel(ylabel)
        interaction_pvalue = model_result.pvalues[f"{CAT_VAR}:I(scaled_particle_count ** 2)"]
        ax.legend()
        plt.show()
    return (plot_poly_predictions,)


@app.cell
def _(np, plot_poly_predictions, site_dataframe, sm, smf):
    def plot_poly_speed():
        # Fix off diagonal terms in random effects covariance matrix to zero:
        re_cov = np.ones((3, 3))
        re_cov[1, 1] = 0
        quad_free = sm.regression.mixed_linear_model.MixedLMParams.from_components(
            np.ones(6), re_cov
        )

        # Fit mixed linear model:
        polynomial_lm = smf.mixedlm(
            "mean_speed ~ C(phenotype, Treatment(reference='CTL')) * (scaled_particle_count + I(scaled_particle_count**2))",
            site_dataframe, groups=site_dataframe["experiment"],
            re_formula="~ scaled_particle_count + I(scaled_particle_count**2)"
        ).fit(method=["bfgs"], reml=True, free=quad_free)
        print(polynomial_lm.summary())

        # Plot regression with CI over interaction terms:
        plot_poly_predictions(polynomial_lm, "Speed")

        return polynomial_lm

    test_mlm_res = plot_poly_speed()
    return


@app.cell
def _(plot_poly_predictions, site_dataframe, smf):
    def plot_anni():
        # Fit mixed linear model:
        polynomial_lm = smf.mixedlm(
            "anni ~ C(phenotype, Treatment(reference='CTL')) * (scaled_particle_count + I(scaled_particle_count**2))",
            site_dataframe, groups=site_dataframe["experiment"],
            re_formula="~scaled_particle_count + I(scaled_particle_count**2)"
        ).fit(method=["lbfgs"])

        # Print summary:
        print(polynomial_lm.summary())

        # Plot regression with CI over interaction terms:
        plot_poly_predictions(polynomial_lm, "ANNI")

    plot_anni()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
