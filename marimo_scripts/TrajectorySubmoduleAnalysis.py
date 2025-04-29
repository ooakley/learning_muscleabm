import marimo

__generated_with = "0.12.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import analysis_utils

    import colorcet as cc
    import numpy as np
    import matplotlib.pyplot as plt
    return analysis_utils, cc, np, plt


@app.cell
def _(plt):
    plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
    plt.rcParams['figure.dpi'] = 600
    plt.rc('font', weight='bold')
    return


@app.cell
def _(analysis_utils):
    TIMESTEPS = 3000
    MESH_NUMBER = 64
    final_index = TIMESTEPS - 1
    matrix_list, trajectory_list = analysis_utils.get_model_outputs(0, timesteps=TIMESTEPS, mesh_number=MESH_NUMBER, superiterations=1)  
    return MESH_NUMBER, TIMESTEPS, final_index, matrix_list, trajectory_list


@app.cell
def _(trajectory_list):
    trajectory_list[0].columns
    return


@app.cell
def _(trajectory_list):
    sorted_dataframe = trajectory_list[0].sort_values(['particle', 'frame'])
    return (sorted_dataframe,)


@app.cell
def _(sorted_dataframe):
    sorted_dataframe
    return


@app.cell
def _(sorted_dataframe):
    sorted_dataframe["actin_mag"]
    return


@app.cell
def _(np, sorted_dataframe):
    speed = np.asarray(sorted_dataframe["actin_mag"])
    acceleration = np.diff(speed)
    polarisation = np.asarray(sorted_dataframe["polarity_extent"])

    angle = np.asarray(sorted_dataframe["orientation"])
    angle_change = np.abs(np.diff(angle))
    return acceleration, angle, angle_change, polarisation, speed


@app.cell
def _(np):
    rng = np.random.default_rng()
    rng.choice(5, 3)
    return (rng,)


@app.cell
def _(np):
    def get_bin_data(x, y, bin_number = 40):
        # Get binned data across x values:
        bin_count, bin_edges = np.histogram(x, bins=bin_number, density=False)
        y_sum, _ = np.histogram(x, bins=bin_number, weights=y, density=False)
        mean_y = y_sum / bin_count

        # Get bin width:
        half_width = (bin_edges[1] - bin_edges[0]) /2

        return bin_edges[:-1] + half_width, mean_y
    return (get_bin_data,)


@app.cell
def _(np):
    def get_quantile_data(x, y, bin_number=40):
        # Get quantiles as bin edges:
        quantiles = np.linspace(0, 1, bin_number + 1)
        bin_edges = np.quantile(x, quantiles)

        # Get binned data across x values:
        bin_count, _ = np.histogram(x, bins=bin_edges, density=False)
        y_sum, _ = np.histogram(x, bins=bin_edges, weights=y, density=False)
        mean_y = y_sum / bin_count

        return bin_edges, mean_y
    return (get_quantile_data,)


@app.cell
def _(acceleration, get_quantile_data, plt, speed):
    # Get binned data:
    speed_bins, acc_by_speed = get_quantile_data(speed[:-1], acceleration)

    # Plot binned data as line plot:
    fig1, ax1 = plt.subplots(figsize=(4, 3), dpi=600)

    for i, y in enumerate(acc_by_speed):
        ax1.hlines(y, speed_bins[i], speed_bins[i + 1], color='k')
        ax1.scatter((speed_bins[i] + speed_bins[i + 1]) / 2, y, c='k', s=5)

    # Set plot appearance variables:
    ax1.spines[['right', 'top']].set_visible(False)

    xlims = (0, 2.5)
    ax1.hlines(0, *xlims, linestyles='--', color='k')
    ax1.set_xlim(xlims)

    ax1.spines[['left', 'bottom']].set_linewidth(2)
    ax1.tick_params(width=2, labelsize=8)

    ax1.locator_params(axis='y', nbins=6)

    ax1.set_xlabel("|α|", weight='bold')
    ax1.set_ylabel("⟨|δα|⟩", weight='bold')

    # Show plot:
    plt.show()
    return acc_by_speed, ax1, fig1, i, speed_bins, xlims, y


@app.cell
def _(angle_change, get_quantile_data, plt, polarisation):
    # Get binned data:
    polarisation_bins, angular_acc_by_polarisation = get_quantile_data(polarisation[:-1], angle_change)

    # Plot binned data as line plot:
    fig2, ax2 = plt.subplots(figsize=(4, 3), dpi=600)

    for j, averaged_acc in enumerate(angular_acc_by_polarisation):
        ax2.hlines(averaged_acc, polarisation_bins[j], polarisation_bins[j+1], color='k')
        ax2.scatter((polarisation_bins[j] + polarisation_bins[j+1]) / 2, averaged_acc, c='k', s=10)

    # Set plot appearance variables:
    ax2.spines[['right', 'top']].set_visible(False)

    xlims_fig2 = (0, 1)
    ax2.hlines(0, *xlims_fig2, linestyles='--', color='k')
    ax2.set_xlim(xlims_fig2)
    ax2.set_ylim(0, None)

    ax2.spines[['left', 'bottom']].set_linewidth(2)
    ax2.tick_params(width=2, labelsize=8)

    ax2.set_xlabel("|p|", weight='bold')
    ax2.set_ylabel("⟨|δθ|⟩", weight='bold')

    # ax2.locator_params(axis='y', nbins=6)

    # Show plot:
    plt.show()
    return (
        angular_acc_by_polarisation,
        averaged_acc,
        ax2,
        fig2,
        j,
        polarisation_bins,
        xlims_fig2,
    )


@app.cell
def _(angle_change, get_quantile_data, plt, speed, speed_bins):
    # Get binned data:
    _, angular_acc_by_speed = get_quantile_data(speed[:-1], angle_change)

    # Plot binned data as line plot:
    fig3, ax3 = plt.subplots(figsize=(4, 3), dpi=600)

    for k, averaged_ac in enumerate(angular_acc_by_speed):
        ax3.hlines(averaged_ac, speed_bins[k], speed_bins[k+1], color='k')
        ax3.scatter((speed_bins[k] + speed_bins[k+1]) / 2, averaged_ac, c='k', s=10)

    # Set plot appearance variables:
    ax3.spines[['right', 'top']].set_visible(False)

    xlims_fig3 = (0, 2.5)
    ax3.hlines(0, *xlims_fig3, linestyles='--', color='k')
    ax3.set_xlim(xlims_fig3)
    ax3.set_ylim(0, None)

    ax3.spines[['left', 'bottom']].set_linewidth(2)
    ax3.tick_params(width=2, labelsize=8)

    ax3.set_xlabel("|α|", weight='bold')
    ax3.set_ylabel("⟨|δθ|⟩", weight='bold')

    # ax2.locator_params(axis='y', nbins=6)

    # Show plot:
    plt.show()
    return angular_acc_by_speed, averaged_ac, ax3, fig3, k, xlims_fig3


@app.cell
def _(acceleration, get_quantile_data, plt, polarisation, polarisation_bins):
    # Get binned data:
    _, acc_by_polarisation = get_quantile_data(polarisation[:-1], acceleration)

    # Plot binned data as line plot:
    fig4, ax4 = plt.subplots(figsize=(4, 3), dpi=600)

    for l, pol_acc in enumerate(acc_by_polarisation):
        ax4.hlines(pol_acc, polarisation_bins[l], polarisation_bins[l+1], color='k')
        ax4.scatter((polarisation_bins[l] + polarisation_bins[l+1]) / 2, pol_acc, c='k', s=1)

    # Set plot appearance variables:
    ax4.spines[['right', 'top']].set_visible(False)

    xlims_fig4 = (0, 1)
    ax4.hlines(0, *xlims_fig4, linestyles='--', color='k')
    ax4.set_xlim(xlims_fig4)
    ax4.set_ylim(None, None)

    ax4.spines[['left', 'bottom']].set_linewidth(2)
    ax4.tick_params(width=2, labelsize=8)

    ax4.set_xlabel("|p|", weight='bold')
    ax4.set_ylabel("⟨|δα|⟩", weight='bold')

    # ax2.locator_params(axis='y', nbins=6)

    # Show plot:
    plt.show()
    return acc_by_polarisation, ax4, fig4, l, pol_acc, xlims_fig4


if __name__ == "__main__":
    app.run()
