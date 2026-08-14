import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import json
    import subprocess

    import pandas as pd
    import numpy as np
    import colorcet as cc

    import scipy.stats

    import matplotlib.pyplot as plt

    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.ticker import AutoLocator, MaxNLocator

    from datetime import datetime
    return datetime, json, np, os, pd, subprocess


@app.cell
def _():
    import matplotlib as mpl

    # Font formatting:
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = "cmr10"
    mpl.rcParams['font.size'] = 8
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams['axes.unicode_minus'] = False

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
def _(datetime, os, subprocess):
    # PNG 300 dpi
    # A4 dimensions: 8.27 × 11.69 inches
    # Image dimensions: 160 x ? mm
    # Metadata: date, script, github branch id, og experiment source
    OUT_DIRPATH = "plotting_scripts/gaussian_process/out"
    CONTROL_PALETTE = "#1A85FF"
    RD_PALETTE = "#D41159"
    PIXEL_SIZE = 0.3469 * 2  # Pixel size in µm
    MM_UNIT = 1/25.4  # Millimeters in inches, for matplotlib

    if not os.path.exists(OUT_DIRPATH):
        os.mkdir(OUT_DIRPATH)

    # Get current commit hash:
    commit_hash = subprocess.run("git rev-parse --short HEAD", shell=True, capture_output=True)
    commit_hash = commit_hash.stdout.decode("utf-8")[:-1]

    METADATA_DICTIONARY = {
        "creator": "Omar El Oakley",
        "script": "gp_sobol_indices.py",
        "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_commit_hash": commit_hash
    }
    return


@app.cell
def _():
    METRICS_TO_PLOT = [
        "speeds",
        "meander_ratios",
        "ann_indices",
        "coherency"
    ]

    METRICS_LABELS = [
        "Speed",
        "MR",
        "ANNI",
        "Coherency"
    ]
    return METRICS_LABELS, METRICS_TO_PLOT


@app.cell
def _(METRICS_TO_PLOT, json, np, os):
    EXPERIMENT_DIRPATH = "model_experiments/2026-05-31-collisions_shape"

    with open(os.path.join(EXPERIMENT_DIRPATH, "config.json")) as json_file:
        config_dict = json.load(json_file)

    si_array = []
    sTi_array = []
    for metric_name in METRICS_TO_PLOT:
        # Read in Sobol' indices:
        si_filepath = os.path.join(EXPERIMENT_DIRPATH, "gaussian_process_models", metric_name, "sobol_i.npy")
        si = np.load(si_filepath)
        sTi_filepath = os.path.join(EXPERIMENT_DIRPATH, "gaussian_process_models", metric_name, "sobol_Ti.npy")
        sTi = np.load(sTi_filepath)
        si_array.append(si)
        sTi_array.append(sTi)

    si_array = np.stack(si_array, axis=1)
    sTi_array = np.stack(sTi_array, axis=1)
    return config_dict, sTi_array, si_array


@app.cell
def _(config_dict):
    parameter_list = [parameter_range[0] for parameter_range in config_dict["gridsearch_parameters"]]
    return (parameter_list,)


@app.function
def bold_format(number):
    if number < 0.1:
        return f"{number:.2f}"
    else:
        return f"\\textbf{{{number:.2f}}}"


@app.cell
def _(METRICS_LABELS, np, parameter_list, pd, si_array):
    def print_si_array():
        # Generate dataframe:
        cat_si_array = np.concatenate([si_array, np.sum(si_array, axis=0, keepdims=True)], axis=0)
        si_dataframe = pd.DataFrame(cat_si_array, columns=METRICS_LABELS)
        parameter_column = pd.DataFrame(parameter_list + ["$\Sigma$$S_i$"], columns=["Parameter"])
        full_dataframe = pd.concat([parameter_column, si_dataframe], axis=1)
        print(full_dataframe.to_latex(index=False, float_format=bold_format))

    print_si_array()
    return


@app.cell
def _(METRICS_LABELS, np, parameter_list, pd, sTi_array):
    def print_sti_array():
        # Generate dataframe:
        cat_sTi_array = np.concatenate([sTi_array, np.sum(sTi_array, axis=0, keepdims=True)], axis=0)
        sTi_dataframe = pd.DataFrame(cat_sTi_array, columns=METRICS_LABELS)
        parameter_column = pd.DataFrame(parameter_list + ["$\Sigma$$S_{T_i}$"], columns=["Parameter"])
        full_dataframe = pd.concat([parameter_column, sTi_dataframe], axis=1)
        print(full_dataframe.to_latex(index=False, float_format=bold_format))

    print_sti_array()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
