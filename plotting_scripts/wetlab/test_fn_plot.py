import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import skimage

    import numpy as np
    import tifffile as tfl
    return skimage, tfl


@app.cell
def _():
    sample_fn_ctl = "/camp/home/eloaklo/home/shared/eloaklo/data/OEO20260427/OEO20260427_1/A1-Site_0/img_channel001_position000_time000000000_z000.tif"

    aa_fn_ctl = "/camp/home/eloaklo/home/shared/eloaklo/data/OEO20260427/OEO20260427_1/A3-Site_0/img_channel001_position008_time000000000_z000.tif"

    sample_fn_rd = "/camp/home/eloaklo/home/shared/eloaklo/data/OEO20260427/OEO20260427_1/A4-Site_0/img_channel001_position012_time000000000_z000.tif"

    aa_fn_rd = "/camp/home/eloaklo/home/shared/eloaklo/data/OEO20260427/OEO20260427_1/A6-Site_0/img_channel001_position020_time000000000_z000.tif"
    return sample_fn_ctl, sample_fn_rd


@app.cell
def _(sample_fn_ctl, sample_fn_rd, tfl):
    ctl_fn = tfl.imread(sample_fn_ctl)
    rd_fn = tfl.imread(sample_fn_rd)
    return (ctl_fn,)


@app.cell
def _(ctl_fn, skimage):
    ctl_background = skimage.restoration.rolling_ball(ctl_fn)
    corrected_ctl = ctl_fn - ctl_background
    return corrected_ctl, ctl_background


@app.cell
def _(ctl_background, plt):
    plt.imshow(ctl_background)
    return


@app.cell
def _(corrected_ctl, plt):
    plt.hist(corrected_ctl.flatten())
    return


@app.cell
def _(corrected_ctl, plt):
    plt.imshow(corrected_ctl, vmin=500, vmax=7500, cmap='gray')
    return


app._unparsable_cell(
    r"""
    }ctl_aa = tfl.imread(aa_fn_ctl)
    rd_aa = tfl.imread(aa_fn_rd)
    """,
    name="_"
)


@app.cell
def _():
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _(ctl_aa, plt):
    plt.hist(ctl_aa.flatten())
    return


@app.cell
def _(ctl_aa, plt):
    def ctl_plot():
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(ctl_aa, vmin=10000, vmax=15000)
        plt.show()

    ctl_plot()
    return


@app.cell
def _(plt, rd_aa):
    def rd_plot():
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rd_aa, vmin=10000, vmax=15000)
        plt.show()

    rd_plot()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
