import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import subprocess

    import skimage
    import sklearn

    import scipy.stats

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import colorcet as cc
    import tifffile as tfl

    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    from datetime import datetime
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    from sklearn.neighbors import NearestNeighbors
    return cc, datetime, np, os, plt, scipy, skimage, sklearn, subprocess, tfl


@app.cell
def _(datetime, subprocess):
    # PNG 300 dpi
    # A4 dimensions: 8.27 × 11.69 inches
    # Metadata: date, script, github branch id, og experiment source
    OUT_DIRPATH = "plotting_scripts/out"
    CONTROL_PALETTE = "#1A85FF"
    RD_PALETTE = "#D41159"
    PIXEL_SIZE = 0.3469 * 2  # Pixel size in µm
    SAMPLE_EXPERIMENT = "/camp/home/eloaklo/home/shared/eloaklo/analysed_data/OEO20260313"
    MM_UNIT = 1/25.4  # Millimeters in inches, for matplotlib

    # Get current commit hash:
    commit_hash = subprocess.run("git rev-parse --short HEAD", shell=True, capture_output=True)
    commit_hash = commit_hash.stdout.decode("utf-8")[:-1]

    METADATA_DICTIONARY = {
        "creator": "Omar El Oakley",
        "script": "wetlab_02_plots.py",
        "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_experiment": SAMPLE_EXPERIMENT,
        "current_commit_hash": commit_hash
    }

    seaborn_palette = [CONTROL_PALETTE, RD_PALETTE]
    return METADATA_DICTIONARY, MM_UNIT, OUT_DIRPATH, PIXEL_SIZE


@app.cell
def _():
    FLUORESCENCE_CLIP = 2.5
    return (FLUORESCENCE_CLIP,)


@app.cell
def _(np, os, tfl):
    SAMPLE_FOLDERPATH = "/camp/home/eloaklo/bentleyk/home/shared/eloaklo/data/OEO20260313/OEO20260313_1"

    # Get list of sites:
    site_list = os.listdir(SAMPLE_FOLDERPATH)
    site_list.sort()
    site_list = site_list[:72]

    # Get rough sample of images:
    sample_images = []
    for index, site_folder in enumerate(site_list):
        # Get image filenames:
        site_path = os.path.join(SAMPLE_FOLDERPATH, site_folder)
        image_filenames = os.listdir(site_path)
        image_filenames.sort()
        image_filenames = image_filenames[:-1]

        # Load images:
        for load_filename in image_filenames:
            sample_image = tfl.imread(os.path.join(site_path, load_filename))
            sample_images.append(sample_image)

        break

    sample_images = np.stack(sample_images, axis=0)
    return (sample_images,)


@app.cell
def _(FLUORESCENCE_CLIP, np, sample_images, sklearn):
    # Get representative pixel distribution, remove outlier pixels:
    pixel_distribution = sample_images.flatten()
    filter_threshold = np.quantile(pixel_distribution, 0.995)
    filtered_pixel_distribution = pixel_distribution[pixel_distribution < filter_threshold]

    # Fit mixture model (for fluorescent structures & background), subsampling for speed:
    print("Fitting Gaussian mixture model for pixel fluorescence distribution...", flush=True)
    gm_model = sklearn.mixture.GaussianMixture(n_components=2, random_state=0)
    gm_model.fit(np.log(filtered_pixel_distribution)[::128].reshape(-1, 1))
    fluorescence_mean = np.max(gm_model.means_)
    fluorescence_variance = np.max(gm_model.covariances_)
    fluorescence_std = np.sqrt(fluorescence_variance)
    print(f"Log fluorescent pixel mean: {fluorescence_mean}")
    print(f"Log fluorescent pixel variance: {fluorescence_variance}")
    print(f"Log fluorescent pixel standard deviation: {fluorescence_std}")

    # Take three sigmas of log fluorescence distribution:
    lower_fl = np.exp(fluorescence_mean - (FLUORESCENCE_CLIP * fluorescence_std))
    upper_fl = np.exp(fluorescence_mean + (FLUORESCENCE_CLIP * fluorescence_std))
    print(f"Fluorescent pixel lower threshold: {lower_fl}")
    print(f"Fluorescent pixel upper threshold: {upper_fl}")
    return filtered_pixel_distribution, gm_model, lower_fl, upper_fl


@app.cell
def _(
    METADATA_DICTIONARY,
    MM_UNIT,
    OUT_DIRPATH,
    datetime,
    filtered_pixel_distribution,
    gm_model,
    np,
    os,
    plt,
    scipy,
):
    def plot_pixel_distribution():
        # Set up figure:
        fig, axs = plt.subplots(2, 1, figsize=(160 * MM_UNIT, 90 * MM_UNIT), layout="constrained", sharex=True)

        # Plot data:
        pix_min = np.min(filtered_pixel_distribution[::128])
        pix_max = np.max(filtered_pixel_distribution[::128])
        logbins = np.geomspace(pix_min, pix_max, 25)
        axs[0].hist(filtered_pixel_distribution[::128], bins=logbins, density=True)

        # Format axes:
        # ax.set_xscale('log')
        axs[0].set_xlim(pix_min, pix_max)
        axs[0].set_yticks([0.00, 0.01], labels=["0.00", "0.01"])

        # Label axes:
        axs[0].set_ylabel("Histogram Density")
        axs[0].set_yticks([])

        # Get PDF values:
        pix_min = np.min(filtered_pixel_distribution[::128])
        pix_max = np.max(filtered_pixel_distribution[::128])
        log_min = np.log(pix_min)
        log_max = np.log(pix_max)
        pdf_input = np.linspace(log_min, log_max, 1000)

        # -- Get background PDF:
        bg_mean = np.min(gm_model.means_)
        bg_std = np.sqrt(np.min(gm_model.covariances_))
        bg_pdf = scipy.stats.norm.pdf(pdf_input, bg_mean, bg_std)

        # -- Get fluorescence PDF:
        fl_mean = np.max(gm_model.means_)
        fl_std = np.sqrt(np.max(gm_model.covariances_))
        fl_pdf = scipy.stats.norm.pdf(pdf_input, fl_mean, fl_std)

        # Plot PDFs:
        axs[1].plot(np.exp(pdf_input), bg_pdf, label="Background Distribution", color='k', alpha=0.5)
        axs[1].plot(np.exp(pdf_input), fl_pdf, label="Fluorescence Distribution", color='r')
        axs[1].legend()

        # # Format axes:
        axs[1].set_ylim(0, 3.45)
        axs[1].set_yticks([])

        # Label axes:
        axs[1].set_ylabel("PDF Density")
        axs[1].set_xlabel("Pixel Intensity")
    
        # Save figure:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(os.path.join(OUT_DIRPATH, "pixel_histogram.png"), dpi=300, metadata=METADATA_DICTIONARY, transparent=True)
        plt.show()

    plot_pixel_distribution()
    return


@app.cell
def _(
    METADATA_DICTIONARY,
    MM_UNIT,
    OUT_DIRPATH,
    cc,
    datetime,
    lower_fl,
    np,
    os,
    plt,
    sample_images,
    skimage,
    upper_fl,
):
    def get_registration_image(input_image):
        normalised_image = np.clip(input_image, lower_fl, upper_fl)
        normalised_image = (normalised_image - lower_fl) / (upper_fl - lower_fl)
        thresholded_image = normalised_image > 0.25
        thresholded_image = skimage.morphology.closing(thresholded_image)
        thresholded_image = skimage.morphology.opening(thresholded_image)
        binary_dilation = skimage.morphology.dilation(thresholded_image)
        registration_image = binary_dilation ^ thresholded_image
        return registration_image

    def plot_registration_thresholding():
        # Set up figure:
        fig, axs = plt.subplots(1, 2, figsize=(160 * MM_UNIT, 85 * MM_UNIT), layout="constrained")

        # Plot sample image:
        axs[0].imshow(sample_images[0], cmap='gray', vmin=lower_fl, vmax=upper_fl)

        # Plot outlined image:
        registration_image = np.zeros((1024, 1024))
        for i in range(96):
            outline_mask = get_registration_image(sample_images[i])
            registration_image[outline_mask] = i

        # Convert array to hours:
        registration_image *= (2.5) / 60
        image_object = axs[1].imshow(registration_image, cmap=cc.m_CET_L16)

        # Plot colorbar:
        fig.colorbar(
            image_object, ax=axs.flatten(),
            label="Time Elapsed (h)", ticks=[0, 2, 4],
            fraction=0.1, shrink=0.8, pad=0.025
        )

        # Format axes:
        axs[0].set_axis_off()
        axs[1].set_axis_off()

        # Save figure:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(os.path.join(OUT_DIRPATH, "registration_thresh.png"), dpi=300, metadata=METADATA_DICTIONARY, transparent=True)
        plt.show()

    plot_registration_thresholding()
    return


@app.cell
def _():
    # TODO: stage shift plots
    return


@app.cell
def _(
    METADATA_DICTIONARY,
    MM_UNIT,
    OUT_DIRPATH,
    PIXEL_SIZE,
    datetime,
    lower_fl,
    np,
    os,
    plt,
    sample_images,
    scipy,
    skimage,
    upper_fl,
):
    # Do LoG and peak finding:
    NUCLEUS_DIAMETER = 12.5  # in µm
    QUALITY_THRESHOLD = -4  # cutoff for the value of the logarithm of laplacian of gaussian filtering


    def calculate_2D_LoG(x, y, sigma):
        exponent = - ((x**2 + y**2) / (2 * (sigma**2)))
        factor = -(1 / (np.pi * sigma**2)) * (1 + exponent)
        log_kernel = factor * np.exp(exponent)
        return log_kernel


    def get_kernel(nuclei_size, pixel_size):
        # Determining sigma of kernel:
        nuclei_radius = nuclei_size / 2  # in µm
        sigma_µm = nuclei_radius / np.sqrt(2)
        sigma_pixel = sigma_µm / pixel_size

        # Calculating kernel size:
        # We want the kernel radius to be about 3 sigmas wide, to capture the majority of
        # the envelope. However, for larger radii, this is *very* slow, so we compromise with
        # 1.5 sigma.
        kernel_extent = (2 * np.ceil(sigma_pixel * 1.5)) + 1
        side_array = np.arange(-kernel_extent, kernel_extent + 1)
        xx, yy = np.meshgrid(side_array, side_array)

        # Calculating kernel:
        return calculate_2D_LoG(xx, yy, sigma_pixel)

    def filter_image(image, kernel):
        # Convolve with LoG kernel:
        pad_width = int((kernel.shape[0] - 1) / 2)
        padded_image = np.pad(image, pad_width, mode='median')
        image_filtered = scipy.signal.fftconvolve(padded_image, -kernel, mode='valid', axes=None)
        return image_filtered

    def plot_log_filter():
        fig, axs = plt.subplots(1, 2, figsize=(160 * MM_UNIT, 80 * MM_UNIT), layout="constrained")

        # Get sample normalised image:
        normalised_image = np.clip(sample_images[0], lower_fl, upper_fl)
        normalised_image = (normalised_image - lower_fl) / (upper_fl - lower_fl)

        # Apply laplacian of gaussian and plot:
        kernel = get_kernel(NUCLEUS_DIAMETER, PIXEL_SIZE)
        filtered_image = filter_image(normalised_image, kernel)
        extent = np.max([np.abs(filtered_image.min()), np.abs(filtered_image.max())])
        axs[0].imshow(filtered_image.T, cmap='gray', vmin=-extent, vmax=extent)
        axs[0].set_xlim(0, 1024)
        axs[0].set_ylim(0, 1024)
        axs[0].set_axis_off()

        # Get peaks:
        maxima = skimage.feature.peak_local_max(filtered_image)
        maxima_values = []
        for maximum in maxima:
            maxima_values.append(filtered_image[tuple(maximum)])
        maxima_values = np.array(maxima_values)

        # Plot peaks over LoG filtered image:
        valid_mask = np.log(maxima_values) > QUALITY_THRESHOLD
        axs[1].imshow(filtered_image.T, cmap='gray', vmin=-extent, vmax=extent)
        axs[1].scatter(maxima[~valid_mask, 0], maxima[~valid_mask, 1], c='k', s=0.5, label="Invalid Local Peaks")
        axs[1].scatter(maxima[valid_mask, 0], maxima[valid_mask, 1], c='r', s=0.5, label="Valid Local Peaks")
        axs[1].set_xlim(0, 1024)
        axs[1].set_ylim(0, 1024)
        axs[1].set_axis_off()
        axs[1].legend()

        # Save:
        METADATA_DICTIONARY["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(os.path.join(OUT_DIRPATH, "log_filter_peak.png"), dpi=300, metadata=METADATA_DICTIONARY, transparent=True)
        plt.show()

    plot_log_filter()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
