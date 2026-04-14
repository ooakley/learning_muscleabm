import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import tifffile

    import numpy as np
    import matplotlib.pyplot as plt
    return np, os, plt, tifffile


@app.cell
def _():
    SPARSE_FILEPATH = "/camp/home/eloaklo/home/shared/eloaklo/data/OEO20260319"
    CONFLUENT_FILEPATH = "/camp/home/eloaklo/home/shared/eloaklo/data/OEO20260325"
    return (CONFLUENT_FILEPATH,)


@app.cell
def _(CONFLUENT_FILEPATH, os):
    def get_image_paths(base_path):
        directories = os.listdir(base_path)
        directories.sort()

        ctl_imagepaths = []
        rd_imagepaths = []
        for well_directory in directories:
            print(well_directory)
            well_path = os.path.join(base_path, well_directory)
            site_directories = os.listdir(well_path)
            site_directories.sort()

            for site in site_directories:
                print(f"--- {site}")
                site_path = os.path.join(well_path, site)
                if not os.path.isdir(site_path):
                    continue
                site_files = os.listdir(site_path)
                site_files.sort()
                image_path = os.path.join(site_path, site_files[0])

                if well_directory[:4] == "G019":
                    rd_imagepaths.append(image_path)
                else:
                    ctl_imagepaths.append(image_path)

        return ctl_imagepaths, rd_imagepaths

    ctl_imagepaths, rd_imagepaths = get_image_paths(CONFLUENT_FILEPATH)
    return ctl_imagepaths, rd_imagepaths


@app.cell
def _(ctl_imagepaths, rd_imagepaths, tifffile):
    control_images = []
    for image_path in ctl_imagepaths:
        control_images.append(tifffile.imread(image_path))

    rd_images = []
    for image_path in rd_imagepaths:
        rd_images.append(tifffile.imread(image_path))
    return control_images, rd_images


@app.cell
def _(control_images, np, plt):
    def plot_image(array, ax=None):
        # Create figure if none present:
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))

        # Clip:
        upper_limit = np.quantile(array, 0.99)
        lower_limit = np.quantile(array, 0.01)

        # Generate image and format axes:
        ax.imshow(array, cmap='gray', vmin=lower_limit, vmax=upper_limit)
        ax.set_xticks([])
        ax.set_yticks([])

    plot_image(control_images[9])
    plt.show()
    return (plot_image,)


@app.cell
def _(control_images, plot_image, plt):
    def plot_grid(images, num=3):
        fig, axs = plt.subplots(num, num, figsize=(15, 15), layout="constrained")

        count = 0
        for i in range(num):
            for j in range(num):
                plot_image(images[count], axs[i, j])
                count += 1
    
        plt.show()

    plot_grid(control_images)
    return (plot_grid,)


@app.cell
def _(plot_grid, rd_images):
    plot_grid(rd_images)
    return


@app.cell
def _(well_directories):
    well_directories
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
