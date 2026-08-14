import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import shutil
    import math
    import sys

    import h5py

    import numpy as np

    import matplotlib.pyplot as plt
    return h5py, math, np, os, shutil


@app.cell
def _():
    from PIL import Image
    return (Image,)


@app.cell
def _():
    import imageio.v3 as iio
    return (iio,)


@app.cell
def _(math, os):
    check_dirpath = "model_experiments/2026-05-22-matrix_shape/run_data"
    test_file = "ann_indices.npy"

    experiment_dirpath = "model_experiments/2026-05-20-collisions_shape"
    run_data_dirpath = os.path.join(experiment_dirpath, "run_data")

    SAMPLE_COUNT = 2 ** 17
    HASH_COUNT = 1000

    hash_directory_count = math.ceil(SAMPLE_COUNT / HASH_COUNT)
    hash_directory_ids = list(range(hash_directory_count))
    return (
        HASH_COUNT,
        SAMPLE_COUNT,
        check_dirpath,
        experiment_dirpath,
        hash_directory_ids,
        run_data_dirpath,
        test_file,
    )


@app.cell
def _(
    HASH_COUNT,
    SAMPLE_COUNT,
    check_dirpath,
    hash_directory_ids,
    os,
    test_file,
):
    def check_integrity():
        failed_runs = []
        for hash_id in hash_directory_ids:
            print(f"Checking hash {hash_id}...")
            for subhash_id in range(HASH_COUNT):
                # Calculate run ID, skip if out of range:
                run_id = (hash_id * HASH_COUNT) + subhash_id
                if run_id >= SAMPLE_COUNT:
                    continue

                # Copy image files from run data to image data:
                source_dirpath = os.path.join(check_dirpath, str(hash_id), str(run_id))
                if not os.path.exists(os.path.join(source_dirpath, test_file)):
                    print(f"Run {test_file} not completed...")
                    failed_runs.append(run_id)
        return failed_runs

    failed_runs = check_integrity()
    return


@app.cell
def _(np):
    load_com_trajectory = np.load("model_experiments/2026-05-20-collisions_shape/com_trajectory_images.npz")
    return (load_com_trajectory,)


@app.cell
def _(Image, load_com_trajectory):
    img = Image.fromarray(load_com_trajectory[str(3)])
    img
    return


@app.cell
def _(
    HASH_COUNT,
    SAMPLE_COUNT,
    experiment_dirpath,
    hash_directory_ids,
    iio,
    np,
    os,
    run_data_dirpath,
):
    trajectory_images = {}
    com_trajectory_images = {}
    for hash_id in hash_directory_ids:
        print(f"Processing hashed folder: {hash_id}...")
        for subhash_id in range(HASH_COUNT):
            # Calculate run ID, skip if out of range:
            run_id = (hash_id * HASH_COUNT) + subhash_id
            if run_id >= SAMPLE_COUNT:
                continue

            # Copy image files from run data to image data:
            source_dirpath = os.path.join(run_data_dirpath, str(hash_id), str(run_id))
            trajectory_images[str(run_id)] = iio.imread(os.path.join(source_dirpath, "trajectory.png"))
            com_trajectory_images[str(run_id)] = iio.imread(os.path.join(source_dirpath, "com_trajectory.png"))

    np.savez_compressed(os.path.join(experiment_dirpath, "trajectory_images.npz"), **trajectory_images)
    np.savez_compressed(os.path.join(experiment_dirpath, "com_trajectory_images.npz"), **com_trajectory_images)
    return


@app.cell
def _():
    test_imagepath = "model_experiments/2026-05-20-collisions_shape/run_data/0/0/trajectory.png"
    return (test_imagepath,)


@app.cell
def _(iio, test_imagepath):
    test_image_array = iio.imread(test_imagepath)
    return


@app.cell
def _(h5py, os):
    with h5py.File("test_file.hdf5", "w-") as root_file:
        print(root_file.keys())
        root_file.create_group("image_data")
        print(root_file.keys())

    os.remove("test_file.hdf5")
    return


@app.cell
def _():
    return


@app.cell
def _(experiment_dirpath, os):

    image_data_dirpath = os.path.join(experiment_dirpath, "image_data")

    if not os.path.exists(image_data_dirpath):
        os.mkdir(image_data_dirpath)
    return (image_data_dirpath,)


@app.cell
def _():
    return


@app.cell
def _(hash_directory_ids):
    hash_directory_ids
    return


@app.cell
def _(
    HASH_COUNT,
    SAMPLE_COUNT,
    hash_directory_ids,
    image_data_dirpath,
    os,
    run_data_dirpath,
    shutil,
):
    for hash_id in hash_directory_ids:
        print(f"Processing hashed folder: {hash_id}...")
        for subhash_id in range(HASH_COUNT):
            # Calculate run ID, skip if out of range:
            run_id = (hash_id * HASH_COUNT) + subhash_id
            if run_id >= SAMPLE_COUNT:
                continue

            # Copy image files from run data to image data:
            source_dirpath = os.path.join(run_data_dirpath, str(hash_id), str(run_id))
            target_dirpath = os.path.join(image_data_dirpath, str(hash_id), str(run_id))
            if not os.path.exists(target_dirpath):
                os.makedirs(target_dirpath)

            # Skip if completed:
            trajectory_copied = os.path.exists(os.path.join(target_dirpath, "trajectory.png"))
            com_trajectory_copied = os.path.exists(os.path.join(target_dirpath, "com_trajectory.png"))
            if trajectory_copied and com_trajectory_copied:
                continue

            # --- Copy trajectory image:
            shutil.copy2(
                os.path.join(source_dirpath, "trajectory.png"),
                os.path.join(target_dirpath, "trajectory.png")
            )
            # --- Copy center-of-mass trajectory image:
            shutil.copy2(
                os.path.join(source_dirpath, "com_trajectory.png"),
                os.path.join(target_dirpath, "com_trajectory.png"),
            )
    return


@app.cell
def _():
    import tarfile
    return (tarfile,)


@app.cell
def _(tarfile):
    with tarfile.open("model_experiments/2026-05-20-collisions_shape/image_data.tar", "r:") as tar:
        print(tar.getnames()[:10])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
