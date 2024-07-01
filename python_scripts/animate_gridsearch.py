import imageio

ITERATION_COUNT = 7200


def main():
    print("Reading images...")
    frames = []
    for folder_id in range(1, ITERATION_COUNT+1):
        if folder_id % 100 == 0:
            print(folder_id)
        image = imageio.v2.imread(f'./fileOutputs/{folder_id}/trajectories.png')
        frames.append(image)

    print("Collating images...")
    imageio.mimsave('./fast.gif', frames, duration=75, loop=1)


if __name__ == "__main__":
    main()
