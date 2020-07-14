from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
import os


def part_1():
    # Use the array data from the first image in this dataset:
    one_image = load_sample_image(
        "/home/divya/Desktop/Computer-Vision/Assignment-4/t1.png")
    print('Image shape: {}'.format(one_image.shape))

    patches = image.extract_patches_2d(one_image, (2, 2))
    print('Patches shape: {}'.format(patches.shape))

    # Here are just two of these patches:
    print(patches[1])


if __name__ == "__main__":
    part_1()
