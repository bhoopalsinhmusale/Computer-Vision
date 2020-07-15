import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
import matplotlib.image as mpimg
import os


def part_1():
    img = mpimg.imread("t1.png")
    print(img.shape)
    patch = extract_patches_2d(img, patch_size=(256, 256), max_patches=1)
    print(patch.shape)
    plt.subplot(1, 1, 1)
    plt.imshow(patch[0], cmap='gray')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    os.system("clear")

    part_1()
