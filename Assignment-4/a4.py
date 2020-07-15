import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
import matplotlib.image as mpimg
import os
import matplotlib.patches as patches


def part_1():
    images = ["t1.png",  "t1_v2.png",  "t1_v3.png", " t2.png"]
    img = mpimg.imread(images[0])
    fig, ax = plt.subplots(1)
    ax.imshow(img, cmap='gray')
    ax.set_title("{}".format(images[0]))

    rect = patches.Rectangle(
        (10, 90), 25, 25, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.show()


if __name__ == "__main__":
    os.system("clear")

    part_1()
