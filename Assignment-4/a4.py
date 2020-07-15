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

    x = 0
    y = 50
    height, width = 40, 50
    noise_patch = patches.Rectangle(
        (x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(noise_patch)
    noise_patch = np.copy(img[y:y+height, x:x+width])

    x = 90
    y = 80
    signal_patch = patches.Rectangle(
        (x, y), width, height, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(signal_patch)
    signal_patch = np.copy(img[y:y+height, x:x+width])

    snr = np.mean(signal_patch) / np.std(noise_patch)
    ax.set_title("SNR=%.2f" % snr)

    plt.show()

    '''plt.subplot(1, 2, 1)
    plt.imshow(noise_patch, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(signal_patch, cmap='gray')
    plt.show()'''


if __name__ == "__main__":
    os.system("cls")

    part_1()
