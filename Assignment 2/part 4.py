import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from scipy.interpolate import interp2d


def translation(I, p=0, q=0):
    print("P={}".format(p))
    print("Q={}".format(q))

    img = imread(I)
    #img = cers[100:300, 100:300]
    height, width = img.shape

    x = np.linspace(0, height, height)
    y = np.linspace(0, width, width)

    ones = np.ones(width)

    A = np.array([x, y, ones]).T
    T = np.array([[1, 0, p],
                  [0, 1, q],
                  [0, 0, 1]]).T

    final_img = np.dot(A, T)

    f = interp2d(final_img.T[0, :], final_img.T[1, :],
                 img, kind='cubic', fill_value=0)
    znew = f(x, y)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(znew)
    plt.title("Transformed")
    plt.show()


if __name__ == "__main__":
    translation(I="I1.png", p=100.36, q=100.40)
