import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from scipy.interpolate import interp2d
from scipy.interpolate import interpn
plt.rcParams.update({'font.size': 20})

# Part 2-A


def translation(I, p=0, q=0):
    print("P={}".format(p))
    print("Q={}".format(q))

    img = imread(I)
    # img = cers[100:300, 100:300]
    height, width = img.shape

    x = np.linspace(0, height, height)
    y = np.linspace(0, width, width)

    ones = np.ones(width)

    A = np.array([x, y, ones]).T
    T = np.array([[1, 0, p],
                  [0, 1, q],
                  [0, 0, 1]]).T

    final_img = np.dot(A, T)
    print("asssssss")
    print(final_img.T[0, :].shape)
    f = interp2d(final_img.T[0, :], final_img.T[1, :],
                 img, kind='cubic', fill_value=0)
    znew = f(x, y)
    '''plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(znew)
    plt.title("Transformed")
    plt.show()
    ssd_ = ssd(img, znew)
    print(ssd_)'''
    return znew


# Part 2-C
def rotation(I, theta=45):
    img = imread(I)
    #img = img[100:300, 100:300]
    height, width = img.shape

    xs = np.arange(img.shape[0])
    ys = np.arange(img.shape[1])

    # matrix
    theta = np.deg2rad(theta)
    c = np.cos(theta)
    s = np.sin(theta)
    m = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    out = np.array([[m.dot(np.array([x, y, 1])) for y in ys]
                    for x in xs])[:, :, :-1]

    znew = interpn((xs, ys), img, out, method='nearest',
                   bounds_error=False, fill_value=0)

    '''plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(znew)
    plt.title("Transformed")
    plt.show()'''

    # ssd_ = ssd(img, znew)
    # print(ssd_)
    # return znew'''
    return znew


def part_4b(img):
    '''height, width = img.shape
    # creating the u and v vector
    u = v = np.nan*np.ones(height)

    # Calculating the u and v arrays for the good features obtained n the previous step.
    for l in height:
        j, i = l.ravel()
        # calculating the derivatives for the neighbouring pixels
        # since we are using  a 3*3 window, we have 9 elements for each derivative.
        IX =
        IX = ([Ix[i-1, j-1], Ix[i, j-1], Ix[i-1, j-1], Ix[i-1, j], Ix[i, j], Ix[i+1, j],
               Ix[i-1, j+1], Ix[i, j+1], Ix[i+1, j-1]])  # The x-component of the gradient vector
        IY = ([Iy[i-1, j-1], Iy[i, j-1], Iy[i-1, j-1], Iy[i-1, j], Iy[i, j], Iy[i+1, j],
               Iy[i-1, j+1], Iy[i, j+1], Iy[i+1, j-1]])  # The Y-component of the gradient vector
        IT = ([It[i-1, j-1], It[i, j-1], It[i-1, j-1], It[i-1, j], It[i, j], It[i+1, j],
               It[i-1, j+1], It[i, j+1], It[i+1, j-1]])  # The XY-component of the gradient vector

        # Using the minimum least squares solution approach
        LK = (IX, IY)
        LK = np.matrix(LK)
        LK_T = np.array(np.matrix(LK))  # transpose of A
        LK = np.array(np.matrix.transpose(LK))

        A1 = np.dot(LK_T, LK)  # Psedudo Inverse
        A2 = np.linalg.pinv(A1)
        A3 = np.dot(A2, LK_T)

        # we have the vectors with minimized square error
        (u[i, j], v[i, j]) = np.dot(A3, IT)
        print()'''


if __name__ == "__main__":
    # translation("I1.png", p=110, q=110)


    #znew = rotation("I1.png")
    znew = translation("I1.png", p=110, q=110)
    plt.imshow(znew)
    plt.title("Transformed")
    plt.show()
    rotation("I1.png", 10)
