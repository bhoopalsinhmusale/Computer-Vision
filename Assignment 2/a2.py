from numpy.linalg import inv
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.io import imread
from scipy.interpolate import interp2d
from scipy.interpolate import interpn
plt.rcParams.update({'font.size': 20})


# Part 1(A,B,C)
def joint_histogram(I, J, num_bins=16):
    minmax_range = None
    if I.shape != J.shape:
        raise AssertionError("The inputs must be the same size.")

    I = I.reshape((I.shape[0]*I.shape[1], 1)).astype(float)
    J = J.reshape((J.shape[0]*J.shape[1], 1)).astype(float)

    if minmax_range is None:
        minmax_range = np.array([min(min(I), min(J)), max(max(I), max(J))])

    I = (I-minmax_range[0]) / (minmax_range[1]-minmax_range[0])
    J = (J-minmax_range[0]) / (minmax_range[1]-minmax_range[0])

    # and this will make them integers in the [0 (num_bins-1)] range
    I = np.round(I*(num_bins-1)).astype(int)
    J = np.round(J*(num_bins-1)).astype(int)

    n = I.shape[0]
    hist_size = np.array([num_bins, num_bins])

    # initialize the joint histogram to all zeros
    p = np.zeros(hist_size)

    for k in range(n):
        p[I[k], J[k]] = p[I[k], J[k]] + 1

    print("Verify n x p={}".format(np.sum(p)))

    norma_jhist = p/n
    jhist = p
    return jhist, norma_jhist


# Part 2-A
def ssd(A, B):
    dif = A.ravel() - B.ravel()
    return np.sum(np.square(A - B))


# Part 2-B
def pearson_correlation(x, y):
    # used from data mining course
    x = x.ravel()
    y = y.ravel()
    n = len(x)
    x_sum = float(np.sum(x))
    y_sum = float(np.sum(y))
    x_sq_sum = float(np.sum(np.square(x)))
    y_sq_sum = float(np.sum(np.square(y)))
    prd_xy_sum = float(np.sum(np.multiply(x, y)))
    top = float(prd_xy_sum - ((x_sum * y_sum) / n))
    bot = float(np.sqrt((x_sq_sum - np.square(x_sum) / n)
                        * (y_sq_sum - np.square(y_sum) / n)))
    if bot == 0:
        return 0
    else:
        correlation = top / bot
        return correlation


# Part 2-C
def MI(I, J):
    jh, _ = joint_histogram(I, J)

    pxy = jh / float(np.sum(jh))
    # print("Normalized Joint Histogram={}".format(np.sum(pxy)))

    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    return mi


# Part 3-A,B
def rigid_transform(theta=0, omega=0, phi=0, dx=0, dy=0, dz=0):
    print("theta={}".format(theta))
    print("omega={}".format(omega))
    print("phi={}".format(phi))
    print("dx={}".format(dx))
    print("dy={}".format(dy))
    print("dz={}".format(dz))
    size = 20
    a = np.linspace(0, size, size)
    b = np.linspace(0, size, size)
    c = np.linspace(0, size, size)
    x, y, z = np.meshgrid(a, b, c)

    ones = np.ones((size, size, size))

    # print(x.shape)
    # print(ones.shape)

    A = np.array([x, y, z, ones]).T

    # print(A.shape)
    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([[1, 0, 0, 0],
                   [0, np.cos(theta), -np.sin(theta), 0],
                   [0, np.sin(theta), np.cos(theta), 0],
                   [0, 0, 0, 1]])

    RY = np.array([[np.cos(omega), 0, -np.sin(omega), 0],
                   [0, 1, 0, 0],
                   [np.sin(omega), 0, np.cos(omega), 0],
                   [0, 0, 0, 1]])

    RZ = np.array([[np.cos(phi), -np.sin(phi), 0, 0],
                   [np.sin(phi), np.cos(phi), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX.T, RY.T), RZ.T)
    # Translation matrix
    T = np.array([[1, 0, 0, dx],
                  [0, 1, 0, dy],
                  [0, 0, 1, dz],
                  [0, 0, 0, 1]]).T

    final_A = np.dot(A, np.dot(R, T))

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(A.T[0, :], A.T[1, :], A.T[2, :], color="black")
    plt.setp(ax, xticks=[i for i in range(0, 25, 5)],
             yticks=[i for i in range(0, 25, 5)], zticks=[i for i in range(0, 22, 5)])
    ax.set_title("Original 3D grid")

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(final_A.T[0, :], final_A.T[1, :], final_A.T[2, :], color="Red")
    # plt.setp(ax, xticks=[i for i in range(0, 25, 5)],
    #         yticks=[i for i in range(0, 25, 5)], zticks=[i for i in range(0, 22, 2)])
    ax.set_title("Transformed 3D grid")
    plt.suptitle("Part 3-A,B Rigid Transform")
    plt.show()


# Part 3-C
def affine_transform(slice=0, theta=0, omega=0, phi=0, dx=0, dy=0, dz=0):
    '''print("slice={}".format(slice))
    print("theta={}".format(theta))
    print("omega={}".format(omega))
    print("phi={}".format(phi))
    print("dx={}".format(dx))
    print("dy={}".format(dy))
    print("dz={}".format(dz))'''
    size = 20
    a = np.linspace(0, size, size)
    b = np.linspace(0, size, size)
    c = np.linspace(0, size, size)
    x, y, z = np.meshgrid(a, b, c)

    ones = np.ones((size, size, size))

    # print(x.shape)
    # print(ones.shape)

    A = np.array([x, y, z, ones]).T

    # print(A.shape)
    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([[1, 0, 0, 0],
                   [0, np.cos(theta), -np.sin(theta), 0],
                   [0, np.sin(theta), np.cos(theta), 0],
                   [0, 0, 0, 1]])

    RY = np.array([[np.cos(omega), 0, -np.sin(omega), 0],
                   [0, 1, 0, 0],
                   [np.sin(omega), 0, np.cos(omega), 0],
                   [0, 0, 0, 1]])

    RZ = np.array([[np.cos(phi), -np.sin(phi), 0, 0],
                   [np.sin(phi), np.cos(phi), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX.T, RY.T), RZ.T)
    # Translation matrix
    T = np.array([[1, 0, 0, dx],
                  [0, 1, 0, dy],
                  [0, 0, 1, dz],
                  [0, 0, 0, 1]]).T
    # Scale matrix
    S = np.array([[slice, 0, 0, 0],
                  [0, slice, 0, 0],
                  [0, 0, slice, 0],
                  [0, 0, 0, 1]]).T

    final_A = np.dot(A, np.dot(S, np.dot(R, T)))

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(A.T[0, :], A.T[1, :], A.T[2, :], color="black")
    plt.setp(ax, xticks=[i for i in range(0, 25, 5)],
             yticks=[i for i in range(0, 25, 5)], zticks=[i for i in range(0, 22, 5)])
    ax.set_title("Original 3D grid")
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(final_A.T[0, :], final_A.T[1, :], final_A.T[2, :], color="Red")

    # plt.setp(ax, xticks=[i for i in range(0, 25, 5)],
    #         yticks=[i for i in range(0, 25, 5)], zticks=[i for i in range(0, 22, 2)])
    ax.set_title("Transformed 3D grid")
    plt.suptitle("Part 3-A,C Affine Transform")
    # plt.show()
    M1 = np.array([[0.9045, -0.3847, -0.1840, 10.0000],
                   [0.2939, 0.8750, -0.3847, 10.0000],
                   [0.3090, 0.2939, 0.9045, 10.0000],
                   [0, 0, 0, 1.0000]])

    M2 = np.array([[-0.0000, -0.2598, 0.1500, -3.0000],
                   [0.0000, -0.1500, -0.2598, 1.5000],
                   [0.3000, -0.0000, 0.0000, 0],
                   [0, 0, 0, 1.0000]])

    M3 = np.array([[0.7182, -1.3727, -0.5660, 1.8115],
                   [-1.9236, -4.6556, -2.5512, 0.2873],
                   [-0.6426, -1.7985, -1.6285, 0.7404],
                   [0, 0, 0, 1.0000]])

    # print(np.dot(inv(M1).T, T.T).T)


# Part 4-A
def translation(I, p=0, q=0):
    # print("P={}".format(p))
    # print("Q={}".format(q))

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
    # print("asssssss")
    # print(final_img.T[0, :].shape)
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


# Part 4-B
def regis_trans(i1, i2):
    ssds = []
    p = q = 0
    i1 = imread(i1)
    stepsize = 0.000005
    n_iter = 100

    for _ in range(n_iter):
        translated_i2 = translation(i2, p, q)
        sub = translated_i2 - i1
        ssds.append(ssd(i1, translated_i2))

        # compute gradient
        x_grad, y_grad = np.gradient(translated_i2)
        ssd_p = 2 * np.sum(np.multiply(sub, x_grad))
        ssd_q = 2 * np.sum(np.multiply(sub, y_grad))

        p -= stepsize * ssd_p
        q -= stepsize * ssd_q

    return ssds


# Part 4-C
def rotation(img, theta=45):
    #img = imread(I)
    # img = img[100:300, 100:300]
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


# Part 4-D
def ssd_rotat(i1, i2):

    i1 = imread(i1)
    i2 = imread(i2)
    ssds = []
    theta = 0
    stepsize = 0.000005
    n_iter = 100
    for _ in range(n_iter):
        # move i2 and save SSD
        rotated_i2 = rotation(i2, theta)
        sub = rotated_i2 - i1
        ssds.append(ssd(i1, rotated_i2))

        # compute gradient
        x_grad, y_grad = np.gradient(rotated_i2)

        # print(i2)
        # print(i2.shape)
        x, y = np.mgrid[0:i2.shape[0], 0:i2.shape[1]]
        angle = np.deg2rad(theta)
        s, c = np.sin(angle), np.cos(angle)
        ssd_theta = 2 * np.sum(np.multiply(sub,
                                           np.multiply(x_grad, -(x * s + y * c)) +
                                           np.multiply(y_grad, x * c - y * s)))
        # print("ssd_theta: ", ssd_theta)
        # print("theta: ", theta)

        # update parameters
        theta -= stepsize * ssd_theta * 0.01

    return ssds


# Part 4-E
def rigid_transform(i, p, q, theta):
    """Return a new image corresponding to image i both translated and rotated"""
    len_x, len_y = i.shape
    x, y = np.mgrid[0:len_x, 0:len_y]

    t = np.array([[1, 0, p],
                  [0, 1, q],
                  [0, 0, 1]])

    angle = np.deg2rad(theta)
    s, c = np.sin(angle), np.cos(angle)
    r = np.array([[c, s, 0],
                  [-s, c, 0],
                  [0, 0, 1]])

    T = t @ r

    axes = np.vstack([np.ravel(x), np.ravel(y), np.ones(len_x * len_y)])
    data_points = axes[:-1].T
    new_axes = np.apply_along_axis(lambda col: T @ col.T, 0, axes)[:-1]
    new_x, new_y = new_axes[0].reshape(
        len_x, len_y), new_axes[1].reshape(len_x, len_y)
    new_i = interpolate.griddata(data_points, np.ravel(
        i), (new_x, new_y), method='cubic', fill_value=0)

    return new_i


def part_1():
    # Part 1(A,B,C)
    img1 = imread("I1.png")
    # I = np.array(img1)
    print(img1.shape)

    img2 = imread("J1.png")
    # J = np.array(im_frame)
    print(img2.shape)

    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.title("img 1")
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.imshow(img2)
    plt.title("img 2")
    plt.colorbar()

    myjh, _ = joint_histogram(img1, img2)
    plt.subplot(2, 2, 3)
    plt.imshow(np.log(myjh))
    plt.title("Joint Histogram")
    plt.suptitle("Part-1\n nxp={}".format(np.sum(myjh)))
    plt.tight_layout()
    plt.show()

    # Following code is for testing with different pairs of images
    '''for i in range(2, 7):
        img1 = imread('I{}.jpg'.format(i))
        # img1 = imread("I1.png")
        # I = np.array(img1)
        print('I{}.jpg'.format(i))
        print(img1.shape)

        img2 = imread('J{}.jpg'.format(i))
        # J = np.array(im_frame)
        print('J{}.jpg'.format(i))
        print(img2.shape)

        plt.subplot(2, 2, 1)
        plt.imshow(img1)
        plt.title("img 1")
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.imshow(img2)
        plt.title("img 2")
        plt.colorbar()

        myjh, _ = joint_histogram(img1, img2)
        plt.subplot(2, 2, 3)
        plt.imshow(np.log(myjh))
        plt.title("Joint Histogram")
        plt.suptitle("Part-1\n nxp={}".format(np.sum(myjh)))
        plt.tight_layout()
        plt.show()'''


def part_2():
    # Part 2-A,B,C,D
    img1 = imread("I1.png")
    img2 = imread("J1.png")
    ssd = ssd(img1, img2)
    corr = pearson_correlation(img1, img2)
    mi = MI(img1, img2)
    print("SSD={} | CORR={} | MI={}".format(ssd, corr, mi))


def part_3():
    # Part 3-A, B
    rigid_transform(theta=90, omega=0,
                    phi=0, dx=0, dy=0, dz=0)

    # Part 3-C
    affine_transform(slice=-20, theta=90, omega=0,
                     phi=0, dx=0, dy=0, dz=0)


def part_4():
    # Part 4-A
    img = imread("I1.png")
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")

    plt.subplot(1, 2, 2)
    znew = translation("I1.png", p=110, q=110)
    plt.imshow(znew)
    plt.title("Transformed")
    plt.suptitle("Part-4 A\nTranslation")
    plt.tight_layout()
    plt.show()

    # Part 4 B,C,D
    images = ["BrainMRI_1.jpg", "BrainMRI_2.jpg",
              "BrainMRI_3.jpg", "BrainMRI_4.jpg"]

    # Part 4-B
    ssd_pts = []
    for i in range(1, len(images)):

        ssd_i = regis_trans(images[0], images[i])
        ssd_pts.append(ssd_i)

    plt.subplot(1, 2, 1)
    for i, ssd_i in enumerate(ssd_pts):
        plt.plot(ssd_i, label="BrainMRI_{}".format(i+2))

    plt.suptitle('Part 4-B\nminimizing SSD (translation)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Part 4-C
    '''img = imread("I1.png")
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")

    znew = rotation("I1.png")
    plt.subplot(1, 2, 2)
    plt.imshow(znew)
    plt.title("Transformed")
    plt.suptitle("Part-4 C\nRotation")
    plt.tight_layout()
    plt.show()'''

    # Part 4-D
    '''ssd_pts = []
    for i in range(1, len(images)):
        ssd_i = ssd_rotat(
            images[0], images[i])
        ssd_pts.append(ssd_i)

    plt.subplot(1, 2, 1)
    for i, ssd_i in enumerate(ssd_pts):
        plt.plot(ssd_i, label="BrainMRI_{}".format(i+2))

    plt.suptitle('Part 4-D\nminimizing SSD (rotations)')
    plt.legend()'
    plt.show()'''

    # Part 4-E


if __name__ == "__main__":

    # part_1()

    # part_2()

    # part_3()

    part_4()

    """Describe the SSD curve, is it strictly decreasing, and if not, why?
       discuss the quality of your registration
       stepsize or local optimum
    """
