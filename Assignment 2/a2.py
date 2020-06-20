import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Part 1(A,B,C)


def joint_histogram(I, J, num_bins=16, minmax_range=None):

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

    print("aaaaaaaa%", np.sum(p))

    norma_jhist = p/n
    jhist = p
    return jhist, norma_jhist


# Part 2-A
def ssd(A, B):
    dif = A.ravel() - B.ravel()
    return np.dot(dif, dif)


# Part 2-B(from data mining course)
def pearson_correlation(x, y):
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
             yticks=[i for i in range(0, 25, 5)], zticks=[i for i in range(0, 22, 2)])

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(final_A.T[0, :], final_A.T[1, :], final_A.T[2, :], color="Red")
    plt.setp(ax, xticks=[i for i in range(0, 25, 5)],
             yticks=[i for i in range(0, 25, 5)], zticks=[i for i in range(0, 22, 2)])
    plt.suptitle("Rigid Transform")
    plt.show()


def affine_transform(slice=0, theta=0, omega=0, phi=0, dx=0, dy=0, dz=0):
    print("slice={}".format(slice))
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
             yticks=[i for i in range(0, 25, 5)], zticks=[i for i in range(0, 22, 2)])

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(final_A.T[0, :], final_A.T[1, :], final_A.T[2, :], color="Red")
    plt.setp(ax, xticks=[i for i in range(0, 25, 5)],
             yticks=[i for i in range(0, 25, 5)], zticks=[i for i in range(0, 22, 2)])
    plt.suptitle("Affine Transform")
    plt.show()


if __name__ == "__main__":
    # Part 2(SSD,CORR,MI)
    '''for i in range(2, 7):
        im_frame = Image.open('I{}.jpg'.format(i))
        I = np.array(im_frame)
        print(I.shape)

        im_frame = Image.open('J{}.jpg'.format(i))
        J = np.array(im_frame)
        print(J.shape)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        my_joint_hist, _ = joint_histogram(I, J, 50)
        jhist = np.histogram2d(np.ravel(I), np.ravel(J), 50)
        ax1.imshow(np.log(my_joint_hist))
        ax1.set_title("My joint hist")

        ax2.imshow(np.log(jhist[0]))
        ax2.set_title("NP joint hist")
        plt.suptitle("Part 1 \nI{}.jpg".format(i))
        ax3.imshow(I, "gray")
        ax4.imshow(J, "gray")
        fig.tight_layout()
        print(np.sum(jhist[0]))
        print(np.sum(my_joint_hist))
        plt.show()

    im_frame = Image.open('I1.png')
    I = np.array(im_frame)
    print(I.shape)

    im_frame = Image.open('J1.png')
    J = np.array(im_frame)
    print(J.shape)

    ssd = ssd(I, J)
    print("SSD={}".format(ssd))

    corr = pearson_correlation(I, J)
    print("CORR={}".format(corr))

    mi = MI(I, J)
    print("MI={}".format(mi))'''

    # Part 3-B
    rigid_transform(theta=90, omega=0,
                    phi=0, dx=0, dy=0, dz=0)

    # Part 3-C
    affine_transform(slice=-20, theta=90, omega=0,
                     phi=0, dx=0, dy=0, dz=0)
