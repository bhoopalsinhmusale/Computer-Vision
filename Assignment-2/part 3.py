import matplotlib.pyplot as plt
import numpy as np


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
    print("ssss")
    print(final_A.T.shape)
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
    # Part 3-B
    rigid_transform(theta=90, omega=0,
                    phi=0, dx=0, dy=0, dz=0)

    # Part 3-C
    affine_transform(slice=-20, theta=90, omega=0,
                     phi=0, dx=0, dy=0, dz=0)
