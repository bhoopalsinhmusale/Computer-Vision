import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


def joint_histogram(A, B, binA, binB):

    A = A.ravel()
    B = B.ravel()
    A2 = A.copy()
    B2 = B.copy()

    # assign bins
    for i in range(1, len(binA)):
        Ai = np.where(np.bitwise_and(A > binA[i-1], A <= binA[i]))
        A2[Ai] = i-1
    for i in range(1, len(binB)):
        Bi = np.where(np.bitwise_and(B > binB[i-1], B <= binB[i]))
        B2[Bi] = i-1
    JH = np.zeros((len(binA)-1, len(binB)-1))
    # calculate joint histogram
    for i in range(len(A)):
        JH[A2[i], B2[i]] += 1
    # calculate histogram for A
    HA = np.zeros(len(binA)-1)
    for i in range(len(A)):
        HA[A2[i]] += 1
    # calculate histogram for B
    HB = np.zeros(len(binB)-1)
    for i in range(len(B)):
        HB[B2[i]] += 1
    return JH, HA, HB


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
    # p = p/n

    return p


def ssd(A, B):
    dif = A.ravel() - B.ravel()
    return np.dot(dif, dif)


if __name__ == "__main__":
    '''for i in range(2, 7):
        im_frame = Image.open('I{}.jpg'.format(i))
        I = np.array(im_frame)
        print(I.shape)

        im_frame = Image.open('J{}.jpg'.format(i))
        J = np.array(im_frame)
        print(J.shape)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        my_joint_hist = joint_histogram(I, J, 50)
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
        plt.show()'''
    im_frame = Image.open('I1.png')
    I = np.array(im_frame)
    print(I.shape)

    im_frame = Image.open('J1.png')
    J = np.array(im_frame)
    print(J.shape)

    kal = ssd(I, J)
    print(kal)
    print(kal)
