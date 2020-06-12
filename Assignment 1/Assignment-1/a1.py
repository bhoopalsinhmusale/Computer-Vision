import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

slice = 200
slice_axial = slice
slice_sagittal = slice
slice_coronal = slice


def histeq_processor(img):

    nbr_bins = 256

    imhist, bins = np.histogram(img.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]

    original_shape = img.shape
    img = np.interp(img.flatten(), bins[:-1], cdf)
    img = img/255.0
    return img.reshape(original_shape)


def part_1(brain, slice_arg, view, histeq):
    global slice
    global slice_axial
    global slice_sagittal
    global slice_coronal
    slice = slice_arg
    slice_axial = slice_arg
    slice_sagittal = slice_arg
    slice_coronal = slice_arg

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(5, 5), constrained_layout=True)

    if histeq:
        # print("hist")
        if view == 'axial':

            axial_brain = histeq_processor(brain[:, :, slice_arg])
            axial_image = ax1.imshow(axial_brain.T)
            axial_image.set_data(axial_brain.T)
            ax1.set_title("Axial slice {}".format(slice_arg))
            ax2.axis('off')
            ax3.axis('off')
            ax4.axis('off')
            # fig.canvas.mpl_connect('scroll_event', onscroll)
            # fig.tight_layout()
            # plt.show()

        elif view == 'sagittal':
            # sagittal_brain = histeq_processor(brain[slice_sagittal, :, :])
            sagittal_image = ax2.imshow(histeq_processor(
                np.rot90(brain[slice_sagittal, :, :])))
            sagittal_image.set_data(histeq_processor(
                np.rot90(brain[slice_sagittal, :, :])))
            ax2.set_title("Sagittal slice {}".format(slice_sagittal))
            ax1.axis('off')
            ax3.axis('off')
            ax4.axis('off')

        elif view == 'coronal':
            coronal_image = ax3.imshow(histeq_processor(
                np.rot90(brain[:, slice_sagittal, :])))
            coronal_image.set_data(histeq_processor(
                np.rot90(brain[:, slice_coronal, :])))
            ax3.set_title("Coronal slice {}".format(slice_coronal))
            ax1.axis('off')
            ax2.axis('off')
            ax4.axis('off')
            # fig.canvas.mpl_connect('scroll_event', onscroll)
            # fig.tight_layout()
            # plt.show()
        elif view == 'all':
            axial_brain = histeq_processor(brain[:, :, slice_arg])
            axial_image = ax1.imshow(axial_brain.T)
            axial_image.set_data(axial_brain.T)
            ax1.set_title("Axial slice {}".format(slice_arg))

            sagittal_image = ax2.imshow(histeq_processor(
                np.rot90(brain[slice_sagittal, :, :])))
            sagittal_image.set_data(histeq_processor(
                np.rot90(brain[slice_sagittal, :, :])))
            ax2.set_title("Sagittal slice {}".format(slice_sagittal))

            coronal_image = ax3.imshow(histeq_processor(
                np.rot90(brain[:, slice_sagittal, :])))
            coronal_image.set_data(histeq_processor(
                np.rot90(brain[:, slice_coronal, :])))
            ax3.set_title("Coronal slice {}".format(slice_coronal))
            ax4.axis('off')

        ax4.axis('off')

    else:
        if view == 'axial':
            axial_image = ax1.imshow(brain[:, :, slice_arg].T)
            axial_image.set_data(brain[:, :, slice_arg].T)
            ax1.set_title("Axial slice {}".format(slice_arg))
            ax2.axis('off')
            ax3.axis('off')
            ax4.axis('off')
            # fig.canvas.mpl_connect('scroll_event', onscroll)
            # fig.tight_layout()
            # plt.show()

        elif view == 'sagittal':
            sagittal_image = ax2.imshow(np.rot90(brain[slice_sagittal, :, :]))
            sagittal_image.set_data(np.rot90(brain[slice_sagittal, :, :]))
            ax2.set_title("Sagittal slice {}".format(slice_sagittal))
            ax1.axis('off')
            ax3.axis('off')
            ax4.axis('off')

        elif view == 'coronal':
            coronal_image = ax3.imshow(np.rot90(brain[:, slice_sagittal, :]))
            coronal_image.set_data(np.rot90(brain[:, slice_coronal, :]))
            ax3.set_title("Coronal slice {}".format(slice_coronal))
            ax1.axis('off')
            ax2.axis('off')
            ax4.axis('off')
            # fig.canvas.mpl_connect('scroll_event', onscroll)
            # fig.tight_layout()
            # plt.show()
        elif view == 'all':
            axial_image = ax1.imshow(brain[:, :, slice_axial].T)
            axial_image.set_data(brain[:, :, slice_axial].T)
            ax1.set_title("Axial slice {}".format(slice_axial))

            sagittal_image = ax2.imshow(np.rot90(brain[slice_sagittal, :, :]))
            sagittal_image.set_data(np.rot90(brain[slice_sagittal, :, :]))
            ax2.set_title("Sagittal slice {}".format(slice_sagittal))

            coronal_image = ax3.imshow(np.rot90(brain[:, slice_sagittal, :]))
            coronal_image.set_data(np.rot90(brain[:, slice_coronal, :]))
            ax3.set_title("Coronal slice {}".format(slice_coronal))
            ax4.axis('off')

    def onscroll(event):
        global slice_axial
        global slice_sagittal
        global slice_coronal

        ax_no = None
        for i, ax in enumerate([ax1, ax2, ax3]):
            if ax == event.inaxes:
                ax_no = i+1
                # print("Click is in axes ax{}".format(ax_no))

            if ax_no == 1:
                if event.button == 'up':
                    slice_axial = (slice_axial + 5)
                    # print(slice_axial)
                else:
                    slice_axial = (slice_axial - 5)
                    # print(slice_axial)
                update_axial()
            elif ax_no == 2:
                if event.button == 'up':
                    slice_sagittal = (slice_sagittal + 5)
                    # print(slice_sagittal)
                else:
                    slice_sagittal = (slice_sagittal - 5)
                    # print(slice_sagittal)
                update_sagittal()
            elif ax_no == 3:
                if event.button == 'up':
                    slice_coronal = (slice_coronal + 5)
                    # print(slice_coronal)
                else:
                    slice_coronal = (slice_coronal - 5)
                    # print(slice_coronal)
                update_coronal()

    def onclick(event):
        global slice_axial
        global slice_sagittal
        global slice_coronal

        ax_no = None
        for i, ax in enumerate([ax1, ax2, ax3]):
            if ax == event.inaxes:
                ax_no = i+1
                # print("Click is in axes ax{}".format(ax_no))

            if ax_no == 1:
                if event.key == 'up':
                    slice_axial = (slice_axial + 5)
                    # print(slice_axial)
                else:
                    slice_axial = (slice_axial - 5)
                    # print(slice_axial)
                update_axial()
            elif ax_no == 2:
                if event.key == 'up':
                    slice_sagittal = (slice_sagittal + 5)
                    # print(slice_sagittal)
                else:
                    slice_sagittal = (slice_sagittal - 5)
                    # print(slice_sagittal)
                update_sagittal()
            elif ax_no == 3:
                if event.key == 'up':
                    slice_coronal = (slice_coronal + 5)
                    # print(slice_coronal)
                else:
                    slice_coronal = (slice_coronal - 5)
                    # print(slice_coronal)
                update_coronal()

    def update_axial():
        if histeq:
            axial_brian = histeq_processor(brain[:, :, slice_axial])
            axial_image.set_data(axial_brian.T)
            ax1.set_title("Axial slice {}".format(slice_axial))
        else:
            ax1.set_title("Axial slice {}".format(slice_axial))
            axial_image.set_data(brain[:, :, slice_axial].T)
        # ax1.set_ylabel('slice {}' .format(slice))'''
        axial_image.axes.figure.canvas.draw()

    def update_sagittal():
        if histeq:
            sagittal_image.set_data(
                histeq_processor(np.rot90(brain[slice_sagittal, :, :])))
            ax2.set_title("Sagittal slice {}".format(slice_sagittal))
        else:
            sagittal_image.set_data(np.rot90(brain[slice_sagittal, :, :]))
            ax2.set_title("Sagittal slice {}".format(slice_sagittal))

        # ax1.set_ylabel('slice {}' .format(slice))
        sagittal_image.axes.figure.canvas.draw()

    def update_coronal():
        if histeq:
            ax3.set_title("Coronal slice {}".format(slice_coronal))
            coronal_image.set_data(histeq_processor(
                np.rot90(brain[:, slice_coronal, :])))
        else:
            ax3.set_title("Coronal slice {}".format(slice_coronal))
            coronal_image.set_data(np.rot90(brain[:, slice_coronal, :]))
        # ax1.set_ylabel('slice {}' .format(slice))
        coronal_image.axes.figure.canvas.draw()

    fig.canvas.mpl_connect('scroll_event', onscroll)
    fig.canvas.mpl_connect('key_press_event', onclick)
    plt.suptitle("Part 1 with all bonus parts\nHisteq={}".format(histeq))
    plt.show()


def part_2a(brain, slice_arg):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))
    axial_image = ax1.imshow(np.rot90(brain[:, :, slice_arg]))
    axial_image.set_data(np.rot90(brain[:, :, slice_arg]))
    ax1.set_title("raw")

    f2 = np.fft.fft2(np.rot90(brain[:, :, slice_arg]))
    fshift = np.fft.fftshift(f2)
    new_img = 20 * np.log10(np.abs(fshift))

    fft2_image = ax2.imshow(new_img)
    fft2_image.set_data(new_img)
    ax2.set_title("frequency domain of raw")
    plt.suptitle("Part 2a")
    fig.tight_layout()
    plt.show()


def part_2b(brain, slice_arg):

    f2 = np.fft.fft2(brain[:, :, slice_arg].T)
    fshift = np.fft.fftshift(f2)
    rotim = 10 * np.log(np.abs(fshift))

    sz_x = rotim.shape[0]
    sz_y = rotim.shape[1]
    [X, Y] = np.mgrid[0:sz_x, 0:sz_y]
    xpr = X - int(sz_x) // 2
    ypr = Y - int(sz_y) // 2
    count = 1
    for sigma in range(1, 25, 5):
        gaussfilt = np.exp(-((xpr**2+ypr**2)/(2*sigma**2)))/(2*np.pi*sigma**2)
        plt.subplot(1, 5, count)

        ft_brain = np.fft.fftshift(np.fft.fft2(brain[:, :, slice_arg].T))

        ft_brain *= gaussfilt

        inverse_ft_brain = np.fft.ifft2(np.fft.ifftshift(ft_brain))

        plt.imshow(np.rot90(inverse_ft_brain.T).real)
        plt.title('sigma='+str(sigma))
        count = count + 1
    plt.tight_layout()
    plt.suptitle("Part 2b")

    plt.show()


def edge_detector(im):
    F1 = np.fft.fft2((im).astype(float))
    F2 = np.fft.fftshift(F1)

    (w, h) = im.shape
    half_w, half_h = int(w/2), int(h/2)

    n = 3
    F2[half_w-n:half_w+n+1, half_h-n:half_h+n+1] = 0

    # final_im = (20*np.log10(0.1 + F2)).astype(int)
    final_im = np.fft.ifft2(np.fft.ifftshift(F2)).real
    # final_im = (F2-F1).real
    return final_im


def part_2c():
    file_names = ["t1.nii", "t2.nii", "tof.nii", "swi.nii", "bold.nii"]

    for name in file_names:
        slice_arg = 250
        epi_image = nib.load(name)
        epi_data = epi_image.get_data()
        if name == "bold.nii":
            slice_arg = 35
        elif name == "tof.nii":
            slice_arg = 162
        elif name == "swi.nii":
            slice_arg = 250
        # print("NNNNNNNNName {} Slice {}".format(name, slice_arg))
        # print(epi_data.shape)

        brain = np.asarray(epi_data)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(5, 5))

        ax1.imshow(np.rot90(brain[:, :, slice_arg]))
        ax1.set_title("raw slice {}".format(slice_arg))

        f2 = np.fft.fft2(np.rot90(brain[:, :, slice_arg]))
        fshift = np.fft.fftshift(f2)
        rotim = 10 * np.log(np.abs(fshift))

        ax2.imshow(rotim)
        ax2.set_title("fft")

        img = brain[:, :, slice_arg]
        imgs_final = edge_detector(np.rot90(img))
        ax3.imshow(imgs_final)  # twilight
        ax3.set_title("edge-detect")

        sz_x = rotim.shape[0]
        sz_y = rotim.shape[1]
        [X, Y] = np.mgrid[0: sz_x, 0: sz_y]
        xpr = X - int(sz_x) // 2
        ypr = Y - int(sz_y) // 2
        sigma = 25
        gaussfilt = np.exp(-((xpr**2+ypr**2)/(2*sigma**2)))/(2*np.pi*sigma**2)
        ft_brain = np.fft.fftshift(np.fft.fft2(brain[:, :, slice_arg].T))
        ft_brain *= gaussfilt
        inverse_ft_brain = np.fft.ifft2(np.fft.ifftshift(ft_brain))
        ax4.imshow(np.rot90(inverse_ft_brain.T.real))
        ax4.set_title("blrrur")

        plt.suptitle("Part 2c\n{}".format(name))
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":

    # Part 1a with all bonus parts
    epi_image = nib.load('t1.nii')
    epi_data = epi_image.get_fdata()
    # print(epi_data.shape)
    brain = np.asarray(epi_data)
    # view parameter can be = all/axial/sagittal/coronal
    # histeq parameter can be = True/False
    part_1(brain=brain, slice_arg=250, view='all', histeq=False)

    # Part 2a
    epi_image = nib.load('t2.nii')
    epi_data = epi_image.get_fdata()
    # print(epi_data.shape)
    brain = np.asarray(epi_data)
    part_2a(brain, 250)

    # Part 2b
    epi_image = nib.load('swi.nii')
    epi_data = epi_image.get_data()
    # print(epi_data.shape)
    brain = np.asarray(epi_data)
    part_2b(brain, 250)

    # Part 2c
    part_2c()
