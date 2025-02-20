from dipy.denoise.noise_estimate import estimate_sigma
from dipy.denoise.nlmeans import nlmeans
import nipype.interfaces.fsl as fsl
import itk
import SimpleITK as sitk
import cv2
import nibabel as nib
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({'font.size': 10})


def my_bilateral_filter(src, window_size=5, sigma_d=5, sigma_r=50):
    '''
    src : input noisy image
    window_size : window size
    sigma_d : smoothing weight factor
    sigma_r : range weight factor
    retruns : filtered output image
    '''
    height = src.shape[0]
    width = src.shape[1]
    filtered_image = np.empty([height, width])
    window_boundary = int(np.ceil(window_size/2))
    for i in range(height):
        for j in range(width):

            normalization_counter = 0
            filtered_pixel = 0

            for k in range(i - window_boundary, i + window_boundary):
                for l in range(j - window_boundary, j + window_boundary):
                    if (k >= 0 and k < height and l >= 0 and l < width):
                        smoothing_weight_dist = math.sqrt(
                            np.power((i - k), 2) + np.power((j - l), 2))
                        smoothing_weight = math.exp(
                            -smoothing_weight_dist/(2 * (sigma_d ** 2)))

                        range_weight_dist = (
                            abs(int(src[i][j]) - int(src[k][l]))) ** 2
                        range_weight = math.exp(-range_weight_dist /
                                                (2 * (sigma_r ** 2)))

                        bilateral_weight = smoothing_weight * range_weight

                        neighbor_pixel = src[k, l]

                        filtered_pixel += neighbor_pixel * bilateral_weight

                        normalization_counter += bilateral_weight

            filtered_pixel = filtered_pixel / normalization_counter

            filtered_image[i][j] = int(round(filtered_pixel))

    return filtered_image


def part_1():
    if not os.path.exists('part-1-output-images'):
        os.makedirs('part-1-output-images')
    images = ["t1.png",  "t1_v2.png",  "t1_v3.png", "t2.png", "flair.png"]
    for i in images:
        img = cv2.imread(i, 0)
        print(img.shape)
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)
              ) = plt.subplots(2, 3, figsize=(8, 8))
        im = ax1.imshow(img, cmap='gray')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        x = 10
        y = 10
        width, height = 40, 40
        noise_patch = patches.Rectangle(
            (x, y), width, height, linewidth=1, edgecolor='r', facecolor='r', alpha=0.6)
        ax1.add_patch(noise_patch)
        noise_patch = np.copy(img[y:y+height, x:x+width])

        x = 90
        y = 80
        signal_patch = patches.Rectangle(
            (x, y), width, height, linewidth=1, edgecolor='g', facecolor='g', alpha=0.6)
        ax1.add_patch(signal_patch)
        signal_patch = np.copy(img[y:y+height, x:x+width])

        snr = np.mean(signal_patch) / np.std(noise_patch)

        ax1.set_title("{} SNR={:.2f}".format(i, snr))

        im = ax2.imshow(signal_patch, cmap='gray')
        ax2.set_title("Signal patch, mean={:.2f}".format(np.std(signal_patch)))
        ax2.set_xticks([])
        ax2.set_yticks([])
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax2.spines[axis].set_linewidth(4)
            ax2.spines[axis].set_color('g')

        im = ax3.imshow(noise_patch, cmap='gray')
        ax3.set_title("Noise patch, std={:.2f}".format(np.std(noise_patch)))
        ax3.set_xticks([])
        ax3.set_yticks([])
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax3.spines[axis].set_linewidth(4)
            ax3.spines[axis].set_color('r')

        ax4.imshow(img, cmap='gray')
        ax4.set_title("Noisy g")

        filtered_image = my_bilateral_filter(img, 5, 12.0, 16.0)
        ax5.imshow(img-filtered_image, cmap='gray')
        ax5.set_title("Noise n")

        ax6.imshow(filtered_image, cmap='gray')
        ax6.set_title("my_bilateral f=g-n")

        plt.savefig(
            "part-1-output-images/Output-{}".format(i))

        plt.suptitle("Part-1 : Denoising")
        plt.show()


def my_otsu(src):
    '''
    src : input image
    returns segmented output image
    '''
    pixel_number = src.shape[0] * src.shape[1]
    mean_weigth = 1.0/pixel_number
    his, bins = np.histogram(src, np.arange(0, 257))
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    for t in bins[1:-1]:
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weigth
        Wf = pcf * mean_weigth

        mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf)
        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value
    output_img = src.copy()
    print("Final Threshold={}".format(final_thresh))
    output_img[src > final_thresh] = 255
    output_img[src < final_thresh] = 0
    return output_img


def part_2():
    if not os.path.exists('part-2-output-images'):
        os.makedirs('part-2-output-images')
    images = ["t1.png",  "t1_v2.png",  "t1_v3.png", "t2.png", "flair.png"]
    for i in images:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
        img = cv2.imread(i, 0)
        output_img = my_otsu(img)
        # plt.subplot(121)
        ax1.imshow(img, cmap='gray')
        ax1.set_title(i)

        # plt.subplot(122)
        # plt.imshow(img, cmap='jet', interpolation='none', alpha=0.7)
        ax2.imshow(output_img, cmap='gray')
        ax2.set_title("Otsu’s method Output")
        plt.savefig(
            "part-2-output-images/Output-{}".format(i))
        plt.suptitle("Part-2 : Segmentation")
        plt.show()


def vessel_segmentation(p, name):
    ip_img = itk.imread(p, itk.F)
    Img_type = type(ip_img)
    size = ip_img.GetImageDimension()
    # Hessian Pixel Type
    Hessian_PType = itk.SymmetricSecondRankTensor[itk.D, size]
    # Hessian Image Type
    Hessian_IType = itk.Image[Hessian_PType, size]
    Filter = itk.HessianToObjectnessMeasureImageFilter[Hessian_IType, Img_type].New(
    )
    Filter.SetBrightObject(True)
    Filter.SetScaleObjectnessMeasure(True)
    # Initial the threshoulds
    Filter.SetAlpha(0.5)
    Filter.SetBeta(1.0)
    Filter.SetGamma(5.0)
    # Filter.Set(abs(HIT))
    # multi  nlm filter
    multi_NLM_Filter = itk.MultiScaleHessianBasedMeasureImageFilter[Img_type, Hessian_IType, Img_type].New(
    )
    multi_NLM_Filter.SetInput(ip_img)
    multi_NLM_Filter.SetHessianToMeasureFilter(Filter)
    multi_NLM_Filter.SetSigmaStepMethodToLogarithmic()
    val_min = 0.2
    multi_NLM_Filter.SetSigmaMinimum(val_min)
    val_max = 3.
    multi_NLM_Filter.SetSigmaMaximum(val_max)
    sStps = 8
    multi_NLM_Filter.SetNumberOfSigmaSteps(sStps)

    OP_Pixel = itk.UC
    OP_Image = itk.Image[OP_Pixel, size]

    multi_NLM_Filter2 = itk.RescaleIntensityImageFilter[Img_type, OP_Image].New(
    )
    multi_NLM_Filter2.SetInput(multi_NLM_Filter)
    # Applying threshould filter
    filter = itk.BinaryThresholdImageFilter[OP_Image, OP_Image].New()
    filter.SetInput(multi_NLM_Filter2.GetOutput())
    segment_threshold = 40
    filter.SetLowerThreshold(segment_threshold)
    filter.SetUpperThreshold(255)
    filter.SetOutsideValue(0)
    filter.SetInsideValue(255)

    output_image = 'final_' + str(name)+'.mha'
    itk.imwrite(filter.GetOutput(), output_image)


def part_3():

    slice = 243
    os.system("sudo chmod -R 777 './part-3-data/'")

    os.chdir("part-3-data/")
    print(os.system("pwd"))

    # Skull Stripping of tof.nii
    if not os.path.exists('ss_tof.nii'):
        btr = fsl.BET()
        btr.inputs.in_file = 'tof.nii'
        btr.inputs.frac = 0.03
        btr.inputs.vertical_gradient = 0.7
        btr.inputs.out_file = 'ss_tof.nii'
        res = btr.run()

    # Skull Stripping of swi.nii
    if not os.path.exists('ss_swi.nii.gz'):
        btr = fsl.BET()
        btr.inputs.in_file = 'swi.nii'
        btr.inputs.frac = 0.04
        btr.inputs.out_file = 'ss_swi.nii'
        res = btr.run()

    epi_image = nib.load('tof.nii')
    img = epi_image.get_data()
    plt.subplot(1, 2, 1)
    plt.imshow(img[slice, :, :].T, cmap='gray')
    plt.title("Orginal tof.nii")

    epi_image = nib.load('ss_tof.nii.gz')
    img = epi_image.get_data()
    plt.subplot(1, 2, 2)
    plt.imshow(img[slice, :, :].T,  cmap='gray')
    plt.title("Skull Stripped")

    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()

    slice = 250
    epi_image = nib.load('swi.nii')
    img = epi_image.get_data()
    plt.subplot(1, 2, 1)
    plt.imshow(np.flip(img[slice, :, :].T), cmap='gray')
    plt.title("Orginal swi.nii")

    epi_image = nib.load('ss_swi.nii.gz')
    img = epi_image.get_data()
    plt.subplot(1, 2, 2)
    plt.imshow(np.flip(img[slice, :, :].T),  cmap='gray')
    plt.title("Skull Stripped")

    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()

    # Denoising of tof.nii using dipy
    epi_image = nib.load('ss_tof.nii.gz')
    ss_tof_img = epi_image.get_data()
    sigma = estimate_sigma(ss_tof_img, N=16)
    clear_ss_tof_img = nlmeans(ss_tof_img, sigma)
    nifti_img = nib.Nifti1Image(clear_ss_tof_img, epi_image.affine)
    nib.save(nifti_img, "clear_ss_tof_img.nii.gz")

    # Denoising of swi.nii using dipy
    epi_image = nib.load('ss_swi.nii.gz')
    ss_swi_img = epi_image.get_data()
    sigma = estimate_sigma(ss_swi_img, N=16)
    clear_ss_swi_img = nlmeans(ss_swi_img, sigma)
    nifti_img = nib.Nifti1Image(clear_ss_swi_img, epi_image.affine)
    nib.save(nifti_img, "clear_ss_swi_img.nii.gz")

    # Vessel Segmentation of tof using itk and sitk
    vessel_segmentation('clear_ss_tof_img.nii.gz', 'tof')
    vessel_segmentation('clear_ss_swi_img.nii.gz', 'swi')

    os.chdir("..")
    print(os.system("pwd"))


if __name__ == "__main__":
    os.system("clear")

    part_1()

    part_2()

    part_3()
