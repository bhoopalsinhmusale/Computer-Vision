
import nibabel as nib
import numpy as np
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import SimpleITK as sitk
import itk
from skimage.restoration import (denoise_bilateral, denoise_nl_means)
import math


def vSeg(p):

    # Outpot file
    output_image = 't2_nl_ostu.mha'
    input_image = itk.imread(p, itk.F)
    # initials
    i = 0
    round = 0.5
    s_minimum = 0.2
    s_maximum = 3.
    sigmaSteps = 8
    threshold = 40
    # threshold2=50
    type = type(input_image)
    size = input_image.GetImageDimension()
    # Hessian Pixel Type
    HType = itk.SymmetricSecondRankTensor[itk.D, size]
    # Hessian Image Type
    HIT = itk.Image[HType, size]
    NLM_Filter = itk.HessianToObjectnessMeasureImageFilter[HIT, type].New(
    )
    NLM_Filter.SetBrightObject(True)
    NLM_Filter.SetScaleObjectnessMeasure(True)
    # Initial the threshoulds
    NLM_Filter.SetAlpha(0.5)
    NLM_Filter.SetBeta(1.0)
    NLM_Filter.SetGamma(5.0)
    # NLM_Filter.Set(abs(HIT))
    # multi  nlm filter
    multiFilter = itk.MultiScaleHessianBasedMeasureImageFilter[type, HIT, type].New(
    )
    multiFilter.SetInput(input_image)
    multiFilter.SetHessianToMeasureFilter(NLM_Filter)
    multiFilter.SetSigmaStepMethodToLogarithmic()
    multiFilter.SetSigmaMinimum(s_minimum)
    multiFilter.SetSigmaMaximum(s_maximum)
    multiFilter.SetNumberOfSigmaSteps(sigmaSteps)

    OPixelType = itk.UC
    OImageType = itk.Image[OPixelType, size]

    multiFilter2 = itk.RescaleIntensityImageFilter[type, OImageType].New()
    multiFilter2.SetInput(multiFilter)
    # Applying threshould filter
    filter = itk.BinaryThresholdImageFilter[OImageType, OImageType].New()
    filter.SetInput(multiFilter2.GetOutput())
    filter.SetLowerThreshold(threshold)
    filter.SetUpperThreshold(255)
    filter.SetOutsideValue(0)
    filter.SetInsideValue(255)
    # filter.SetUpperThreshold(200)
    # filter.SetOutsideValue(50)
    # filter.SetInsideValue(200)

    itk.imwrite(filter.GetOutput(), output_image)

# Regenerate image based on threshold


def regenerate_img(img, ts):
    row, col = img.shape
    y = np.zeros((row, col))
    for i in range(0, row):
        for j in range(0, col):
            if img[i, j] >= ts:
                y[i, j] = 255
            else:
                y[i, j] = 0
    return y


if __name__ == '__main__':
    img = nib.load('part-3-data/tof.nii')
    img1 = nib.load('part-3-data/swi.nii')

    if True:
        imgData = img.get_fdata()
        img1Data = img1.get_fdata()
        img1Affine = img1.affine
        h = img.header
        nbImg = h.get_data_shape()
        nbImg_h = nbImg[2]
        i = 1
    for sl in range(0, nbImg_h):
        slice_h = imgData[:, :, sl]
        sx = denoise_nl_means(slice_h, patch_size=7, patch_distance=5,
                              h=0.1, multichannel=False, fast_mode=True)
        sy = denoise_nl_means(slice_h, patch_size=7, patch_distance=5,
                              h=0.1, multichannel=False, fast_mode=True)

        nl_h = np.hypot(sx, sy)

        img1Data[:, :, sl] = nl_h
        i += 1

    img1 = nib.Nifti1Image(img1Data, affine=img1Affine, header=h)
    nib.save(img1, 't2_nl_mean')
    vSeg('t2_nl_mean.nii')
