import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import scipy.stats as stats
import scipy.signal as signal
from scipy.ndimage import rotate
import os
from shutil import copyfile
import shutil
import wget
import subprocess
import matplotlib.image as mpimg

plt.rcParams.update({'font.size': 10})

no = 3


def part_2a(filePath="/home/divya/Desktop/Computer-Vision/Assignment-3/part1-data"):
    os.chdir(filePath)
    print("\n"+str(os.system("pwd")))
    fmri = nib.load("clean_bold.nii.gz")
    events = pd.read_csv("events.tsv", delimiter='\t')
    events = events.to_numpy()
    tr = fmri.header.get_zooms()[3]
    ts = np.zeros(int(tr*fmri.shape[3]))
    for i in np.arange(0, events.shape[0]):
        if events[i, 3] == 'FAMOUS' or events[i, 3] == 'UNFAMILIAR' or events[i, 3] == 'SCRAMBLED':
            ts[int(events[i, 0])] = 1

    plt.plot(ts)
    plt.xlabel('time(seconds)')
    plt.show()

    hrf = pd.read_csv(
        "/home/divya/Desktop/Computer-Vision/Assignment-3/part1-data/hrf.csv", header=None)
    hrf = hrf.to_numpy().reshape(len(hrf),)

    conved = signal.convolve(ts, hrf, mode='full')
    conved = conved[0:ts.shape[0]]
    plt.plot(ts)
    plt.plot(conved*3.2, lineWidth=3)
    plt.xlabel('time(seconds)')
    plt.show()

    conved = conved[0::2]
    img = fmri.get_fdata()

    meansub_img = img - np.expand_dims(np.mean(img, 3), 3)
    meansub_conved = conved - np.mean(conved)
    corrs = np.sum(meansub_img*meansub_conved, 3)/(np.sqrt(np.sum(meansub_img *
                                                                  meansub_img, 3))*np.sqrt(np.sum(meansub_conved*meansub_conved)))

    fig = plt.figure(figsize=(20, 20))
    plt.axis('off')
    for i in range(1, 6*8 + 1):
        fig.add_subplot(8, 6, i)
        plt.imshow(np.rot90(corrs[:, :, i+7]), vmin=-0.25, vmax=0.25)
        plt.title("Slice={}".format(i+7))
        plt.subplots_adjust(hspace=1)
    plt.show()
    return corrs


def part_2b(filePath="/home/divya/Desktop/Computer-Vision/Assignment-3/part1-data"):
    fmri = nib.load('bold.nii.gz')
    events = pd.read_csv('events.tsv', delimiter='\t')
    events = events.to_numpy()
    tr = fmri.header.get_zooms()[3]
    ts = np.zeros(int(tr*fmri.shape[3]))
    for i in np.arange(0, events.shape[0]):
        if events[i, 3] == 'FAMOUS' or events[i, 3] == 'UNFAMILIAR' or events[i, 3] == 'SCRAMBLED':
            ts[int(events[i, 0])] = 1

    plt.plot(ts)
    plt.xlabel('time(seconds)')
    plt.show()

    hrf = pd.read_csv(
        '/home/divya/Desktop/Computer-Vision/Assignment-3/part1-data/hrf.csv', header=None)
    hrf = hrf.to_numpy().reshape(len(hrf),)

    conved = signal.convolve(ts, hrf, mode='full')
    conved = conved[0:ts.shape[0]]
    plt.plot(ts)
    plt.plot(conved*3.2, lineWidth=3)
    plt.xlabel('time(seconds)')
    plt.show()

    conved = conved[0::2]
    img = fmri.get_fdata()

    meansub_img = img - np.expand_dims(np.mean(img, 3), 3)
    meansub_conved = conved - np.mean(conved)
    corrs = np.sum(meansub_img*meansub_conved, 3)/(np.sqrt(np.sum(meansub_img *
                                                                  meansub_img, 3))*np.sqrt(np.sum(meansub_conved*meansub_conved)))

    fig = plt.figure(figsize=(20, 20))
    plt.axis('off')
    for i in range(1, 6*8 + 1):
        fig.add_subplot(8, 6, i)
        plt.imshow(np.rot90(corrs[:, :, i+7]), vmin=-0.25, vmax=0.25)
        plt.title("Slice={}".format(i+7))

        plt.subplots_adjust(hspace=1)
    plt.show()
    return corrs


def dataset_download():
    for i in range(1, no):
        i = str(i).zfill(2)
        '''if os.path.exists('/home/divya/Desktop/Computer-Vision/Assignment-3/part2-data/subject{}'.format(i)):
            shutil.rmtree(
                '/home/divya/Desktop/Computer-Vision/Assignment-3/part2-data/subject{}'.format(i))'''
        if not os.path.exists('/home/divya/Desktop/Computer-Vision/Assignment-3/part2-data/subject{}'.format(i)):
            os.makedirs(
                '/home/divya/Desktop/Computer-Vision/Assignment-3/part2-data/subject{}'.format(i))
        os.chdir(
            "/home/divya/Desktop/Computer-Vision/Assignment-3/part2-data/subject{}".format(i))
        print("\n"+str(os.system("pwd")))

        copyfile("/home/divya/Desktop/Computer-Vision/Assignment-3/part1-data/pipeline.sh",
                 "pipeline.sh".format(i))
        if not os.path.exists('t1.nii.gz'.format(i)):
            T1_url = "https://openneuro.org/crn/datasets/ds000117/snapshots/1.0.3/files/sub-{}:ses-mri:anat:sub-{}_ses-mri_acq-mprage_T1w.nii.gz".format(
                i, i)
            wget.download(
                T1_url, 't1.nii.gz'.format(i))

        if not os.path.exists('bold.nii.gz'.format(i)):
            bold_url = "https://openneuro.org/crn/datasets/ds000117/snapshots/1.0.3/files/sub-{}:ses-mri:func:sub-{}_ses-mri_task-facerecognition_run-01_bold.nii.gz".format(
                i, i)
            wget.download(
                bold_url, 'bold.nii.gz'.format(i))

        if not os.path.exists('events.tsv'.format(i)):
            even_url = "https://openneuro.org/crn/datasets/ds000117/snapshots/1.0.3/files/sub-{}:ses-mri:func:sub-{}_ses-mri_task-facerecognition_run-01_events.tsv".format(
                i, i)
            wget.download(
                even_url, 'events.tsv'.format(i))

        #os.system("sudo chmod -R 777 './'".format(i))
        if not os.path.exists("clean_bold.nii.gz"):
            os.system("./pipeline.sh".format(i))
        return 0


def part_3():
    hrf = pd.read_csv(
        "/home/divya/Desktop/Computer-Vision/Assignment-3/part1-data/hrf.csv", header=None)
    hrf = hrf.to_numpy().reshape(len(hrf),)

    for i in range(1, no):
        i = str(i).zfill(2)
        os.chdir(
            "/home/divya/Desktop/Computer-Vision/Assignment-3/part2-data/subject{}".format(i))
        print("\n"+str(os.system("pwd")))
        fmri = nib.load("clean_bold.nii.gz")
        events = pd.read_csv("events.tsv", delimiter='\t')
        events = events.to_numpy()
        tr = fmri.header.get_zooms()[3]
        ts = np.zeros(int(tr*fmri.shape[3]))
        for i in np.arange(0, events.shape[0]):
            if events[i, 3] == 'FAMOUS' or events[i, 3] == 'UNFAMILIAR' or events[i, 3] == 'SCRAMBLED':
                ts[int(events[i, 0])] = 1
        conved = signal.convolve(ts, hrf, mode='full')
        conved = conved[0:ts.shape[0]]
        conved = conved[0::2]
        img = fmri.get_fdata()

        meansub_img = img - np.expand_dims(np.mean(img, 3), 3)
        meansub_conved = conved - np.mean(conved)
        corrs = np.sum(meansub_img*meansub_conved, 3)/(np.sqrt(np.sum(meansub_img *
                                                                      meansub_img, 3))*np.sqrt(np.sum(meansub_conved*meansub_conved)))

        corrs_nifti = nib.Nifti1Image(corrs, fmri.affine)
        nib.save(corrs_nifti, 'corrs.nii.gz')

        os.system(
            "flirt -in corrs.nii.gz -ref t1.nii.gz -applyxfm -init epireg.mat -out corrs_in_t1.nii.gz")

        os.system("afni -com 'OPEN_WINDOW A.axialimage'\
            -com 'OPEN_WINDOW A.sagittalimage'\
            -com 'OPEN_WINDOW A.coronalimage'\
            -com 'SET_DICOM_XYZ A 17.900 62.688 1.632'\
            -com 'SET_PBAR_ALL A.-5844 1.0 Spectrum:red_to_blue'\
            -com 'SWITCH_UNDERLAY t1.nii.gz+orig'\
            -com 'SEE_OVERLAY A.-'\
            -com 'SWITCH_OVERLAY corrs_in_t1.nii.gz+orig'\
            -com 'SEE_OVERLAY A.+'\
            -com 'SET_THRESHOLD A.01506 1'\
            -com 'SAVE_PNG A.axialimage axialimage.png blowup=5'\
            -com 'SAVE_PNG A.sagittalimage sagittalimage.png blowup=5'\
            -com 'SAVE_PNG A.coronalimage coronalimage.png blowup=5'\
            -com 'QUIT'")
        print("Subject{} is completed".format(i))
        '''epi_image = nib.load('corrs_in_t1.nii.gz')
        tof_t1_img = epi_image.get_data()

        plt.subplot(1, 1, 1)
        plt.imshow(np.flip(tof_t1_img[100, :, :]).T)
        plt.title("tof.nii")
        plt.show()'''
        return 0


def plot_corrs_in_t1():
    fig, axes = plt.subplots(16, 3)
    for i in range(1, no):
        r = i
        print("R=%", r)
        i = str(i).zfill(2)

        os.chdir(
            "/home/divya/Desktop/Computer-Vision/Assignment-3/part2-data/subject{}".format(i))
        print("\n"+str(os.system("pwd")))

        plt.subplot(16, 3, r+1)
        img = mpimg.imread("axialimage.png")
        plt.imshow(np.fliplr(img))
        plt.title("Subject{}-axialimage.png".format(i))

        plt.subplot(16, 3, r+1)
        img = mpimg.imread("sagittalimage.png")
        plt.imshow(img)
        plt.title("Subject{}-sagittalimage.png".format(i))

        plt.subplot(16, 3, r+1)
        img = mpimg.imread("coronalimage.png")
        plt.imshow(np.fliplr(img))
        plt.title("Subject{}-coronalimage.png".format(i))

    plt.tight_layout()
    plt.suptitle("Final Output")
    plt.show()
    return 0


if __name__ == "__main__":
    os.system("clear")
    # part_2a()

    # part_2b()

    # dataset_download()
    '''completed = 1
    if dataset_download() == 0:
        completed = part_3()
    if completed == 0:
        plot_corrs_in_t1()'''

    plot_corrs_in_t1()
