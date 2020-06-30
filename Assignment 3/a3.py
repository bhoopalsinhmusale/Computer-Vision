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

plt.rcParams.update({'font.size': 10})


def part_2a():
    fmri = nib.load('part1-data/clean_bold.nii.gz')
    events = pd.read_csv(
        'part1-data/events.tsv', delimiter='\t')
    events = events.to_numpy()
    tr = fmri.header.get_zooms()[3]
    ts = np.zeros(int(tr*fmri.shape[3]))
    for i in np.arange(0, events.shape[0]):
        if events[i, 3] == 'FAMOUS' or events[i, 3] == 'UNFAMILIAR' or events[i, 3] == 'SCRAMBLED':
            ts[int(events[i, 0])] = 1

    plt.plot(ts)
    plt.xlabel('time(seconds)')
    plt.show()

    hrf = pd.read_csv('part1-data/hrf.csv', header=None)
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


def part_2b():
    fmri = nib.load('part1-data/bold.nii.gz')
    events = pd.read_csv(
        'part1-data/events.tsv', delimiter='\t')
    events = events.to_numpy()
    tr = fmri.header.get_zooms()[3]
    ts = np.zeros(int(tr*fmri.shape[3]))
    for i in np.arange(0, events.shape[0]):
        if events[i, 3] == 'FAMOUS' or events[i, 3] == 'UNFAMILIAR' or events[i, 3] == 'SCRAMBLED':
            ts[int(events[i, 0])] = 1

    plt.plot(ts)
    plt.xlabel('time(seconds)')
    plt.show()

    hrf = pd.read_csv('part1-data/hrf.csv', header=None)
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


def dataset_download():
    for i in range(1, 2):
        i = str(i).zfill(2)
        if os.path.exists('part2-data/subject{}'.format(i)):
            shutil.rmtree('part2-data/subject{}'.format(i))
        os.makedirs('part2-data/subject{}'.format(i))

        copyfile("part1-data/pipeline.sh",
                 "part2-data/subject{}/pipeline.sh".format(i))
        if not os.path.exists('part2-data/subject{}/t1.nii.gz'.format(i)):
            T1_url = "https://openneuro.org/crn/datasets/ds000117/snapshots/1.0.3/files/sub-{}:ses-mri:anat:sub-{}_ses-mri_acq-mprage_T1w.nii.gz".format(
                i, i)
            wget.download(
                T1_url, 'part2-data/subject{}/t1.nii.gz'.format(i))

        if not os.path.exists('part2-data/subject{}/bold.nii.gz'.format(i)):
            bold_url = "https://openneuro.org/crn/datasets/ds000117/snapshots/1.0.3/files/sub-{}:ses-mri:func:sub-{}_ses-mri_task-facerecognition_run-01_bold.nii.gz".format(
                i, i)
            wget.download(
                bold_url, 'part2-data/subject{}/bold.nii.gz'.format(i))

        if not os.path.exists('part2-data/subject{}/events.tsv'.format(i)):
            even_url = "https://openneuro.org/crn/datasets/ds000117/snapshots/1.0.3/files/sub-{}:ses-mri:func:sub-{}_ses-mri_task-facerecognition_run-01_events.tsv".format(
                i, i)
            wget.download(
                even_url, 'part2-data/subject{}/events.tsv'.format(i))

        #os.system("sudo chmod 755 'part2-data/subject{}/pipeline.sh'".format(i))
        subprocess.call(["./part2-data/subject{}/pipeline.sh".format(i)])


if __name__ == "__main__":
    os.system("clear")
    # part_2a()

    # part_2b()

    dataset_download()
