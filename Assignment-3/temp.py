import numpy as np
import pandas as pd
import nibabel as nib
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
plt.rcParams.update({'font.size': 10})


def corr_volume(im, cv):
    """Compute the correlation between an f-MRI image and a convolution, voxel by voxel"""
    ci = im - np.expand_dims(np.mean(im, 3), 3)
    cc = cv - np.mean(cv)
    corr = np.sum(ci * cc, 3) / (np.sqrt(np.sum(ci * ci, 3) + 1e-14) *
                                 np.sqrt(np.sum(cc * cc) + 1e-14))
    return corr


def run():
    """Localize task activation: find out which brain area responds to a task"""
    # load data files
    # f-mri image after pre-processing
    fmri = nib.load(
        '/home/divya/Desktop/Computer-Vision/Assignment-3/part2-data/subject01/clean_bold.nii.gz')
    task = pd.read_csv(
        '/home/divya/Desktop/Computer-Vision/Assignment-3/part2-data/subject01/events.tsv', delimiter='\t').to_numpy()
    # hemodynamic response function
    hrf = pd.read_csv(
        '/home/divya/Desktop/Computer-Vision/Assignment-3/part1-data/hrf.csv', header=None)
    hrf = hrf.to_numpy().reshape(len(hrf),)

    # repetition time (time interval between 2 volumes)
    tr = fmri.header.get_zooms()[3]
    n = int(tr * fmri.shape[3])      # number of seconds in the time series
    # initialize the time series (a function of seconds)
    ts = np.zeros(n)

    # create an ideal time series
    # stimulus type we are interested in
    stimuli = ('FAMOUS', 'UNFAMILIAR', 'SCRAMBLED')
    # filter out the tasks of interest
    mask = np.isin(task[:, 3], stimuli)
    # time when the stimulus is shown to the subject
    for onset in task[mask][:, 0]:
        ts[int(onset)] = 1

    # convolve the time series with hrf
    convolved = signal.convolve(ts, hrf, mode='full')
    convolved = convolved[0:len(ts)]

    f, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
    ax[0].plot(ts)
    ax[0].set_xlabel('time(seconds)')
    ax[1].plot(ts)
    ax[1].plot(convolved * 2.6, c='r', lineWidth=0.7)
    ax[1].set_xlabel('time(seconds)')
    plt.show()

    # correlate convolved triggers with signal in each voxel
    convolved = convolved[0::2]  # tr = 2 seconds per volume
    im = fmri.get_fdata()
    corr_map = corr_volume(im, convolved)

    f, ax = plt.subplots(nrows=7, ncols=7, figsize=(18, 22))
    for z in range(7, 53):
        row, col = (z - 7) // 7, (z - 7) % 7
        ax[row, col].imshow(np.rot90(corr_map[:, :, z]),
                            origin='lower', vmin=-0.25, vmax=0.25)

    f.delaxes(ax[6, -1])
    f.delaxes(ax[6, -2])
    f.delaxes(ax[6, -3])
    plt.show()


if __name__ == "__main__":
    run()
