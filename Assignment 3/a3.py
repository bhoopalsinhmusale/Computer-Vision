import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
from scipy.ndimage import rotate


def part_2a():
    plt.rcParams.update({'font.size': 20})
    fmri = nib.load('/home/divya/Desktop/images/clean_bold.nii.gz')
    events = pd.read_csv(
        '/home/divya/Desktop/images/events.tsv', delimiter='\t')
    events = events.to_numpy()
    tr = fmri.header.get_zooms()[3]
    ts = np.zeros(int(tr*fmri.shape[3]))
    for i in np.arange(0, events.shape[0]):
        if events[i, 3] == 'FAMOUS' or events[i, 3] == 'UNFAMILIAR' or events[i, 3] == 'SCRAMBLED':
            ts[int(events[i, 0])] = 1
    plt.plot(ts)
    plt.xlabel('time(seconds)')

    conved = signal.convolve(ts, hrf, mode='full')
    conved = conved[0:ts.shape[0]]
    plt.plot(ts)
    plt.plot(conved*3.2, lineWidth=3)
    plt.xlabel('time(seconds)')
    conved = conved[0::2]
    img = fmri.get_data()
    meansub_img = img - np.expand_dims(np.mean(img, 3), 3)
    meansub_conved = conved - np.mean(conved)
    corrs = np.sum(meansub_img*meansub_conved, 3)/(np.sqrt(np.sum(meansub_img *
                                                                  meansub_img, 3))*np.sqrt(np.sum(meansub_conved*meansub_conved)))
    plt.imshow(np.rot90(np.max(corrs, axis=2)))
    plt.colorbar()


if __name__ == "__main__":
    part_2a()
