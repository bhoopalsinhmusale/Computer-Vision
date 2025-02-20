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
import threading

import matplotlib.image as mpimg

plt.rcParams.update({'font.size': 10})


def part_2a():

    fmri = nib.load("part-2-data/clean_bold.nii.gz")
    events = pd.read_csv("part-2-data/events.tsv", delimiter='\t')
    events = events.to_numpy()
    tr = fmri.header.get_zooms()[3]
    ts = np.zeros(int(tr*fmri.shape[3]))
    for i in np.arange(0, events.shape[0]):
        if events[i, 3] == 'FAMOUS' or events[i, 3] == 'UNFAMILIAR' or events[i, 3] == 'SCRAMBLED':
            ts[int(events[i, 0])] = 1

    plt.plot(ts)
    plt.xlabel('time(seconds)')
    plt.show()

    hrf = pd.read_csv("part-2-data/hrf.csv", header=None)
    hrf = hrf.to_numpy().reshape(len(hrf),)

    conved = signal.convolve(ts, hrf, mode='full')
    conved = conved[0:ts.shape[0]]
    plt.plot(ts)
    plt.plot(conved*3.2, linewidth=1)
    plt.xlabel('time(seconds)')
    plt.show()

    conved = conved[0::2]
    img = fmri.get_fdata()

    meansub_img = img - np.expand_dims(np.mean(img, 3), 3)
    meansub_conved = conved - np.mean(conved)
    corrs = np.sum(meansub_img*meansub_conved, 3)/(np.sqrt(np.sum(meansub_img *
                                                                  meansub_img, 3))*np.sqrt(np.sum(meansub_conved*meansub_conved)))
    # code to display each slice Individual
    '''if not os.path.exists('part-2-output-images'):
        os.makedirs('part-2-output-images')
    for i in range(1, 47):
        plt.subplot(1, 1, 1)
        plt.imshow(np.rot90(corrs[:, :, i+7]),
                   vmin=-0.25, vmax=0.25)
        plt.title("Slice={}".format(i+7))
        plt.savefig("part-2-output-images/part-2-A-Slice={}".format(i+7))
        plt.show()'''

    axes = []
    fig = plt.figure(figsize=(20, 20))

    for a in range(6*8):

        axes.append(fig.add_subplot(6, 8, a+1))
        plt.imshow(np.rot90(corrs[:, :, a+7]),
                   vmin=-0.25, vmax=0.25)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("part-2-A")
    plt.suptitle(
        "part-2-A-localize task activation(clean_bold.nii.gz)\nSlices-7 to 54")
    plt.show()
    return 0


def part_2b():
    fmri = nib.load('part-2-data/bold.nii.gz')
    events = pd.read_csv('part-2-data/events.tsv', delimiter='\t')
    events = events.to_numpy()
    tr = fmri.header.get_zooms()[3]
    ts = np.zeros(int(tr*fmri.shape[3]))
    for i in np.arange(0, events.shape[0]):
        if events[i, 3] == 'FAMOUS' or events[i, 3] == 'UNFAMILIAR' or events[i, 3] == 'SCRAMBLED':
            ts[int(events[i, 0])] = 1

    plt.plot(ts)
    plt.xlabel('time(seconds)')
    plt.show()

    hrf = pd.read_csv('part-2-data/hrf.csv', header=None)
    hrf = hrf.to_numpy().reshape(len(hrf),)

    conved = signal.convolve(ts, hrf, mode='full')
    conved = conved[0:ts.shape[0]]
    plt.plot(ts)
    plt.plot(conved*3.2, linewidth=1)
    plt.xlabel('time(seconds)')
    plt.show()

    conved = conved[0::2]
    img = fmri.get_fdata()

    meansub_img = img - np.expand_dims(np.mean(img, 3), 3)
    meansub_conved = conved - np.mean(conved)
    corrs = np.sum(meansub_img*meansub_conved, 3)/(np.sqrt(np.sum(meansub_img *
                                                                  meansub_img, 3))*np.sqrt(np.sum(meansub_conved*meansub_conved)))

    # code to display each slice Individual
    '''if not os.path.exists('part-2-output-images'):
        os.makedirs('part-2-output-images')
    for i in range(1, 47):
        plt.subplot(1, 1, 1)
        plt.imshow(np.rot90(corrs[:, :, i+7]),
                   vmin=-0.25, vmax=0.25)
        plt.title("Slice={}".format(i+7))
        plt.savefig("part-2-output-images/part-2-B-Slice={}".format(i+7))
        plt.show()'''

    axes = []
    fig = plt.figure(figsize=(20, 20))

    for a in range(6*8):

        axes.append(fig.add_subplot(6, 8, a+1))
        plt.imshow(np.rot90(corrs[:, :, a+7]),
                   vmin=-0.25, vmax=0.25)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("part-2-B")
    plt.suptitle(
        "part-2-B-localize task activation(bold.nii.gz)\nSlices-7 to 54")
    plt.show()
    return corrs


no = 16  # subject count


def dataset_download_preprocess():
    '''
    Funtion to automatically create directory and download all subjects from openneuro.org.
    Adterwards it copies and execute pipeline.sh in each subject's directory
    '''
    if not os.path.exists('part-3-data'):
        os.makedirs('part-3-data')

    os.system("sudo chmod -R 777 './part-3-data/'")
    print("\n"+str(os.system("pwd")))
    for i in range(1, no+1):
        i = str(i).zfill(2)
        '''if os.path.exists('part-3-data/subject{}'.format(i)):
            shutil.rmtree(
                'part-3-data/subject{}'.format(i))'''

        if not os.path.exists('part-3-data/subject{}'.format(i)):
            os.makedirs(
                'part-3-data/subject{}'.format(i))
        # os.chdir(
        #    "/home/divya/Desktop/Computer-Vision/Assignment-3/part-3-data/subject{}".format(i))
        # print("\n"+str(os.system("pwd")))

        copyfile("part-2-data/pipeline.sh",
                 "part-3-data/subject{}/pipeline.sh".format(i))
        if not os.path.exists('part-3-data/subject{}/t1.nii.gz'.format(i)):
            T1_url = "https://openneuro.org/crn/datasets/ds000117/snapshots/1.0.3/files/sub-{}:ses-mri:anat:sub-{}_ses-mri_acq-mprage_T1w.nii.gz".format(
                i, i)
            wget.download(
                T1_url, 'part-3-data/subject{}/t1.nii.gz'.format(i))

        if not os.path.exists('part-3-data/subject{}/bold.nii.gz'.format(i)):
            bold_url = "https://openneuro.org/crn/datasets/ds000117/snapshots/1.0.3/files/sub-{}:ses-mri:func:sub-{}_ses-mri_task-facerecognition_run-01_bold.nii.gz".format(
                i, i)
            wget.download(
                bold_url, 'part-3-data/subject{}/bold.nii.gz'.format(i))

        if not os.path.exists('part-3-data/subject{}/events.tsv'.format(i)):
            even_url = "https://openneuro.org/crn/datasets/ds000117/snapshots/1.0.3/files/sub-{}:ses-mri:func:sub-{}_ses-mri_task-facerecognition_run-01_events.tsv".format(
                i, i)
            wget.download(
                even_url, 'part-3-data/subject{}/events.tsv'.format(i))

        os.chdir("part-3-data/subject{}".format(i))
        print("\n"+str(os.system("pwd")))

        os.system("sudo chmod -R 777 './'".format(i))
        if not os.path.exists("clean_bold.nii.gz".format(i)):
            os.system("./pipeline.sh".format(i))
        os.chdir("..")
        os.chdir("..")
    return 0


def correlation_map_registration_overlay():  # fully Automated
    '''
    Funtion to automatically perform all AFNI GUI tasks for each subject
    1. Compute correlation map
    2. Save corr.nii.gz
    3. Brings the correlation map into the subject’s T1 space
    4. Automated GUI AFNI open and overlay and underlay image selection
    5. Automated AFNI DICOM XYZ position selection
    6. Automated AFNI threshold selection
    7. Automated AFNI saves all images axial,sagittal and coronal
    8. Automated AFNI close at the end
    '''
    print("\n"+str(os.system("pwd")))
    hrf = pd.read_csv(
        "part-2-data/hrf.csv", header=None)
    hrf = hrf.to_numpy().reshape(len(hrf),)

    for i in range(1, no+1):
        i = str(i).zfill(2)
        fmri = nib.load("part-3-data/subject{}/clean_bold.nii.gz".format(i))
        events = pd.read_csv(
            "part-3-data/subject{}/events.tsv".format(i), delimiter='\t')
        events = events.to_numpy()
        tr = fmri.header.get_zooms()[3]
        ts = np.zeros(int(tr*fmri.shape[3]))
        for j in np.arange(0, events.shape[0]):
            if events[j, 3] == 'FAMOUS' or events[j, 3] == 'UNFAMILIAR' or events[j, 3] == 'SCRAMBLED':
                ts[int(events[j, 0])] = 1
        conved = signal.convolve(ts, hrf, mode='full')
        conved = conved[0:ts.shape[0]]
        conved = conved[0::2]
        img = fmri.get_fdata()

        meansub_img = img - np.expand_dims(np.mean(img, 3), 3)
        meansub_conved = conved - np.mean(conved)
        corrs = np.sum(meansub_img*meansub_conved, 3)/(np.sqrt(np.sum(meansub_img *
                                                                      meansub_img, 3))*np.sqrt(np.sum(meansub_conved*meansub_conved)))

        corrs_nifti = nib.Nifti1Image(corrs, fmri.affine)
        nib.save(corrs_nifti, "part-3-data/subject{}/corrs.nii.gz".format(i))

        os.chdir("part-3-data/subject{}".format(i))
        print("\n"+str(os.system("pwd")))

        # Command for brining the correlation map into the subject’s T1 space
        os.system(
            "flirt -in corrs.nii.gz -ref t1.nii.gz -applyxfm -init epireg.mat -out corrs_in_t1.nii.gz")

        # Command for Opening AFNI GUI and performing all taks
        os.system("afni -com 'OPEN_WINDOW A.axialimage'\
            -com 'OPEN_WINDOW A.sagittalimage'\
            -com 'OPEN_WINDOW A.coronalimage'\
            -com 'SET_DICOM_XYZ A 17.900 62.688 1.632'\
            -com 'SET_XHAIRS A.OFF'\
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
        os.chdir("..")
        os.chdir("..")
        print("\n"+str(os.system("pwd")))
    return 0


def plot_corrs_in_t1():
    '''
    Funtion to plot all saved images from funtion correlation_map_registration_overlay() for each subject.
    '''
    fig = plt.figure()
    print("\n"+str(os.system("pwd")))
    for r in range(1, no+1):
        print("\n"+str(os.system("pwd")))

        img = np.fliplr(mpimg.imread(
            "part-3-data/subject{}/axialimage.png".format(str(r).zfill(2))))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(
            "Axial".format(str(r).zfill(2)))
        # axes[r-1, c].set_aspect(aspect=0.6)

        img = mpimg.imread(
            "part-3-data/subject{}/sagittalimage.png".format(str(r).zfill(2)))
        plt.subplot(1, 3, 2)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Sagittal".format(str(r).zfill(2)))
        # axes[r-1, c+1].set_aspect(aspect=0.6)

        img = np.fliplr(mpimg.imread(
            "part-3-data/subject{}/coronalimage.png".format(str(r).zfill(2))))
        plt.subplot(1, 3, 3)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Coronal".format(str(r).zfill(2)))

        plt.suptitle("Subject{}".format(str(r).zfill(2)))
        plt.savefig(
            "Subject{}".format(str(r).zfill(2), str(r).zfill(2)))
        plt.show()
    return 0


def group_average():
    os.system("sudo chmod -R 777 './part-3-data'")
    for i in range(1, no+1):
        i = str(i).zfill(2)
        copyfile("part-2-data/MNI152_2009_template.nii.gz",
                 "part-3-data/subject{}/MNI152_2009_template.nii.gz".format(i))
        os.chdir("part-3-data/subject{}".format(i))
        print("\n"+str(os.system("pwd")))
        if not os.path.exists('corrs_in_MNI152.nii.gz'):
            os.system(
                "fslsplit MNI152_2009_template.nii.gz Resized_MNI152_2009_template")

            os.system(

                "flirt -in bet.nii.gz -ref Resized_MNI152_2009_template0000.nii.gz -out t1_in_MNI152.nii -omat MNI152.mat")
            os.system(
                "flirt -in corrs_in_t1.nii.gz -ref Resized_MNI152_2009_template0000.nii.gz -applyxfm -init MNI152.mat -out corrs_in_MNI152.nii.gz")
        os.chdir("..")
        os.chdir("..")
    print("\n"+str(os.system("pwd")))

    os.system(
        "3dMean -prefix grand_avgerage.nii.gz part-3-data/subject01/corrs_in_MNI152.nii.gz\
                part-3-data/subject02/corrs_in_MNI152.nii.gz\
                part-3-data/subject03/corrs_in_MNI152.nii.gz\
                part-3-data/subject04/corrs_in_MNI152.nii.gz\
                part-3-data/subject05/corrs_in_MNI152.nii.gz\
                part-3-data/subject06/corrs_in_MNI152.nii.gz\
                part-3-data/subject07/corrs_in_MNI152.nii.gz\
                part-3-data/subject08/corrs_in_MNI152.nii.gz\
                part-3-data/subject09/corrs_in_MNI152.nii.gz\
                part-3-data/subject10/corrs_in_MNI152.nii.gz\
                part-3-data/subject11/corrs_in_MNI152.nii.gz\
                part-3-data/subject12/corrs_in_MNI152.nii.gz\
                part-3-data/subject13/corrs_in_MNI152.nii.gz\
                part-3-data/subject14/corrs_in_MNI152.nii.gz\
                part-3-data/subject15/corrs_in_MNI152.nii.gz\
                part-3-data/subject16/corrs_in_MNI152.nii.gz -overwrite")
    copyfile("part-2-data/MNI152_2009_template.nii.gz",
             "MNI152_2009_template.nii.gz")

    # Command for Opening AFNI GUI and performing all taks
    os.system("afni -com 'OPEN_WINDOW A.axialimage'\
            -com 'OPEN_WINDOW A.sagittalimage'\
            -com 'OPEN_WINDOW A.coronalimage'\
            -com 'SET_DICOM_XYZ A 0 42 48'\
            -com 'SET_PBAR_ALL A.-1981 1.0 Spectrum:red_to_blue FLIP'\
            -com 'SWITCH_UNDERLAY MNI152_2009_template.nii.gz+tlrc'\
            -com 'SEE_OVERLAY A.-'\
            -com 'SWITCH_OVERLAY grand_avgerage.nii.gz+tlrc'\
            -com 'SEE_OVERLAY A.+'\
            -com 'SET_THRESHOLD A.00812 1'\
            -com 'SAVE_PNG A.axialimage grand_avgerage_axialimage.png blowup=5'\
            -com 'SAVE_PNG A.sagittalimage grand_avgerage_sagittalimage.png blowup=5'\
            -com 'SAVE_PNG A.coronalimage grand_avgerage_coronalimage.png blowup=5'\
            -com 'QUIT'")
    fig = plt.figure()
    img = np.fliplr(mpimg.imread(
        "grand_avgerage_axialimage.png"))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(
        "Axial")
    # axes[r-1, c].set_aspect(aspect=0.6)

    img = mpimg.imread(
        "grand_avgerage_sagittalimage.png")
    plt.subplot(1, 3, 2)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Sagittal")
    # axes[r-1, c+1].set_aspect(aspect=0.6)

    img = np.fliplr(mpimg.imread(
        "grand_avgerage_coronalimage.png"))
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Coronal")

    plt.suptitle("Grand-Average-Output")
    plt.savefig(
        "Grand-Average-Output")
    plt.show()
    return 0


def part_3():
    thread = threading.Thread(target=dataset_download_preprocess)
    thread.start()
    print("dataset_download_preprocess is running.")
    thread.join()
    print("dataset_download_preprocess thread has finished.")

    thread = threading.Thread(target=correlation_map_registration_overlay)
    thread.start()
    print("correlation_map_registration_overlay is running.")
    thread.join()
    print("correlation_map_registration_overlay thread has finished.")

    thread = threading.Thread(target=plot_corrs_in_t1)
    thread.start()
    print("plot_corrs_in_t1 is running.")
    thread.join()
    print("plot_corrs_in_t1 thread has finished.")

    thread = threading.Thread(target=group_average)
    thread.start()
    print("group_average is running.")
    thread.join()
    print("group_average thread has finished.")


if __name__ == "__main__":
    os.system("clear")
    # part_2a()

    # part_2b()

    part_3()

    # correlation_map_registration_overlay()

    # dataset_download_preprocess()
