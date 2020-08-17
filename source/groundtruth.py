# -*- coding: utf-8 -*-
# !/usr/bin/env python

import sys, time, os
from os.path import expanduser

home = expanduser("~")

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import find_boundaries


def get_segmentation(path, filename):
    """
    Load groundtruth on BSD500 for the specified image
    """

    f = loadmat(path + filename)
    data = f['groundTruth'][0]
    groundtruth = []
    for img in data:
        groundtruth.append(img[0][0][0])

    # 4: Output
    return groundtruth


def get_segment_from_filename(filename):
    """
    Load groundtruth on BSD500 for the specified image
    """

    path = os.getcwd() + "/../../data/groundTruth/"
    list_dir = os.listdir(path)
    filename = filename + '.mat'

    segments = []
    for folder in list_dir:

        list_img = os.listdir(path + folder + "/")
        if (filename in list_img):
            segments.extend(get_segmentation(path + folder + "/", filename))

    return segments


# ---------------------------------
# when executed, run the main script
# ---------------------------------
if __name__ == '__main__':
    filename = "2092"
    segments = get_segment_from_filename(filename)
    plt.imshow(segments[5])
    plt.show()
