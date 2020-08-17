import cv2
import ot
import numpy as np
import multiprocessing
import skimage.segmentation as skim
import networkx as nx
from pyemd import emd, emd_samples
import ot
import skimage.future.graph as grph
import matplotlib.pyplot as plt
import pdb
from joblib import Parallel, delayed
from skimage import io
from skimage.measure import regionprops
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from scipy.signal import fftconvolve as convolve
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale


def color_histogram(img, n_bins):
    w, _ = np.histogramdd(img, bins=n_bins)
    w = np.array(w, dtype=np.float)
    w = w / w.sum()

    return w


def bin2bin_hist_comp(hist1, hist2, method):
    hist2 = np.float32(hist2 / hist2.sum())
    return cv2.compareHist(hist1, hist2, method)


def color_3dhistogram(coor, n_bins):
    """
    Transforms the color pixels of a region from a color space (LAB ou RGB) to a 3D histogram of n_bins per axe.
    :param coor: The value of each color channel per pixel
    :param n_bins: Number of bins per color axe to quantize the color image
    :return: A list containing the weight (w), which is the normalized number of pixels per 3D bin; and the position of
             the bin in the new 3D frame.
    """
    hist, _ = np.histogramdd(coor, bins=n_bins, range=((0, 255), (0, 255), (0, 255)))   # range=((0, 255), (0, 255), (0, 255)))
    pos = np.where(hist > 0)
    w = hist[pos] / hist[pos].sum()

    return [w, pos]


# def color_3dhistogram(coor, n_bins):
#     """
#     Transforms the color pixels of a region from a color space (LAB ou RGB) to a 3D histogram of n_bins per axe.
#     :param coor: The value of each color channel per pixel
#     :param n_bins: Number of bins per color axe to quantize the color image
#     :return: A list containing the weight (w), which is the normalized number of pixels per 3D bin; and the position of
#              the bin in the new 3D frame.
#     """
#     coor = np.uint8(np.float32(coor)/(256 / n_bins))
#     w = []
#     pos = []
#
#     while coor.shape[0] != 0:
#         x, y, z = coor[0]
#         index = np.where((coor[:, 0] == x) & (coor[:, 1] == y) & (coor[:, 2] == z))
#         coor = np.delete(coor, index, axis=0)
#         w.append(len(index[0]))
#         pos.append([x, y, z])
#
#     pos = np.array(pos)
#     w = np.array(w, dtype=np.float)
#     w = w/w.sum()
#     return [w, pos]





def slic_superpixel(img, n_regions, convert2lab):
    '''
    Divide the input image into n_regions using SLIC superpixel technique
    :param img: Color input image in RGB
    :param n_regions: Number of desired regions
    :param convert2lab: Boolean parameter to indicate if work on the LAB color space. If false it work on RGB color space
    :return: A matrix of the image size with the label for each pixel
    '''
    regions_slic = skim.slic(img, n_segments=n_regions, convert2lab=convert2lab, slic_zero=True) #, compactness=0.1, sigma=0
    print('SLIC number of regions: {}'.format(len(np.unique(regions_slic))))

    return regions_slic


def img_preparation(img):
    """
    Resize the input image to an image of 640x480 pixels, transforms it from BGR to RGB and removes some noise from it.
    :param img: 3D image in BGR color space
    :return: The transformed image
    """
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (640, 480))
    # img = cv2.fastNlMeansDenoisingColored(img, img, 1, 1, 1, 1)  # Configuration for threshold distribution
    # img = cv2.fastNlMeansDenoisingColored(img, img, 11, 3, 1, 21)  # Configuration for spectral clustering

    return img


def get_mst(rag):
    rag_MST = nx.minimum_spanning_tree(rag)
    print('Number of MST edges:', rag_MST.number_of_edges())

    return rag_MST


def cost_matrix_texture(n_freq, n_angles):
    signature = np.ones((n_freq, n_angles))
    pos1 = pos2 = np.where(signature >= 0)
    pos1 = np.float32(pos1)
    pos2 = np.float32(pos2)

    sz = n_freq * n_angles
    CM = np.zeros((sz, sz))
    for ii in range(sz):
        for jj in range(sz):
            delta_freq = pos1[0][ii] - pos2[0][jj]
            delta_theta = min(np.abs(pos1[1][ii] - pos2[1][jj]),
                              n_angles - np.abs(pos1[1][ii] - pos2[1][jj]))
            CM[ii, jj] = np.abs(delta_freq) + np.abs(delta_theta)  # the deltas had a value of 0.001
    return CM


# Function for EMD (classic) version for normalized histograms [Rubner et. al.]
def em_dist_Rubner(signature, CM):
    w1 = np.float64(signature[0] / signature[0].sum())
    w2 = np.float64(signature[1] / signature[1].sum())
    return ot.emd2(w1, w2, CM, processes=-1)


# Function for EMD (new) version for non-normalized histograms [Pele et. al.]
def em_dist_Pele(signature, CM):
    dist = emd(signature[0], signature[1], CM, extra_mass_penalty=1)
    return dist


# Function for EMD (classic) version for normalized histograms [Mine]
def em_dist_mine(signature, CM):
    w1 = np.float64(signature[0] / signature[0].sum())
    w2 = np.float64(signature[1] / signature[1].sum())
    #     w1 = np.float64(softmax(signature[0]))
    #     w2 = np.float64(softmax(signature[1]))
    return ot.emd2(w1, w2, CM, processes=-1) + np.abs(signature[0].sum() - signature[1].sum())  # The abs() was multiplied by 2
