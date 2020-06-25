# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import cv2
import pdb

def calc_luminance(img):
    L = img[:, :, 0] * 0.2125 + img[:, :, 0] * 0.7154  + img[:, :, 0] * 0.0721
    return L

def normalize_img(img, rows, cols):
    img_nrmzd = (img * (np.sqrt(rows*cols))) / np.sqrt(np.sum(np.abs(img)**2))
    return img_nrmzd

def linear_normalization(arr, nmax, nmin):
    minarr = np.min(arr)
    maxarr = np.max(arr)
    return (arr - minarr) * ((nmax - nmin)/ (maxarr - minarr)) + nmin

def linear_normalization2(arr):
    arr //= 2.
    arr += 128.

    return arr

def hexencode(rgb):
    """Transform an RGB tuple to a hex string (html color)"""
    r=int(rgb[0])
    g=int(rgb[1])
    b=int(rgb[2])
    return '#%02x%02x%02x' % (r,g,b)

def img2complex_colorspace(img, color_space):
    # see https://docs.opencv.org/trunk/de/d25/imgproc_color_conversions.html#color_convert_rgb_hsv for color space transformation info

#     img = img_as_float32(img.astype('uint8')) # The input image must be float32 between [0, 1]

    if color_space == 'HV':
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        ch1 = np.deg2rad(img_hsv[:, :, 0]) # Change the H channel from deg to rag #img_hsv[:, :, 0]#
        ch2 = img_hsv[:, :, 1] * 255. #* 100. # Change the S channel between [0, 1]
        ch3 = img_hsv[:, :, 2] * 255. #* 100. # Change the V channel between [0, 1]

        luminance = ch2
        chrominance = ch3 * np.exp(1j * ch1)
        chrominance_real = chrominance.real
        chrominance_imag = chrominance.imag
        cs = 'HSV'

    if color_space == 'HLS':
        img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype('float32')
        ch1 = np.deg2rad(img_hls[:, :, 0]) * 2# Change the H channel from deg to rag #img_hsv[:, :, 0]#
        ch2 = img_hls[:, :, 1] #* 255. #* 100. # Change the L channel between [0, 1]
        ch3 = img_hls[:, :, 2] #* 255. #* 100. # Change the S channel between [0, 1]

        luminance = ch2
        chrominance = ch3 * np.exp(1j * ch1)
        chrominance_real = chrominance.real
        chrominance_imag = chrominance.imag
        cs = 'HLS'

    if color_space == 'HS':
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype('float32')
        ch1 = np.deg2rad(img_hsv[:, :, 0])*2 # Change the H channel from deg to rag img_hsv[:, :, 0]#
        ch2 = img_hsv[:, :, 1] #* 255. #* 100. # Change the S channel between [0, 1]
        ch3 = img_hsv[:, :, 2] #* 255. #* 100. # Change the V channel between [0, 1]

        luminance = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype('float32')#calc_luminance(img) #ch3#ch3 #
        chrominance = ch2 * np.exp(1j * ch1)
        chrominance_real = chrominance.real
        chrominance_imag = chrominance.imag
        cs = 'HSV'

    if color_space == 'LAB':
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        ch1 = img_lab[:, :, 0]  # L channel between [0, 100]
        ch2 = img_lab[:, :, 1]  # a channel between [-127, 127]
        ch3 = img_lab[:, :, 2]  # b channel between [-127, 127]

        luminance = ch1
        chrominance = ch2 + (1j * ch3)
        chrominance_real = chrominance.real
        chrominance_imag = chrominance.imag
        cs = 'LAB'

#     fig, axs = plt.subplots(1, 3, figsize=(9, 3), dpi=300)
#     axs[0].imshow(ch1, cmap='gray')
#     axs[1].imshow(ch2, cmap='gray')
#     axs[2].imshow(ch3, cmap='gray')
#     fig.suptitle('Image channels in '+ cs +' color space')

    return luminance, chrominance_real, chrominance_imag


def img2complex_normalized_colorspace(img, img_shape, color_space='HS'):

    rows, cols, channels = img_shape
    lum, chrom_r, chrom_i = img2complex_colorspace(img, color_space)

    ##################################  Luminance and chrominance normalization ##################################
    lum = linear_normalization(lum, 255., 0.)
    chrom_r = linear_normalization2(chrom_r)
    chrom_i = linear_normalization2(chrom_i)

    img_2ch = np.array((lum, chrom_r, chrom_i))
    img_2ch_norm = normalize_img(img_2ch, rows, cols)

    return img_2ch_norm


def tonemap(img, lum, f, m, a, c):
    lum = lum[..., np.newaxis]
    Cva = np.mean(img, axis=-1)
    Lav = np.mean(lum)
    Llav = np.log(Lav)
    Lmin = np.min(lum)
    Lmax = np.max(lum)

    f = np.exp(-f)
    if m == 0:
        k = (np.log(Lmax)-Llav) / (np.log(Lmax)-np.log(Lmin))
        m = 0.3 +0.7* (k **1.4)
    pdb.set_trace()
    I_l = np.sum(c * img , (1-c) * lum)
    I_g = c * Cva + (1-c) * Lav
    I_a = a + I_l + (1-a) * I_g
    return I_a

def calc_semisat(lum, m=0, f=0):
    lum = lum
#     Cva = np.mean(img, axis=-1)
    Lav = np.mean(lum)
    Llav = np.log(Lav)
    Lmin = np.min(lum)
    Lmax = np.max(lum)

    f = np.exp(-f)
    pdb.set_trace()
    if m == 0:
        k = (np.log(Lmax)-Llav) / (np.log(Lmax)-np.log(Lmin))
        print(k)
        m = 0.3 + 0.7 * (k ** 1.4)

    return f * (Lav ** m)


def gaussian_color_model(arr):
    gauss_model_from_rgb = np.array([[0.006, 0.63, 0.31],
                                     [0.19, 0.18, -0.37],
                                     [0.22, -0.44, 0.06]])
    return np.dot(arr, gauss_model_from_rgb.T.copy())


def gaussian_shadow_color_model(arr):
    # arr = img_as_float32(arr)
    arr[np.where(arr == 0)] = 1
    gcm = gaussian_color_model(arr)
    gscm = np.zeros((gcm.shape))
    gscm[:, :, 0] = np.log(gcm[:, :, 0])
    gscm[:, :, 1] = np.divide(gcm[:, :, 1], gcm[:, :, 0])
    gscm[:, :, 2] = np.divide(np.subtract((gcm[:, :, 0] * gcm[:, :, 2]), gcm[:, :, 1]**2), gcm[:, :, 0]**2)
    return gscm


def reshape4clustering(feature, rows, cols):
    if feature.ndim == 3 and feature.shape[-1] == 3:
        feature = feature.reshape((rows*cols, 3))
        feature_norm = (feature - feature.mean(axis=0))/feature.std(axis=0)
    elif feature.ndim == 3 and feature.shape[-1] == 4:
        feature = feature.reshape((rows*cols, 4))
        feature_norm = (feature - feature.mean(axis=0))/feature.std(axis=0)
    else:
        feature = feature.reshape(rows*cols)
        feature_norm = (feature - feature.mean())/feature.std()
    return feature