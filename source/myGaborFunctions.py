import numpy as np
from scipy.signal import fftconvolve as convolve
from scipy.ndimage import gaussian_filter
from skimage.morphology import square, opening
from joblib import Parallel, delayed

__all__ = ['makeGabor_filter', 'makeGabor_filterbank', 'applyGabor_filter', 'applyGabor_filterbank']


def makeGabor_filter(frequency=0.1, angle=0, alpha=0.1, beta=0.1, freq_bandwidth=None, angle_bandwidth=None,
                     freq_crossing_point=0.5, ang_crossing_point=0.5, n_stds=3.5):
    if freq_bandwidth is not None:
        k = 2 ** freq_bandwidth
        gamma = (1 / np.pi) * np.sqrt(np.log(1 / freq_crossing_point)) * ((k + 1) / (k - 1))
        alpha = frequency / gamma
    if angle_bandwidth is not None:
        angle_bandwidth = np.deg2rad(angle_bandwidth)
        eta = (1 / np.pi) * np.sqrt(np.log(1 / ang_crossing_point)) * (1 / np.tan(angle_bandwidth / 2))
        beta = frequency / eta

    sigma_x = 1 / (np.sqrt(2) * alpha)
    sigma_y = 1 / (np.sqrt(2) * beta)

    x0 = np.ceil(max(np.abs(n_stds * sigma_x * np.cos(angle)), np.abs(n_stds * sigma_y * np.sin(angle)), 1))
    y0 = np.ceil(max(np.abs(n_stds * sigma_y * np.cos(angle)), np.abs(n_stds * sigma_x * np.sin(angle)), 1))
    y, x = np.mgrid[-y0:y0 + 1, -x0:x0 + 1]

    xr = x * np.cos(angle) + y * np.sin(angle)
    yr = -x * np.sin(angle) + y * np.cos(angle)

    norm_const = (alpha * beta) / np.pi
    gauss = np.exp(- (alpha ** 2 * xr ** 2 + beta ** 2 * yr ** 2))
    gabor = gauss * np.exp(1j * 2 * np.pi * frequency * xr)
    gabor_norm = gabor * norm_const

    params = {'frequency': frequency, 'angle': angle, 'sigma_x': sigma_x, 'sigma_y': sigma_y}

    return gabor_norm, params


def makeGabor_filterbank(min_period=2., max_period=40., freq_bandwidth=1, angle_bandwidth=45, freq_crossing_point=0.5,
                         ang_crossing_point=0.5, n_stds=3.5):
    k = 2 ** freq_bandwidth
    angle_bandwidth = np.deg2rad(angle_bandwidth)

    gamma = (1 / np.pi) * np.sqrt(np.log(1 / freq_crossing_point)) * ((k + 1) / (k - 1))
    eta = (1 / np.pi) * np.sqrt(np.log(1 / ang_crossing_point)) * (1 / np.tan(angle_bandwidth / 2.))

    f_high = 1 / min_period
    f_low = 1 / max_period
    frequencies = np.array([k ** -ii * f_high for ii in range(20)], dtype=np.float)
    frequencies = frequencies[f_low < frequencies]

    n = np.int(np.round(np.sqrt((eta * np.pi ** 2) ** 2 / (4 * np.log(1 / ang_crossing_point)))))
    angles = [ii * (np.pi / int(n)) for ii in range(n)]

    gabor_filters = []
    for ii, freq in enumerate(frequencies):
        for jj, theta in enumerate(angles):
            alpha = freq / gamma
            beta = freq / eta
            gabor = makeGabor_filter(frequency=freq, angle=theta, alpha=alpha, beta=beta, n_stds=n_stds)
            gabor_filters.append(gabor)

    return gabor_filters, frequencies, angles


def applyGabor_filter(image, filtr, resp_type, smooth, morph_opening, se_z):
    img_padded = np.pad(image, (
    (filtr[0].shape[0] // 2, filtr[0].shape[0] // 2), (filtr[0].shape[1] // 2, filtr[0].shape[1] // 2)),
                        mode='symmetric')
    resp = np.vectorize(complex)(convolve(img_padded, filtr[0].real, mode='valid'),
                                 convolve(img_padded, filtr[0].imag, mode='valid'))
    #     resp = convolve(img_padded, filtr[0], mode='valid')

    #     resp /= (filtr[1].get('sigma')**2)#np.sqrt(filtr[1].get('sigma_x'))*2

    # print(filtr[0].shape)
    if resp_type == 'L2':
        resp_1 = np.abs(resp)

    elif resp_type == 'real':
        resp_1 = resp.real
    elif resp_type == 'imag':
        resp_1 = resp.imag
    elif resp_type == 'square':
        resp_1 = resp.real ** 2 + resp.imag ** 2
    elif resp_type == 'complete':
        resp_1 = resp

    if morph_opening:
        # print('opening')
        period = np.int(se_z / filtr[1].get('frequency'))
        selem = square(period)
        resp_opened = opening(resp_1, selem=selem)
    else:
        resp_opened = resp_1

    if smooth:
        sigma_s = 1. * filtr[1].get('sigma_x')

        if np.iscomplexobj(resp_1):
            resp_smth = np.copy(resp_1)
            resp_smth.real = gaussian_filter(resp_opened.real, sigma=sigma_s, truncate=1.5, mode='reflect')
            resp_smth.imag = gaussian_filter(resp_opened.imag, sigma=sigma_s, truncate=1.5, mode='reflect')
        else:
            resp_smth = gaussian_filter(resp_opened, sigma=sigma_s, truncate=1.5, mode='reflect')

    else:
        resp_smth = resp_opened

    return np.array(resp_smth)


def applyGabor_filterbank(img, filterbank, resp_type, smooth, morph_opening, se_z):
    num_cores = -1
    responses = Parallel(n_jobs=num_cores)(
        delayed(applyGabor_filter)(img, g_filter, resp_type, smooth, morph_opening, se_z) for g_filter in filterbank)

    return np.array(responses)
