import cv2
import ot
import numpy as np
import multiprocessing
import skimage.segmentation as skim
import networkx as nx
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

# Functions for Gabor filterbank generation

def _sigma_prefactor(bandwidth):
    b = bandwidth
    # See http://www.cs.rug.nl/~imaging/simplecell.html
    return 1.0 / np.pi * np.sqrt(np.log(2) / 2.0) * \
           (2.0 ** b + 1) / (2.0 ** b - 1)


def genGabor(freq=0.1, theta=0, b=1, n=5, gamma=None, eta=None, p1=None, p2=None, n_stds=3.):
    k = 2 ** b

    if gamma is None and p1 is not None:
        gamma = np.float((1 / np.pi) * (k + 1) / (k - 1) * np.sqrt(-np.log(p1)))
    else:
        p1 = np.exp(-(gamma * np.pi * (1 - k) / (k + 1)) ** 2)
        gamma = np.float((1 / np.pi) * (k + 1) / (k - 1) * np.sqrt(-np.log(p1)))
    if eta is None and p2 is not None:
        eta = np.float(2 * n * np.sqrt(-np.log(p2)) / np.pi ** 2)
    else:
        p2 = np.exp(-((eta * (np.pi) ** 2) / (2 * n)) ** 2)
        eta = np.float(2 * n * np.sqrt(-np.log(p2)) / np.pi ** 2)

    #     sigma_x = gamma/(np.sqrt(2) * freq)
    #     sigma_y = eta/(np.sqrt(2) * freq)

    sigma_x = _sigma_prefactor(b) / freq
    sigma_y = _sigma_prefactor(b) / freq
    x0 = np.ceil(max(np.abs(n_stds * sigma_x * np.cos(theta)), np.abs(n_stds * sigma_y * np.sin(theta)), 1))
    y0 = np.ceil(max(np.abs(n_stds * sigma_y * np.cos(theta)), np.abs(n_stds * sigma_x * np.sin(theta)), 1))
    y, x = np.mgrid[-y0:y0 + 1, -x0:x0 + 1]

    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)

    gauss = (freq ** 2 / (np.pi * gamma * eta)) * np.exp(-((freq / gamma) ** 2 * xr ** 2 + (freq / eta) ** 2 * yr ** 2))
    gabor = gauss * np.exp(1j * 2 * np.pi * freq * xr)
    gabor /= np.sum(np.abs(gabor), dtype='float')

    return gabor


def makeGabor_filterbank(scale=1., min_period=2., max_period=40., b=1, n=5, gamma=None, eta=None, p1=None, p2=None,
                         n_stds=3.):
    k = 2 ** b
    f_high = 1 / (min_period * scale)
    f_low = 1 / (max_period * scale)

    if gamma is None and p1 is not None:
        gamma = np.float((1 / np.pi) * (k + 1) / (k - 1) * np.sqrt(-np.log(p1)))
    else:
        p1 = np.exp(-(gamma * np.pi * (1 - k) / (k + 1)) ** 2)
        gamma = np.float((1 / np.pi) * (k + 1) / (k - 1) * np.sqrt(-np.log(p1)))
        print(p1)
    if eta is None and p2 is not None:
        eta = np.float(2 * n * np.sqrt(-np.log(p2)) / np.pi ** 2)
    else:
        p2 = np.exp(-((eta * (np.pi) ** 2) / (2 * n)) ** 2)
        eta = np.float(2 * n * np.sqrt(-np.log(p2)) / np.pi ** 2)
        print(p2)

    frequencies = np.array([k ** -ii * f_high for ii in range(20)], dtype=np.float)
    angles = [ii * (np.pi / n) for ii in range(n)]
    frequencies = frequencies[f_low < frequencies]

    gabor_filters = []
    #     filter_params = []
    for ii, freq in enumerate(frequencies):
        for jj, theta in enumerate(angles):
            sigma = gamma / freq
            params = {'freq': freq, 'angle': theta, 'sigma': sigma}
            gabor = genGabor(freq=freq, theta=theta, gamma=gamma, eta=eta, n_stds=n_stds)
            gabor_filters.append((gabor, params))
    #             filter_params.append()

    frequencies *= scale

    return gabor_filters, frequencies, angles


def applyFilter(img, filtr, scale=1., resp_type='complex', non_linear=False, smooth=False):
    img_padded = np.pad(img, (
    (filtr[0].shape[0] // 2, filtr[0].shape[0] // 2), (filtr[0].shape[1] // 2, filtr[0].shape[1] // 2)), mode='symmetric')
    #     resp = np.vectorize(complex)(convolve(img_padded, filtr[0].real, mode='valid'), convolve(img_padded, filtr[0].imag, mode='valid'))
    resp = convolve(img_padded, filtr[0], mode='valid')

    if resp_type == 'complex':
        resp = np.abs(resp)
    elif resp_type == 'real':
        resp = resp.real
    elif resp_type == 'imag':
        resp = resp.imag
    elif resp_type == 'square':
        resp = resp.real ** 2 + resp.imag ** 2
    elif resp_type == 'complete':
        resp = resp

    #     # Normalize between [0, 1]
    #     resp -= resp.min()
    #     resp /= resp.max()

    if non_linear:
        alpha = 2.5
        if np.iscomplexobj(resp):
            resp = np.tanh(np.abs(resp) * alpha) * np.exp(1j * np.angle(resp))
        else:
            resp = np.tanh(resp * alpha)

    #         # Normalize between [0, 1]
    #         resp -= resp.min()
    #         resp /= resp.max()

    if smooth:
        sigma_s = filtr[1].get('sigma')
        if np.iscomplexobj(resp):
            resp.real = gaussian_filter(resp.real, sigma=sigma_s, truncate=3., mode='mirror')
            resp.imag = gaussian_filter(resp.imag, sigma=sigma_s, truncate=3., mode='mirror')
        else:
            resp = gaussian_filter(resp, sigma=sigma_s, truncate=4., mode='mirror')

    #         # Normalize between [0, 1]
    #         resp -= resp.min()
    #         resp /= resp.max()

    if int(scale) > 1:
        resp = rescale(resp, 1 / scale, anti_aliasing=True, multichannel=False,
                       mode='reflect')  # The antialiasing method is the best for downscaling

    return np.array(resp)


def rescale_img(img, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    new_size = (width, height)
    img_scaled = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)  # INTER_CUBIC#
    return img_scaled


def apply_filterbank(img, filterbank, scale, resp_type='complex', non_linear=False, smooth=False):
    num_cores = -1

    if int(scale) > 1:
        img_scaled = rescale_img(img, scale)
    else:
        img_scaled = img

    responses = Parallel(n_jobs=num_cores)(
        delayed(applyFilter)(img_scaled, g_filter, scale, resp_type, non_linear, smooth) for g_filter in filterbank)

    return np.array(responses)


def gabor_bank(freq=0.1, n_angles=6, n_scales=4, sigma=10):

    angles = np.deg2rad(np.linspace(0, 180, n_angles, endpoint=False))
    scales = [1, 3, 5, 7]  #(np.arange(n_scales) + 1)  #
    gk = []
    for s in scales:
        for a in angles:
            gk.append(gabor_kernel(frequency=freq, theta=a, sigma_x=sigma, sigma_y=sigma, n_stds=s))

    return np.array(gk)


def convolve_gabor(image, kernel):
    image = (image - image.mean()) / image.std()
    gabor_reponse = np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 + ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
    gabor_reponse = np.log(1 + gabor_reponse)  # Smoothing strategy
    return gabor_reponse - gabor_reponse.mean()


def img2complex_colorspace(img, color_space):
    #     img = img_as_float32(img.astype('uint8')) # The input image must be float32 between [0, 1]

    if color_space == 'HV':
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        ch1 = np.deg2rad(img_hsv[:, :, 0] * 2)  # Change the H channel from deg to rag
        ch2 = img_hsv[:, :, 1]  # * 100. # Change the S channel between [0, 100]
        ch3 = img_hsv[:, :, 2]  # * 100. # Change the V channel between [0, 100]

        luminance = ch2
        chrominance = ch3 * np.exp(1j * ch1)

        cs = 'HSV'

    if color_space == 'HS':
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        ch1 = np.deg2rad(img_hsv[:, :, 0] * 2)  # Change the H channel from deg to rag
        ch2 = img_hsv[:, :, 1]  # * 100. # Change the S channel between [0, 100]
        ch3 = img_hsv[:, :, 2]  # * 100. # Change the V channel between [0, 100]

        luminance = ch3
        chrominance = ch2 * np.exp(1j * ch1)

        cs = 'HSV'

    if color_space == 'LAB':
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        ch1 = img_lab[:, :, 0]  # L channel between [0, 100]
        ch2 = img_lab[:, :, 1]  # a channel between [-127, 127]
        ch3 = img_lab[:, :, 2]  # b channel between [-127, 127]

        luminance = ch1
        chrominance = ch2 + (1j * ch3)

        cs = 'LAB'

    #     fig, axs = plt.subplots(1, 3, figsize=(9, 3), dpi=300)
    #     axs[0].imshow(ch1, cmap='gray')
    #     axs[1].imshow(ch2, cmap='gray')
    #     axs[2].imshow(ch3, cmap='gray')
    #     fig.suptitle('Image channels in '+ cs +' color space')

    return luminance, chrominance


def gabor_saliency_model(image, freq=0.1, n_angles=6, n_scales=4, sigma=10):
    num_cores = multiprocessing.cpu_count()
    color_space = 'LAB'
    lum, chrom = img2complex_colorspace(image, color_space)

    # Filterbank parameters
    scale = 1.
    min_period = 2.
    max_period = 25.
    bandwidth = 1
    n_angles = 5
    gamma = eta = 1  # 0.55 # None  #
    p1 = p2 = None  # 0.9  #
    n_stds = 1.

    gabor_filters, frequencies, angles = makeGabor_filterbank(scale, min_period, max_period, bandwidth, n_angles, p1=p1,
                                                              p2=p2, gamma=gamma, eta=eta, n_stds=n_stds)

    g_responses = apply_filterbank(chrom, gabor_filters, scale, resp_type='complete', non_linear=True, smooth=True)
    pdb.set_trace()
    # gabor_kernels = gabor_bank(freq, n_angles, n_scales, sigma)

    # gabor_responses = np.array(Parallel(n_jobs=num_cores, require='sharedmem')(
    #     delayed(convolve_gabor)(image, kernel) for kernel in gabor_kernels))
    # gabor_model = gabor_responses.sum(axis=0)
    gabor_model = np.abs(g_responses.sum(axis=0))
    pdb.set_trace()
    return gabor_model


def update_edges_weight(img, regions, rag, convert2lab, texture, n_bins, method):
    """
    Obtain 3D color histogram of each superpixel region, then it computes the color distance between neighbor regions.
    :param img: Input image in RGB
    :param regions: label of each region
    :param rag: Region adjacency graph of the image
    :param convert2lab: Boolean parameter to indicate if work on the LAB color space. If false it work on RGB color space
    :param n_bins: Number of bins per color axe to quantize the color image
    :return: Regions adjacency graph with the edges weights updated
    """

    if convert2lab:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        if texture:
            print('Computing Texture Model')
            text_model = gabor_saliency_model(img, n_angles=6, n_scales=4)#[:, :, 0]
            plt.figure()
            plt.imshow(text_model)


    regions_ids = np.unique(regions)
    num_cores = multiprocessing.cpu_count()
    rag_weighted = rag.copy()

    if method == 'OT':
        # Get 3d color histograms
        hist = np.array(Parallel(n_jobs=num_cores, require='sharedmem')(delayed(color_3dhistogram)(img[regions == i], n_bins) for i in regions_ids))

        # Compute the Optimal Transport (EMD) between neighbor regions
        for e in list(rag.edges()):
            cost_matrix = ot.dist(np.array(hist[e[0]][1], dtype='int').T, np.array(hist[e[1]][1], dtype='int').T, 'sqeuclidean')
            dist = ot.emd2(hist[e[0]][0], hist[e[1]][0], cost_matrix, processes=num_cores)
            dist += np.abs(np.mean(text_model[np.where(regions == e[0])]) - np.mean(text_model[np.where(regions == e[1])])) * 100
            pdb.set_trace()

            rag_weighted[e[0]][e[1]]['weight'] = dist

    if method == 'KL':
        # Get 3d color histograms
        hist = np.array(Parallel(n_jobs=num_cores, require='sharedmem')(delayed(color_histogram)(img[regions == i], n_bins) for i in regions_ids), dtype=np.float32)

        # # Compute the Optimal Transport (EMD) between neighbor regions
        # for e in list(rag.edges()):
        #     divergence = cv2.compareHist(hist[e[0]], hist[e[1]], 5)
        #     rag_weighted[e[0]][e[1]]['weight'] = divergence

        # Compute the Optimal Transport (EMD) between neighbor regions
        for e in list(rag.edges()):
            divergence = cv2.compareHist(hist[e[0]], hist[e[1]], 5)
            rag_weighted[e[0]][e[1]]['weight'] = divergence

    return rag_weighted


def get_graph(img, regions, graph_type, convert2lab, neighbors, radius):

    if convert2lab:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    if graph_type == 'rag':
        graph = rag_networkx(regions)
        graph = set_node_edge_attr(graph, img, regions)

    if graph_type == 'complete':
        graph = graph_complete_networkx(regions)
        graph = set_node_edge_attr(graph, img, regions)

    if graph_type == 'knn':
        graph = graph_knn(regions, neighbors)
        graph = set_node_edge_attr(graph, img, regions)

    if graph_type == 'eps':
       graph = graph_epsilon(regions, radius)
       graph = set_node_edge_attr(graph, img, regions)

    return graph


def graph_epsilon(regions, radius):
    region_props = regionprops(regions + 1)
    centroid = np.array([region_props[ii].centroid for ii in range(len(region_props))])
    knn_mat = radius_neighbors_graph(centroid, radius, mode='connectivity', include_self=False)
    epsilon_graph = nx.from_scipy_sparse_matrix(knn_mat, 'weight')

    return epsilon_graph


def graph_knn(regions, neighbors):
    region_props = regionprops(regions + 1)
    centroid = np.array([region_props[ii].centroid for ii in range(len(region_props))])
    knn_mat = kneighbors_graph(centroid, neighbors, mode='connectivity', include_self=False)
    knn_graph = nx.from_scipy_sparse_matrix(knn_mat, 'weight')

    return knn_graph


def set_node_edge_attr(graph, img, regions):
    region_props = regionprops(regions + 1)

    for n in graph:

        graph.nodes[n].update({'labels': [n],
                            'pixel count': 0,
                            'total color': np.array([0, 0, 0], dtype=np.double),
                            'centroid': np.array([0, 0], dtype=np.double)})

    for index in np.ndindex(regions.shape):
        current = regions[index]
        graph.nodes[current]['pixel count'] += 1
        graph.nodes[current]['total color'] += img[index]

    for n in graph:
        graph.nodes[n]['mean color'] = (graph.nodes[n]['total color'] / graph.nodes[n]['pixel count'])
        graph.nodes[n]['centroid'] = (region_props[n].centroid)

    nx.set_edge_attributes(graph, 1, 'weight')
    # nx.set_edge_attributes(rag, 1, 'similarity')

    return graph


def rag_networkx(regions):
    rag = grph.RAG(regions, connectivity=2)
    print('Number of edges:', rag.number_of_edges())

    return rag


def graph_complete_networkx(regions):
    complete_graph = nx.complete_graph(len(np.unique(regions)))
    print('Number of edges:', complete_graph.number_of_edges())

    return complete_graph


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