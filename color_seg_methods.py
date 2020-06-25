import numpy as np
import networkx as nx
import itertools
import pandas as pd
import matplotlib.pyplot as plt

import pdb
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from scipy.stats import gamma, lognorm
from skimage.future import graph
from sklearn.cluster import SpectralClustering, KMeans


# from gap import gap


def threshold_graphcut(rag, cut_level, regions):
    print('\n ### Gamma threshold graph cut:')

    weights = list(nx.get_edge_attributes(rag, 'weight').values())
    # weights = np.array(weights, dtype=float)
    print('Max edge weight:', max(weights), 'Min edge weight:', min(weights))
    rag_aftercut = rag.copy()

    # fit
    # params = lognorm.fit(weights)#, floc=0
    params = gamma.fit(weights, loc=0)#, floc=0
    print(params)
    # print params
    # thresh = lognorm.ppf(cut_level, *params)
    thresh = gamma.ppf(cut_level, *params)
    # thresh = gamma.ppf(cut_level, *params)
    print('Threshold:', thresh)
    graph.cut_threshold(regions, rag_aftercut, thresh, in_place=True)
    print('Number of edges after cut:', rag_aftercut.number_of_edges())

    return rag_aftercut, thresh, params


def graph2regions(rag, regions):
    comps = nx.connected_components(rag)
    map_array = np.arange(regions.max() + 1, dtype=regions.dtype)
    for i, nodes in enumerate(comps):
        for node in nodes:
            for label in rag.nodes[node]['labels']:
                map_array[label] = i

    new_regions = map_array[regions]
    print('Number of regions after graph cut:', len(np.unique(new_regions)))

    return new_regions


def spectral_clustering(rag, weights, sigma_method, regions):
    print('\n ### Spectral Clustering:')

    mean_color = np.array(dict(rag.nodes(data='mean color')).values())
    centroid = np.array(dict(rag.nodes(data='centroid')).values())
    node_features = np.column_stack((centroid, mean_color))

    # print node_features / np.std(node_features, axis=0)
    # print 'Region features dimension:', node_features.shape

    # gaps, s_k, K = gap.gap_statistic(node_features / np.std(node_features, axis=0), refs=None, B=20, K=range(1, 35), N_init=10)
    # bestKValue = gap.find_optimal_k(gaps, s_k, K)
    # print('Optimal number of clusters GAP-KMEANS:', bestKValue)
    #
    # if bestKValue == 1:
    #     # gaps, s_k, K = gap.gap_statistic(mean_color, refs=None, B=10, K=range(1, 35), N_init=10)
    #     bestKValue = 25
    #     print('Optimal number of clusters GAP-KMEANS:', bestKValue)

    opt_k, gapdisp = optimalK(node_features, 10, 30)
    print('Optimal number of clusters GAP-STAT:', opt_k)
    bestKValue = opt_k

    # pixel_colors = mean_color
    # norm = colors.Normalize(vmin=-1., vmax=1.)
    # norm.autoscale(pixel_colors)
    # pixel_colors = norm(pixel_colors).tolist()
    # fig = plt.figure()
    # axis = fig.add_subplot(1, 1, 1, projection="3d")
    # axis.scatter(mean_color[:, 0], mean_color[:, 1], mean_color[:, 2], marker="o", edgecolor='k')
    # plt.show()

    if sigma_method == 'global':

        sigma = global_sigma(weights, len(np.unique(regions)))
        print('sigma:', sigma)

        adj_mat = nx.adjacency_matrix(rag, weight='weight')
        aff_mat = adj_mat.copy()
        aff_mat.data = np.exp(- np.square(adj_mat.data) / np.square(sigma))
        # aff_mat.data = np.power(np.exp(- adj_mat.data / (sigma / 2)), 3)  # 3 is the dimension data (3d color space)
        clustering = SpectralClustering(bestKValue, assign_labels='discretize', affinity='precomputed', n_init=100, n_jobs=-1).fit(aff_mat)

    elif sigma_method == 'local':

        aff_mat = local_sigma(rag)
        clustering = SpectralClustering(bestKValue, assign_labels='discretize', affinity='precomputed', n_init=100, n_jobs=-1).fit(aff_mat)

    print('Number of regions after clustering:', len(np.unique(clustering.labels_)))

    rag_clustering = rag.copy()
    nx.set_node_attributes(rag_clustering, dict(enumerate(clustering.labels_)), 'labels')
    map_array = np.array(nx.get_node_attributes(rag_clustering, 'labels').values(), dtype=regions.dtype)
    new_regions = map_array[regions]

    return rag_clustering, new_regions, aff_mat


def optimalK(data, nrefs, maxClusters):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    pdb.set_trace()
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_

        # Calculate gap statistic
        #         gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        gap = np.mean(np.log(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

    return (gaps.argmax() + 1,
            resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal


def global_sigma(distances, n_points):
    return max(distances) / (n_points ** (1/np.float(3)))  # 3 is the dimension data (3d color space)


def local_sigma(rag):

    adj_mat = nx.adjacency_matrix(rag, weight='weight')
    aff_mat = adj_mat.copy()
    aff = []
    aff_mat = aff_mat.tocoo()
    for i, j, v in zip(aff_mat.row, aff_mat.col, aff_mat.data):
        dist_i = np.percentile(aff_mat.data[aff_mat.row == i], [25, 50, 75])
        dist_j = np.percentile(aff_mat.data[aff_mat.row == j], [25, 50, 75])
        dist_i[1] = dist_i[1] * 2
        dist_j[1] = dist_j[1] * 2
        sigma_i = dist_i.sum() / 4
        sigma_j = dist_j.sum() / 4

        aff.append(np.exp(-np.square(v) / (sigma_i * sigma_j)))

    aff_mat.data = np.array(aff)

    return aff_mat


def normalized_graphcut(rag, weights, sigma_method, regions):
    print('\n ### Normalized Graph Cut:')

    if sigma_method == 'global':

        sigma = global_sigma(weights, len(np.unique(regions)))
        print('sigma:', sigma)

        adj_mat = nx.adjacency_matrix(rag, weight='weight')
        aff_mat = adj_mat.copy()
        # aff_mat.data = np.exp(- np.square(adj_mat.data) / np.square(sigma))
        aff_mat.data = np.power(np.exp(- adj_mat.data / (sigma / 2)), 3)  # 3 is the dimension data (3d color space)

    elif sigma_method == 'local':

        aff_mat = local_sigma(rag)

    rag_normalized = nx.from_scipy_sparse_matrix(aff_mat)
    for i in np.unique(regions):
        rag_normalized.nodes[i]['labels'] = [i]

    new_regions = graph.cut_normalized(regions, rag_normalized)

    return rag_normalized, new_regions, aff_mat