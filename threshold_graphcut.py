import os
import time
import datetime

from computation_support import *
from metrics import *
from groundtruth import *
from myGaborFunctions import *
from color_transformations import *
from graph_operations import *
from plot_save_figures import *
from color_seg_methods import *
import pdb

if __name__ == '__main__':

    indir = '../data/myFavorite_BSDimages/dir/'
    in_imgs = os.listdir(indir)

    # Generate Gabor filter bank: parameters
    min_period = 2.
    max_period = 25.
    fb = 0.7  #1  #
    ab = 30  #45 #
    c1 = 0.65
    c2 = 0.65
    stds = 3.0
    print('Generating Gabor filterbank')
    gabor_filters, frequencies, angles = makeGabor_filterbank(min_period, max_period, fb, ab, c1, c2, stds)

    n_freq = len(frequencies)
    n_angles = len(angles)

    # Compute ground distance matrix
    ground_distance = cost_matrix_texture(n_freq, n_angles)

    for im_file in in_imgs:
        time_total = time.time()

        print('##############################', im_file, '##############################')

        ''' Read and preparing the image '''
        img = io.imread(indir + im_file)
        rows, cols, channels = img.shape
        # img = cv2.imread(indir + im_file, cv2.CV_8S)
        # img = img_preparation(img)

        ''' Computing superpixel regions '''
        # Superpixels function parameters
        n_regions = 500 * 8
        convert2lab = True
        texture = False
        regions = slic_superpixel(img, n_regions, convert2lab)

        ''' Computing Graph '''
        # Graph function parameters
        graph_type = 'knn'  # Choose: 'complete', 'knn', 'rag'
        kneighbors = 8
        radius = 10

        graph = get_graph(img, regions, graph_type, kneighbors, radius)

        # Complex image transformation parameters
        color_space = 'HS'
        img_complex = img2complex_normalized_colorspace(img, (rows, cols, channels), color_space)

        # output_url += '_%dfreq_%dang' % (n_freq, n_angles)

        # Gabor image transformation parameters
        r_type = 'L2'  # 'real'
        gsmooth = True
        opn = True
        selem_size = 1
        num_cores = -1
        gabor_responses = np.array(Parallel(n_jobs=num_cores, prefer='processes')(
        delayed(applyGabor_filterbank)(img_channel, gabor_filters, resp_type=r_type, smooth=gsmooth,
                                       morph_opening=opn, se_z=selem_size) for img_channel in img_complex))

        g_responses_norm = normalize_img(gabor_responses, rows, cols)
        g_energies = np.moveaxis(np.array(g_responses_norm), 0, -1)
        g_energies_sum = np.moveaxis(np.sum(g_energies, axis=-1), 0, -1)

        # # 3D Region histogram parameters
        # n_bins = 8

        ''' Updating edges weights with optimal transport '''
        method = 'OT'
        graph_weighted = update_edges_weight(regions, graph, g_energies_sum, ground_distance, method)
        weights = nx.get_edge_attributes(graph_weighted, 'weight').values()

        ''' Computing Minimum Spanning Tree '''
        graph_mst = get_mst(graph_weighted)
        weights_mst = nx.get_edge_attributes(graph_mst, 'weight').values()

        ''' FIRST METHOD: Lognorm distribution threshold: Graph cut on complete RAG '''
        # First method parameters
        cut_level = 0.9  # set threshold at the 99.9% quantile level
        graph_aftercut, thresh, params = threshold_graphcut(graph_weighted, cut_level, regions)
        regions_afercut = graph2regions(graph_aftercut, regions)

        ''' SECOND METHOD: Lognorm distribution threshold: Graph cut on complete RAG '''
        # Second method parameters
        rag_mst_aftercut, thresh_mst, params_mst = threshold_graphcut(graph_mst, cut_level, regions)
        regions_mst_aftercut = graph2regions(rag_mst_aftercut, regions)

        groundtruth_segments = np.array(get_segment_from_filename(im_file))

        # Evaluate metrics
        m = metrics(None, regions_mst_aftercut, groundtruth_segments)
        m.set_metrics()
        # m.display_metrics()

        metrics_values = m.get_metrics()
        ##############################################################################
        '''Visualization Section: show and/or save images'''
        # General Params
        save_fig = True
        fontsize = 20
        file_name = im_file[:-4]

        outdir = 'outdir/' + method + '/graph_' + graph_type + '/threshold_graphcut/computation_support/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Show Input image
        fig_title = 'Input Image'
        show_and_save_img(img, fig_title, fontsize, save_fig, outdir, file_name)

        # Show SLIC result
        fig_title = 'Superpixel Regions'
        img_name = '_slic'
        show_and_save_regions(img, regions, fig_title, img_name, fontsize, save_fig, outdir, file_name)

        # Show Graph with uniform weight
        fig_title = 'Graph (' + graph_type + ')'
        img_name = '_raw_' + graph_type
        colbar_lim = (0, 1)
        show_and_save_imgraph(img, regions, graph, fig_title, img_name, fontsize, save_fig, outdir, file_name, colbar_lim)

        # Show Graph with updated weights
        fig_title = 'Weighted Graph (' + graph_type + ')'
        img_name = '_weighted_' + graph_type
        colbar_lim = (min(weights), max(weights))
        show_and_save_imgraph(img, regions, graph_weighted, fig_title, img_name, fontsize, save_fig, outdir, file_name, colbar_lim)

        # Show MST Graph with updated weights
        fig_title = 'MST Graph (' + graph_type + ')'
        img_name = '_mst_' + graph_type
        colbar_lim = (min(weights), max(weights))
        show_and_save_imgraph(img, regions, graph_mst, fig_title, img_name, fontsize, save_fig, outdir, file_name, colbar_lim)

        # # Show one SLIC region and its neighbors
        # region = 109
        # show_and_save_some_regions(img, regions, region, rag, save_fig, outdir, file_name)

        ##############################################################################
        # First method visualization section
        outdir = 'outdir/' + method + '/graph_' + graph_type + '/threshold_graphcut/results/'

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # # Edges weight distribution
        # fig_title = 'Edges Weight Distribution'
        # img_name = '_weight_dist'
        # show_and_save_dist(weights, thresh, params, fig_title, img_name, fontsize, save_fig, outdir, file_name)

        # RAG after cut
        fig_title = 'RAG after cut'
        img_name = '_thr_graph_aftercut'
        colbar_lim = (min(weights), max(weights))
        show_and_save_imgraph(img, regions, graph_aftercut, fig_title, img_name, fontsize, save_fig, outdir, file_name, colbar_lim)

        fig_title = 'Segmentation Result '
        img_name = '_graphcut_result'
        show_and_save_result(img, regions_afercut, fig_title, img_name, fontsize, save_fig, outdir, file_name)

        # f = open(outdir + '00-params_setup.txt', 'wb')
        # f.write('THRESHOLD GRAPHCUT <PARAMETER SETUP> <%s> \n \n' % datetime.datetime.now())
        # f.write('Superpixels function parameters: \n')
        # f.write(' n_regions = %i ~ %i \n convert2lab = %r \n\n' % (n_regions, len(np.unique(regions)), convert2lab))
        # f.write('Graph function parameters: \n')
        # if graph_type == 'knn':
        #     f.write(' graph_type = %s \n kneighbors = %i \n\n' % (graph_type, kneighbors))
        # elif graph_type == 'eps':
        #     f.write(' graph_type = %s \n radius = %i \n\n' % (graph_type, radius))
        # else:
        #     f.write(' graph_type = %s \n\n' % graph_type)
        # f.write('Color quantification parameters: \n')
        # f.write(' method = %s \n n_bins = %i \n\n' % (method, n_bins))
        # f.write('Particular parameters: \n')
        # f.write(' cut_level = %i ' % (cut_level * 100))
        # f.close()
        ##############################################################################
        # Second method visualization section
        outdir = 'outdir/' + method + '/graph_' + graph_type + '/threshold_graphcut/results_mst/'

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # # Edges weight distribution
        # fig_title = 'Edges Weight Distribution (MST)'
        # img_name = '_mst_weight_dist'
        # show_and_save_dist(weights_mst, thresh_mst, params_mst, fig_title, img_name, fontsize, save_fig, outdir, file_name)

        # RAG after cut
        fig_title = 'RAG after cut (MST)'
        img_name = '_mst_rag_aftercut'
        colbar_lim = (min(weights_mst), max(weights_mst))
        show_and_save_imgraph(img, regions, rag_mst_aftercut, fig_title, img_name, fontsize, save_fig, outdir, file_name, colbar_lim)

        fig_title = 'Segmentation Result (MST) '
        img_name = '_graphcut_mst_result'
        show_and_save_result(img, regions_mst_aftercut, fig_title, img_name, fontsize, save_fig, outdir, file_name)

        # f = open(outdir + '00-params_setup.txt', 'wb')
        # f.write('THRESHOLD GRAPHCUT <PARAMETER SETUP> <%s> \n \n' % datetime.datetime.now())
        # f.write('Superpixels function parameters: \n')
        # f.write(' n_regions = %i ~ %i \n convert2lab = %r \n\n' % (n_regions, len(np.unique(regions)), convert2lab))
        # f.write('Graph function parameters: \n')
        # if graph_type == 'knn':
        #     f.write(' graph_type = %s \n kneighbors = %i \n\n' % (graph_type, kneighbors))
        # elif graph_type == 'eps':
        #     f.write(' graph_type = %s \n radius = %i \n\n' % (graph_type, radius))
        # else:
        #     f.write(' graph_type = %s \n\n' % graph_type)
        # f.write('Color quantification parameters: \n')
        # f.write(' method = %s \n n_bins = %i \n\n' % (method, n_bins))
        # f.write('Particular parameters: \n')
        # f.write(' cut_level = %i ' % (cut_level * 100))
        # f.close()

        # plt.show()
        # plt.close('all')

