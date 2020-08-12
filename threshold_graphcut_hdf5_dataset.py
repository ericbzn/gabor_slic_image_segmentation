import os
import time
import datetime
from pathlib import Path

import h5py

from computation_support import *
from myGaborFunctions import *
from color_transformations import *
from metrics import *
from groundtruth import *
from graph_operations import *
from plot_save_figures import *
from color_seg_methods import *
import pdb

if __name__ == '__main__':
    num_cores = -1

    num_imgs = 7

    hdf5_dir = Path('../data/hdf5_datasets/')

    if num_imgs is 500:
        # Path to whole Berkeley image data set
        hdf5_indir_im = hdf5_dir / 'complete' / 'images'
        hdf5_indir_feat = hdf5_dir / 'complete' / 'features'
        num_imgs_dir = 'complete/'

    elif num_imgs is 7:
        # Path to my 7 favourite images from the Berkeley data set
        hdf5_indir_im = hdf5_dir / '7images/' / 'images'
        hdf5_indir_feat = hdf5_dir / '7images/' / 'features'
        num_imgs_dir = '7images/'

    print('Reading Berkeley image data set')
    t0 = time.time()
    # Read hdf5 file and extract its information
    images_file = h5py.File(hdf5_indir_im / "Berkeley_images.h5", "r+")
    image_vectors = np.array(images_file["/images"])
    img_shapes = np.array(images_file["/image_shapes"])
    img_ids = np.array(images_file["/image_ids"])

    images = Parallel(n_jobs=num_cores)(
        delayed(np.reshape)(img, (shape[0], shape[1], shape[2])) for img, shape in zip(image_vectors, img_shapes))

    t1 = time.time()
    print('Reading hdf5 image data set time: %.2fs' % (t1 - t0))

    input_files = os.listdir(hdf5_indir_feat)
    for features_input_file in input_files:
        with h5py.File(hdf5_indir_feat / features_input_file, "r+") as features_file:
            print('Reading Berkeley features data set')
            print('File name: ', features_input_file)
            t0 = time.time()
            feature_vectors = np.array(features_file["/gabor_features"])
            feature_shapes = np.array(features_file["/feature_shapes"])

            features = Parallel(n_jobs=num_cores)(
                delayed(np.reshape)(features, (shape[0], shape[1])) for features, shape in
                zip(feature_vectors, feature_shapes))
            t1 = time.time()
            print('Reading hdf5 features data set time: %.2fs' % (t1 - t0))

            n_freq = features_file.attrs['num_freq']
            n_angles = features_file.attrs['num_angles']

            # Compute ground distance matrix
            ground_distance = cost_matrix_texture(n_freq, n_angles)

            # Superpixels function parameters
            n_regions = 500 * 8
            convert2lab = True
            texture = False

            # Graph function parameters
            graph_type = 'knn'  # Choose: 'complete', 'knn', 'rag'
            kneighbors = 8
            radius = 10

            gabor_features_norm = Parallel(n_jobs=num_cores)(
                delayed(np.reshape)(features, (shape[0], shape[1], n_freq * n_angles, shape[2])) for features, shape in
                zip(feature_vectors, img_shapes))

            metrics_values = []
            for im_file, img, g_energies in zip(img_ids, images, gabor_features_norm):
                time_total = time.time()

                print('##############################', im_file, '##############################')

                ''' Computing superpixel regions '''
                regions = slic_superpixel(img, n_regions, convert2lab)

                ''' Computing Graph '''
                graph = get_graph(img, regions, graph_type, kneighbors, radius)

                g_energies_sum = np.sum(g_energies, axis=-1)

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
                vals = m.get_metrics()
                metrics_values.append((vals['recall'], vals['precision']))

                ##############################################################################
                '''Visualization Section: show and/or save images'''
                # General Params
                save_fig = True
                fontsize = 20
                file_name = im_file

                outdir = 'outdir/' + num_imgs_dir + 'threshold_graphcut/' + method + '/graph_' + graph_type + '/' + features_input_file[:-3] + '/computation_support/'
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
                # outdir = 'outdir/' + num_imgs_dir + input_file + '/' + method + '/graph_' + graph_type + '/threshold_graphcut/results/'
                outdir = 'outdir/' + num_imgs_dir + 'threshold_graphcut/' + method + '/graph_' + graph_type + '/' + features_input_file[:-3] + '/results/'


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
                # outdir = 'outdir/' + num_imgs_dir + input_file + '/' + method + '/graph_' + graph_type + '/threshold_graphcut/results_mst/'
                outdir = 'outdir/' + num_imgs_dir + 'threshold_graphcut/' + method + '/graph_' + graph_type + '/' + features_input_file[:-3] + '/results_mst/'


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
                plt.close('all')

            # outdir = 'outdir/' + num_imgs_dir + features_input_file[:-3] + '/' + method + '/graph_' + graph_type + '/threshold_graphcut/results_mst/'
            outdir = 'outdir/' + num_imgs_dir + 'threshold_graphcut/' + method + '/graph_' + graph_type + '/' + features_input_file[:-3] + '/results_mst'

            metrics_values = np.array(metrics_values)
            recall = metrics_values[:, 0]
            precision = metrics_values[:, 1]

            plt.figure(dpi=180)
            plt.plot(np.arange(len(image_vectors)) + 1, recall, '-o', c='k', label='recall')
            plt.plot(np.arange(len(image_vectors)) + 1, precision, '-o', c='r', label='precision')
            plt.title('Thr graphcut P/R histogram')
            plt.xlabel('Rmax: %.3f, Rmin: %.3f, Rmean: %.3f, Rmed: %.3f, Rstd: %.3f \n Pmax: %.3f, Pmin: %.3f, Pmean: %.3f, Pmed: %.3f, Pstd: %.3f ' % (
                    recall.max(), recall.min(), recall.mean(), np.median(recall), recall.std(), precision.max(),
                    precision.min(), precision.mean(), np.median(precision), precision.std()))
            plt.ylim(0, 1.05)
            plt.legend()
            plt.grid()
            plt.savefig(outdir + 'Thr_graphcut_PR_hist.png', bbox_inches='tight')

            plt.close('all')

