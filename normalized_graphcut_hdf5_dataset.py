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
            graph_type = 'rag'  # Choose: 'complete', 'knn', 'rag'
            kneighbors = 4
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

                ''' FIFTH METHOD: Normalized cut on complete RAG '''
                # Fifth method parameters
                sigma_method = 'global'  # Choose 'global' or 'local'
                graph_ncut, regions_ncut, aff_matrix = normalized_graphcut(graph_weighted, weights, sigma_method, regions)

                ''' SIXTH METHOD: Spectral Clustering on MST RAG '''
                # Sixth method parameters
                graph_mst_ncut, regions_mst_ncut, aff_matrix_mst = normalized_graphcut(graph_weighted, weights, sigma_method, regions)

                groundtruth_segments = np.array(get_segment_from_filename(im_file))

                # Evaluate metrics
                if len(np.unique(regions_mst_ncut)) == 1:
                    metrics_values.append((0., 0.))
                else:
                    m = metrics(None, regions_mst_ncut, groundtruth_segments)
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

                # outdir = 'outdir/' + num_imgs_dir + input_file + '/' + method + '/graph_' + graph_type + '/normalized_graphcut/computation_support/'
                outdir = 'outdir/' + num_imgs_dir + 'normalized_graphcut/' + method + '/graph_' + graph_type + '/' + features_input_file[:-3] + '/computation_support/'

                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                # Show Input image
                fig_title = 'Input Image'
                show_and_save_img(img, fig_title, fontsize, save_fig, outdir, im_file)

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

                ##############################################################################
                # Third method visualization section
                if sigma_method == 'global':
                    outdir = 'outdir/' + num_imgs_dir + 'normalized_graphcut/' + method + '/graph_' + graph_type + '/' + features_input_file[:-3] + '/global_sigma/results/'

                elif sigma_method == 'local':
                    outdir = 'outdir/' + num_imgs_dir + 'normalized_graphcut/' + method + '/graph_' + graph_type + '/' + features_input_file[:-3] + '/local_sigma/results/'

                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                fig_title = 'Affinity Matrix'
                img_name = '_aff_mat'
                show_and_save_affmat(aff_matrix, fig_title, img_name, fontsize, save_fig, outdir, file_name)

                fig_title = 'Normalized GraphCut Result '
                img_name = '_ncut_result'
                show_and_save_result(img, regions_ncut, fig_title, img_name, fontsize, save_fig, outdir, file_name)

                # f = open(outdir + '00-params_setup.txt', 'wb')
                # f.write('NORMALIZED GRAPHCUT <PARAMETER SETUP> <%s> \n \n' % datetime.datetime.now())
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
                # f.write(' sigma_method = %s ' % sigma_method)
                # f.close()
                #############################################################################
                # Forth method visualization section
                if sigma_method == 'global':
                    outdir = 'outdir/' + num_imgs_dir + 'normalized_graphcut/' + method + '/graph_' + graph_type + '/' + features_input_file[:-3] + '/global_sigma/results_mst/'

                elif sigma_method == 'local':
                    outdir = 'outdir/' + num_imgs_dir + 'normalized_graphcut/' + method + '/graph_' + graph_type + '/' + features_input_file[:-3] + '/local_sigma/results_mst/'

                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                fig_title = 'Affinity Matrix (MST)'
                img_name = '_aff_mat_mst'
                show_and_save_affmat(aff_matrix_mst, fig_title, img_name, fontsize, save_fig, outdir, file_name)

                fig_title = 'Normalized GraphCut Result (MST)'
                img_name = '_mst_ncut_result'
                show_and_save_result(img, regions_mst_ncut, fig_title, img_name, fontsize, save_fig, outdir, file_name)

                # f = open(outdir + '00-params_setup.txt', 'wb')
                # f.write('NORMALIZED GRAPHCUT <PARAMETER SETUP> <%s> \n \n' % datetime.datetime.now())
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
                # f.write(' sigma_method = %s ' % sigma_method)
                # f.close()
                # plt.show()
                plt.close('all')

            if sigma_method == 'global':
                outdir = 'outdir/' + num_imgs_dir + 'normalized_graphcut/' + method + '/graph_' + graph_type + '/' + features_input_file[:-3] + '/global_sigma/results/'
            elif sigma_method == 'local':
                outdir = 'outdir/' + num_imgs_dir + 'normalized_graphcut/' + method + '/graph_' + graph_type + '/' + features_input_file[:-3] + '/local_sigma/results/'

            if not os.path.exists(outdir):
                os.makedirs(outdir)
            metrics_values = np.array(metrics_values)
            recall = metrics_values[:, 0]
            precision = metrics_values[:, 1]

            plt.figure(dpi=180)
            plt.plot(np.arange(len(image_vectors)) + 1, recall, '-o', c='k', label='recall')
            plt.plot(np.arange(len(image_vectors)) + 1, precision, '-o', c='r', label='precision')
            plt.title('Thr graphcut P/R histogram')
            plt.xlabel(
                'Rmax: %.3f, Rmin: %.3f, Rmean: %.3f, Rmed: %.3f, Rstd: %.3f \n Pmax: %.3f, Pmin: %.3f, Pmean: %.3f, Pmed: %.3f, Pstd: %.3f ' % (
                    recall.max(), recall.min(), recall.mean(), np.median(recall), recall.std(), precision.max(),
                    precision.min(), precision.mean(), np.median(precision), precision.std()))
            plt.ylim(0, 1.05)
            plt.legend()
            plt.grid()
            plt.savefig(outdir + 'Normalized_graphcut_PR_hist.png', bbox_inches='tight')

            plt.close('all')