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
            n_regions = 500 * 4
            convert2lab = True
            texture = False

            # Graph function parameters
            graph_type = 'knn'  # Choose: 'complete', 'knn', 'rag'
            kneighbors = 8
            radius = 10

            # Segmentation parameters
            method = 'OT'  # Choose: 'OT' for Earth Movers Distance or 'KL' for Kullback-Leiber divergence
            aff_norm_method = 'global'  # Choose: 'global' or 'local'
            graph_mode = 'mst'  # Choose: 'complete' to use whole graph or 'mst' to use Minimum Spanning Tree

            gabor_features_norm = Parallel(n_jobs=num_cores)(
                delayed(np.reshape)(features, (shape[0], shape[1], n_freq * n_angles, shape[2])) for features, shape in
                zip(feature_vectors, img_shapes))

            metrics_values = []
            for im_file, img, g_energies in zip(img_ids, images, gabor_features_norm):
                time_total = time.time()

                print('##############################', im_file, '##############################')

                ''' Computing superpixel regions '''
                regions_slic = slic_superpixel(img, n_regions, convert2lab)

                ''' Computing Graph '''
                graph_raw = get_graph(img, regions_slic, graph_type, kneighbors, radius)

                ''' Updating edges weights with similarity measure (OT/KL) '''
                g_energies_sum = np.sum(g_energies, axis=-1)
                graph_weighted = update_edges_weight(regions_slic, graph_raw, g_energies_sum, ground_distance, method)

                if graph_mode == 'complete':
                    weights = nx.get_edge_attributes(graph_weighted, 'weight').values()
                elif graph_mode == 'mst':
                    # Compute Minimum Spanning Tree
                    graph_mst = get_mst(graph_weighted)
                    weights = nx.get_edge_attributes(graph_mst, 'weight').values()
                    graph_weighted = graph_mst

                ''' Performing Spectral Clustering on weighted graph '''
                aff_matrix, graph_normalized = distance_matrix_normalization(graph_weighted, weights, aff_norm_method, regions_slic)

                t0 = time.time()
                segmentation = SpectralClustering(n_clusters=10, assign_labels='discretize', affinity='precomputed',
                                                  n_init=100, n_jobs=-1).fit(aff_matrix)
                t1 = time.time()
                print(' Computing time: %.2fs' % (t1 - t0))
                regions_spec = get_sgmnt_regions(graph_weighted, segmentation.labels_, regions_slic)

                ''' Evaluation of segmentation'''
                groundtruth_segments = np.array(get_segment_from_filename(im_file))
                m = metrics(None, regions_spec, groundtruth_segments)
                m.set_metrics()
                # m.display_metrics()
                vals = m.get_metrics()
                metrics_values.append((vals['recall'], vals['precision']))

                ##############################################################################
                '''Visualization Section: show and/or save images'''
                # General Params
                save_fig = True
                fontsize = 10
                file_name = im_file

                outdir = 'outdir/' + \
                         num_imgs_dir + \
                         'spectral_clustering/' + \
                         method + '/' + \
                         graph_type + '_graph/' + \
                         features_input_file[:-3] + '/' + \
                         aff_norm_method + '_normalization/' + \
                         graph_mode + '_graph/' +\
                         'computation_support/'

                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                # Show Input image
                fig_title = 'Input Image'
                show_and_save_img(img, fig_title, fontsize, save_fig, outdir, file_name)

                # Show SLIC result
                fig_title = 'Superpixel Regions'
                img_name = '_slic'
                show_and_save_regions(img, regions_slic, fig_title, img_name, fontsize, save_fig, outdir, file_name)

                # Show Graph with uniform weight
                fig_title = 'Graph (' + graph_type + ')'
                img_name = '_raw_' + graph_type
                colbar_lim = (0, 1)
                show_and_save_imgraph(img, regions_slic, graph_raw, fig_title, img_name, fontsize, save_fig, outdir, file_name, colbar_lim)

                # Show Graph with updated weights
                fig_title = graph_mode + ' Weighted Graph (' + graph_type + ')'
                img_name = '_weighted_' + graph_type
                colbar_lim = (min(weights), max(weights))
                show_and_save_imgraph(img, regions_slic, graph_weighted, fig_title, img_name, fontsize, save_fig, outdir, file_name, colbar_lim)

                # fig_title = 'Spectral Graph'
                # img_name = '_spec_graph'
                # show_and_save_spectralgraph(graph, fig_title, img_name, fontsize, save_fig, outdir, file_name)

                fig_title = 'Affinity Matrix'
                img_name = '_aff_mat'
                show_and_save_affmat(aff_matrix, fig_title, img_name, fontsize, save_fig, outdir, file_name)
                ##############################################################################
                # Segmentation results visualization
                outdir = 'outdir/' + \
                         num_imgs_dir + \
                         'spectral_clustering/' + \
                         method + '/' + \
                         graph_type + '_graph/' + \
                         features_input_file[:-3] + '/' + \
                         aff_norm_method + '_normalization/' + \
                         graph_mode + '_graph/' + \
                         'results/'

                if not os.path.exists(outdir):
                    os.makedirs(outdir)


                fig_title = 'Spectral Clustering Result '
                fig_label = (vals['recall'], vals['precision'], (t1 - t0))
                img_name = '_spec_result'
                show_and_save_result(img, regions_spec, fig_title, fig_label, img_name, fontsize, save_fig, outdir, file_name)

                # f = open(outdir + '00-params_setup.txt', 'wb')
                # f.write('SPECTRAL CLUSTERING <PARAMETER SETUP> <%s> \n \n' % datetime.datetime.now())
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

            metrics_values = np.array(metrics_values)
            recall = metrics_values[:, 0]
            precision = metrics_values[:, 1]

            plt.figure(dpi=180)
            plt.plot(np.arange(len(image_vectors)) + 1, recall, '-o', c='k', label='recall')
            plt.plot(np.arange(len(image_vectors)) + 1, precision, '-o', c='r', label='precision')
            plt.title('Spectral clustering P/R histogram')
            plt.xlabel('Rmax: %.3f, Rmin: %.3f, Rmean: %.3f, Rmed: %.3f, Rstd: %.3f \n Pmax: %.3f, Pmin: %.3f, Pmean: %.3f, Pmed: %.3f, Pstd: %.3f ' % (
                    recall.max(), recall.min(), recall.mean(), np.median(recall), recall.std(), precision.max(),
                    precision.min(), precision.mean(), np.median(precision), precision.std()))
            plt.ylim(0, 1.05)
            plt.legend()
            plt.grid()
            plt.savefig(outdir + 'Spectral_clustering_PR_hist.png', bbox_inches='tight')

            plt.close('all')
