import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pdb

from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gamma, lognorm
from skimage import segmentation, color, data
from skimage.future import graph


def show_and_save_img(img, title, fontsize, save, outdir, imfile):
    plt.figure(dpi=180)
    plt.title(title, fontsize=fontsize)
    plt.imshow(img)# ,extent=(0, 1, 1, 0)
    plt.tight_layout()
    # plt.axis('tight')
    plt.axis('off')

    if save:
        plt.savefig(outdir + imfile + '.png', transparent=True)


def show_and_save_some_regions(img, regions, region, rag, save, outdir, imfile):
    superpixels = segmentation.mark_boundaries(img, regions, color=(0, 0, 0), mode='thick')
    superp = np.array(superpixels * 255, np.uint8)

    plt.figure(dpi=180)
    plt.imshow(superp[0:140, 380:540, :], extent=(0, 1, 1, 0))
    plt.tight_layout()
    # plt.axis('tight')
    plt.axis('off')

    if save:
        plt.savefig(outdir + imfile + '_slic_zoom' + '.png', transparent=True)

    superpixels = np.array(superpixels * 50, np.uint8)
    superpixels[regions == region] = superpixels[regions == region] * 5.1
    neighbors = list(rag.neighbors(region))

    plt.figure(dpi=180)
    plt.imshow(superpixels[0:140, 380:540, :], extent=(0, 1, 1, 0))
    plt.tight_layout()
    # plt.axis('tight')
    plt.axis('off')

    if save:
        plt.savefig(outdir + imfile + '_slic_%i' % region + '.png', transparent=True)

    for r in neighbors:
        superpixels[regions == r] = superpixels[regions == r] * 5.1

        plt.figure()
        plt.imshow(superpixels[0:140, 380:540, :], extent=(0, 1, 1, 0))
        plt.tight_layout()
        # plt.axis('tight')
        plt.axis('off')

        if save:
            plt.savefig(outdir + imfile + '_slic_%i' % r + '.png', transparent=True)

    fig = plt.figure(dpi=180)
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = img[regions == region]
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(img[regions == region][:, 0], img[regions == region][:, 1], img[regions == region][:, 2], facecolors=pixel_colors,
                 marker="o", edgecolor='k')
    axis.set_xlim(0, 255)
    axis.set_ylim(0, 255)
    axis.set_zlim(0, 255)

    if save:
        plt.savefig(outdir + imfile + '_hist_%i' % region + '.png', bbox_inches='tight')

    for r in neighbors:

        # 3D Histogram region
        pixel_colors = img[regions == r]
        norm = colors.Normalize(vmin=-1., vmax=1.)
        norm.autoscale(pixel_colors)
        pixel_colors = norm(pixel_colors).tolist()

        axis.scatter(img[regions == r][:, 0], img[regions == r][:, 1], img[regions == r][:, 2], facecolors=pixel_colors, marker="o", edgecolor='k')
        axis.set_xlim(0, 255)
        axis.set_ylim(0, 255)
        axis.set_zlim(0, 255)

        if save:
            plt.savefig(outdir + imfile + '_hist_%i' % r + '.png', bbox_inches='tight')


def show_and_save_regions(img, regions, title, name, fontsize, save, outdir, imfile):
    superpixels = segmentation.mark_boundaries(img, regions, color=(0, 0, 0), mode='thick')
    superpixels = np.array(superpixels * 255, np.uint8)
    plt.figure(dpi=180)
    plt.title(title, fontsize=fontsize)
    plt.imshow(superpixels)#, extent=(0, 1, 1, 0)
    plt.tight_layout()
    # plt.axis('tight')
    plt.axis('off')

    if save:
        plt.savefig(outdir + imfile + name + '.png', transparent=True)

    # Plot the SLIC regions using random colors. Useful to identify the number of a region
    # plt.figure()
    # plt.title(title, fontsize=fontsize)
    # plt.imshow(regions[0:140, 380:540], extent=(0, 1, 1, 0))
    # plt.tight_layout()
    # plt.axis('tight')
    # plt.axis('off')


def show_and_save_imgraph(img, regions, rag, title, name, fontsize, save, outdir, imfile, colbar_lim):
    lc = graph.show_rag(regions, rag, img, edge_width=1, img_cmap=None, border_color=None)
    lc.set_clim(colbar_lim[0], colbar_lim[1])
    plt.colorbar(lc, fraction=0.0355, pad=0.02)
    plt.title(title, fontsize=fontsize)
    plt.imshow(img)  # To see the rag with the image in colors, erase 'extent=(0, 1, 1, 0)', extent=(0, 1, 1, 0)
    plt.tight_layout()
    # plt.axis('tight')
    plt.axis('off')

    if save:
        plt.savefig(outdir + imfile + name + '.png', transparent=True)


def show_and_save_spectralgraph(rag, title, name, fontsize, save, outdir, imfile):
    plt.figure(dpi=180)
    plt.title(title, fontsize=fontsize)
    nx.draw_spectral(rag, with_labels=False, node_size=5, edge_cmap='magma')
    plt.tight_layout()
    # plt.axis('tight')
    plt.axis('off')

    if save:
        plt.savefig(outdir + imfile + name + '.png')


def show_and_save_affmat(aff_mat, title, name, fontsize, save, outdir, imfile):
    aff_mat = aff_mat.toarray()
    plt.figure(dpi=180)
    plt.title(title, fontsize=fontsize)
    plt.imshow(aff_mat, interpolation='gaussian')  #
    plt.clim(0, np.std(aff_mat))
    # plt.axis('off')

    if save:
            plt.savefig(outdir + imfile + name + '.png', bbox_inches='tight')


def show_and_save_result(img, regions, title, label, name, fontsize, save, outdir, imfile):
    out = color.label2rgb(regions, img, kind='avg')
    out = segmentation.mark_boundaries(out, regions, color=(0, 0, 0), mode='thick')

    plt.figure(dpi=180)
    ax = plt.gca()
    ax.imshow(out)
    ax.tick_params(axis='both', which='both', labelsize=7, pad=0.1,
                   length=2)  # , bottom=False, left=False, labelbottom=False, labelleft=False
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(('Recall: %.3f, Precision: %.3f, Time: %.2fs' % label).lstrip('0'), fontsize=fontsize)

    if save:
        plt.savefig(outdir + imfile + name + '.png')

    plt.cla()
    plt.clf()
    plt.close()


def show_and_save_dist(weights, thresh, params, title, name, fontsize, save, outdir, imfile):
    x = np.linspace(0, max(weights), 100)
    # pdf_fitted = gamma.pdf(x, *params)
    pdf_fitted = lognorm.pdf(x, *params)
    thr_pos = lognorm.pdf(thresh, *params)
    plt.figure()
    plt.hist(weights, bins='auto', color='k', density=True)
    plt.plot(x, pdf_fitted, color='r')
    plt.grid(axis='y')
    plt.annotate('threshold', xy=(thresh, thr_pos), xytext=(thresh, pdf_fitted.max()/4), arrowprops=dict(facecolor='red', shrink=2))
    plt.title(title, fontsize=fontsize)

    if save:
        plt.savefig(outdir + imfile + name + '.png', bbox_inches='tight')
