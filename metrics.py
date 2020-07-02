# -*- coding: utf-8 -*-
#!/usr/bin/env python

import sys, time, os
from os.path import expanduser
home = expanduser("~")

from math import *
import numpy as np

from skimage import io, color
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, rectangle, disk
from skimage.measure import regionprops
import matplotlib.pyplot as plt


class metrics:

    """
    Compute the metrics associated to a segmentation for the 
    Berkeley Segmentation Dataset
    """

    def __init__(self, img, lb, segments_truth):

        """
        Class constructor
        
        Load the label image and the manual segmentations, 
        and extract the segments boundaries from the manual 
        segmentations.

        :param img: Original image
        :type img: numpy array
        :param lb: label image
        :type lb: numpy array 
        :param segments_truth: Manual segmentations
        :type segments_truth: list of numpy array
        """

        self.img = img
        self.lb = lb.astype('int')
        self.nx, self.ny = self.lb.shape

        self.segments_truth = segments_truth
        self.img_truth = []
        for segments in self.segments_truth:
            self.img_truth.append(find_boundaries(segments))

        self.n_segments = np.max(self.lb) + 1
         

    # -------------------------------------------
    # Boundary recall and precision
    # -------------------------------------------
          
    def set_boundary_recall(self, size=5):

        """
        Calculate the boundary recall with respect to the ground truth.
        The boundary recall is averaged between all manual segmentations.

        :param size: size of the square neighborhood considered to compute
         the boundary recall (default: size=5)
        :type size: int
        """

        eikonal_bd = dilation(find_boundaries(self.lb), rectangle(size, size))
        self.recall = 0
        for idx, truth in enumerate(self.img_truth):
            self.recall += float(np.sum(eikonal_bd * truth))/float(np.sum(truth))

        self.recall /= len(self.img_truth)


    def set_boundary_precision(self, size=5):

        """
        Compute the boundary precision with respect to the ground truth.
        The boundary precision is averaged between all manual segmentations.

        :param size: size of the square neighborhood considered to compute
         the boundary recall (default: size=5)
        :type size: int
        """

        eikonal_bd = find_boundaries(self.lb)
        self.precision = 0
        global_score = float(np.sum(eikonal_bd))

        for idx, truth in enumerate(self.img_truth):
            intersection = eikonal_bd * dilation(truth, rectangle(5, 5))
            self.precision += float(np.sum(intersection))/global_score

        self.precision /= len(self.img_truth)

    # -------------------------------------------
    # Undersegmentation
    # -------------------------------------------

    def set_undersegmentation(self):

        """
        Compute the undersegmentation metric.
        The boundary precision is averaged between 
        all manual segmentations.
        """

        self.undersegmentation = 0.
        self.undersegmentationNP = 0.

        for truth in self.segments_truth:

            n_labels = int(np.max(truth) + 1)
            area = np.zeros(self.n_segments)
            hist = np.zeros((self.n_segments, n_labels))

            # Intersection matrix
            for x in range(self.nx):
                for y in range(self.ny):

                    idx = int(self.lb[x, y])
                    t_idx = int(truth[x, y])
                    hist[idx, t_idx] += 1
                    area[idx] += 1

            # Undersegmentation (Van den Bergh formula)
            u = 0.
            for k in range(self.n_segments):
                u += area[k] - np.max(hist[k, :])

            u /= self.nx*self.ny
            self.undersegmentation += u

            # Undersegmentation (Neuber and Protzel formula)
            unp = 0.
            for p in range(n_labels):
                for q in range(self.n_segments):
                    unp += min(hist[q, p], np.sum(hist[q, :]) - hist[q, p])

            unp /= self.nx*self.ny
            self.undersegmentationNP += unp        

        self.undersegmentation /= len(self.segments_truth)
        self.undersegmentationNP /= len(self.segments_truth)        
    
    # -------------------------------------------
    # Geometrical criteria
    # -------------------------------------------
                
    def set_density(self):

        """
        Computes the segmentation density
        """
        self.density = np.sum(find_boundaries(self.lb))/float(self.nx*self.ny)


    def perimeter(self):

        """
        Computes the perimeter of the superpixels
        """

        self.perimeters = np.zeros((self.n_segments))

        for i in range(self.nx):
            for j in range(self.ny):

                label = self.lb[i, j]

                # Pixel belonging to an image border
                if(i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1):
                    self.perimeters[label] += 1

                # Pixel at the border of two regions
                elif(self.lb[i - 1, j] != label or self.lb[i + 1, j] != label or
                  self.lb[i, j - 1] != label or self.lb[i, j + 1] != label):
                    self.perimeters[label] += 1
          
    def set_compactness(self):

        """
        Compute compactness
        """

        # Compute the segments perimeters
        self.perimeter()
     
        # Compute compactness
        self.compactness = 0
        max_area = float(self.nx * self.ny)

        for i in range(self.n_segments):

            area = np.sum(self.lb == i)
            perimeter = self.perimeters[i]        
            ratio = area/max_area
            if(perimeter > 0):
                self.compactness += 4*pi*ratio*area/pow(perimeter, 2)

      
    # -------------------------------------------
    # Compute all metrics
    # -------------------------------------------
               
    def set_metrics(self):

        """
        Computes all metrics
        """
        self.set_boundary_recall()
        self.set_boundary_precision()
        self.set_density()
        self.set_undersegmentation()
        self.set_compactness()



    # -------------------------------------------
    # Display metrics
    # -------------------------------------------

    def display_metrics(self):

        """
        Display the computed metrics
        """

        print("Regions: " + str(self.n_segments) + "\n",
         " Recall: " + str(self.recall) + "\n",
         " Precision: " + str(self.precision) + "\n",
         " Undersegmentation (Bergh): " + str(self.undersegmentation) + "\n",
         " Undersegmentation (NP): " + str(self.undersegmentationNP) + "\n",
         " Compactness: " + str(self.compactness) + "\n",
         " Density: " + str(self.density) + "\n")

    # -------------------------------------------
    # Return metrics
    # -------------------------------------------

    def get_metrics(self):

        """
        Returns the computed metrics in a .npy file
        """

        return {"regions": self.n_segments,
          "recall": self.recall,
          "precision": self.precision, 
          "underseg": self.undersegmentation,
          "undersegNP": self.undersegmentationNP,
          "compactness": self.compactness, 
          "density": self.density}


