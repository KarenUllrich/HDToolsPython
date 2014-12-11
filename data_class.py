#!/usr/bin/env python

"""
User is provided with a class to keep all data specific parameters in one place.
"""

__author__  = ["Karen Ullrich"]
__email__   = "karen.ullrich@ofai.at"
__version__ = "Dec 2014"



import numpy as np
from methods.proximity import computeDistanceMatrix


class data(object):
  	"""
    According to Bishop:    X ... data 
                            T ... targets
    """
    def __init__(self, X, T = None, pre_processing = 'centering', distance = None):
        

        self.X = X.T
        self.t = T
        [self.dim, self.n] = self.X.shape
        print "dim = %d and n= %d. If thats not correct try X.T." % (dim, n)
        self.distance = distance
        self.D =None # distance matrix

        # pre-process
        if 'centering' in pre_processing :
            self.centering()
        
        # Computing or not computing thats the question? Its really time intense 
        #if self.distance is not None:
        #        self.D = computeDistanceMatrix(self.X,self.distance)
     
    def centering(self):
        """ Centers the data X. """

        oneV = np.matrix(np.ones(self.n))
        H = np.matrix(np.identity(self.n))-(1./self.n)*oneV.T*oneV  
        self.X = self.X*H
        
        if self.distance is not None:
            self.D = computeDistanceMatrix(self.X,self.distance)

    def computeD(self):
        """ Computes the distance matrix. """

        if self.distance is None:
            print " Distance measure needs to be specified before \
            computing the distance matrix, e.g., object.distance \
            = 'cosine' "
        else:
            self.D = computeDistanceMatrix(self.X,self.distance)

