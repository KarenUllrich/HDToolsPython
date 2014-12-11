#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Provides methods to compute proximity between points of a data set.
"""

import numpy as np
from scipy.spatial.distance import squareform, pdist
import random as r


__author__	= ["Karen Ullrich"]
__email__	= "karen.ullrich@ofai.at"
__version__	= "Dec 2010"


def computeDistanceMatrix(X,distance):
        # compute distance matrix
            # remark: I hate it as is since you save n!² information when you only really got n!    
        "TODO: better solution for fatter and bigger data"
        # for infos on which distances are available go on 
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
        return squareform(pdist(X.T, distance))
        #return pdist(X, distance)


def computeMutualProximity(X = None, distance = None, D = None):
    '''
    TODO: not very effective yet
    '''
    if D is None:
        if distance is None or X is None:
            print 'Distance measure and data set or distance matrix need to be specified'
            ValueError()
        D = computeDistanceMatrix(X,distance)
    else:
        D_MP = D * 0 #.. Distance matrix after applying mutual proximity  
        (n,n) = D.shape
        for i in xrange(n):
            for j in xrange(i,n):
                s1 = find(D[i,:], lambda x: x > D[i,j]);
                s2 = find(D[j,:], lambda x: x > D[i,j]);
                
                D_MP[j, i] = 1 - len(set(s1).intersection(s2))/float(n);
                D_MP[i, j] = D_MP[j, i];
                    
            D_MP[i,i] = 0;

    return D_MP

def find(a, func):
    '''
    In accordance to MATLABs find-function.
    Returns all indexes of a given array X for which a given condition func is true.
    Example usage:
    idx = find(X, lambda x: x > 0)
    '''
    return [i for (i, val) in enumerate(a) if func(val)]


def compute_knn_matrix(D,k):
    """
    Computes the first k nearest neighbors.

    D ... distance matrix
    """

    return np.sort(D,axis=0)[1:k+2] # because first one is always going to be 0 