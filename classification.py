#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Provides a kNN classier.
"""

import numpy as np


__author__	= ["Karen Ullrich"]
__email__	= "karen.ullrich@ofai.at"
__version__	= "Dec 2014"



def kNN(data, k):
    '''
    Performs kNN. Computes LOOCV accuracy.

    k   ...neighborhood size
    D   ...distance matrix
    n   ...number of data instances
    t   ...ground truth/ labels 
    acc ...classification accuracy for LOOCV
    '''
    n = len(data.D)
    num_evaluations = len(k) # How many neigbohood sizes are given
    acc = np.zeros(num_evaluations)
    corr = np.zeros((n, num_evaluations))

    for i in xrange(n):

        ground_truth = data.t[i]
        
        row = data.D[i, :]
        row[i] = float('inf')
        idx = np.argsort(row)
        
        for j in xrange(num_evaluations):

            nn_class = findMostCommonElementOfSet(data.t[idx[:k[j]]])
            if ground_truth == nn_class: 
                acc[j] +=  1./n
                corr[i,j] = 1
    return acc


def findMostCommonElementOfSet(elements):
    '''
    Returns the most common element in a set.'For ties it decides randomly.
    Input:
    a ... list or np.array
    '''
    elementCounter = Counter(elements).most_common()
    highest_count = max([i[1] for i in elementCounter])
    element = [i[0] for i in elementCounter if i[1] == highest_count]
    r.shuffle(element)
    return element[0]