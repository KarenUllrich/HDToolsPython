#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Given a distance matrix D, these functions compute intrinsic dimensionality, relative variance and hubness.
"""

import numpy as np
from methods.proximity import compute_knn_matrix

__author__	= ["Karen Ullrich"]
__email__	= "karen.ullrich@ofai.at"
__version__	= "Dec 2010"


def MLE(D):
	"""
	Maximum Likelihood Estimation of Intrinsic Dimension
	Levina and Brickel, NIPS 2010 

	The variable choice is in accordance with the paper.

	D 	 	... distance matrix
	"""

	# D must be given, make sure you use the right distance function

	n,n = D.shape
    
    # Set neighborhood range to search in
    
    k1 = 6
    k2 = 12

    # Compute matrix of log nearest neighbor distances

    T_k_matrix = np.log(compute_knn_matrix(D,k=k2))

    # Compute the ML estimate, equation (8) and (9)
	
	mhat = 0.
	for k in xrange(k1-1,k2):
		for i in xrange(n):
			mhat_k = 0.
			for j in xrange(k-1):
				mhat_k +=  T_k_matrix[k,i]-T_k_matrix[j,i]

			mhat += (1./(k-1.) *mhat_k)**(-1)
		
	mhat = mhat/n/(k2-k1+1)

	return mhat # mhat = d



def realativeVariance(D):
	"""
	Computes the relative Variance of the distance distribution as defined in 

	The Concentration of Fractional Distances [Francois et. al., 2007]
	"""
	var = np.std(squareform(D))
	mean = np.mean(D)

	return np.sqrt(var)/mean

def hubness( D, k=5):
	"""	
	Computes the hubness of a distance matrix using its k nearest neighbors as defined in 
	
	Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data [Radovanovic et. al., 2010]
	"""

    n,n = D.shape

    # Compute the number of k-occurrences N_k(x) of each data point x. 
    
    N_k = np.zeros(n)
    occurrences = np.argsort(D,axis=0)[1:k+1]
    
    for i in xrange(n):
    	N_k[i] = occurrences.count(i)
    

    hubness = third_momentum(N_k)

    return hubness, N_k

def third_momentum(distribution):
	"""
	Computes the standardized third momentum of a probability distribution. 

	According page 2492 in 
	Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data [Radovanovic et. al., 2010]
	"""
	
	mu = np.mean(distribution)
	distribution = distribution - mu
	expectation3 = np.mean(distribution)**3 #-... law of big numbers

	sigma3 = np.std(distribution)**3

	return expectation3 / sigma3