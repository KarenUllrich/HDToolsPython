#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Provides sampling methods e.g. bootstrapping of a data set.
"""

import numpy as np
import random as r

__author__	= ["Karen Ullrich"]
__email__	= "karen.ullrich@ofai.at"
__version__	= "Dec 2014"


def bootstrap(data,m,f,i,seed=None):
    """ 
    Returns a random subset of the original data set. 
    
    Thereby always includes point i as first point. The number of features
    can be restricted as well. 
    """

    (dim,n) = data.X.shape

     # dataset without point i
    X_i = np.concatenate((data.X[:,:i],data.X[:,i+1:]), axis=1)
    t_i = np.concatenate((data.t[:i],data.t[i+1:]))

    while True:
        idx = np.arange(n-1)
        r.seed(seed)
        r.shuffle(idx)
        idx = idx[:m]
        X_ = np.concatenate((data.X[:,i].reshape((dim,1)),X_i[:,idx]), axis=1)
        t_ = np.concatenate((data.t[i].reshape(1),t_i[idx]))
        idx = np.arange(dim)
        r.seed(seed)
        r.shuffle(idx)
        idx = idx[:f]

        yield X_[idx,:],t_