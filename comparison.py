#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:29:16 2019

@author: yanyifan
"""

import numpy as np
import time
from admm import Admm_rpca
from svd_method import Svd_rpca
from primitive_grad_opt import Grad_rpca
from inexact_augmented_lagrange_multiplier import inexact_augmented_lagrange_multiplier

if __name__ == '__main__':
    np.random.seed(0)
    scale = 100
    a = np.random.uniform(size=(scale,scale))
    np.random.seed(None)
    mu = 1/np.sqrt(scale)
    
    rpca1 = Svd_rpca(a,mu,constrain = False)
    time1 = time.time()
    rpca1.fit()
    time2 = time.time()
    print("Time cost for SVD method >>>",time2-time1)
    print("Loss for SVD method >>>",rpca1.loss())
    rank = np.linalg.matrix_rank(rpca1.Z)
    print("The rank for convex form is :",rank)
    print("===============================")
    
    
    rpca2 = Admm_rpca(a,rank,mu)
    time1 = time.time()
    rpca2.fit()
    time2 = time.time()
    print("Time cost for ADMM method >>>",time2-time1)
    print("Loss for ADMM method >>>",rpca2.loss())
    
    print("===============================")
    
    
    time1 = time.time()
    A,E = inexact_augmented_lagrange_multiplier(a,lmbda = mu,tol = 1e-3,Z = rpca2.Z)
    time2 = time.time()
    loss = np.sum(np.abs(E)) + mu * np.linalg.norm(A,ord='nuc')
    print("Time cost for gradient method >>>",time2-time1)
    print("Loss for IALM method >>>",loss)