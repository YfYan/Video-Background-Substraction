#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:55:39 2020

@author: yanyifan
"""

from PIL import Image
from admm import Admm_rpca
from svd_method import Svd_rpca
from primitive_grad_opt import Grad_rpca
from inexact_augmented_lagrange_multiplier import inexact_augmented_lagrange_multiplier
import imageio
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import numpy as np

def get_pic_matrix(dir_):
    pil_im=Image.open(dir_,mode = 'r')
    im=np.array(pil_im,dtype = 'float64')
    return im

def rgb2gray(rgb):
	r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	return gray/255


if __name__ == '__main__':
    names = sorted(glob("./ShoppingMall/*.bmp"))
    chip = len(names)
    #chip = 48
    d1, d2, channels = imageio.imread(names[0]).shape
    #d1,d2,channels = np.array(Image.open(names[0])).shape

    X = np.zeros((d1,d2,chip))
    for i in range(chip):
        im_matrix = imageio.imread(names[i])
        im_matrix = rgb2gray(im_matrix)
        X[:,:,i] = im_matrix
    print("data loaded")
    X = X.reshape((d1*d2, chip))
    mu = 0.001
    # A, E = inexact_augmented_lagrange_multiplier(X)
    # loss = np.sum(np.abs(E)) + mu * np.linalg.norm(A,ord='nuc')
    # print("The loss is ",loss)
    rank = 12
    rpca = Svd_rpca(X,mu = mu,constrain=False)
    rpca.fit()
    A = rpca.Z
    print(rpca.loss())
    A = A.reshape(d1,d2,chip) * 255.0

    #im_set = []
    for i in range(10):
        im2 = A[:, :, i]
        # contain = False
        # for item in im_set:
        #     print(np.sum(np.abs(im2 - item)))
        #     if np.sum(np.abs(im2 - item)) < 1e-3:
        #         contain = True
        #         break
        #
        # if contain==False:
        im2 = Image.fromarray(im2)
        plt.imshow(im2)
        plt.show()
        print(i)