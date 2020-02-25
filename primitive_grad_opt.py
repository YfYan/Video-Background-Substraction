#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 20:07:32 2019

@author: yanyifan
"""

import numpy as np
import time
class Grad_rpca(object):
    
    def __init__(self,A,rank,mu,adam = True):
        self.A = A
        self.rank = rank
        self.mu = mu
        
        self.m = A.shape[0]
        self.n = A.shape[1]
        
        if rank > min(self.m,self.n):
            raise ValueError('The required rank should be lower than the original matrix.')
        
        self.adam = adam
        self.initilize1()
    
    def initilize1(self):
        self.X = np.random.random(size=(self.m,self.rank))
        self.Y = np.random.random(size=(self.rank,self.n))
        
        self.adam_init = False
        self.p = 0.9
    
    def grad_x(self):
        temp = np.sign(np.dot(self.X,self.Y) - self.A)
        # subg = np.random.uniform(low = -0.5,high = 0.5, size = self.A.shape)
        # is_0 = (temp == 0)
        # temp[is_0] = subg[is_0]
        return np.dot(temp,self.Y.T) + self.mu * self.X
    
    def grad_y(self):
        temp = np.sign(np.dot(self.X,self.Y) - self.A)
        # subg = np.random.uniform(low = -0.5,high = 0.5, size = self.A.shape)
        # is_0 = (temp == 0)
        # temp[is_0] = subg[is_0]
        return np.dot(self.X.T,temp) + self.mu * self.Y
    
    def adam_x(self):
        gx = self.grad_x()
        if self.adam_init == False:
            self.xv = gx
            self.xu = gx**2
        else:
            self.xv = 0.9 * self.xv + 0.1 * gx
            self.xu = 0.9 * self.xu + 0.1 * gx**2
        return self.xv/(1-self.p)/np.sqrt(self.xu + 1e-8)
    
    def adam_y(self):
        gy = self.grad_y()
        if self.adam_init == False:
            self.yv = gy
            self.yu = gy**2
        else:
            self.yv = 0.9 * self.yv + 0.1 * gy
            self.yu = 0.9 * self.yu + 0.1 * gy**2
        return self.yv/(1-self.p)/np.sqrt(self.yu + 1e-8)
    
    def fit(self,alpha = 0.01):
        cnt = 0
        flag = True
        tol = 1e-2
        while flag and cnt < 1000:
            if self.adam:
                gx = self.adam_x()
            else:
                gx = self.grad_x()
            sx = np.linalg.norm(gx)
            self.X -= alpha * gx
            
            if self.adam:
                gy = self.adam_y()
            else:
                gy = self.grad_y()
            sy = np.linalg.norm(gy)
            self.Y -= alpha * gy
            
            if self.adam:
                self.adam_init = True
                self.p *= 0.9
            if max(sx,sy) < 1e-4:
                flag = False
            # X_pre = self.X.copy()
            # converge = False
            # c = 0
            # while not converge and c < 1000:
            #     gx = self.grad_x()
            #     sx = np.linalg.norm(gx)
            #     self.X -= gx*alpha
            #     print(self.X)
            #     print(sx)
            #     c+=1
            #     if sx < tol:
            #         converge = True
            # X_diff = np.linalg.norm(X_pre - self.X)
            
            
            # Y_pre = self.Y.copy()
            # self.admm_init = False
            # converge = False
            # c = 0
            # while not converge and c < 100:
            #     gy = self.grad_y()
            #     sy = np.linalg.norm(gy)
            #     self.Y -= gy*alpha
            #     c += 1
            #     if sy < tol:
            #         converge = True
            
            # Y_diff = np.linalg.norm(Y_pre - self.Y)
            
            # if max(X_diff,Y_diff) < 1e-3:
            #     flag = False
            cnt+=1
        print(cnt)        
        self.Z = np.dot(self.X,self.Y)
        
    def loss(self):
        a = np.sum(np.abs(self.A-np.dot(self.X,self.Y))) 
        b = self.mu/2 * np.sum(self.X**2)
        c = self.mu/2 * np.sum(self.Y**2)
        return a + b + c
    
if __name__ == '__main__':   
    np.random.seed(0)
    a = np.random.randint(low = 1,high = 10,size = (9,6))
    np.random.seed(None)
    rpca = Grad_rpca(a,3,1)
    time1 = time.time()
    rpca.fit()
    time2 = time.time()
    print("Time cost >>>",time2-time1)
    print(rpca.loss())