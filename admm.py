#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 13:54:38 2019

@author: yanyifan
"""

import numpy as np
import time
np.random.seed(None)

class Admm_rpca(object):
    
    def __init__(self,A,rank,mu):
        self.A = A
        self.rank = rank
        self.mu = mu
        
        self.m = A.shape[0]
        self.n = A.shape[1]
        
        if rank > min(self.m,self.n):
            raise ValueError('The required rank should be lower than the original matrix.')
        
        self.p = 10
        
        self.initilize1()
        
        self.tol = 1e-2
    
        
    def initilize1(self):
        self.X = np.random.random(size=(self.m,self.rank))
        
        self.Y = np.random.random(size=(self.rank,self.n))
        
        #self.Z = np.random.random(size = self.A.shape)
        self.Z = np.random.random(size=(self.m,self.n))
        
        self.lagragian = np.sign(self.A)/np.linalg.norm(self.A)
        

        
    
    def primal_feasibility(self):
        return np.linalg.norm(self.Z - np.dot(self.X,self.Y))/(self.Z.shape[0]*self.Z.shape[1])
    
    def dual_feasibility(self):
        grad_x = self.mu*self.X - np.dot(self.lagragian,self.Y.T) - self.p*np.dot(self.Z - np.dot(self.X,self.Y),self.Y.T)
        grad_y = self.mu*self.Y - np.dot(self.X.T,self.lagragian) - self.p*np.dot(self.X.T,self.Z - np.dot(self.X,self.Y))
        return max(np.linalg.norm(grad_x)/(self.X.shape[0]*self.X.shape[1]),np.linalg.norm(grad_y)/(self.Y.shape[0]*self.Y.shape[1]))
        
    def x_opt(self):
        pre = self.X.copy()
        temp = np.linalg.inv(self.mu*np.eye(self.rank) + self.p * np.dot(self.Y,self.Y.T))
        temp2 = np.dot(self.lagragian,self.Y.T) + self.p*np.dot(self.Z,self.Y.T)
        self.X = np.dot(temp2,temp)
        return np.linalg.norm(self.X-pre)/np.sqrt(self.X.shape[0]*self.X.shape[1])
    
    def y_opt(self):
        pre = self.Y.copy()
        temp = np.linalg.inv(self.mu*np.eye(self.rank) + self.p * np.dot(self.X.T,self.X))
        temp2 = np.dot(self.X.T,self.lagragian) + self.p*np.dot(self.X.T,self.Z)
        self.Y = np.dot(temp,temp2)
        return np.linalg.norm(self.Y-pre)/np.sqrt(self.Y.shape[0]*self.Y.shape[1])
    
    def z_opt(self):
        pre = self.Z.copy()
        XY = np.dot(self.X,self.Y)
        temp1 = (-1-self.lagragian)/self.p + XY
        temp2 = (1-self.lagragian)/self.p + XY
        
        inter1 = self.A < temp1
        inter2 = self.A > temp2
        
        inter3 = np.bitwise_not(np.bitwise_or(inter1,inter2))
        
        self.Z[inter1] = temp1[inter1]
        self.Z[inter2] = temp2[inter2]
        self.Z[inter3] = self.A[inter3]
        
        # The slower implementation 
        # for i in range(self.m):
        #     for j in range(self.n):
        #         a = self.A[i][j]
        #         l = self.lagragian[i][j]
        #         if (-1-l)/self.p + XY[i][j] > a:
        #             self.Z[i][j] = (-1-l)/self.p + XY[i][j]
        #         elif (1-l)/self.p + XY[i][j] < a:
        #             self.Z[i][j] = (1-l)/self.p + XY[i][j]
        #         else:
        #             self.Z[i][j] = a
        return np.linalg.norm(self.Z-pre)/np.sqrt(self.Z.shape[0]*self.Z.shape[1])

                    
    def one_admm(self):
        x_diff = self.x_opt()
        y_diff = self.y_opt()
        z_diff = self.z_opt()
        
        l_pre = self.lagragian.copy()
        self.lagragian = self.lagragian + self.p*(self.Z-np.dot(self.X,self.Y))
        
        l_diff = np.linalg.norm(self.lagragian - l_pre)/np.sqrt(self.lagragian.shape[0]*self.lagragian.shape[1])
        return max(x_diff,y_diff,z_diff,l_diff) 
        
    def one_convergence(self,threshold = 1e-2):
        flag = True
        cnt = 0
        while flag and cnt < 100:
            diff = self.one_admm()
            #print(diff)
            if diff<threshold:
                flag = False
            cnt+=1
        #print(cnt)
        self.p *= 1.2
    
    def fit(self,tol = 1e-6):
        cnt = 0
        flag = True
        while flag and cnt < 70:
            self.one_convergence()
            cnt+=1
            if self.primal_feasibility() < tol:
                flag = False
            #print(cnt)
            
    def loss(self):
        a = np.sum(np.abs(self.A-self.Z))
        b = self.mu * np.linalg.norm(self.X)**2
        c = self.mu * np.linalg.norm(self.Y)**2
        return a+b+c
    
    
    def fit_p_fixed(self):
        cnt = 0
        flag = True
        while flag and cnt < 100000:
            self.x_opt()
            self.y_opt()
            self.z_opt()
            self.lagragian = self.lagragian + self.p*(self.Z-np.dot(self.X,self.Y))
            gap = max(self.primal_feasibility(),self.dual_feasibility())
            print(gap)
            if  gap < self.tol:
                flag = False
            cnt += 1
            print(cnt)
            
    
if __name__ =='__main__':
    a = np.random.uniform(size = (50,60))
    # u, s, vt = np.linalg.svd(a, full_matrices=True)
    # smat = np.zeros(shape = (9,6))
    # smat[:6,:6] = np.diag(s)
    # print(a)
    # print(np.dot(u,np.dot(smat,vt)))
    rpca = Admm_rpca(a,10,1)
    time1 = time.time()
    rpca.fit_p_fixed()
    time2 = time.time()
    print("Time cost >>>",time2-time1)
    print(rpca.loss())
    
    rpca2 = Admm_rpca(a,10,1)
    time1 = time.time()
    rpca2.fit()
    time2 = time.time()
    print("Time cost >>>",time2-time1)
    print(rpca2.loss())

    