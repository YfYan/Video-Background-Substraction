#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:57:35 2019

@author: yanyifan
"""


import numpy as np
import time
np.random.seed(None)

class Svd_rpca(object):
    
    def __init__(self,A,mu,rank=None,constrain = False):
        self.A = A
        self.rank = rank
        self.mu = mu
        
        self.m = A.shape[0]
        self.n = A.shape[1]
        
        if rank == None and constrain == True:
            raise ValueError("If rank is None then constrain must be False")
        
        
        if rank !=None and rank > min(self.m,self.n):
            raise ValueError('The required rank should be lower than the original matrix.')
        
        self.constrain = constrain
        self.p = 10
        self.initilize()
    
        
    def initilize(self):
        #self.Z = self.A.copy()
        self.Z = np.zeros(shape = self.A.shape)
        self.E = np.zeros(shape = self.A.shape)
        self.lagragian = np.sign(self.A)/np.linalg.norm(self.A)
        
    @staticmethod
    def thresh(A,eps):
       A[A>eps] -= eps
       A[A<-eps] += eps
       A[np.abs(A)<=eps] = 0
       return A
   
    def local_opt(self):
        temp = (self.A - self.E + self.lagragian/self.p).copy()
        u,s,vt = np.linalg.svd(temp)
        s = self.thresh(s,1/self.p)
        smat = np.zeros(shape = self.A.shape)
        
        if self.constrain:
            s = s[:self.rank]
            smat[:self.rank,:self.rank] = np.diag(s)
        else:
            smat[:s.shape[0],:s.shape[0]] = np.diag(s)

        z_pre = self.Z.copy()
        e_pre = self.E.copy()
        l_pre = self.lagragian.copy()
        
        self.Z = np.dot(u,np.dot(smat,vt))
        self.E = self.thresh(self.A - self.Z + self.lagragian/self.p,self.mu/self.p)
        self.lagragian = self.lagragian + self.p*(self.A-self.Z-self.E)
        
        z_diff = np.linalg.norm(z_pre-self.Z)/np.sqrt(self.Z.shape[0]*self.Z.shape[1])
        e_diff = np.linalg.norm(e_pre-self.E)/np.sqrt(self.E.shape[0]*self.E.shape[1])
        l_diff = np.linalg.norm(l_pre-self.lagragian)/np.sqrt(self.lagragian.shape[0]*self.lagragian.shape[1])
        return max(z_diff,e_diff,l_diff)
    
    def fix_p(self,tol = 1e-6):
        z_pre = self.Z.copy()
        e_pre = self.E.copy()
        l_pre = self.lagragian.copy()
        flag = True
        cnt = 0
        while flag and cnt<1000:
            diff = self.local_opt()
            if diff < tol:
                flag = False
            cnt += 1
            #print(cnt)
        self.p*=1.2
        
        z_diff = np.linalg.norm(z_pre-self.Z)
        e_diff = np.linalg.norm(e_pre-self.E)
        l_diff = np.linalg.norm(l_pre-self.lagragian)
        
        return max(z_diff,e_diff,l_diff)
    
    def kkt(self):
        temp = self.A - self.Z - self.E
        return np.linalg.norm(temp)/np.sqrt(self.A.shape[0]*self.A.shape[1])
    
    def fit(self,tol = 1e-6):
        cnt = 0
        flag = True
        while flag and cnt < 100:
            diff=self.fix_p()
            if self.kkt() < tol:
                flag = False
            cnt+=1
            #print(cnt)
            
    def loss(self):
        a = np.sum(np.abs(self.A-self.Z))
        b = self.mu * np.linalg.norm(self.Z,ord='nuc')
        return a+b
            
if __name__ == '__main__':
    a = np.random.uniform(size = (50,60))
    rpca = Svd_rpca(a,mu = 1,rank = 10,constrain = False)
    time1 = time.time()
    rpca.fit()
    time2 = time.time()
    print("Time cost >>>",time2-time1)
    print(rpca.loss())
        