# -*- coding: utf-8 -*-
"""
Created on Tue Jul 05 19:38:00 2016

@author: FF120
"""

import numpy as np

class Perception(object):
    
    def __init__(self, ita=1):
        self.ita_ = ita
    
    '''
    
    李航-统计学习方法-第二章-感知机-例题
    学习到的是一个分类的平面，所以能够预测在训练集中没有出现过的特征。
    '''
    def fit(self,X,y):
        self.n_samples_ = X.shape[0]
        self.n_features_ = X.shape[1]
        self.labels_ = np.unique(y)
        self.n_lables_ = self.labels_.shape[0]
        self.parameters_ = []
        #初始化        
        w = np.zeros(self.n_features_)
        b = 0
        i = 0
        while i < self.n_samples_:
            if y[i] * (np.dot(w,X[i]) + b) <= 0:
                w = w + self.ita_ * y[i] * X[i]
                b = b + self.ita_ * y[i]
                i = 0
                self.parameters_.append((w,b))
            else:
                i = i + 1
                
    def sign(self,x):
        if x>=0:
            return 1
        else:
            return -1
    
    def predict(self,X):
        self.w_,self.b_ = self.parameters_[-1]
        self.f_ = np.dot(self.w_,X) + self.b_
        return self.sign(self.f_)
        

    def look(self):
        for line in self.parameters_:
            print 'w:%s    b: %s'% (line[0],line[1])
        
X = np.array([[3,3],[4,3],[1,1]])
y = np.array([1,1,-1])    

p = Perception()
p.fit(X,y)
print p.f_
print p.predict(np.array([-100,1]))




        