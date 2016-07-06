# -*- coding: utf-8 -*-
"""
Created on Wed Jul 06 17:24:50 2016

@author: FF120
"""
import numpy as np
import math

class DecisionTree(object):
    
    def __init__(self,ita=1):
        self.ita_ = ita
    
    '''计算H(D)
    
    in: x=[n_samples,]
    out: hd
    '''
    def hd(self,x):
        labels,counts = np.unique(x,return_counts=True)
        hd = 0
        for i in range(len(labels)):
            hd += - ( ( counts[i] / float(x.shape[0]) ) * math.log( (counts[i] / float(x.shape[0])),2 ) )
        return hd
    '''计算互信息
    信息增益g(D,A)=H(D)-H(D|A)
    H(D)是整个训练集上类别的熵
    H(D|A)是在A确定的条件下类别信息在D中的条件熵
    
    out:信息增益最大的特征
    '''
    def MutualInformation(self,X,y):
        self.n_features_ = X.shape[1]
        self.n_samples_ = X.shape[0]
        
        #1. 计算整体的H(D)
        self.hd_ = self.hd(y)
        #2. 计算H(D|A)
        self.hda_ = []
        for i in range(self.n_features_): # i代表列，就是特征的维数
            features_range,fr_counts = np.unique(X[:,i],return_counts=True)
            n_features_range = features_range.shape[0]
            sum = 0
            for j in range(n_features_range): # j代表第i维特征去重之后的第j个取值
                sum += (fr_counts[j] / float(self.n_samples_) ) * self.hd(y[X[:,i] == features_range[j]])
            
            self.hda_.append(sum)
        
        #3.计算H(D)-H(D|A)
        self.gda_ = self.hd_ - np.array(self.hda_)
        max_feature = np.argmax( self.gda_ )
        return max_feature
    '''生成测试数据
    青年 1；中年 2；老年 3；
    有工作 1；无工作 0；
    有房子 1；无工作 0；

    信贷情况：
    非常好 3；好 2；一般 1；    

    '''
    def CreateData(self):
        y = [0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]
        X = [   [1,0,0,1],\
                [1,0,0,2],\
                [1,1,0,2],\
                [1,1,1,1],\
                [1,0,0,1],\
                [2,0,0,1],\
                [2,0,0,2],\
                [2,1,1,2],\
                [2,0,1,3],\
                [2,0,1,3],\
                [3,0,1,3],\
                [3,0,1,2],\
                [3,1,0,2],\
                [3,1,0,3],\
                [3,0,0,1]\
            ]
        X = np.array(X)
        y = np.array(y)
        return X,y

dt = DecisionTree()
'''
李航-统计学习方法-决策树-信息增益练习题的数据
'''
X,y = dt.CreateData() 
print dt.MutualInformation(X,y)
        
        
    