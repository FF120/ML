# -*- coding: utf-8 -*-
"""
Created on Mon Jul 04 16:03:50 2016

@author: FF120
"""
import numpy as np

class NativeBayes(object):
    
    '''
    
    没有解决的问题：
        1. 训练样本没有包含全部的特征空间怎么办？这样就没有办法获得全部的条件概率表
        2. 测试数据中的特征没有在训练数据中怎样处理？目前没有处理，测试训练数据中不存在的特征会报错
    '''
    def __init__(self, lamuda=1):
        self.lamuda_ = lamuda
    
    def CheckData(X,y):
        return True
    
    '''训练过程
    
    X = [n_samples,n_features]
    y = [n_samples]
    
    最大似然估计
    '''
    def fit(self, X, y):
        self.n_samples_ = X.shape[0] #样本数量
        self.n_features_ = X.shape[1] #特征的维数
        self.labels_ = np.unique(y)  #去重之后的类别数组
        self.n_labels_ = self.labels_.shape[0]  #共有多少种类别
        self.pyck_ = np.zeros(self.n_labels_)
        #1. 计算各个类别的先验概率
        for i in range(self.n_labels_):
            self.pyck_[i] = np.sum( y == self.labels_[i] ) / float(self.n_samples_)
            
        #2. 计算类别确定下，取各个特征值的条件概率
        XX = np.transpose(X)
        self.feature_range_ = []  # 每一维特征的可能取值的集合
        for feature in XX:  #每个feature是一维特征
            self.feature_range_.append(np.unique(feature))
            
        self.P_ = np.zeros((self.n_labels_,self.n_features_,self.n_samples_))  #存储条件概率表
        
        for i in range(self.n_labels_):        # i表示类别
            for j in range(self.n_features_):      #j表示第j个特征
                aa = XX[j][y == self.labels_[i]]  #提取出类别i对应的特征
                subn = aa.shape[0] # 属于类ck[i]的第j维特征的 个数的和。
                uaa = np.unique(aa)
                for k in range(len(uaa)):            #k表示第j维特征取第K个值
                    self.P_[i][j][k] = np.sum( aa == uaa[k] ) / float(subn)
                    
                    
        return self
        
    '''训练过程
    
    X = [n_samples,n_features]
    y = [n_samples]
    当self.lamuda_取值为0的时候，就是最大似然估计
    贝叶斯估计
    '''
    def fitBayes(self, X, y):
        self.n_samples_ = X.shape[0] #样本数量
        self.n_features_ = X.shape[1] #特征的维数
        self.labels_ = np.unique(y)  #去重之后的类别数组
        self.n_labels_ = self.labels_.shape[0]  #共有多少种类别
        self.pyck_ = np.zeros(self.n_labels_)
        #1. 计算各个类别的先验概率
        for i in range(self.n_labels_):
            self.pyck_[i] = ( np.sum( y == self.labels_[i] ) + self.lamuda_ ) / ( float(self.n_samples_) + self.n_labels_ * self.lamuda_ )
            
        #2. 计算类别确定下，取各个特征值的条件概率
        XX = np.transpose(X)
        self.feature_range_ = []  # 每一维特征的可能取值的集合
        for feature in XX:  #每个feature是一维特征
            self.feature_range_.append(np.unique(feature))
            
        self.P_ = np.zeros((self.n_labels_,self.n_features_,self.n_samples_))  #存储条件概率表
        
        for i in range(self.n_labels_):        # i表示类别
            for j in range(self.n_features_):      #j表示第j个特征
                aa = XX[j][y == self.labels_[i]]  #提取出类别i对应的特征
                subn = aa.shape[0] # 属于类ck[i]的第j维特征的 个数的和。
                uaa = np.unique(aa)
                for k in range(len(uaa)):            #k表示第j维特征取第K个值
                    self.P_[i][j][k] = ( np.sum( aa == uaa[k] ) + self.lamuda_ ) / ( float(subn) + len(uaa) * self.lamuda_)
                    
                    
        return self
    
   
    '''预测过程
    
    X = [n_features]
    out: category of X belongs to
    '''
    def predict(self, X):
        #test_X = np.array([2,11])  ##训练的时候需要保证每维特征都包含特征可能取到的所有值，才能训练出完整的条件概率表
        self.test_result_ = np.ones(X.shape[0])
        for i in range(X.shape[0]):
            self.test_result_[i] = self.pyck_[i] * self.test_result_[i] 
            for k in range(self.n_features_):
                #查找第k维特征的索引，取出条件概率
                arr = np.where(self.feature_range_[k] == X[k])
                index = arr[0][0]  #第k维特征在特征空间中的索引，根据这个索引去查找该条件概率
                self.test_result_[i] = self.test_result_[i] * self.P_[i][k][index]
    
        ins = np.where(self.test_result_ ==  np.max(self.test_result_))
        return self.labels_[ins[0][0]]

    '''参数可视化
    
    '''
    def look(self):
        print '人取值： %s' % self.lamuda_
        print '训练样本：%d' % self.n_samples_
        print '特征维数：%d' % self.n_features_
        print '特征空间：%s' % self.feature_range_
        print '类别数量:%d'  % self.n_labels_
        print '类别:%s' % self.labels_
        print '类别的先验概率：%s' % self.pyck_
        print '条件概率表：%s' % self.P_

X = [[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],[11,22,22,11,11,11,22,22,33,33,33,22,22,33,33]]
X = map(list, zip(*X)) #list的转置
X = np.array(X,dtype=float)
y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])

NB = NativeBayes(lamuda=1)
NB.fitBayes(X,y)
NB.predict(np.array([2,11]))
NB.look()



##=============检查数据=================
#print  X.shape[0] == y.shape[0]
##=============准备数据==============
###1. 计算先验概率
#N = X.shape[0]  #样本的数量 N
#ck = np.unique(y) #不重复的类别
#pyck = np.zeros(len(ck))  #存储各个类别的先验概率
#for i in range(len(ck)):
#    pyck[i] = np.sum( y == ck[i] ) / float(N)
#
###2. 计算条件概率
#feature_num = X.shape[1]  #特征的维数，就是矩阵的列数
#X = np.transpose(X)
#feature_range = [] # 各个特征的所有可能的取值
#for row in X:
#    feature_range.append(np.unique(row))
#    
#
#P = np.zeros((len(ck),feature_num,N))  #存储条件概率表
#for i in range(len(ck)):        # i表示类别
#    for j in range(feature_num):      #j表示第j个特征
#        aa = X[j][y == ck[i]]
#        subn = aa.shape[0] # 属于类ck[i]的第j维特征的 个数的和。
#        uaa = np.unique(aa)
#        for k in range(len(uaa)):            #k表示第j维特征取第K个值
#            P[i][j][k] = np.sum( aa == uaa[k] ) / float(subn)
#
###3. 预测
#test_X = np.array([2,11])  ##训练的时候需要保证每维特征都包含特征可能取到的所有值，才能训练出完整的条件概率表
#test_result = np.ones(test_X.shape[0])
#for i in range(test_X.shape[0]):
#    test_result[i] = pyck[i] * test_result[i] 
#    for k in range(feature_num):
#        #查找第k维特征的索引，取出条件概率
#        arr = np.where(feature_range[k] == test_X[k])
#        index = arr[0][0]  #第k维特征在特征空间中的索引，根据这个索引去查找该条件概率
#        test_result[i] = test_result[i] * P[i][k][index]
#    
#ins = np.where(test_result ==  np.max(test_result))
#print ck[ins[0][0]]


