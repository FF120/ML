# -*- coding: utf-8 -*-
"""
Created on Mon Jul 04 16:03:50 2016

@author: FF120
"""

import numpy as np

X = [[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],[11,23,22,11,11,11,22,22,33,33,33,22,22,33,33]]
X = map(list, zip(*X)) #list的转置
X = np.array(X,dtype=float)
y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
#=============检查数据=================
print  X.shape[0] == y.shape[0]
#=============准备数据==============
##1. 计算先验概率
N = X.shape[0]  #样本的数量 N
ck = np.unique(y) #不重复的类别
pyck = np.zeros(len(ck))  #存储各个类别的先验概率
for i in range(len(ck)):
    pyck[i] = np.sum( y == ck[i] ) / float(N)

##2. 计算条件概率
feature_num = X.shape[1]  #特征的维数，就是矩阵的列数
X = np.transpose(X)
feature_range = [] # 各个特征的所有可能的取值
for row in X:
    feature_range.append(np.unique(row))
    

P = np.zeros((len(ck),feature_num,N))  #存储条件概率表
for i in range(len(ck)):        # i表示类别
    for j in range(feature_num):      #j表示第j个特征
        aa = X[j][y == ck[i]]
        subn = aa.shape[0] # 属于类ck[i]的第j维特征的 个数的和。
        uaa = np.unique(aa)
        for k in range(len(uaa)):            #k表示第j维特征取第K个值
            P[i][j][k] = np.sum( aa == uaa[k] ) / float(subn)

##3. 预测
test_X = np.array([2,11])  ##训练的时候需要保证每维特征都包含特征可能取到的所有值，才能训练出完整的条件概率表
test_result = np.ones(test_X.shape[0])
for i in range(test_X.shape[0]):
    test_result[i] = pyck[i] * test_result[i] 
    for k in range(feature_num):
        #查找第k维特征的索引，取出条件概率
        arr = np.where(feature_range[k] == test_X[k])
        index = arr[0][0]  #第k维特征在特征空间中的索引，根据这个索引去查找该条件概率
        test_result[i] = test_result[i] * P[i][k][index]
    
ins = np.where(test_result ==  np.max(test_result))
print ck[ins[0][0]]


