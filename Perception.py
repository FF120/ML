# -*- coding: utf-8 -*-
"""
Created on Tue Jul 05 19:38:00 2016

@author: FF120
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Perception(object):
    
    '''
    ita range in  (0,1]
    '''
    def __init__(self, ita=1):
        self.ita_ = ita
    
    '''
    
    李航-统计学习方法-第二章-感知机-例题 -- 原始形式
    学习到的是一个分类的平面，所以能够预测在训练集中没有出现过的特征。
    
    感知机得到的结果不唯一，与初始参数w,b的选择和每次选择的点有关，这里初始的w,b选择的都是0，
    每次选择第一个误分类点迭代w,b，得到的结果和书上的例题结果一致。
    实际中，通常使用随机选择的方法。
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
    
    '''感知机算法的对偶形式
    
    X = [n_samples,n_features]
    y = [n_samples,]
    '''
    def fitPair(self,X,y):
        self.n_samples_ = X.shape[0]
        self.n_features_ = X.shape[1]
        self.labels_ = np.unique(y)
        self.n_lables_ = self.labels_.shape[0]
        self.parameters_ = []
        
        gram = np.dot(X,np.transpose(X))
        aerfa = np.zeros(self.n_samples_)
        b = 0
        
        i = 0
#        while i < self.n_samples_:
#            sum = np.zeros(self.n_samples_)
#            for j in range(self.n_samples_):
#                sum += y[j] * np.dot( aerfa,gram[j] )
#            if y[i] * ( np.dot(sum,gram[i]) + b) <= 0:
#                aerfa = aerfa + self.ita_
#                b = b + self.ita_ * y[i]
#                self.parameters_.append((aerfa,b))
#                i = 0
#                print i
#            else:
#                i = i + 1
#                print i
            
    '''符号函数

    '''     
    def sign(self,x):
        if x>=0:
            return 1
        else:
            return -1
    
    def predict(self,X):
        self.w_,self.b_ = self.parameters_[-1]
        self.f_ = np.dot(self.w_,X) + self.b_
        return self.sign(self.f_)
        
    '''查看训练得到的参数
    
    '''
    def look(self):
        for line in self.parameters_:
            print 'w:%s    b: %s'% (line[0],line[1])
    
        
    '''针对二维特征的计算过程可视化
    
    '''
    def update_line(self,num,data,line):
        line.set_xdata(data[num][0])
        line.set_ydata(data[num][1])
        return line,
    
    '''动态呈现分类直线的变化过程
    
    '''
    def lookAnimotionGraph(self,X,y,repeat=False,interval=400):
        markers = ('s', 'o', 'x', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        
        fig1 = plt.figure()
        # 绘制特征的散点图
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], marker=markers[idx],
            alpha=.8, c=colors[idx], label=np.where(cl==1, 'positive', 'negative'))
        # 训练模型，获得参数
        #self.fit(X,y)
        # 绘制每一步迭代过程得到的直线
        self.data = [] 
        l, = plt.plot([], [], 'r-')
        x = []
        y = []
        for w,b in self.parameters_:
            xx = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 10) 
            if (w[0] == 0 and w[1] == 0):
                yy = b
            elif  w[1] == 0:
                yy = b / float( w[1] )
            elif  w[0] == 0:
                yy = b
            else:
                yy = - ( (w[0]/w[1]) * xx + b / w[1])
            x.append(np.min(xx))
            x.append(np.max(xx))
            y.append(np.max(yy))
            y.append(np.min(yy))
            self.data.append((xx,yy))
        #设置坐标轴的取值范围
        x = np.array(x)
        x_min = np.min(np.array([np.min(x),np.min(X[:,0])]))
        x_max = np.max(np.array([np.max(x),np.max(X[:,0])]))
        y = np.array(y)
        y_min = np.min(np.array([np.min(y),np.min(X[:,1])]))
        y_max = np.max(np.array([np.max(y),np.max(X[:,1])]))
        plt.xlim(x_min-0.3*x_min,x_max+0.3*x_max)
        plt.ylim(y_min-0.3*y_min,y_max+0.3*y_max)
        
        line_ani = animation.FuncAnimation(fig1, self.update_line, len(self.data), fargs=(self.data, l),
           interval=interval, blit=True,repeat=repeat)  
        plt.show()
    
    '''静态绘制所有直线
    
    '''
    def lookGraph(self,X,y):
        markers = ('s', 'o', 'x', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        
        # 绘制特征的散点图
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], marker=markers[idx],
                    alpha=.8, c=colors[idx], 
                    label=np.where(cl==1, 'positive', 'negative'))
        # 训练模型，获得参数
        #self.fit(X,y)
        # 绘制每一步迭代过程得到的直线
        self.data = [] 
        for w,b in self.parameters_:
            xx = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 10) 
            if (w[0] == 0 and w[1] == 0):
                yy = np.linspace(b,b,10)
            elif  w[1] == 0:
                yy = np.linspace(b / float( w[1] ), b / float( w[1] ) ,10)
            elif  w[0] == 0:
                yy = np.linspace(b,b,10)
            else:
                yy = - ( (w[0]/w[1]) * xx + b / w[1])
            self.data.append((xx,yy))
            plt.plot(xx,yy,color="red",linewidth=1)  
            
        plt.show()
        #添加分类直线的动态绘制过程
        #....
        
X = np.array([[3,3],[4,3],[1,1]])
y = np.array([1,1,-1])    

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
X = X[0:100,0:2]
y = y[0:100]

p = Perception(ita=1)
p.fit(X,y)
p.lookAnimotionGraph(X,y)






        