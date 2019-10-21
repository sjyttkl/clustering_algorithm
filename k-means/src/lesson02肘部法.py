# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     lesson02肘部法
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2019/10/21
   Description :
==================================================
"""
__author__ = 'songdongdong'

#一，导入库
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import  cdist
import matplotlib.pyplot as plt

#二、生成数据集
cluster1 = np.random.uniform(0.5,1.5,(2,10))#从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开, size: 输出样本数目，为int或元组(tuple)类型，例如，size=(m,n,k), 则输出m*n*k个样本，缺省时输出1个值。
cluster2 = np.random.uniform(3.5,4.5,(2,10))
X = np.hstack((cluster1,cluster2)).T
# print(X.shape) #20*1
# 三、遍历k值并可视化
K = range(1,10)
meandistortions = []
for k in K:
    #愚弄KMeans算法
    kmeans  = KMeans(n_clusters=k)
    kmeans.fit(X)
    #print(np.min(cdist(X,kmeans.cluster_centers_,'euclidean'),axis=1)) #注意这里的维度，axis = 1， 0 表示第一维度，按照第一个维度视角，一维度下所有最小值。1 表示第二个维度，按照第二个维度视角来看，按照第二维度一个一个的最小值。
    print("---")
    meandistortions.append(sum(np.min(cdist(X,kmeans.cluster_centers_,'euclidean'),axis=1))/X.shape[0]) #欧氏距离,计算最小欧氏距离的平均值
plt.plot(K,meandistortions,'bx-')
plt.xlabel('k')

# 设置平均畸变过程
plt.ylabel("Ave Distor")

#增加表头
plt.show()






