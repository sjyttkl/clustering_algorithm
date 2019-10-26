# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     lesson01
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/10/10
   Description :  
==================================================
"""
__author__ = 'sjyttkl'

#导入库

import  numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import silhouette_samples,silhouette_score
# %matplotlib inline
#scikit中的make_blobs方法常被用来生成聚类算法的测试数据，
# 直观地说，make_blobs会根据用户指定的特征数量、中心点数量、范围等来生成几类数据，这些数据可用于测试聚类算法的效果。
#生成数据集
X,y = make_blobs(n_samples=1000,n_features=2,centers=[[-1,-1],[0,0],[1,1],[2,2]],cluster_std=[0.4,0.2,0.2,0.2],random_state=9)
#第四个，是每个簇中心的标准差，
#生成数据散点图
plt.scatter(X[:,0],X[:,1],marker='o')


#三 可视化,利用kmeans进行聚类，MiniBatchKmeans 是小batch进行聚类

for index ,k in enumerate((2,3,4,5)):
    plt.subplot(2, 2, index+1)#一个一个的画这个图，
    #尝试使用  k 值进行聚类
    y_pred = MiniBatchKMeans(n_clusters=k,batch_size=200,random_state=9).fit_predict(X)
    score = metrics.calinski_harabaz_score(X,y_pred)
    #CH指标通过计算类中各点与类中心的距离平方和来度量类内的紧密度，
    # 通过计算各类中心点与数据集中心点距离平方和来度量数据集的分离度，
    # CH指标由分离度与紧密度的比值得到。从而，CH越大代表着类自身越紧密，类与类之间越分散，即更优的聚类结果。
    silhouette_score(X, y_pred)  #分析用来研究聚类结果的类间距离。Silhouette数值度量在相同类中的点，与不同类中的点相比的紧密程度。Silhouette图可视化这一测度，这样就提供了一种评价类个数的方法。 Silhouette值在[-1, 1]内，接近1表示样本远离邻近类，取0表示样本几乎在两个近邻类的决策边界上，取负值表示样本被分在错误的类里。在本例中，我们使用Silhouette分析选择一个类个数参数n_clusters的最优值。
    print(score)

    plt.scatter(X[:,0],X[:,1],c=y_pred)
    plt.text(.99,.01,('k=%d,score :%.2f' %(k,score)),
                      transform = plt.gca().transAxes,size=10,
             horizontalalignment="right")
    #axes对象：transData为数据坐标变换对象 transAxes为子图坐标变换对象
plt.show()



