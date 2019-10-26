# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     kernel_kmeans
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/10/22
   Description :  
==================================================
"""
__author__ = 'songdongdong'
"""
标准的k-均值算法在处理线性可分的数据集时会表现出很好的效果，线性可分表现在数据集上中就是：从属同一个类别的数据点之间是紧致的，而不同的类别之间是分散的。

挑战：当数据集中包含的数据点所构成的簇在形状和密度上都有很大差异时，标准k-均值算法就无法发挥它的效果了。

核方法是，通过一个映射将数据集进行转换，转换成标准k-均值算法可以接收的数据样式，然后再用聚类算法进行处理。这就是核k-均值算法（kernel k-means）。

主要思想：通过一个非线性映射，将输入空间中的数据点映射到一个高维特征空间中，并选取合适的核函数代替非线性映射的内积，在特征空间进行聚类分析。这种将数据映射到高维空间的方法可以突出样本类别之间的特征差异，使得样本在核空间线性可分（或近似线性可分）。

输入：所有数据点A，聚类个数k

输出：k个聚类中心点

1：输入数据通过核函数映射到高维空间得到矩阵B

2：对B进行标准k-均值聚类
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

moon_dat = pd.read_excel("moons.xls",header=None)
moon_dat = np.array(moon_dat)
moon_x = moon_dat[:,0]
moon_y = moon_dat[:,1]

#高斯核函数（2维）
def gaussian_2d(x1, y1, x2, y2, ga):
    gau = np.exp(-ga*((x1-x2)**2 + (y1-y2)**2))

    return gau

#计算拉普拉斯矩阵
def laplace_array(moon_x,moon_y,gamma):

    S = np.zeros((len(moon_x),len(moon_y)))

    for i in range(len(moon_x)):
        for j in range(len(moon_y)):
            S[i][j] = gaussian_2d(moon_x[i],moon_y[i],moon_x[j],moon_y[j],gamma)

    D = np.sum(S, axis=1)
    D = np.squeeze(np.array(D))
    D = np.diag(D) #array是一个1维数组时，结果形成一个以一维数组为对角线元素的矩阵，array是一个二维矩阵时，结果输出矩阵的对角线元素

    return D-S # 使用 对角 矩阵，减去 原来的高斯核生成的矩阵。这就是拉布拉斯矩阵

#实现ratio_cut 普聚类方法，返回归一化后的matrix
def ratio_cut(laplace,k):

    val, vec = np.linalg.eig(laplace) #计算特征值和特征向量
    id = np.argsort(val) #排序后，返回index
    topk_vecs = vec[:,id[0:k:1]]  #(1000,2)
    print(id[0:k:1])
    Sqrt = np.array(topk_vecs) * np.array(topk_vecs) #(1000,2)
    print(np.shape(sum(np.transpose(Sqrt)))) # sum,默认是进行第一维进行求和  (2,1000) ==> (1000)
    print(np.shape(np.transpose(sum(np.transpose(Sqrt))))) #有进行了转置操作因为只有一维，所以维度一直是(1000)
    print(np.shape(np.sqrt(np.transpose(sum(np.transpose(Sqrt)))))) # 有进行了转置操作因为只有一维，所以维度一直是(1000)


    divMat = np.tile(np.sqrt(np.transpose(sum(np.transpose(Sqrt)))), (2, 1))#(2,1000)  tile 在第一维度 进行tile ,第二个维度不变
    divMat = np.transpose(divMat)#(1000,2)

    #divMat = np.transpose(np.sqrt(Sqrt))  直接用这条也是可以的。
    F = np.array(topk_vecs) / divMat

    return F


#kmeas算法
def kmeans(feature,class_num):

    core = []

    # 初始化
    for i in range(class_num):
        core.append(feature[i])

    flag = True

    while flag:

        K1 = np.zeros(feature.shape)
        K2 = np.zeros(feature.shape)
        count1 = 0
        count2 = 0

        for i in range(len(feature)):
            if sum((feature[i]-core[0])**2) <= sum((feature[i]-core[1])**2):
                K1[i] = feature[i]
                count1 += 1
            else:
                K2[i] = feature[i]
                count2 += 1

        sm1 = sum(K1) / count1
        sm2 = sum(K2) / count2

        count = 0

        if (sm1[0] == core[0][0]) & (sm1[1] == core[0][1]):
            count += 1
        else:
            core[0] = sm1#更新均值

        if (sm2[0] == core[1][0]) & (sm2[1] == core[1][1]):
            count += 1
        else:
            core[1] = sm2

        if count == 2:
            flag = False#分类完成，结束循环

    return K1,K2

#画图
def plot_moon(data1,data2):

    id_1 = []
    id_2 = []
    for i in range(len(data1)):
        if (data1[i][0] != 0) & (data1[i][1] != 0):
            id_1.append(i)
    for i in range(len(data2)):
        if (data2[i][0] != 0) & (data2[i][1] != 0):
            id_2.append(i)

    id_1 = np.array(id_1)
    id_2 = np.array(id_2)

    result1 = moon_dat[id_1]
    result2 = moon_dat[id_2]

    plt.figure()
    plt.scatter(result1[:, 0], result1[:, 1], color='b')
    plt.scatter(result2[:, 0], result2[:, 1], color='r')
    plt.show()

    return
if __name__ == '__main__':
    #使用高斯核函数将数据映射到高维，使用谱聚类提高运算效率，最后使用kmeans算法进行分类
    gamma = 10 ** 3
    L = laplace_array(moon_x, moon_y, gamma)
    F = ratio_cut(L, 2)
    K1, K2 = kmeans(F, 2)
    plot_moon(K1, K2)
