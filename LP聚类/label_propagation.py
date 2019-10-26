# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     LP聚类.py
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/10/21
   Description :  Label Propagation聚类,标签传播算法,https://blog.csdn.net/zouxy09/article/details/49105265/
==================================================
"""

"""
半监督学习（Semi-supervised learning）发挥作用的场合是：你的数据有一些有label，一些没有。
而且一般是绝大部分都没有，只有少许几个有label。半监督学习算法会充分的利用unlabeled数据来捕捉我们整个数据的潜在分布。
它基于三大假设：

1）Smoothness平滑假设：相似的数据具有相同的label。
2）Cluster聚类假设：处于同一个聚类下的数据具有相同label。
3）Manifold流形假设：处于同一流形结构下的数据具有相同label。

标签传播算法（label propagation）的核心思想非常简单：相似的数据应该具有相同的label。
LP算法包括两大步骤：
    1）构造相似矩阵；
    2）勇敢的传播吧。
    
Label propagation是基于标传播的一种社区划分算法。Label Propagation Algorithm简称LPA算法，
也可以是说是一种划分小团体的算法。这种社区划分的方法有很多，LPA只是一种最简单的一种。
比如，以微博为例，用户在微博上可以关注感兴趣的人，同样也会被其他人关注，这样用户和用户之间就存在了关系，
使用LPA就可以对用户进行聚类操作，相同兴趣点的用户可以聚类在一起，划分一起之后就可以统一进行推荐了，这样就可以用LPA。
"""

__author__ = 'songdongdong'

import time
import numpy as np


# return k neighbors index
def navie_knn(dataSet, query, k):
    numSamples = dataSet.shape[0]

    ## step 1: calculate Euclidean distance # 欧式度量
    diff = np.tile(query, (numSamples, 1)) - dataSet #(801,2) - (801,2) = （801,2）,
    squaredDiff = diff ** 2
    squaredDist = np.sum(squaredDiff, axis=1)  # sum is performed by row  #（801,）

    ## step 2: sort the distance
    sortedDistIndices = np.argsort(squaredDist) #从小到大排序，返回index
    if k > len(sortedDistIndices): # 如果长度小于 K 则返回全部，
        k = len(sortedDistIndices)

    return sortedDistIndices[0:k] #取出前 k个范围的数据


# build a big graph (normalized weight matrix)
def buildGraph(MatX, kernel_type, rbf_sigma=None, knn_num_neighbors=None):
    """
    根据 核函数 进行构建  图， 相似矩阵构建
    :param MatX: 数据矩阵
    :param kernel_type: 核函数的类型
    :param rbf_sigma: 方差
    :param knn_num_neighbors:  knn 临近数
    :return:
    """
    num_samples = MatX.shape[0] #数据矩阵的个数
    affinity_matrix = np.zeros((num_samples, num_samples), np.float32) #相关关系矩阵
    if kernel_type == 'rbf':
        if rbf_sigma == None:
            raise ValueError('You should input a sigma of rbf kernel!')
        #每个点进行求距离。
        for i in range(num_samples):
            row_sum = 0.0
            for j in range(num_samples):
                diff = MatX[i, :] - MatX[j, :]
                affinity_matrix[i][j] = np.exp(sum(diff ** 2) / (-2.0 * rbf_sigma ** 2))
                row_sum += affinity_matrix[i][j]
            affinity_matrix[i][:] /= row_sum
    elif kernel_type == 'knn': #核函数为 knn
        if knn_num_neighbors == None:
            raise ValueError('You should input a k of knn kernel!')
        for i in range(num_samples):
            k_neighbors = navie_knn(MatX, MatX[i, :], knn_num_neighbors) # 每个点需要和所有点 进行 距离计算，并且返回，前 k 个距离最小 点的 index
            affinity_matrix[i][k_neighbors] = 1.0 / knn_num_neighbors # 前  最前 k 个近距离的点 都设为 0.1
    else:
        raise NameError('Not support kernel type! You can use knn or rbf!')

    return affinity_matrix


# label propagation
def labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type='rbf', rbf_sigma=1.5, \
                     knn_num_neighbors=10, max_iter=500, tol=1e-3):
    """
     label 进行传播，
    :param Mat_Label: 有标签的 数据
    :param Mat_Unlabel: 无标签的 数据
    :param labels:所有的标签
    :param kernel_type: 核函数
    :param rbf_sigma: 方差，一般是高斯核
    :param knn_num_neighbors: knn 邻近数
    :param max_iter: 最大迭代书
    :param tol: 前后传播的差值 到某个值进停止 传播，防止震荡
    :return:
    """
    # initialize
    num_label_samples = Mat_Label.shape[0]
    num_unlabel_samples = Mat_Unlabel.shape[0]
    num_samples = num_label_samples + num_unlabel_samples
    labels_list = np.unique(labels)
    num_classes = len(labels_list)

    MatX = np.vstack((Mat_Label, Mat_Unlabel)) # 有标签 和无标签 组成的 数据矩阵已经形成
    clamp_data_label = np.zeros((num_label_samples, num_classes), np.float32) #制作 打过标签数据矩阵， 并且进行分类
    for i in range(num_label_samples):
        clamp_data_label[i][labels[i]] = 1.0

    label_function = np.zeros((num_samples, num_classes), np.float32) #（801,2） 列代表label,  制作 未打过标签的数据 和 打过标签的数据 都制作成  矩阵，并且把
    label_function[0: num_label_samples] = clamp_data_label #融合 有标签的 数据，
    label_function[num_label_samples: num_samples] = -1 # 没有标签的数据，全都赋值为  -1
    # 有标签 和无标签 组成的 label 矩阵已经形成。
    # graph construction
    affinity_matrix = buildGraph(MatX, kernel_type, rbf_sigma, knn_num_neighbors)

    # start to propagation
    iter = 0
    pre_label_function = np.zeros((num_samples, num_classes), np.float32) # 构建 样本矩阵（801,2） 样本*label
    changed = np.abs(pre_label_function - label_function).sum() #求出 当前 数据集和 之前数据集传播后的 差值 综合 的绝对值
    while iter < max_iter and changed > tol:
        if iter % 1 == 0:
            print("---> Iteration %d/%d, changed: %f" % (iter, max_iter, changed))
        pre_label_function = label_function #传播前，记录先前的  label矩阵分布
        iter += 1

        # propagation
        label_function = np.dot(affinity_matrix, label_function) #利用点乘进行传播(802,2)

        # clamp
        label_function[0: num_label_samples] = clamp_data_label #原来有标签的 label不能变，需要重置回去。

        # check converge
        changed = np.abs(pre_label_function - label_function).sum() #求出差值 综合 的绝对值

    # get terminate label of unlabeled data
    unlabel_data_labels = np.zeros(num_unlabel_samples)
    for i in range(num_unlabel_samples):
        unlabel_data_labels[i] = np.argmax(label_function[i + num_label_samples]) #取出 缺少标签数据，每行最大数 index,(0,1)代表的就是分类。

    return unlabel_data_labels
