# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     DBSCAN聚类
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/10/25
   Description :  
==================================================
"""

"""
DBSCAN 算法是一种基于密度的聚类算法：
　　1.聚类的时候不需要预先指定簇的个数
　　2.最终的簇的个数不确定
DBSCAN算法将数据点分为三类：
　　1.核心点：在半径Eps内含有超过MinPts数目的点。
　　2.边界点：在半径Eps内点的数量小于MinPts,但是落在核心点的邻域内的点。
　　3.噪音点：既不是核心点也不是边界点的点。

DBSCAN算法迭代可视化展示
国外有一个特别有意思的网站：https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/

"""
__author__ = 'songdongdong'


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

def make_datas():
    """
    返回：样本相似度矩阵、样本数、
    :return:
    """
    data = make_moons(n_samples=200,shuffle=True,random_state=2)
    data,label = data

    n, m = data.shape
    all_index = np.arange(n)
    disMatrix = np.zeros([n, n])
    # data = np.delete(data, m - 1, axis=1)
    return disMatrix,n,data
def dis_vec(a, b):
    """
    计算两个向量的距离
    """
    if len(a) != len(b):
        return Exception
    else:
        return np.sqrt(np.sum(np.square(a - b)))
def dis_matx(n,dis):
    """
     计算距离矩阵
    :param n: 输入矩阵 宽度
    :return:
    """
    for i in range(n):
        for j in range(i):
            dis[i, j] = dis_vec(data[i], data[j])
            dis[j, i] = dis[i, j]


def dbscan(s, minpts,all_index,dis):
    """
    密度聚类
    :param s:
    :param minpts:
    :return:
    """
    center_points = []  # 存放最终的聚类结果
    k = 0  # 检验是否进行了合并过程
    for i in range(n):
        if sum(dis[i] <= s) >= minpts:  # 查看距离矩阵的第i行是否满足条件
            if len(center_points) == 0:  # 如果列表长为0，则直接将生成的列表加入
                center_points.append(list(all_index[dis[i] <= s]))
            else:
                for j in range(len(center_points)):  # 查找是否有重复的元素
                    if set(all_index[dis[i] <= s]) & set(center_points[j]):
                        center_points[j].extend(list(all_index[dis[i] <= s]))
                        k = 1  # 执行了合并操作
                if k == 0:
                    center_points.append(list(all_index[dis[i] <= s]))  # 没有执行合并说明这个类别单独加入
                k = 0

    lenc = len(center_points)

    # 以下这段代码是进一步查重，center_points中所有的列表并非完全独立，还有很多重复
    # 那么为何上面代码已经查重了，这里还需查重，其实可以将上面的步骤统一放到这里，但是时空复杂的太高
    # 经过第一步查重后，center_points中的元素数目大大减少，此时进行查重更快！
    k = 0
    for i in range(lenc - 1):
        for j in range(i + 1, lenc):
            if set(center_points[i]) & set(center_points[j]):
                center_points[j].extend(center_points[i])
                center_points[j] = list(set(center_points[j]))
                k = 1

        if k == 1:
            center_points[i] = []  # 合并后的列表置空
        k = 0

    center_points = [s for s in center_points if s != []]  # 删掉空列表即为最终结果

    return center_points


if __name__ == '__main__':
    disMatrix, n, data = make_datas()
    dis_matx(n,disMatrix)
    center_points = dbscan(0.2, 10,np.arange(n),disMatrix)  # 半径和元素数目
    c_n = center_points.__len__()  # 聚类完成后的类别数目
    print(c_n)
    ct_point = []
    color = ['g', 'r', 'b', 'm', 'k']
    noise_point = np.arange(n)  # 没有参与聚类的点即为噪声点
    for i in range(c_n):
        ct_point = list(set(center_points[i]))
        noise_point = set(noise_point) - set(center_points[i])
        print(ct_point.__len__())  # 输出每一类的点个数
        print(ct_point)  # 输出每一类的点
        print("**********")

    noise_point = list(noise_point)

    for i in range(c_n):
        ct_point = list(set(center_points[i]))
        plt.scatter(data[ct_point, 0], data[ct_point, 1], color=color[i])  # 画出不同类别的点
    plt.scatter(data[noise_point, 0], data[noise_point, 1], color=color[c_n], marker='h', linewidths=0.1)  # 画噪声点
    plt.show()

