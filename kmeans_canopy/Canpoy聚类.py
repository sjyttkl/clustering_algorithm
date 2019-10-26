# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     Canpoy聚类
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/10/27
   Description :  
==================================================
"""
"""
Canopy算法是2000年由Andrew McCallum, Kamal Nigam and Lyle Ungar提出来的，
它是对k-means聚类算法和层次聚类算法的预处理。众所周知，kmeans的一个不足之处在于k值需要通过人为的进行调整，
后期可以通过肘部法则（Elbow Method）和轮廓系数（Silhouette Coefficient）来对k值进行最终的确定，
但是这些方法都是属于“事后”判断的，而Canopy算法的作用就在于它是通过事先粗聚类的方式，
为k-means算法确定初始聚类中心个数和聚类中心点。
"""


__author__ = 'songdongdong'

import math
import random
import numpy as np
from datetime import datetime
from pprint import pprint as p
import matplotlib.pyplot as plt

# 随机生成500个二维[0,1)平面点
dataset = np.random.rand(500, 2)


class Canopy:
    def __init__(self, dataset):
        self.dataset = dataset
        self.t1 = 0
        self.t2 = 0

    # 设置初始阈值
    def setThreshold(self, t1, t2):
        if t1 > t2:
            self.t1 = t1
            self.t2 = t2
        else:
            print('t1 needs to be larger than t2!')

    # 使用欧式距离进行距离的计算
    def euclideanDistance(self, vec1, vec2):
        return math.sqrt(((vec1 - vec2)**2).sum())

    # 根据当前dataset的长度随机选择一个下标
    def getRandIndex(self):
        return random.randint(0, len(self.dataset) - 1)

    def clustering(self):
        if self.t1 == 0:
            print('Please set the threshold.')
        else:
            canopies = []  # 用于存放最终归类结果
            # while len(self.dataset) != 0:
            # 20180324修改
            while len(self.dataset) > 1:
                rand_index = self.getRandIndex()
                current_center = self.dataset[rand_index]  # 随机获取一个中心点，定为P点
                current_center_list = []  # 初始化P点的canopy类容器
                delete_list = []  # 初始化P点的删除容器
                self.dataset = np.delete(self.dataset, rand_index,
                                         0)  # 删除随机选择的中心点P
                for datum_j in range(len(self.dataset)):
                    datum = self.dataset[datum_j]
                    distance = self.euclideanDistance(
                        current_center, datum)  # 计算选取的中心点P到每个点之间的距离
                    if distance < self.t1:
                        # 若距离小于t1，则将点归入P点的canopy类
                        current_center_list.append(datum)
                    if distance < self.t2:
                        delete_list.append(datum_j)  # 若小于t2则归入删除容器
                # 根据删除容器的下标，将元素从数据集中删除
                self.dataset = np.delete(self.dataset, delete_list, 0)
                canopies.append((current_center, current_center_list))
        return canopies


def showCanopy(canopies, dataset, t1, t2):
    fig = plt.figure()
    sc = fig.add_subplot(111)
    colors = [
        'brown', 'green', 'blue', 'y', 'r', 'tan', 'dodgerblue', 'deeppink',
        'orangered', 'peru', 'blue', 'y', 'r', 'gold', 'dimgray', 'darkorange',
        'peru', 'blue', 'y', 'r', 'cyan', 'tan', 'orchid', 'peru', 'blue', 'y',
        'r', 'sienna'
    ]
    markers = [
        '*', 'h', 'H', '+', 'o', '1', '2', '3', ',', 'v', 'H', '+', '1', '2',
        '^', '<', '>', '.', '4', 'H', '+', '1', '2', 's', 'p', 'x', 'D', 'd',
        '|', '_'
    ]
    for i in range(len(canopies)):
        canopy = canopies[i]
        center = canopy[0]
        components = canopy[1]
        sc.plot(
            center[0],
            center[1],
            marker=markers[i],
            color=colors[i],
            markersize=10)
        t1_circle = plt.Circle(
            xy=(center[0], center[1]),
            radius=t1,
            color='dodgerblue',
            fill=False)
        t2_circle = plt.Circle(
            xy=(center[0], center[1]), radius=t2, color='skyblue', alpha=0.2)
        sc.add_artist(t1_circle)
        sc.add_artist(t2_circle)
        for component in components:
            sc.plot(
                component[0],
                component[1],
                marker=markers[i],
                color=colors[i],
                markersize=1.5)
    maxvalue = np.amax(dataset)
    minvalue = np.amin(dataset)
    plt.xlim(minvalue - t1, maxvalue + t1)
    plt.ylim(minvalue - t1, maxvalue + t1)
    plt.show()


def main():
    t1 = 0.6
    t2 = 0.4
    gc = Canopy(dataset)
    gc.setThreshold(t1, t2)
    canopies = gc.clustering()
    print('Get %s initial centers.' % len(canopies))
    showCanopy(canopies, dataset, t1, t2)


if __name__ == '__main__':
    t_s = datetime.now()
    main()
    t_e = datetime.now()
    usedtime = t_e - t_s
    print('[%s]' % usedtime)