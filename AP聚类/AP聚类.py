# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     AP聚类
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/10/21
   Description :   Affinity（类同） Propagation
==================================================
"""
__author__ = 'sjyttkl'

"""
AP聚类算法是基于数据点间的"信息传递"的一种聚类算法。
与k-均值算法或k中心点算法不同，AP算法不需要在运行算法之前确定聚类的个数。
AP算法寻找的"examplars"即聚类中心点是数据集合中实际存在的点，作为每类的代表。

在AP算法中有一些特殊名词：

Exemplar：指的是聚类中心，K-Means中的质心。
Similarity：数据点i和点j的相似度记为s(i, j)，是指点j作为点i的聚类中心的相似度。一般使用欧氏距离来计算，一般点与点的相似度值全部取为负值；因此，相似度值越大说明点与点的距离越近，便于后面的比较计算。
Preference：数据点i的参考度称为p(i)或s(i,i)，是指点i作为聚类中心的参考度。一般取s相似度值的中值。
Responsibility：r(i,k)用来描述点k适合作为数据点i的聚类中心的程度。
Availability：a(i,k)用来描述点i选择点k作为其聚类中心的适合程度。
Damping factor(阻尼系数)：主要是起收敛作用的。

AP和K-Means运行时间对比
"""
print("====================================AP和K-Means运行时间对比============================")
from sklearn import metrics

from sklearn.cluster import AffinityPropagation
from sklearn.datasets.samples_generator import make_blobs

# 生成数据
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
                            random_state=0)

# #############################################################################

# Compute Affinity Propagation
af = AffinityPropagation(preference=-50).fit(X) #AffinityPropagation可配置的参数包括：（重点是damping和preference）
cluster_centers_indices = af.cluster_centers_indices_ #:中心样本的指标，聚类的中心索引
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels)) #每一个聚出的类仅包含一个类别的程度度量
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels)) #:每一个类别被指向相同聚出的类的程度度量。
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels)) #上面两个折中 v = 2 * (homogeneity * completeness) / (homogeneity + completeness)
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels)) #调整的兰德系数,ARI取值范围为[-1,1],从广义的角度来讲，ARI衡量的是两个数据分布的吻合程度
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels)) #调整的互信息。利用基于互信息的方法来衡量聚类效果需要实际类别信息，MI与NMI取值范围为[0,1],AMI取值范围为[-1,1]。
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels, metric='sqeuclidean')) #轮廓系数 ,对于一个样本点(b - a)/max(a, b) a平均类内距离，b样本点到与其最近的非此类的距离。silihouette_score返回的是所有样本的该值,取值范围为[-1,1]。
print("Calinski-Harabaz Index: %0.3f " % metrics.calinski_harabaz_score(X, labels,))#在真实的分群label不知道的情况下(内部度量)：CH指标通过计算类中各点与类中心的距离平方和来度量类内的紧密度，通过计算各类中心点与数据集中心点距离平方和来度量数据集的分离度，CH指标由分离度与紧密度的比值得到。从而，CH越大代表着类自身越紧密，类与类之间越分散，即更优的聚类结果。


"""
AP算法的应用场景
图像、文本、生物信息学、人脸识别、基因发现、搜索最优航线、 码书设计以及实物图像识别等领域。
"""

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk') #这里只是设置颜色
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k # 如果 聚类 为 k[0,1,2]，则为 True。
    cluster_center = X[cluster_centers_indices[k]] #根据索引，取出 聚类中心
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.') # 画出聚类出的点
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14) #画点聚类的中心；
    for x in X[class_members]:
         plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col) #画出 连线。

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

"""
总结：

综合来看，由于AP算法不适用均值做质心计算规则，因此对于离群点和异常值不敏感；
同时其初始值不敏感的特性也能保持模型的较好鲁棒性。这两个突出特征使得它可以作为K-Means算法的一个有效补充，
但在大数据量下的耗时过长，这导致它的适用范围只能是少量数据；
虽然通过调整damping（收敛规则）可以在一定程度上提升运行速度（damping值调小），
但由于算法本身的局限性决定了这也只是杯水车薪。
"""