# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     层次聚类
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2019/10/26
   Description :  
==================================================
"""
__author__ = 'songdongdong'

"""
sklearn中实现的是自底向上的层次聚类，实现方法是sklearn.cluster.AgglomerativeClustering。
初始时，所有点各自单独成为一类，然后采取某种度量方法将相近的类进行合并，并且度量方法有多种选择。
合并的过程可以构成一个树结构，其根节点就是所有数据的集合，叶子节点就是各条单一数据。
sklearn.cluster.AgglomerativeClustering中可以通过参数linkage选择不同的度量方法，用来度量两个类之间的距离，
可选参数有ward,complete,average三个。

ward:选择这样的两个类进行合并，合并后的类的离差平方和最小。

complete:两个类的聚类被定义为类内数据的最大距离，即分属两个类的距离最远的两个点的距离。
选择两个类进行合并时，从现有的类中找到两个类使得这个值最小，就合并这两个类。

average:两个类内数据两两之间距离的平均值作为两个类的距离。
同样的，从现有的类中找到两个类使得这个值最小，就合并这两个类。

Agglomerative cluster有一个缺点，就是rich get richer现象，
这可能导致聚类结果得到的类的大小不均衡。
从这个角度考虑，complete策略效果最差，ward得到的类的大小最为均衡。
但是在ward策略下，affinity参数只能是“euclidean”，即欧式距离。
如果在欧氏距离不适用的环境中，average is a good alternative。

另外还应该注意参数affinity，这个参数设置的是计算两个点之间距离时采用的策略，
注意和参数linkage区分，linkage设置的是衡量两个类之间距离时采用的策略，
而点之间的距离衡量是类之间距离衡量的基础。
affinity的可选数值包括 “euclidean”, “l1”, “l2”, “manhattan”, “cosine”,
‘precomputed’. If linkage is “ward”, only “euclidean” is accepted.


"""