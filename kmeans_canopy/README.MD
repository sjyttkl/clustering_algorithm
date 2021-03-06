### Canopy算法



​    Canopy属于一种‘粗’聚类算法，即使用一种简单、快捷的距离计算方法将数据集分为若干可重叠的子集canopy，这种算法不需要指定k值、但精度较低，可以结合K-means算法一起使用：先由Canopy算法进行粗聚类得到k个质心，再使用K-means算法进行聚类。

​    Canopy算法步骤如下：

​        （1）将原始样本集随机排列成样本列表L=[x1,x2,...,xm]（排列好后不再更改），根据先验知识或交叉验证调参设定初始距离阈值T1、T2，且T1>T2 。

​        （2）从列表L中随机选取一个样本P作为第一个canopy的质心，并将P从列表中删除。

​        （3）从列表L中随机选取一个样本Q，计算Q到所有质心的距离，考察其中最小的距离D：

​                    如果D≤T1，则给Q一个弱标记，表示Q属于该canopy，并将Q加入其中；

​                    如果D≤T2，则给Q一个强标记，表示Q属于该canopy，且和质心非常接近，所以将该canopy的质心设为所有强标记样本的中心位置，并将Q从列表L中删除；

​                    如果D>T1，则Q形成一个新的聚簇，并将Q从列表L中删除。

​        （4）重复第三步直到列表L中元素个数为零。

   ![1](D:\Program Files\Python_Workspace\clustering_algorithm\kmeans_canopy\pic\1.png)

​    注意：

​        （1）‘粗’距离计算的选择对canopy的分布非常重要，如选择其中某个属性、其他外部属性、欧式距离等。

​        （2）当T2<D≤T1时，样本不会从列表中被删除，而是继续参与下一轮迭代，直到成为新的质心或者某个canopy的强标记成员。

​        （3）T1、T2的取值影响canopy的重叠率及粒度：当T1过大时，会使样本属于多个canopy，各个canopy间区别不明显；当T2过大时，会减少canopy个数，而当T2过小时，会增加canopy个数，同时增加计算时间。

​        （4）canopy之间可能存在重叠的情况，但是不会存在某个样本不属于任何canopy的情况。

​        （5）Canopy算法可以消除孤立点，即删除包含样本数目较少的canopy，往往这些canopy包含的是孤立点或噪音点。

