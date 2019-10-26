## DBSCAN方法及应用

### 1.DBSCAN密度聚类简介

DBSCAN 算法是一种基于密度的聚类算法：
　　1.聚类的时候不需要预先指定簇的个数
　　2.最终的簇的个数不确定
DBSCAN算法将数据点分为三类：
　　1.核心点：在半径Eps内含有超过MinPts数目的点。
　　2.边界点：在半径Eps内点的数量小于MinPts,但是落在核心点的邻域内的点。
　　3.噪音点：既不是核心点也不是边界点的点。

如下图所示：图中黄色的点为边界点，因为在半径Eps内，它领域内的点不超过MinPts个，我们这里设置的MinPts为5；而中间白色的点之所以为核心点，是因为它邻域内的点是超过MinPts（5）个点的，它邻域内的点就是那些黄色的点

![1](D:\Program Files\Python_Workspace\clustering_algorithm\DBSCAN\pic\1.png)

### 2.DBSCAN算法的流程

1.将所有点标记为核心点、边界点或噪声点；
		2.删除噪声点；
		3.为距离在Eps之内的所有核心点之间赋予一条边；
		4.每组连通的核心点形成一个簇；
		5.将每个边界点指派到一个与之关联的核心点的簇中（哪一个核心点的半径范围之内）。

![2](D:\Program Files\Python_Workspace\clustering_algorithm\DBSCAN\pic\2.png)

![3](D:\Program Files\Python_Workspace\clustering_algorithm\DBSCAN\pic\3.png)

![4](D:\Program Files\Python_Workspace\clustering_algorithm\DBSCAN\pic\4.png)

![5](D:\Program Files\Python_Workspace\clustering_algorithm\DBSCAN\pic\5.png)

![6](D:\Program Files\Python_Workspace\clustering_algorithm\DBSCAN\pic\6.png)

### DBSCAN算法的主要思想是，

认为密度稠密的区域是一个聚类，各个聚类是被密度稀疏的区域划分开来的。 也就是说，密度稀疏的区域构成了各个聚类之间的划分界限。与K-means等算法相比，

该算法的主要优点包括：可以自主计算聚类的数目，不需要认为指定；不要求类的形状是凸的，可以是任意形状的。

DBSCAN中包含的几个关键概念包括core sample，non-core sample，min_sample，eps。

+ core samle是指，在该数据点周围eps范围内，至少包含min_sample个其他数据点，则该点是core sample， 这些数据点称为core sample的邻居。与之对应的，non-core sample是该点周围eps范围内，所包含的数据点个数少于min_sample个。从定义可知，core sample是位于密度稠密区域的点。

+ 一个聚类就是一个core sample的集合，这个集合的构建过程是一个递归的构成。
   首先，找到任意个core sample，然后从它的邻居中找到core sample， 接着递归的从这些邻居中的core sample的邻居中继续找core sample。 要注意core sample的邻居中不仅有其他core sample，也有一些non-core smaple， 也正是因为这个原因，聚类集合中也包含少量的non-core sample，它们是聚类中core sample的邻居， 但自己不是core sample。这些non-core sample构成了边界。

+ 在确定了如何通过单一core sample找到了一个聚类后，下面描述DBSCAN算法的整个流程。 
  + 首先，扫描数据集找到任意一个core sample，以此core sample为起点，按照上一段描述的方法进行扩充，确定一个聚类。
  + 然后，再次扫描数据集，找到任意一个不属于以确定类别的core sample，重复扩充过程。
  + 再次确定一个聚类。 迭代这个过程，直至数据集中不再包含有core sample。 这也是为什么DBSCAN不用认为指定聚类数目的原因。

+ DBSCAN算法包含一定的非确定性。数据中的core sample总是会被分配到相同的聚类中的，哪怕在统一数据集上多次运行DBSCAN。其不确定性主要体现在non-core sample的分配上。 一个non-core sample可能同时是两个core sample的邻居，而这两个core sample隶属于不同的聚类。
+  DBSCAN中，这个non-core sample会被分配给首先生成的那个聚类，而哪个聚类先生成是随机的。

sklearn中DBSCAN的实现中，邻居的确定使用的ball tree和kd-tree思想，这就避免了计算距离矩阵。

### DBSCAN的主要优点有：

　+  可以对任意形状的稠密数据集进行聚类，相对的，K-Means之类的聚类算法一般只适用于凸数据集。
　+  可以在聚类的同时发现异常点，对数据集中的异常点不敏感。
　+  聚类结果没有偏倚，相对的，K-Means之类的聚类算法初始值对聚类结果有很大影响。

### DBSCAN的主要缺点有：

+ 如果样本集的密度不均匀、聚类间距差相差很大时，聚类质量较差，这时用DBSCAN聚类一般不适合。
+ 如果样本集较大时，**聚类收敛时间较长**，此时可以对搜索最近邻时建立的KD树或者球树进行规模限制来改进。
+ 调参相对于传统的K-Means之类的聚类算法稍复杂，不同的参数组合对最后的聚类效果有较大影响。