# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     label_propagation_MPI
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/10/25
   Description :  LP算法MPI并行实现 ： https://blog.csdn.net/zouxy09/article/details/49105265/#commentsedit
==================================================
"""
__author__ = 'songdongdong'

"""
这里，我们测试的是LP的变身版本。从公式，我们可以看到，第二项PULYL迭代过程并没有发生变化，所以这部分实际上从迭代开始就可以计算好，从而避免重复计算。不过，不管怎样，LP算法都要计算一个UxU的矩阵PUU和一个UxC矩阵FU的乘积。当我们的unlabeled数据非常多，而且类别也很多的时候，计算是很慢的，同时占用的内存量也非常大。另外，构造Graph需要计算两两的相似度，也是O(n2)的复杂度，当我们数据的特征维度很大的时候，这个计算量也是非常客观的。所以我们就得考虑并行处理了。而且最好是能放到集群上并行。那如何并行呢？

对算法的并行化，一般分为两种：数据并行和模型并行。
数据并行很好理解，就是将数据划分，每个节点只处理一部分数据，例如我们构造图的时候，计算每个数据的k近邻。例如我们有1000个样本和20个CPU节点，那么就平均分发，让每个CPU节点计算50个样本的k近邻，然后最后再合并大家的结果。可见这个加速比也是非常可观的。
模型并行一般发生在模型很大，无法放到单机的内存里面的时候。例如庞大的深度神经网络训练的时候，就需要把这个网络切开，然后分别求解梯度，最后有个leader的节点来收集大家的梯度，再反馈给大家去更新。当然了，其中存在更细致和高效的工程处理方法。在我们的LP算法中，也是可以做模型并行的。假如我们的类别数C很大，把类别数切开，让不同的CPU节点处理，实际上就相当于模型并行了。

那为啥不切大矩阵PUU，而是切小点的矩阵FU，因为大矩阵PUU没法独立分块，并行的一个原则是处理必须是独立的。 矩阵FU依赖的是所有的U，而把PUU切开分发到其他节点的时候，每次FU的更新都需要和其他的节点通信，这个通信的代价是很大的（实际上，很多并行系统没法达到线性的加速度的瓶颈是通信！线性加速比是，我增加了n台机器，速度就提升了n倍）。但是对类别C也就是矩阵FU切分，就不会有这个问题，因为他们的计算是独立的。只是决定样本的最终类别的时候，将所有的FU收集回来求ma

所以，在下面的代码中，是同时包含了数据并行和模型并行的雏形的。另外，还值得一提的是，我们是迭代算法，那决定什么时候迭代算法停止？除了判断收敛外，我们还可以让每迭代几步，就用测试label测试一次结果，看模型的整体训练性能如何。特别是判断训练是否过拟合的时候非常有效。因此，代码中包含了
好了，代码终于来了。大家可以搞点大数据库来测试，如果有MPI集群条件的
下面的代码依赖numpy、scipy（用其稀疏矩阵加速计算）和mpi4py。其中mpi4py需要依赖openmpi和Cpython

"""
# ***************************************************************************
# *
# * Description: label propagation
# * Author: Zou Xiaoyi (zouxy09@qq.com)
# * Date:   2015-10-15
# * HomePage: http://blog.csdn.net/zouxy09
# *
# **************************************************************************

import os, sys, time
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, eye
import operator
import cPickle as pickle
import mpi4py.MPI as MPI

#
#   Global variables for MPI
#

# instance for invoking MPI related functions
comm = MPI.COMM_WORLD
# the node rank in the whole community
comm_rank = comm.Get_rank()
# the size of the whole community, i.e., the total number of working nodes in the MPI cluster
comm_size = comm.Get_size()


# load mnist dataset
def load_MNIST():
    import gzip
    f = gzip.open("mnist.pkl.gz", "rb")
    train, val, test = pickle.load(f)
    f.close()

    Mat_Label = train[0]
    labels = train[1]
    Mat_Unlabel = test[0]
    groundtruth = test[1]
    labels_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    return Mat_Label, labels, labels_id, Mat_Unlabel, groundtruth


# return k neighbors index
def navie_knn(dataSet, query, k):
    numSamples = dataSet.shape[0]

    ## step 1: calculate Euclidean distance
    diff = np.tile(query, (numSamples, 1)) - dataSet
    squaredDiff = diff ** 2
    squaredDist = np.sum(squaredDiff, axis=1)  # sum is performed by row

    ## step 2: sort the distance
    sortedDistIndices = np.argsort(squaredDist)
    if k > len(sortedDistIndices):
        k = len(sortedDistIndices)
    return sortedDistIndices[0:k]


# build a big graph (normalized weight matrix)
# sparse U x (U + L) matrix
def buildSubGraph(Mat_Label, Mat_Unlabel, knn_num_neighbors):
    num_unlabel_samples = Mat_Unlabel.shape[0]
    data = []
    indices = []
    indptr = [0]
    Mat_all = np.vstack((Mat_Label, Mat_Unlabel))
    values = np.ones(knn_num_neighbors, np.float32) / knn_num_neighbors
    for i in range(num_unlabel_samples):
        k_neighbors = navie_knn(Mat_all, Mat_Unlabel[i, :], knn_num_neighbors)
        indptr.append(np.int32(indptr[-1]) + knn_num_neighbors)
        indices.extend(k_neighbors)
        data.append(values)
    return csr_matrix((np.hstack(data), indices, indptr))


# build a big graph (normalized weight matrix)
# sparse U x (U + L) matrix
def buildSubGraph_MPI(Mat_Label, Mat_Unlabel, knn_num_neighbors):
    num_unlabel_samples = Mat_Unlabel.shape[0]
    local_data = []
    local_indices = []
    local_indptr = [0]
    Mat_all = np.vstack((Mat_Label, Mat_Unlabel))
    values = np.ones(knn_num_neighbors, np.float32) / knn_num_neighbors
    sample_offset = np.linspace(0, num_unlabel_samples, comm_size + 1).astype('int')
    for i in range(sample_offset[comm_rank], sample_offset[comm_rank + 1]):
        k_neighbors = navie_knn(Mat_all, Mat_Unlabel[i, :], knn_num_neighbors)
        local_indptr.append(np.int32(local_indptr[-1]) + knn_num_neighbors)
        local_indices.extend(k_neighbors)
        local_data.append(values)
    data = np.hstack(comm.allgather(local_data))
    indices = np.hstack(comm.allgather(local_indices))
    indptr_tmp = comm.allgather(local_indptr)
    indptr = []
    for i in range(len(indptr_tmp)):
        if i == 0:
            indptr.extend(indptr_tmp[i])
        else:
            last_indptr = indptr[-1]
            del (indptr[-1])
            indptr.extend(indptr_tmp[i] + last_indptr)
    return csr_matrix((np.hstack(data), indices, indptr), dtype=np.float32)


# label propagation
def run_label_propagation_sparse(knn_num_neighbors=20, max_iter=100, tol=1e-4, test_per_iter=1):
    # load data and graph
    print("Processor %d/%d loading graph file..." % (comm_rank, comm_size))
    # Mat_Label, labels, Mat_Unlabel, groundtruth = loadFourBandData()
    Mat_Label, labels, labels_id, Mat_Unlabel, unlabel_data_id = load_MNIST()
    if comm_size > len(labels_id):
        raise ValueError("Sorry, the processors must be less than the number of classes")
    # affinity_matrix = buildSubGraph(Mat_Label, Mat_Unlabel, knn_num_neighbors)
    affinity_matrix = buildSubGraph_MPI(Mat_Label, Mat_Unlabel, knn_num_neighbors)

    # get some parameters
    num_classes = len(labels_id)
    num_label_samples = len(labels)
    num_unlabel_samples = Mat_Unlabel.shape[0]

    affinity_matrix_UL = affinity_matrix[:, 0:num_label_samples]
    affinity_matrix_UU = affinity_matrix[:, num_label_samples:num_label_samples + num_unlabel_samples]

    if comm_rank == 0:
        print("Have %d labeled images, %d unlabeled images and %d classes" % (num_label_samples, num_unlabel_samples, num_classes))

    # divide label_function_U and label_function_L to all processors
    class_offset = np.linspace(0, num_classes, comm_size + 1).astype('int')

    # initialize local label_function_U
    local_start_class = class_offset[comm_rank]
    local_num_classes = class_offset[comm_rank + 1] - local_start_class
    local_label_function_U = eye(num_unlabel_samples, local_num_classes, 0, np.float32, format='csr')

    # initialize local label_function_L
    local_label_function_L = lil_matrix((num_label_samples, local_num_classes), dtype=np.float32)
    for i in range(num_label_samples):
        class_off = int(labels[i]) - local_start_class
        if class_off >= 0 and class_off < local_num_classes:
            local_label_function_L[i, class_off] = 1.0
    local_label_function_L = local_label_function_L.tocsr()
    local_label_info = affinity_matrix_UL.dot(local_label_function_L)
    print("Processor %d/%d has to process %d classes..." % (comm_rank, comm_size, local_label_function_L.shape[1]))

    # start to propagation
    iter = 1
    changed = 100.0
    evaluation(num_unlabel_samples, local_start_class, local_label_function_U, unlabel_data_id, labels_id)
    while True:
        pre_label_function = local_label_function_U.copy()

        # propagation
        local_label_function_U = affinity_matrix_UU.dot(local_label_function_U) + local_label_info

        # check converge
        local_changed = abs(pre_label_function - local_label_function_U).sum()
        changed = comm.reduce(local_changed, root=0, op=MPI.SUM)
        status = 'RUN'
        test = False
        if comm_rank == 0:
            if iter % 1 == 0:
                norm_changed = changed / (num_unlabel_samples * num_classes)
                print("---> Iteration %d/%d, changed: %f" % (iter, max_iter, norm_changed))
            if iter >= max_iter or changed < tol:
                status = 'STOP'
                print("************** Iteration over! ****************")
            if iter % test_per_iter == 0:
                test = True
            iter += 1
        test = comm.bcast(test if comm_rank == 0 else None, root=0)
        status = comm.bcast(status if comm_rank == 0 else None, root=0)
        if status == 'STOP':
            break
        if test == True:
            evaluation(num_unlabel_samples, local_start_class, local_label_function_U, unlabel_data_id, labels_id)
    evaluation(num_unlabel_samples, local_start_class, local_label_function_U, unlabel_data_id, labels_id)


def evaluation(num_unlabel_samples, local_start_class, local_label_function_U, unlabel_data_id, labels_id):
    # get local label with max score
    if comm_rank == 0:
        print("Start to combine local result...")
    local_max_score = np.zeros((num_unlabel_samples, 1), np.float32)
    local_max_label = np.zeros((num_unlabel_samples, 1), np.int32)
    for i in range(num_unlabel_samples):
        local_max_label[i, 0] = np.argmax(local_label_function_U.getrow(i).todense())
        local_max_score[i, 0] = local_label_function_U[i, local_max_label[i, 0]]
        local_max_label[i, 0] += local_start_class

    # gather the results from all the processors
    if comm_rank == 0:
        print("Start to gather results from all processors")
    all_max_label = np.hstack(comm.allgather(local_max_label))
    all_max_score = np.hstack(comm.allgather(local_max_score))

    # get terminate label of unlabeled data
    if comm_rank == 0:
        print("Start to analysis the results...")
        right_predict_count = 0
        for i in range(num_unlabel_samples):
            if i % 1000 == 0:
                print("***", all_max_score[i])
            max_idx = np.argmax(all_max_score[i])
            max_label = all_max_label[i, max_idx]
            if int(unlabel_data_id[i]) == int(labels_id[max_label]):
                right_predict_count += 1
        accuracy = float(right_predict_count) * 100.0 / num_unlabel_samples
        print("Have %d samples, accuracy: %.3f%%!" % (num_unlabel_samples, accuracy))


if __name__ == '__main__':
    run_label_propagation_sparse(knn_num_neighbors=20, max_iter=30)
