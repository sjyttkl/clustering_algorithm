# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     k-means
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/7/17
   Description :  https://www.cnblogs.com/qianyin123/p/9553805.html
==================================================
"""
__author__ = 'sjyttkl'


import time

import codecs
import shutil
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

'''
sklearn里面的TF-IDF主要用到了两个函数：CountVectorizer()和TfidfTransformer()。
    CountVectorizer是通过fit_transform函数将文本中的词语转换为词频矩阵。
    矩阵元素weight[i][j] 表示j词在第i个文本下的词频，即各个词语出现的次数。
    通过get_feature_names()可看到所有文本的关键字，通过toarray()可看到词频矩阵的结果。
    TfidfTransformer也有个fit_transform函数，它的作用是计算tf-idf值。
'''
#########################################################################
    #                           第一步 计算TFIDF
if __name__ == "__main__":
    corpus = []  # 文档预料 空格连接
    # 读取预料 一行预料为一个文档
    for line in open('../data/BaiduSpider_Result.txt', 'r',encoding="utf-8").readlines():
        print(line)
        corpus.append(line.strip())
    # print corpus
    time.sleep(5)

    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()

    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()

    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()

    # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()

    resName = "../data/BaiduTfidf_Result.txt"
    result = codecs.open(resName, 'w', 'utf-8')
    for j in range(len(word)):
        result.write(word[j] + ' ')
    result.write('\r\n\r\n')

    # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    for i in range(len(weight)):
        print("-------这里输出第", i, u"类文本的词语tf-idf权重------")
        for j in range(len(word)):
            result.write(str(weight[i][j]) + ' ')
        result.write('\r\n\r\n')

    result.close()

    ########################################################################
    #                               第二步 聚类Kmeans

    print('Start Kmeans:')
    from sklearn.cluster import KMeans

    clf = KMeans(n_clusters=20)
    s = clf.fit(weight)
    print(s)
    # 20个中心点
    print(clf.cluster_centers_)

    # 每个样本所属的簇
    print(clf.labels_)
    i = 1
    while i <= len(clf.labels_):
        print(i, clf.labels_[i - 1])
        i = i + 1
    # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    print(clf.inertia_)
