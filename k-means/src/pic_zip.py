# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     pic_zip
  email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/10/21
   Description :
==================================================
"""
__author__ = 'songdongdong'

# 一 ，导入工具库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
import matplotlib.image as mpimg
#二、导入图片，设定参数
n_colors = 64
china = load_sample_image('flower.jpg')
# china = mpimg.imread('../data/girl.jpg')
china = np.array(china,dtype=np.float64) /255 #对图像进行归一化，把图像0-255的数据，进行归一化
w,h,d = original_shape = tuple(china.shape)
assert d == 3
image_array = np.reshape(china,(w*h,d))
image_array_sample = shuffle(image_array,random_state=1)[:1000] #随机去 1000条
kmeans = KMeans(n_clusters=n_colors,random_state=0).fit(image_array_sample)
labels = kmeans.predict(image_array)

#三、压缩聚类
def recreate_image(codebook,labels,w,h):
    d = codebook.shape[1]
    image = np.zeros((w,h,d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx +=1
    return image


#四，可视化
plt.figure(1)
plt.clf()
ax = plt.axes([0,0,1,1])
plt.axis('off')
plt.title('Original image(96,615 colors)')
plt.imshow(china)


plt.figure(2)
plt.clf()
ax = plt.axes([0,0,1,1])
plt.axis("off")
plt.title("Quantized image(64 colors ,K-Means)")
print(kmeans.cluster_centers_)
print(len(labels),labels)
plt.imshow(recreate_image(kmeans.cluster_centers_,labels,w,h))

plt.show()