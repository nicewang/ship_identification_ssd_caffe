# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图
img = cv2.imread('testSet/0004.jpg', 0)
plt.subplot(221),plt.imshow(img,'gray'),plt.title('Original(gray)')
plt.xticks([]),plt.yticks([])

# 将读取的二维图像数组转换为1维数组
img1 = img.reshape((img.shape[0]*img.shape[1],1))
img1 = np.float32(img1)

# define criteria = (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

# 选取初始聚类中心
# ---cv2.KMEANS_PP_CENTERS ; cv2.KMEANS_RANDOM_CENTERS
flags = cv2.KMEANS_RANDOM_CENTERS

# k聚类
compactness,labels,centers = cv2.kmeans(img1,2,None,criteria,10,flags)

img2 = labels.reshape((img.shape[0],img.shape[1]))
plt.subplot(222),plt.imshow(img2,'gray'),plt.title('Kmeans(k=2)')
plt.xticks([]),plt.yticks([])

# k聚类
compactness,labels,centers = cv2.kmeans(img1,3,None,criteria,10,flags)

img3 = labels.reshape((img.shape[0],img.shape[1]))
plt.subplot(223),plt.imshow(img3,'gray'),plt.title('Kmeans(k=3)')
plt.xticks([]),plt.yticks([])

# k聚类
compactness,labels,centers = cv2.kmeans(img1,4,None,criteria,10,flags)

img4 = labels.reshape((img.shape[0],img.shape[1]))
plt.subplot(224),plt.imshow(img4,'gray'),plt.title('Kmeans(k=4)')
plt.xticks([]),plt.yticks([])

plt.show()
