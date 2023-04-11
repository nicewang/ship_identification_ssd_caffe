# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 读取RGB图
img = cv2.imread('testSet/0004.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(221),plt.imshow(img),plt.title('Original')
plt.xticks([]),plt.yticks([])


# 将读取的RGB图像数组转换为1维数组
Z = img.reshape((-1,3))
Z = np.float32(Z)
# print Z.shape

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# k聚类
ret,label,center = cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

img2 = label.reshape((img.shape[0],img.shape[1]))
plt.subplot(222),plt.imshow(img2),plt.title('Kmeans(k=2)')
plt.xticks([]),plt.yticks([])

# k聚类
ret,label,center = cv2.kmeans(Z,3,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

img3 = label.reshape((img.shape[0],img.shape[1]))
plt.subplot(223),plt.imshow(img3),plt.title('Kmeans(k=3)')
plt.xticks([]),plt.yticks([])

# k聚类
ret,label,center = cv2.kmeans(Z,4,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

img4 = label.reshape((img.shape[0],img.shape[1]))
plt.subplot(224),plt.imshow(img4),plt.title('Kmeans(k=4)')
plt.xticks([]),plt.yticks([])

plt.show()
