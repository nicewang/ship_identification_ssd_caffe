# -*- coding: utf-8 -*-
#code:myhaspl@myhaspl.com
#中值滤波
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("testfolder2/0.jpg")
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# #加上椒盐噪声
# #灰阶范围
# h = img.shape[1]
# w = img.shape[0]
# new_img = np.array(img)
# #噪声点数量
# noisecount=50000
# for k in xrange(0,noisecount):
#     xi = int(np.random.uniform(0,new_img.shape[1]))
#     xj = int(np.random.uniform(0,new_img.shape[0]))
#     new_img[xj,xi]=255

#滤波去噪
# lb_img = cv2.bilateralFilter(img,3,sigmaColor=3,sigmaSpace=3)
lb_img = cv2.medianBlur(img,1)

plt.imshow(lb_img, cmap='gray')
plt.show()
