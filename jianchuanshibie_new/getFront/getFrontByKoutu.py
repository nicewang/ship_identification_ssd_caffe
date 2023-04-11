# -*- coding: utf-8 -*-
import cv2
import  numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10,2))

img = cv2.imread('testSet/0013.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rows,cols,channels = img.shape
plt.subplot(131)
plt.imshow(img)
plt.title("Original")

# 转换hsv
hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
# 获取mask
lower_blue = np.array([78,43,46])
upper_blue = np.array([110,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
cv2.imshow('Mask',mask)

# 腐蚀膨胀（也可换成膨胀腐蚀）
erode = cv2.erode(mask,None,iterations=1)
cv2.imshow('Erode',erode)
dilate = cv2.dilate(erode,None,iterations=1)
cv2.imshow('Dilate',dilate)

# 遍历替换
img_back1 = 255*np.ones((rows,cols,3),np.uint8)
img_back2 = 255*np.ones((rows,cols,3),np.uint8)
for i in range(rows):
    for j in range(cols):
        if dilate[i,j] == 0:
            img_back1[i,j] = img[i,j]
for i in range(rows):
    for j in range(cols):
        if dilate[i,j] == 255:
            img_back2[i,j] = img[i,j]
plt.subplot(132)
plt.imshow(img_back1)
plt.title("Result1")
plt.subplot(133)
plt.imshow(img_back2)
plt.title("Result2")

plt.subplots_adjust(hspace=0.4)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
