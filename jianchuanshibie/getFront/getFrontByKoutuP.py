# coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))

# 载入图像
img = cv2.imread("testSet/0013.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_o = img.copy()
# 获取图像的高和宽
h, w = img.shape[:2]
# 显示原始图像
plt.subplot(231)
plt.imshow(img)
plt.title("Original")

# 进行滤波去掉噪声
blured = cv2.blur(img, (5, 5))
# 显示低通滤波后的图像
plt.subplot(232)
plt.imshow(blured)
plt.title("Blur")

# 获取掩码图
# 掩码长和宽都比输入图像多两个像素点，满水填充不会超出掩码的非零边缘
mask = np.zeros((h + 2, w + 2), np.uint8)
# 进行泛洪填充
cv2.floodFill(blured, mask, (w - 1, h - 1), (255, 255, 255), (2, 2, 2), (3, 3, 3), 8)
# 显示掩码图
plt.subplot(233)
plt.imshow(blured)
plt.title("FloodFill")

# 得到灰度图
gray = cv2.cvtColor(blured, cv2.COLOR_RGB2GRAY)
# 显示灰度图
cv2.imshow("Gray",gray)

# 定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
# 开闭运算（也可尝试先闭后开，效果有时不太一样）
# 先开运算去除背景噪声
opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
# 显示开运算结果
cv2.imshow("Opened",opened)
# 再继续闭运算填充目标内的孔洞
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
# 显示闭运算结果
cv2.imshow("Closed",closed)

# 求二值图
ret, binary = cv2.threshold(closed, 250, 255, cv2.THRESH_BINARY)
# 显示二值图
cv2.imshow('Binary', binary)

# 找到轮廓
_, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 绘制轮廓
img_contours = img_o.copy()
cv2.drawContours(img_contours, contours, -1, (0, 0, 255), 3)
plt.subplot(234)
plt.imshow(img_contours)
plt.title("Contours")

# 根据之前二值图结果，抠图
img_back1 = 255*np.ones((h,w,3),np.uint8)
for i in range(h):
    for j in range(w):
        if binary[i,j] == 255:
            img_back1[i,j] = img[i,j]
img_back2 = 255 * np.ones((h, w, 3), np.uint8)
for i in range(h):
    for j in range(w):
        if binary[i,j] == 0:
            img_back2[i,j] = img[i,j]
# 显示抠图结果
plt.subplot(235)
plt.imshow(img_back1)
plt.title("Result1")
plt.subplot(236)
plt.imshow(img_back2)
plt.title("Result2")

plt.subplots_adjust(hspace=0.4)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
