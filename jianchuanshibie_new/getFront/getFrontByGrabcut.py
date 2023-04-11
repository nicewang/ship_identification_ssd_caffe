import numpy as np
import cv2
from matplotlib import pyplot as plt
import vtk

plt.figure(figsize=(10,2))

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (50,50,450,290)

img1 = cv2.imread('testSet/0001.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
mask11 = np.zeros(img1.shape[:2],np.uint8)

img2 = cv2.imread('testSet/0013.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
mask21 = np.zeros(img2.shape[:2],np.uint8)

img3 = cv2.imread('testSet/1004.jpg')
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
mask31 = np.zeros(img3.shape[:2],np.uint8)

cv2.grabCut(img1,mask11,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
cv2.grabCut(img2,mask21,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
cv2.grabCut(img3,mask31,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask12 = np.where((mask11==2)|(mask11==0),0,1).astype('uint8')
img1 = img1*mask12[:,:,np.newaxis]

mask22 = np.where((mask21==2)|(mask21==0),0,1).astype('uint8')
img2 = img2*mask22[:,:,np.newaxis]

mask32 = np.where((mask31==2)|(mask31==0),0,1).astype('uint8')
img3 = img3*mask32[:,:,np.newaxis]

plt.subplot(131)
plt.imshow(img1)
plt.colorbar()

plt.subplot(132)
plt.imshow(img2)
plt.colorbar()

plt.subplot(133)
plt.imshow(img3)
plt.colorbar()

plt.subplots_adjust(hspace=0.4)
plt.show()
