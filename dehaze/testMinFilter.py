import cv2
import numpy as np
import minFilter as mf
import pylab as pl

img = cv2.imread("testfolder3/0.jpg")
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = np.array(img)
img_out = mf.zmMinFilterGray(img)

pl.imshow(img_out)
pl.show()
