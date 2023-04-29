import cv2
import numpy as np
import Segmentation as seg
import darkChannel as dc

img = cv2.imread("testfolder3/0.jpg")
img_darkChannel = dc.deHaze(img / 255.0) * 255
cv2.imwrite("testfolder3_0_darkChannel.jpg", img_darkChannel)
img = np.array(img)
img_darkChannel_seg = seg.dieheSegment(img,8,8)
cv2.imwrite("testfolder3_0_darkChannel_8x8dieheseg.jpg", img_darkChannel_seg)
