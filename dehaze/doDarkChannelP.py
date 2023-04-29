from darkChannelP import *

img = cv2.imread("testfolder3/0.jpg")
img = np.array(img)
m = segment(img,4,4)
cv2.imwrite("testfolder3_0_darkChannelP_4X4.jpg", m)
