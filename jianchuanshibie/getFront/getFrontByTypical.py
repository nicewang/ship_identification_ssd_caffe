import cv2
import matplotlib.pyplot as plt

print('Load Image')

imgFile = 'testSet/0009.jpg'

img = cv2.imread(imgFile)

cRange = 256

rows,cols,channels = img.shape
print('rows,cols,channels:',rows,cols,channels)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgCanny = cv2.Canny(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

imgLap = cv2.Laplacian(imgGray, cv2.CV_8U)

threshold, imgOtsu = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

imgAdapt = cv2.adaptiveThreshold(imgGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

plt.subplot(2,3,1), plt.imshow(img,cmap= 'gray'), plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2), plt.imshow(imgLap,cmap = 'gray'), plt.title('Laplacian Edge'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3), plt.imshow(imgCanny,cmap= 'gray'), plt.title('Canny Edge'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4), plt.imshow(imgOtsu,cmap = 'gray'), plt.title('Otsu Method'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5), plt.imshow(imgAdapt,cmap = 'gray'), plt.title('Adaptive Gaussian Threshold'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(2,3,1), plt.imshow(img), plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2), plt.imshow(imgLap), plt.title('Laplacian Edge'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3), plt.imshow(imgCanny), plt.title('Canny Edge'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4), plt.imshow(imgOtsu), plt.title('Otsu'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5), plt.imshow(imgAdapt), plt.title('Adaptive Gaussian Threshold'), plt.xticks([]), plt.yticks([])
plt.show()

print('Goodbye!')
