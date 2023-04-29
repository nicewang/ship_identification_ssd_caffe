from medianFilter import *
from travelFolder import *

if __name__ == '__main__':
    imgs2 = travel_imgs("testfolder2/")
    list_files(imgs2)
    imgs3 = travel_imgs("testfolder3/")
    list_files(imgs3)
    imgs4 = travel_imgs("testfolder4/")
    list_files(imgs4)
    imgs5 = travel_imgs("testfolder5/")
    list_files(imgs5)

    print "Defog testfolder2..."
    for i in xrange(imgs2.shape[0]):
        m = deHaze(cv2.imread(imgs2[i]) / 255.0) * 255
        cv2.imwrite("medianFilter/" + imgs2[i], m)
    print "Defog testfolder2 Completed!"

    print "Defog testfolder3..."
    for i in xrange(imgs3.shape[0]):
        m = deHaze(cv2.imread(imgs3[i]) / 255.0) * 255
        cv2.imwrite("medianFilter/" + imgs3[i], m)
    print "Defog testfolder3 Completed!"

    print "Defog testfolder4..."
    for i in xrange(imgs4.shape[0]):
        m = deHaze(cv2.imread(imgs4[i]) / 255.0) * 255
        cv2.imwrite("medianFilter/" + imgs4[i], m)
    print "Defog testfolder4 Completed!"

    print "Defog testfolder5..."
    for i in xrange(imgs5.shape[0]):
        m = deHaze(cv2.imread(imgs5[i]) / 255.0) * 255
        cv2.imwrite("medianFilter/" + imgs5[i], m)
    print "Defog testfolder5 Completed!"