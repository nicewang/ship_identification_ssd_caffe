from darkChannel import *
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
        m = deHaze(cv2.imread(imgs2[i]) / 255.0, w=0.9) * 255
        cv2.imwrite("darkChannel/" + imgs2[i], m)
    print "Defog testfolder2 Completed!"

    print "Defog testfolder3..."
    for i in xrange(imgs3.shape[0]):
        m = deHaze(cv2.imread(imgs3[i]) / 255.0, w=0.9) * 255
        cv2.imwrite("darkChannel/" + imgs3[i], m)
    print "Defog testfolder3 Completed!"

    print "Defog testfolder4..."
    for i in xrange(imgs4.shape[0]):
        m = deHaze(cv2.imread(imgs4[i]) / 255.0, w=0.9) * 255
        cv2.imwrite("darkChannel/" + imgs4[i], m)
    print "Defog testfolder4 Completed!"

    print "Defog testfolder5..."
    for i in xrange(imgs5.shape[0]):
        m = deHaze(cv2.imread(imgs5[i]) / 255.0, w=0.9) * 255
        cv2.imwrite("darkChannel/" + imgs5[i], m)
    print "Defog testfolder5 Completed!"

    """m = deHaze(cv2.imread("1.jpg") / 255.0) * 255
    cv2.imwrite("11.jpg" , m)

    m = deHaze(cv2.imread("2.jpg") / 255.0) * 255
    cv2.imwrite("22.jpg", m)

    m = deHaze(cv2.imread("3.jpg") / 255.0) * 255
    cv2.imwrite("33.jpg", m)

    m = deHaze(cv2.imread("4.jpg") / 255.0) * 255
    cv2.imwrite("44.jpg", m)

    m = deHaze(cv2.imread("18.jpg") / 255.0) * 255
    cv2.imwrite("1818.jpg", m) """

    # time1 = datetime.datetime.now()
    # img = cv2.imread("testfolder3/0.jpg")
    # m = deHaze(img / 255.0) * 255
    # cv2.imwrite("test_defog/testfolder3_0.jpg", m)
    # time2 = datetime.datetime.now()
    # print time2-time1
