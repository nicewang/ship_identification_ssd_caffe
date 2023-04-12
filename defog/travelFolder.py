import numpy as np
import os

def travel_imgs(path):
    cate = [path + x for x in os.listdir(path)]
    imgs = []
    for idx, folder in enumerate(cate):
        print('reading the images:%s' % (folder))
        imgs.append(folder)
    return np.asarray(imgs)

def list_files(files):
    for i in xrange(files.shape[0]):
        print files[i]
