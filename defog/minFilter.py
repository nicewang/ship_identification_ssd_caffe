# -*- coding: utf-8 -*-
import cv2
import numpy as np

def zmMinFilterGray(src, r=7):
    '''最小值滤波，r是滤波器半径'''
    if r <= 0:
        return src
    h, w = src.shape[:2]
    I = src
    res = np.minimum(I, I[[0] + range(h - 1), :])
    res = np.minimum(res, I[range(1, h) + [h - 1], :])
    I = res
    res = np.minimum(I, I[:, [0] + range(w - 1)])
    res = np.minimum(res, I[:, range(1, w) + [w - 1]])
    return zmMinFilterGray(res, r - 1)
