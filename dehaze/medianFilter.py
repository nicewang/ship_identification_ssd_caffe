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


def guidedfilter(I, p, r, eps):
    '''引导滤波'''
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def getV1(m, r, eps, p):  # 输入rgb图像，值范围[0,1]
    '''计算大气光幕(雾浓度)F(x)、大气遮罩图像V1和光照值A, V1 = 1-t/A'''

    M = np.min(m, 2)  # 得到暗通道图像

    A_median = cv2.medianBlur(M, 1)
    B = A_median - cv2.medianBlur(abs(A_median - M), 1)
    F = np.minimum(p * B, M)
    F = np.maximum(F, 0)

    V1 = guidedfilter(M, zmMinFilterGray(M, 7), r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()


    return F, A


def deHaze(m, r=81, eps=0.001, p=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    F, A = getV1(m, r, eps, p)  # 得到大气光幕和大气光照
    for k in range(3):
        Y[:, :, k] = (m[:, :, k] - F) / (1 - F / A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校正,默认不进行该操作
    return Y
