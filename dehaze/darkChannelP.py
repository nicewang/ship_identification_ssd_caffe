# -*- coding: utf-8 -*-
import cv2
import numpy as np

def zmMinFilterGray(src, r=1):
    '''最小值滤波，r是滤波器半径'''
    if r <= 0:
        return src
    h, w = src.shape[:2]
    I = src
    I_tmp = I[[0] + range(h - 1), :]
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


def getV1(m, r=81, eps=0.001, w=0.95, maxV1=0.80):  # 输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)  # 得到暗通道图像
    # V1 = np.mean(m, 2)
    V1 = guidedfilter(V1, zmMinFilterGray(V1, 1), r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()

    V1 = np.minimum(V1 * w, maxV1)  # 对值范围进行限制

    return V1, A


def deHaze(m, V1, A, bGamma=False):
    Y = np.zeros(m.shape)
    for k in range(3):
        Y[:, :, k] = (m[:, :, k] - V1) / (1 - V1 / A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校正,默认不进行该操作
    return Y


def segment(data, x_counts, y_counts):
    x = data.shape[0]
    y = data.shape[1]

    x_sub = x / x_counts
    y_sub = y / y_counts
    # 四舍五入
    if x - x_sub * x_counts > x_sub / 2.0:
        x_counts = x_counts + 1
    if y - y_sub * y_counts > y_sub / 2.0:
        y_counts = y_counts + 1

    A_amount = 0
    for i in xrange(x_counts):
        for j in xrange(y_counts):
            if i == x_counts - 1 and j == y_counts - 1:
                data_sub = data[x_sub * i:, y_sub * j:, :]
                V1, A = getV1(data_sub / 255.0)
                A_amount = A_amount + A
            elif i == x_counts - 1:
                data_sub = data[x_sub * i:, y_sub * j:y_sub * j + y_sub, :]
                V1, A = getV1(data_sub / 255.0)
                A_amount = A_amount + A
            elif j == y_counts - 1:
                data_sub = data[x_sub * i:x_sub * i + x_sub, y_sub * j:, :]
                V1, A = getV1(data_sub / 255.0)
                A_amount = A_amount + A
            else:
                data_sub = data[x_sub * i:x_sub * i + x_sub, y_sub * j:y_sub * j + y_sub, :]
                V1, A = getV1(data_sub / 255.0)
                A_amount = A_amount + A

    A_avr = A_amount / np.float(x_counts*y_counts)
    V1_all, A = getV1(data / 255.0, maxV1=0.20)

    data_out = deHaze(data, V1_all, A_avr) * 255

    return data_out
