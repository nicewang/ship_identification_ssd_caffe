# -*- coding: UTF-8 -*-
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

def get_AP_me(pred):

    mpre = np.concatenate(([0.], pred, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    ap = 0.0
    cnt = 0
    for i in range(len(mpre)):
        ap += mpre[i]
        cnt += 1
    return ap / cnt

def get_AP(pred, rec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], pred, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    plt.plot(mrec, mpre)
    plt.show()

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_AP_1(pred, rec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], pred, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    ap = 0.0
    for i in range(mpre.size - 2, 0, -1):
        ap += (mrec[i] - mrec[i-1]) * mpre[i]
    return ap

file = open(sys.argv[1])
pred = []
rec = []
cnt = 0
for line in file.readlines():
    if cnt == 0:
        cnt += 1
        continue
    cnt += 1
    recall, precision = line.strip('\n').split('\t')
    rec.append(float(recall))
    pred.append(float(precision))

a = []
b = []
cnt = 0.0
for i in range(1000):
    a.append(cnt)
    b.append(cnt*cnt)
    cnt += 0.1
plt.plot(a,b)
plt.show()
# 召回率曲线，横轴为召回率，纵轴为准确率
plt.plot(rec, pred)
plt.show()
pred.reverse()
rec.reverse()
plt.plot(rec, pred)
plt.show()

ap_me = get_AP_me(pred)
print("ap_me:"+str(ap_me))
ap = get_AP(pred, rec)
print("ap:"+str(ap))
ap_1 = get_AP_1(pred, rec)
print("ap_1:"+str(ap_1))
