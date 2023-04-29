# -*- coding: utf-8 -*-
from darkChannel import *

def segment(data, x_counts, y_counts):
    x = data.shape[0]
    y = data.shape[1]

    x_sub = x / x_counts
    y_sub = y / y_counts
    #四舍五入
    if x-x_sub*x_counts > x_sub/2.0:
        x_counts = x_counts + 1
    if y-y_sub*y_counts > y_sub/2.0:
        y_counts = y_counts + 1
    
    for i in xrange(x_counts):
        for j in xrange(y_counts):
            if i == x_counts-1 and j == y_counts-1:
                data_sub = data[x_sub*i:,y_sub*j:,:]
                m = deHaze(data_sub / 255.0) * 255
                data[x_sub * i:, y_sub * j:,:] = m
            elif i == x_counts-1:
                data_sub = data[x_sub*i:,y_sub*j:y_sub*j+y_sub,:]
                m = deHaze(data_sub / 255.0) * 255
                data[x_sub * i:, y_sub * j:y_sub * j + y_sub,:] = m
            elif j == y_counts-1:
                data_sub = data[x_sub*i:x_sub*i+x_sub,y_sub*j:,:]
                m = deHaze(data_sub / 255.0) * 255
                data[x_sub * i:x_sub * i + x_sub, y_sub * j:,:] = m
            else:
                data_sub = data[x_sub*i:x_sub*i+x_sub,y_sub*j:y_sub*j+y_sub,:]
                m = deHaze(data_sub / 255.0) * 255
                data[x_sub * i:x_sub * i + x_sub, y_sub * j:y_sub * j + y_sub,:] = m

    return data


def dieheSegment(data, x_counts, y_counts):
    x = data.shape[0]
    y = data.shape[1]

    x_sub = x / x_counts
    y_sub = y / y_counts
    #四舍五入
    if x-x_sub*x_counts > x_sub/2.0:
        x_counts = x_counts + 1
    if y-y_sub*y_counts > y_sub/2.0:
        y_counts = y_counts + 1

    data_copy = data

    x_cover = x_sub / 16
    y_cover = y_sub / 16

    for i in xrange(x_counts):
        if i == 0:
            for j in xrange(y_counts):
                if j == 0:
                    data_sub = data[:x_sub+x_cover,:y_sub+y_cover,:]
                    m = deHaze(data_sub / 255.0) * 255
                    data_copy[:x_sub,:y_sub,:] = m[:x_sub,:y_sub,:]
                elif j == y_counts-1:
                    data_sub = data[:x_sub+x_cover,y_sub*j-y_cover:,:]
                    m = deHaze(data_sub / 255.0) * 255
                    data_copy[:x_sub,y_sub*j:,:] = m[:x_sub,y_cover:,:]
                else:
                    data_sub = data[:x_sub+x_cover,y_sub*j-y_cover:y_sub*j+y_sub+y_cover,:]
                    m = deHaze(data_sub / 255.0) * 255
                    data_copy[:x_sub,y_sub*j:y_sub*j+y_sub,:] = m[:x_sub,y_cover:y_cover+y_sub,:]
        elif i == x_counts-1:
            for j in xrange(y_counts):
                if j == 0:
                    data_sub = data[x_sub*i-x_cover:,:y_sub+y_cover,:]
                    m = deHaze(data_sub / 255.0) * 255
                    data_copy[x_sub*i:,:y_sub,:] = m[x_cover:,:y_sub,:]
                elif j == y_counts-1:
                    data_sub = data[x_sub*i-x_cover:,y_sub*j-y_cover:,:]
                    m = deHaze(data_sub / 255.0) * 255
                    data_copy[x_sub*i:,y_sub*j:,:] = m[x_cover:,y_cover:,:]
                else:
                    data_sub = data[x_sub*i-x_cover:,y_sub*j-y_cover:y_sub*j+y_sub+y_cover,:]
                    m = deHaze(data_sub / 255.0) * 255
                    data_copy[x_sub*i:,y_sub*j:y_sub*j+y_sub,:] = m[x_cover:,y_cover:y_cover+y_sub,:]
        else:
            for j in xrange(y_counts):
                if j == 0:
                    data_sub = data[x_sub*i-x_cover:x_sub*i+x_sub+x_cover,:y_sub+y_cover,:]
                    m = deHaze(data_sub / 255.0) * 255
                    data_copy[x_sub*i:x_sub*i+x_sub,:y_sub,:] = m[x_cover:x_cover+x_sub,:y_sub,:]
                elif j == y_counts-1:
                    data_sub = data[x_sub*i-x_cover:x_sub*i+x_sub+x_cover,y_sub*j-y_cover:,:]
                    m = deHaze(data_sub / 255.0) * 255
                    data_copy[x_sub*i:x_sub*i+x_sub,y_sub*j:,:] = m[x_cover:x_cover+x_sub,y_cover:,:]
                else:
                    data_sub = data[x_sub*i-x_cover:x_sub*i+x_sub+x_cover,y_sub*j-y_cover:y_sub*j+y_sub+y_cover,:]
                    m = deHaze(data_sub / 255.0) * 255
                    data_copy[x_sub*i:x_sub*i+x_sub,y_sub*j:y_sub*j+y_sub,:] = m[x_cover:x_cover+x_sub,y_cover:y_cover+y_sub,:]

    return data_copy
