import cv2
import numpy as np
from sklearn.cluster import KMeans
from synthetic_test import approx_on_core
from numpy.random import Generator, PCG64
rg = Generator(PCG64())

k=2

# colors = [[255,0,0],[255,255,0],[255,255,255],[0,255,0],[0,255,255],[0,0,255]]
def cluster_imag(P,c,pred):
    P_ = np.zeros_like(P)
    for i in range(k):
        P_[pred==i] = c[i]
        print(np.count_nonzero(pred==i))
    return P_

def cluster_imag2(P, c, pred):
    P_ = np.zeros_like(P)
    I = np.linspace(0,len(P_)-1,k+1,dtype=int)
    for i in range(k):
        s,e = I[i],I[i+1]
        P_[s:e] = c[i]
    return P_

def predict_centers(P,c):
    n, d = np.shape(P)
    k = len(c)

    P_ = np.expand_dims(P, 0)
    P_ = np.repeat(P_, k, 0)
    C = np.expand_dims(c, 1).repeat(n, 1)
    l = np.sqrt(np.sum((P_ - C) ** 2, 2))
    return np.argmin(l, 0)

# Photo by me
A = cv2.imread('./imgs/draw.jpg')
n,m,d = np.shape(A)
# A = A[:2200]

#   A = rg.uniform(0,32,((512,512,3))).astype(np.uint8)
#   a = A[256-12:256+12,256-12:256+12]
#   A[256-12:256+12,256-12:256+12] = rg.uniform(200,225,np.shape(a)).astype(np.uint8)

cv2.imwrite('./motivation/draw/orig.png',A)
#  cv2.imshow('orig', cv2.resize(A,(512,512)))
# n,m,d = np.shape(A)
# d = 1
P = np.reshape(A,(n*m,d))

k_means = KMeans(init="k-means++", n_clusters=k, n_init=1)  # for seed equal zero fails.
pred = k_means.fit_predict(P)
c = k_means.cluster_centers_
P_ = cluster_imag(P,c,pred)
P_ = np.reshape(P_,np.shape(A))
cv2.imwrite('./motivation/draw/kmeans.png',P_)
#  cv2.imshow('kmeans',cv2.resize(P_,(512,512)))
# cv2.waitKey(-1)

c = approx_on_core(P,k)
pred = predict_centers(P,c)
P_ = cluster_imag(P,c,pred)
P_ = np.reshape(P_,np.shape(A))
cv2.imwrite('./motivation/draw/our.png',P_)
# cv2.imshow('our',cv2.resize(P_,(512,512)))
# cv2.waitKey(-1)