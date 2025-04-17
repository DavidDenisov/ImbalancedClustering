import cv2
import numpy as np
from divisive import approx_on_core, kmeans
from numpy.random import Generator, PCG64
from copy import deepcopy as copy
from sklearn.metrics import silhouette_score

rg = Generator(PCG64())

depth = 7


# colors = [[255,0,0],[255,255,0],[255,255,255],[0,255,0],[0,255,255],[0,0,255]]


def predict_centers(P, c):
    n, d = np.shape(P)
    k = len(c)

    P_ = np.expand_dims(P, 0)
    P_ = np.repeat(P_, k, 0)
    C = np.expand_dims(c, 1).repeat(n, 1)
    l = np.sum((P_ - C) ** 2, 2)
    return np.argmin(l, 0)


labels = []


class Tree:
    def __init__(self):
        self.left = None
        self.right = None
        self.centers = None

    def predict(self, X):
        if len(X) == 0:
            return np.empty(0)
        if self.left is None:
            return np.ones_like(X) * self.centers
        n, d = np.shape(X)
        x = np.expand_dims(X, 0)
        x = np.repeat(x, 2, 0)
        C = np.expand_dims(self.centers, 1).repeat(n, 1)
        l = np.sum((x - C) ** 2, 2)
        pred = np.argmin(l, 0)
        mask = pred == 0

        X_l, X_r = X[mask], X[np.logical_not(mask)]
        ans = np.zeros_like(X)
        ans[mask] = self.left.predict(X_l)
        ans[np.logical_not(mask)] = self.right.predict(X_r)
        return ans

    def __predict_label(self, X):
        if len(X) == 0:
            return np.empty(0)
        if self.left is None:
            l = len(labels) + 1
            labels.append(l)
            return np.ones(len(X)) * l

        n, d = np.shape(X)
        x = np.expand_dims(X, 0)
        x = np.repeat(x, 2, 0)
        C = np.expand_dims(self.centers, 1).repeat(n, 1)
        l = np.sum((x - C) ** 2, 2)
        pred = np.argmin(l, 0)
        mask = pred == 0

        X_l, X_r = X[mask], X[np.logical_not(mask)]
        ans = np.zeros(len(X))
        ans[mask] = self.left.__predict_label(X_l)
        ans[np.logical_not(mask)] = self.right.__predict_label(X_r)
        return ans

    def predict_label(self, X):
        global labels
        labels = []
        return self.__predict_label(X)


def split(func, data, cluster=False):
    if not cluster:
        return func(data, 2)

    c = func(data, 2)
    P = np.expand_dims(data, 0).repeat(len(c), 0)
    opt_ = np.expand_dims(c, 1).repeat(len(data), 1)
    miss = np.sum((P - opt_)**2,2)
    arg = np.argmin(miss, 0)
    return c, [data[arg == 0], data[arg == 1]], arg


def divisive(func, data, depth, cluster=True):
    P = [np.copy(data)]
    T = Tree()
    L = [T]
    for i in range(depth):
        L_ = []
        new = []
        for leaf, p in zip(L, P):
            if len(np.unique(p, axis=0)) == 1:
                continue
            c, p_, _ = split(func, p, cluster)
            new.extend(p_)
            leaf.centers = c
            leaf.left = Tree()
            leaf.left.centers = c[0]
            leaf.right = Tree()
            leaf.right.centers = c[1]
            L_.extend([leaf.left, leaf.right])

        L = L_
        P = copy(new)
    return P, T


def divisive_choice(F, data, depth, cluster=True):
    P = [np.copy(data)]
    T = Tree()
    L = [T]
    for _ in range(depth):

        L_ = []
        new = []
        for leaf, p in zip(L, P):
            if len(np.unique(p, axis=0)) == 1:
                continue
            c, p_ = None, None
            b = -1
            for i_, f in enumerate(F):
                c_, p__, l = split(f, p, cluster)
                # print(np.count_nonzero(l==0),np.count_nonzero(l==1))
                if len(p) > 2:
                    try:
                        cur = silhouette_score(p, l, sample_size=1024, n_jobs=-1)
                    except ValueError:
                        cur = silhouette_score(p, l, sample_size=None, n_jobs=-1)
                else:
                    c, p_ = copy(c_), copy(p__)
                    break
                if cur >= b:
                    b = cur
                    c, p_ = copy(c_), copy(p__)
            # print(i)
            new.extend(p_)
            leaf.centers = c
            leaf.left = Tree()
            leaf.left.centers = c[0]
            leaf.right = Tree()
            leaf.right.centers = c[1]
            L_.extend([leaf.left, leaf.right])

        L = L_
        P = copy(new)
    return P, T


def img_diff(a, b):
    a_ = a.astype(np.float32)
    b_ = b.astype(np.float32)
    diff = np.abs(a_ - b_)
    diff[diff > 255] = 255
    return diff.astype(np.uint8)


if __name__ == '__main__':
    A = cv2.imread('../imgs/boat.tiff')

    A = A[1:-1,1:-1]
    # A = cv2.resize(A,(512,1024))
    # A = np.concatenate((A1,A1),axis=1)
    # A = A[250:-250,250:-50,:]
    # A = cv2.resize(A, (900, 900))

    n, m, d = np.shape(A)
    # A = A[:2200]

    #   A = rg.uniform(0,32,((512,512,3))).astype(np.uint8)
    #   a = A[256-12:256+12,256-12:256+12]
    #   A[256-12:256+12,256-12:256+12] = rg.uniform(200,225,np.shape(a)).astype(np.uint8)
    cv2.imwrite('../choice/real/orig.png', A)
    # cv2.imshow('orig', cv2.resize(A,(512,512)))
    P = np.reshape(A, (n * m, d))

    F = [kmeans, approx_on_core]
    _, t = divisive_choice(F, P, depth)
    img = t.predict(P)

    P_ = np.reshape(img, np.shape(A))
    # P__ = img_diff(P_,A)
    print('choice')
    cv2.imwrite('../choice/real/choice.png', P_)
    # cv2.imshow('kmeans',cv2.resize(P__,(512,512)))

    _, t = divisive(kmeans, P, depth)
    img = t.predict(P)
    P_ = np.reshape(img, np.shape(A))
    # P__ = img_diff(P_,A)
    print('kmeans')
    cv2.imwrite('../choice/real/kmeans.png', P_)

    _, t = divisive(approx_on_core, P, depth)
    img = t.predict(P)
    P_ = np.reshape(img, np.shape(A))
    # P__ = img_diff(P_,A)
    print('our')
    cv2.imwrite('../choice/real/our.png', P_)
    # cv2.imshow('our',cv2.resize(P_,(512,512)))
