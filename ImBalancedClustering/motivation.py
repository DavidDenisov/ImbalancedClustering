import numpy as np
from numpy.random import Generator, PCG64
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from algorithms import approx_solver, split_by_centers
from time import perf_counter as time

rg = Generator(PCG64())


def k_means_split(P,k):
    modle = KMeans(init="k-means++", n_clusters=k, n_init=1)
    pred = modle.fit_predict(P)
    ans = []
    for i in range(k):
        mask = pred==i
        ans.append(P[mask])

    return ans


def sample_circle(c,r,num,rng=rg):
    #   theta = np.linspace(0, 2 * np.pi, nun)
    #   r = rg.uniform(0,r,nun)
    #   x, y = r * np.cos(theta), r * np.sin(theta)
    #   x+=c[0]
    #   y+=c[1]
    #   P = np.concatenate(([x], [y]), 0).T

    # random angle
    alpha = 2 * np.pi * rng.random(num)
    # random radius
    r = r * np.sqrt(rng.random(num))
    # calculating coordinates
    x = r * np.cos(alpha) + c[0]
    y = r * np.sin(alpha) + c[1]
    P = np.concatenate(([x], [y]), 0).T

    return P

if __name__ == '__main__':
    folder = './motivation/loss_ill/linear/1250_'
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.rcParams.update({'font.size': 50})

    dpi = 125
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.gca().set_aspect('equal')

    # mng.full_screen_toggle()

    n1,n2 = 1250,25
    A = sample_circle([0,0],1,n1)
    B = sample_circle([2.25, 0], 0.1, n2)

    # A = rg.uniform(0,1,(10*x,d))
    # B = rg.uniform(1.25,1.5,(x,d))

    #   A = rg.normal(0, 0.15, (100*x, d))
    #   B = rg.normal(0.75, 0.015, (x, d))

    # C = rg.normal(2, 0.15, (n2, d))

    plt.scatter(A[:,0], A[:,1], c='r', s=250)
    plt.scatter(B[:, 0], B[:, 1], c='b', s=250)
    plt.savefig(folder+"data.png", dpi=dpi, bbox_inches='tight')
    plt.show()

    plt.xlabel("x")
    plt.ylabel("y")

    #
    P = np.concatenate([A,B], 0)

    #
    loss = 0
    P_ = k_means_split(P, 2)
    for p_, c in zip(P_, ['r', 'b','y']):
        c_ = np.mean(p_, 0)
        m = np.sum((p_ - c_) ** 2, 1)
        loss += m.sum()#/(np.log2(1+len(m)))**2
        plt.scatter(p_[:, 0], p_[:, 1], c=c, s=250)
        plt.scatter(c_[0], c_[1], c='k', s=2000)

    plt.savefig(folder+"k_means.png", dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.xlabel("x")
    plt.ylabel("y")
    print(loss)

    loss = 0
    t = time()
    C = approx_solver(P, 2,log=False)
    print(np.round(time()-t,3))
    P_ = split_by_centers(P, C)
    for p_, c, c_ in zip(P_, ['r', 'b','y'], C):
        # c__ = np.mean(p_,0)
        m =np.sum((p_ - c_) ** 2, 1)
        loss += m.sum()#/(np.log2(1+len(m)))**2
        plt.scatter(p_[:, 0], p_[:, 1], c=c, s=250)

    plt.scatter(C[:, 0], C[:, 1], c='k', s=2000)
    plt.savefig(folder+"our.png", dpi=dpi, bbox_inches='tight')
    plt.show()
    print(loss)


