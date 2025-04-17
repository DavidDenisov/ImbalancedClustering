from synthetic_test import kmeans,approx_on_core,bi_criteria,plot_error
from motivation import sample_circle
import numpy as np
from copy import deepcopy as copy
from time import perf_counter as time
from HiPart.clustering import DePDDP
from sklearn.cluster import AgglomerativeClustering,SpectralClustering
import matplotlib.pyplot as plt
from pathlib import Path

save_file = 'res/hierarchical/real.npz'
test_type = 'real'
img_folder = './figs/hierarchical/real_'

from imblearn.datasets import fetch_datasets
data_sets = fetch_datasets()
data = data_sets['webpage'].data
dpi = 125

from numpy.random import Generator, PCG64
rg = Generator(PCG64())

def split(func,data,cluster=False):
    if len(data) ==1:
        return [[data[0]]]
    if len(data)==2:
        return [[data[0]],[data[1]]]

    if not cluster:
        return func(data, 2)

    c = func(data, 2)
    P = np.expand_dims(data, 0).repeat(len(c), 0)
    opt_ = np.expand_dims(c, 1).repeat(len(data), 1)
    miss = np.linalg.norm(P-opt_,2,2)
    arg = np.argmin(miss,0)
    L = np.unique(arg)
    return [data[arg==l] for l in L]

def divisive(func,data,depth,cluster=True):
    P = [np.copy(data)]
    for _ in range(depth):
        new = []
        for p in P:
            res = split(func,p,cluster)
            new.extend(res)
        P = copy(new)
    return P


def calc_miss(P):
    s = 0
    for p in P:
        c = np.mean(p,0)
        m = np.sum((p - c) ** 2, 1)
        s+= m.mean()
    return s/len(P)

def spectral(P,k):
    spectral = SpectralClustering(n_clusters=k,eigen_solver="arpack",affinity="nearest_neighbors")
    clustered_class = spectral.fit_predict(P)
    L = np.unique(clustered_class)
    return [P[clustered_class == l] for l in L]

def one_test(i,depth=1):
    if test_type == 'synthetic':
        n1,n2 = i*250,i*10
        A = sample_circle([0,0],1,n1)
        B = sample_circle([2.25, 0], 0.1, n2)
        C = sample_circle([2.25, 2.25], 0.1, n2)
        P = np.concatenate((A,B,C),0)
    else:
        P = rg.choice(data, 250*i, replace=False)

    t = time()
    p = divisive(approx_on_core,P,depth)
    t1 = time()-t
    a1 = calc_miss(p)

    t = time()
    p2 = divisive(kmeans, P, depth)
    t2 = time() - t
    a2 = calc_miss(p2)

    t = time()
    p3 = divisive(bi_criteria, P, depth)
    t3 = time() - t
    a3 = calc_miss(p3)

    t = time()
    try:
        clustered_class = DePDDP(max_clusters_number=pow(2,depth)).fit_predict(P)
    except RuntimeError:
        clustered_class = np.zeros(len(P))
    t4 = time() - t
    L = np.unique(clustered_class)
    p4 = [P[clustered_class==l] for l in L]
    a4 = calc_miss(p4)

    model = AgglomerativeClustering(n_clusters=pow(2,depth), linkage="ward")
    t = time()
    clustered_class = model.fit_predict(P)
    t5 = time() - t
    L = np.unique(clustered_class)
    p5 = [P[clustered_class == l] for l in L]
    a5 = calc_miss(p5)

    t = time()
    p6 = divisive(spectral, P, depth,cluster=False)
    t6 = time() - t
    a6 = calc_miss(p6)

    return [a1,a2,a3,a4,a5,a6],[t1,t2,t3,t4,t5,t6]

def run(count=100,depth=2):
    one_test(1, depth)
    print_every = 1
    checks = np.arange(2,10,1)
    T, M = [[] for _ in checks], [[] for _ in checks]
    t = time()
    for i in range(count):
        if i%print_every==0 and i>0:
            print('Done on:',np.round(100*i/count,4),', took:', np.round(time()-t,4))
            t = time()

        for j,r in enumerate(checks):
            c_m, c_t = one_test(r, depth)
            T[j].append(c_t)
            M[j].append(c_m)

    return np.array(T),np.array(M),checks


def plot(T,M,S,img_folder,test_type):
    labels = ['APPROX ON CORESET','K-Means','BI-Criteria','DePDDP','Ward','Spectral']
    # Print the time resultS.

    plt.figure()
    plot_error(S, T[:, :, 0], color="black", label=labels[0])
    plot_error(S, T[:, :, 1], color="blue", label=labels[1])
    plot_error(S, T[:, :, 2], color="cyan", label=labels[2])
    plot_error(S, T[:, :, 3], color="yellow", label=labels[3])
    plot_error(S, T[:, :, 4], color="magenta", label=labels[4])
    plot_error(S, T[:, :, 5], color="green", label=labels[5])

    if test_type == 'real':
        plt.ylim(bottom=0, top=0.9)
    else:
        plt.ylim(bottom=0,top=0.4)

    plt.ylabel("Computation time (s)")
    plt.xlabel("Value of x")
    plt.legend(loc='upper center', ncol=2, fancybox=True,bbox_to_anchor=(0.5, 1.38))
    plt.savefig(img_folder+ "_time.png", dpi=dpi, bbox_inches='tight')
    # plt.show()

    plt.figure()
    plot_error(S, M[:, :, 0], color="black", label=labels[0])
    plot_error(S, M[:, :, 1], color="blue", label=labels[1])
    plot_error(S, M[:, :, 2], color="cyan", label=labels[2])
    plot_error(S, M[:, :, 3], color="yellow", label=labels[3])
    plot_error(S, M[:, :, 4], color="magenta", label=labels[4])
    plot_error(S, M[:, :, 5], color="green", label=labels[5])

    # plt.ylim(bottom=0.5,top=1)
    if test_type == 'real':
        plt.ylim(bottom=3, top=15)
    else:
        plt.ylim(bottom=0.01, top=0.5)
    # plt.ylim(bottom=10, top=90)
    plt.ylabel("Loss")
    plt.xlabel("Value of x")
    # Shrink current axis by 20%
    plt.legend(loc='upper center', ncol=2, fancybox=True,bbox_to_anchor=(0.5, 1.38))
    plt.savefig(img_folder + "_loss.png", dpi=dpi, bbox_inches='tight')
    # plt.show()



def run_all_tests():
    global test_type

    for test_type in ['synthetic','real']:
        file = './res/hierarchical/'+test_type+'.npz'
        a, b, c = run()
        np.savez_compressed(file, a=a, b=b, c=c)


def plot_all_tests():
    plt.rcParams.update({'lines.linewidth': 20})
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.rcParams.update({'font.size': 40})

    for test_type in ['synthetic','real']:
        file = './res/hierarchical/'+test_type+'.npz'
        img_folder = './plots/hierarchical/'
        Path(img_folder).mkdir(parents=True, exist_ok=True)

        dict = np.load(file)
        a, b, c = [dict[key] for key in ['a', 'b', 'c']]

        plot(a, b, c, img_folder+test_type,test_type)



if __name__ == '__main__':
    # run_all_tests()
    plot_all_tests()


