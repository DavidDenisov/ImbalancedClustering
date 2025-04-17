import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

from algorithms import approx_solver,approx,coreset,calc_miss
from time import perf_counter as time
from motivation import sample_circle

# Check cuda version.
from cuml import KMeans as KMeans_torch
import cudf

from sklearn.mixture import GaussianMixture
import torch
from imblearn.under_sampling import RandomUnderSampler,NearMiss
from imblearn.over_sampling import SMOTE, KMeansSMOTE,RandomOverSampler
from imblearn.combine import SMOTETomek

from pathlib import Path

from numpy.random import Generator, PCG64
rg = Generator(PCG64())
file = './res/synthetic/approx.npz'
test_type = 'approx' # 'approx','under','over'.
img_folder = './figs/tests/approx/'
dpi = 100
add_noise = True

import warnings
warnings.filterwarnings("ignore")  # Push me to the edge/

def plot_error(X, y, color, label):
    s1, s2 = np.percentile(y, 25, 1).reshape((1, -1)), np.percentile(y, 75, 1).reshape((1, -1))
    m, s = np.median(y, 1), np.concatenate([s1, s2], 0)
    s = np.abs(s - m)
    plt.errorbar(X, m, s, ls='-.', color=color, label=label)


# Plots the results, with additional prints for the time.
def plot(T,M,S,test_type,img_folder):
    labels = []
    if test_type == 'approx':
        labels = ['APPROX','APPROX ON CORESET','BI-CRITERIA','K-Means','K-Means-GPU','Gaussian']

    if test_type == 'under':
        labels = ['APPROX','APPROX ON CORESET','Random-Under','NearMiss-1','NearMiss-2','NearMiss-3']

    if test_type == 'over':
        labels = ['APPROX', 'APPROX ON CORESET','Random-Over', 'SMOTE', 'KMeansSMOTE', 'SMOTETomek']

    if len(labels)==0:
        raise RuntimeError('None supported test type used, options are: approx, under, and over.')

    # Print the time resultS.

    plt.figure()
    plot_error(S, T[:, :, 0], color="black", label=labels[0])
    plot_error(S, T[:, :, 1], color="blue", label=labels[1])
    plot_error(S, T[:, :, 2], color="cyan", label=labels[2])
    plot_error(S, T[:, :, 3], color="yellow", label=labels[3])
    plot_error(S, T[:, :, 4], color="magenta", label=labels[4])
    plot_error(S, T[:, :, 5], color="green", label=labels[5])

    plt.ylim(bottom=0,top=0.4)
    plt.ylabel("Computation time (s)")
    plt.xlabel("Value of x")
    plt.legend(loc='upper center', ncol=2, fancybox=True,bbox_to_anchor=(0.5, 1.38))
    plt.savefig(img_folder +test_type+ "_time.png", dpi=dpi, bbox_inches='tight')
    # plt.show()

    if add_noise and test_type != 'approx':
        M+= rg.normal(0,1e-3,np.shape(M))

    fig = plt.figure()
    plot_error(S, M[:, :, 0], color="black", label=labels[0])
    plot_error(S, M[:, :, 1], color="blue", label=labels[1])
    plot_error(S, M[:, :, 2], color="cyan", label=labels[2])
    plot_error(S, M[:, :, 3], color="yellow", label=labels[3])
    plot_error(S, M[:, :, 4], color="magenta", label=labels[4])
    plot_error(S, M[:, :, 5], color="green", label=labels[5])

    if test_type == 'approx':
        plt.ylim(bottom=0.9, top=2.6)
        # pass

    if test_type == 'over':
        pass
        # plt.ylim(bottom=0.9985, top=1.0015)
    # plt.ylim(bottom=0.995,top=1.005)
    # plt.ylim(bottom=0.995,top=1.025)
    plt.ylabel("Loss (relative to APPROX)")
    plt.xlabel("Value of x")
    # Shrink current axis by 20%
    plt.legend(loc='upper center', ncol=2, fancybox=True,bbox_to_anchor=(0.5, 1.38))
    plt.savefig(img_folder + test_type + "_loss.png", dpi=dpi, bbox_inches='tight')
    # plt.show()


def kmeans(P,k):
    kmeans = KMeans(init="k-means++", n_clusters=k, n_init=1)
    kmeans.fit(P)
    return kmeans.cluster_centers_


def kmeans_gpu(P,k):
    kmeans_float = KMeans_torch(n_clusters=k, output_type='numpy')
    P_t = cudf.DataFrame(P)
    kmeans_float.fit(P_t)
    return kmeans_float.cluster_centers_

def opt(P,k):
    return approx_solver(P,k,True)

def bi_criteria(P,k):
    return approx(P,k,64,True)

def approx_on_core(P,k):
    C,w = coreset(P,k,int(256/(k-1)))
    #   r = w<1e-3/len(P)
    #   w = np.delete(w,r)
    #   C = np.delete(C,r,0)
    ans = approx_solver(C, k, True,w=w)
    return ans


def gaussian(P,k):
    m = GaussianMixture(n_components=k,init_params='k-means++')
    return m.fit_predict(P)


def calc_miss_parititon(P,mask):
    loss = 0

    for i in np.unique(mask):
        p = P[mask ==i]
        c = np.mean(p,0)
        m = np.sum((p - c) ** 2)
        # loss+= m.sum() / ((np.log2(1 + len(m))) ** 2)
        loss += m.mean()

    return np.round(loss,4)

def comp_approx(P,k):
    torch.cuda.synchronize()
    t = time()
    c1 = opt(P, k)
    torch.cuda.synchronize()
    t1 = time() - t
    m1 = calc_miss(P, c1)

    torch.cuda.synchronize()
    t = time()
    c2 = approx_on_core(P, k)
    torch.cuda.synchronize()
    t2 = time() - t
    m2 = calc_miss(P, c2)

    torch.cuda.synchronize()
    t = time()
    c3 = bi_criteria(P, k)
    torch.cuda.synchronize()
    t3 = time() - t
    m3 = calc_miss(P, c3)

    torch.cuda.synchronize()
    t = time()
    c4 = kmeans(P, k)
    torch.cuda.synchronize()
    t4 = time() - t
    m4 = calc_miss(P, c4)

    torch.cuda.synchronize()
    t = time()
    c5 = kmeans_gpu(P, k)
    torch.cuda.synchronize()
    t5 = time() - t
    m5 = calc_miss(P, c5)

    torch.cuda.synchronize()
    t = time()
    c6 = gaussian(P, k)
    torch.cuda.synchronize()
    t6 = time() - t
    m6 = calc_miss_parititon(P, c6)

    return [t1, t2, t3, t4, t5, t6], [m1 / m1, m2 / m1, m3 / m1, m4 / m1, m5 / m1, m6 / m1]

def comp_under(ob_data, ob_target,k):
    torch.cuda.synchronize()
    t = time()
    c1 = opt(ob_data, k)
    torch.cuda.synchronize()
    t1 = time() - t
    m1 = calc_miss(ob_data, c1)

    torch.cuda.synchronize()
    t = time()
    c2 = approx_on_core(ob_data, k)
    torch.cuda.synchronize()
    t2 = time() - t
    m2 = calc_miss(ob_data, c2)

    t = time()
    s = RandomUnderSampler()
    d, _ = s.fit_resample(ob_data, ob_target)
    c3 = kmeans(d, k)
    torch.cuda.synchronize()
    t3 = time() - t
    m3 = calc_miss(ob_data, c3)

    t = time()
    s = NearMiss(version=1)
    d, _ = s.fit_resample(ob_data, ob_target)
    c4 = kmeans(d, k)
    torch.cuda.synchronize()
    t4 = time() - t
    m4 = calc_miss(ob_data, c4)

    t = time()
    s = NearMiss(version=2)
    d, _ = s.fit_resample(ob_data, ob_target)
    c5 = kmeans(d, k)
    torch.cuda.synchronize()
    t5 = time() - t
    m5 = calc_miss(ob_data, c5)

    t = time()
    s = NearMiss(version=3)
    d, _ = s.fit_resample(ob_data, ob_target)
    c6 = kmeans(d, k)
    torch.cuda.synchronize()
    t6 = time() - t
    m6 = calc_miss(ob_data, c6)

    return [t1, t2, t3, t4, t5, t6], [m1 / m1, m2 / m1, m3 / m1, m4 / m1, m5 / m1, m6 / m1]

def comp_over(ob_data, ob_target,k):
    torch.cuda.synchronize()
    t = time()
    c1 = opt(ob_data, k)
    torch.cuda.synchronize()
    t1 = time() - t
    m1 = calc_miss(ob_data, c1)

    torch.cuda.synchronize()
    t = time()
    c2 = approx_on_core(ob_data, k)
    torch.cuda.synchronize()
    t2 = time() - t
    m2 = calc_miss(ob_data, c2)

    t = time()
    d, _ = RandomOverSampler().fit_resample(ob_data, ob_target)
    c3 = kmeans(d, k)
    torch.cuda.synchronize()
    t3 = time() - t
    m3 = calc_miss(ob_data, c3)

    try:
        t = time()
        d, _ = SMOTE().fit_resample(ob_data, ob_target)
        c4 = kmeans(d, k)
        torch.cuda.synchronize()
        t4 = time() - t
        m4 = calc_miss(ob_data, c4)
    except (RuntimeError,ValueError):
        t4 = np.inf
        m4 = np.inf

    try:
        t = time()
        d, _ = KMeansSMOTE().fit_resample(ob_data, ob_target)
        c5 = kmeans(ob_data, k)
        torch.cuda.synchronize()
        t5 = time() - t
        m5 = calc_miss(ob_data, c5)
    except (RuntimeError,ValueError):
        t5 = np.inf
        m5 = np.inf

    try:
        t = time()
        d, _ = SMOTETomek().fit_resample(ob_data, ob_target)
        c6 = kmeans(ob_data, k)
        torch.cuda.synchronize()
        t6 = time() - t
        m6 = calc_miss(ob_data, c6)
    except (RuntimeError,ValueError):
        t6 = np.inf
        m6 = np.inf

    return [t1, t2, t3, t4, t5, t6], [m1 / m1, m2 / m1, m3 / m1, m4 / m1, m5 / m1, m6 / m1]

def one_test(x,k):
    A = sample_circle([0, 0], 1, 25 * x)
    B = sample_circle([2.25, 0], 0.1, x)
    P = np.concatenate((A, B), 0)
    L = np.concatenate((np.zeros(25*x),np.ones(x)))

    if test_type == 'approx':
        return comp_approx(P,k)

    if test_type == 'under':
        return comp_under(P,L,k)

    if test_type == 'over':
        return comp_over(P,L,k)

    raise RuntimeError('None supported test type used, options are: approx, under, and over.')

def run(count=100,k=2):
    print(test_type)
    one_test(25, k)
    print_every = 1
    checks = np.arange(10,51,5)
    T, M = [[] for _ in checks], [[] for _ in checks]
    t = time()
    for i in range(count):
        if i%print_every==0 and i>0:
            print('Done on:',np.round(100*i/count,4),', took:', np.round(time()-t,4))
            t = time()

        for j,r in enumerate(checks):
            c_t, c_m = one_test(r, k)
            T[j].append(c_t)
            M[j].append(c_m)

    return np.array(T),np.array(M),checks

def run_all_tests():
    global file, test_type, img_folder

    for test_type in ['approx','under','over']:
        file = './res/synthetic/'+test_type+'.npz'
        T, M, S = run()
        np.savez_compressed(file, T=T, M=M, S=S)


def plot_all_tests():
    plt.rcParams.update({'lines.linewidth': 20})
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.rcParams.update({'font.size': 40})

    for test_type in ['approx','under','over']:
        file = './res/synthetic/'+test_type+'.npz'
        img_folder = './plots/stand_alone/synthetic/'
        Path(img_folder).mkdir(parents=True, exist_ok=True)

        dict = np.load(file)
        T, M, S = [dict[key] for key in ['T', 'M', 'S']]

        plot(T, M, S, test_type, img_folder)



if __name__ == '__main__':
    # run_all_tests()
    plot_all_tests()