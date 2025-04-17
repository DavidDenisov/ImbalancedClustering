from imblearn.datasets import fetch_datasets
from synthetic_test import comp_approx,comp_under,comp_over,plot
from time import perf_counter as time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from numpy.random import Generator, PCG64
rg = Generator(PCG64())

test_type = 'under' # 'approx','under','over'.

data_sets = fetch_datasets()
data = data_sets['webpage']
data,target = data.data,data.target.astype('int8')
k = len(np.unique(target))

file = './res/real_mean/under.npz'
img_folder = './figs/tests/real_mean/'
dpi = 100

def test_one(x):
    I = rg.choice(len(data),x,replace=False)
    ob_data = data[I]
    ob_target = target[I]

    if test_type == 'approx':
        return comp_approx(ob_data, k)

    if test_type == 'under':
        return comp_under(ob_data, ob_target, k)

    if test_type == 'over':
        return comp_over(ob_data, ob_target, k)

    raise RuntimeError('Un-supported test type used, options are: approx, under, and over.')

def run(count=100):
    test_one(400)
    print_every = 1
    checks = np.arange(400,801,100)
    T, M = [[] for _ in checks], [[] for _ in checks]
    t = time()
    for i in range(count):
        if i%print_every==0 and i>0:
            print('Done on:',np.round(100*i/count,4),', took:', np.round(time()-t,4))
            t = time()

        for j,r in enumerate(checks):
            c_t, c_m = test_one(r)
            T[j].append(c_t)
            M[j].append(c_m)

    return np.array(T),np.array(M),checks

def run_all_tests():
    global file, test_type, img_folder

    for test_type in ['approx','under','over']:
        file = './res/real/'+test_type+'.npz'
        T, M, S = run()
        np.savez_compressed(file, T=T, M=M, S=S)


def plot_all_tests():
    plt.rcParams.update({'lines.linewidth': 20})
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.rcParams.update({'font.size': 40})

    for test_type in ['approx','under','over']:
        file = './res/real/'+test_type+'.npz'
        img_folder = './plots/stand_alone/real/'
        Path(img_folder).mkdir(parents=True, exist_ok=True)

        dict = np.load(file)
        T, M, S = [dict[key] for key in ['T', 'M', 'S']]

        plot(T, M, S, test_type, img_folder)



if __name__ == '__main__':
    # run_all_tests()
    plot_all_tests()
