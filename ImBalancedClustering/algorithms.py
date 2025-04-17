import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# from torch_kmeans import KMeans as KMeans_torch
from time import perf_counter as time
from numpy.random import Generator, PCG64
from itertools import combinations
from sklearn.cluster import KMeans

rg = Generator(PCG64())

t1, t2 = 0, 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calc_loss_log(C_opt, P, n, k, size, w=None):
    P, C_opt = torch.tensor(P, device=device), torch.tensor(C_opt, device=device)
    P_ = torch.unsqueeze(P, 0).repeat((k, 1, 1))
    #   P__ = torch.unsqueeze(P, 0).repeat((size, 1, 1))
    #   C_old = torch.unsqueeze(C_opt, 2).repeat((1, 1, n, 1))
    #
    #   l_ = torch.sum((P_ - C_old) ** 2, 3).sqrt()
    #   p = l_.argmin(1).to(torch.int8)
    #   C = torch.zeros((size,k,d), dtype=torch.float32, device=device)
    #   for j in range(k):
    #       mask = p == j
    #       mask = torch.unsqueeze(mask, 2).repeat((1, 1, d))
    #       p_ = torch.where(mask, P__, torch.zeros_like(P__))
    #       m = torch.count_nonzero(mask,1)
    #       C[:,j,:] = p_.sum(1)/m

    C_ = torch.unsqueeze(C_opt, 2).repeat((1, 1, n, 1))
    l = torch.sum((P_ - C_) ** 2, 3)# .sqrt()
    ans = l.min(1)
    l, p = ans[0], ans[1].to(torch.int8)
    if w is not None:
        w = torch.tensor(w, device=device, dtype=torch.float32)
        l*= w

    miss = torch.zeros(size, dtype=torch.float32, device=device)
    for j in range(k):
        mask = p == j
        m = torch.count_nonzero(mask, 1)
        l_ = torch.where(mask, l, torch.zeros_like(l))
        miss += (l_.sum(1)) / (torch.log2(m + 1) ** 2)
    i = torch.argmin(miss)
    m, c = miss[i], C_opt[i]
    return m.to('cpu').item(), c.to('cpu').numpy()


def calc_loss(C_opt, P, n, k, size, w=None):
    d = len(P[0])
    P, C_opt = torch.tensor(P, device=device), torch.tensor(C_opt, device=device)
    P_ = torch.unsqueeze(P, 0).repeat((k, 1, 1))

    C_ = torch.unsqueeze(C_opt, 2).repeat((1, 1, n, 1))
    l = torch.sum((P_ - C_) ** 2, 3)# .sqrt()
    ans = l.min(1)
    l, p = ans[0], ans[1].to(torch.int8)
    if w is not None:
        w = torch.tensor(w, device=device, dtype=torch.float32)
        l*= w

    miss = torch.zeros(size, dtype=torch.float32, device=device)
    for j in range(k):
        mask = p == j
        m = torch.count_nonzero(mask, 1)
        l_ = torch.where(mask, l, torch.zeros_like(l))
        miss += (l_.sum(1)) / (1 + m)
    i = torch.argmin(miss)
    m, c = miss[i], C_opt[i]
    return m.to('cpu').item(), c.to('cpu').numpy()


def fast_mean(A, l):
    # return np.mean(A,0)
    return np.einsum('ij->j', A) / l


def approx_solver(P, k, log=True, w=None):
    n = len(P)
    d = len(P[0])

    P = P.astype(np.float32)

    P_ = np.unique(P, axis=0)
    if len(P_)<=k:
        return P_

    C_opt = np.array(list(combinations(P_, k)))
    #
    # C_opt = np.array([rg.choice(P_, k, False) for _ in range(128_00)])

    size = len(C_opt)
    expected_mem = 8 * size * n * d * k

    av = torch.cuda.mem_get_info()[0]
    count = int(8 * expected_mem / av)  # To account for matrix copy and operations.


    # count = np.min([count,size])
    #   print(expected_mem,av)
    #   print(count)
    if count <= 1:
        opt = [C_opt]
    else:
        opt = np.array_split(C_opt, count)
    miss, C_opt = [], []
    # print([len(c) for c in opt])
    for C in opt:
        # print(len(C))
        if log:
            m, c = calc_loss_log(C, P, n, k, len(C), w)
        else:
            m, c = calc_loss(C, P, n, k, len(C), w)
        miss.append(m)
        C_opt.append(c)
    # print(np.min(miss))
    i = np.argmin(miss)
    return C_opt[i]


def robust_mean(P, k, s):
    p_np = rg.choice(P, s)
    place = int(s * 15 / (16 * k))

    #   p_0 = np.expand_dims(p_np, 0)
    #   p_1 = np.repeat(p_0,s,0)
    #
    #   p_ = p_1-np.transpose(p_1,(1,0,2))
    #   L = np.linalg.norm(p_,2,2)
    #   L_ = np.partition(L,place,1)
    #   l = L_[:,:place]
    #   l_ = np.sum(l,1)
    #   i_ = np.argmin(l)

    p = torch.tensor(p_np, device=device, dtype=torch.float32)
    p_1 = torch.unsqueeze(p, 0).repeat((s, 1, 1))
    p_ = p_1 - torch.transpose(p_1, 0, 1)
    L = torch.sum(p_**2,dim=2)
    l = torch.topk(L, place, 1, largest=False, sorted=False)[0]
    l = l.sum(1)
    i = torch.argmin(l).to('cpu').item()
    return p_np[i]


t1 = 0


def approx_32(P, k, s=64, try_solve=True):
    if len(P) <= 2 * s:
        if try_solve:
            #   kmeans = KMeans(init="k-means++", n_clusters=k, n_init=1)
            #   kmeans.fit(P)
            #   return kmeans.cluster_centers_
            return approx_solver(P, k, log=True)
        else:
            return P

    # P = P.astype(np.float32)

    c = robust_mean(P, k, s)
    place = int(len(P) * 3 / (4 * k))

    P_, c_ = torch.tensor(P, device=device, dtype=torch.float32), torch.tensor(c, device=device, dtype=torch.float32)
    miss = P_ - c_
    L = torch.sum(miss**2,dim=1)
    l = torch.topk(L, place, largest=True, sorted=False)[1]
    rest = P_[l].to('cpu').numpy()

    #   miss = P-c
    #   L = np.linalg.norm(miss, 2, 1)
    #   place = int(len(P)*3/(4*k))
    #   L_ = np.argpartition(L, place)
    #   l= L_[place:]
    #   rest = P[l]

    ans_rest = approx_32(rest, k, s=s)
    return np.append(ans_rest, np.expand_dims(c, 0), 0)


def approx(P, k, s=64, try_solve=True):
    return approx_32(P.astype(np.float32), k, s=s, try_solve=try_solve)


def sen_coreset(P, sample, s):
    n = len(P)
    if n <= sample:
        return P, np.ones(n)

    p = s / s.sum()
    #   p[p < 1e-3 / n] = 0
    #   p += 1e-2 / n
    #   p = p / p.sum()

    ind = rg.choice(n, sample, True, p)
    hist = np.histogram(ind, bins=range(n + 1))[0].flatten()
    indxs = np.nonzero(hist)[0]
    W = hist[indxs]
    W = W / (sample * p[indxs])
    # W = W*n/np.sum(W)
    return P[indxs], W



def coreset(P, k, size=100):
    #   module = KMeans_torch('k-means++', 1, n_clusters=k, verbose=False)
    #   P_t = torch.tensor(np.expand_dims(P, 0), dtype=torch.float32, device=device)
    #   module = module.fit(P_t)
    #   B = module._result.centers[0].to('cpu').numpy()  # Get protected attributes, a new best practice.
    #   P_ = module.predict(P_t)[0].to('cpu').numpy()

    #   B = approx(P,k)
    #
    #   n = len(P)
    #   P_ = np.expand_dims(P, 0)
    #   P_ = np.repeat(P_, len(B), 0)
    #   C = np.expand_dims(B, 1).repeat(n, 1)
    #   l = np.sqrt(np.sum((P_ - C) ** 2, 2))
    #   P_ = np.argmin(l, 0)

    kmeans = KMeans(init="k-means++", n_clusters=k, n_init=1)
    P_ = kmeans.fit_predict(P)
    B = kmeans.cluster_centers_

    l = np.zeros(len(P))
    w = []
    for i, c in enumerate(B):
        mask = P_ == i
        m = np.count_nonzero(mask)
        w.append(m)
        p = P[mask]
        l[mask] = np.sum((p-c)**2,1)/ (np.log2(1 + m) ** 2)
        l[mask] += 1/m

    C,w = sen_coreset(P,size,l)
    return C, w


def split_by_centers(P, c):
    n, d = np.shape(P)
    k = len(c)

    P_ = np.expand_dims(P, 0)
    P_ = np.repeat(P_, k, 0)
    C = np.expand_dims(c, 1).repeat(n, 1)
    l = np.sum((P_ - C) ** 2, 2)
    p = np.argmin(l, 0)

    ans = []
    for i in range(k):
        mask = p == i
        p_ = P[mask]
        ans.append(p_)

    return ans


def calc_miss(P, c):
    loss = 0
    P_ = split_by_centers(P, c)
    k = len(c)
    for p_, c_ in zip(P_, c):
        if len(p_) == 0:
            continue
        c = np.mean(p_,0)
        m = np.sum((p_ - c) ** 2, 1)
        # loss+= m.sum() / ((np.log2(1 + len(m))) ** 2)
        loss += m.mean()

    return loss/k


# from cProfile import Profile
# from pstats import SortKey, Stats

def k_means_split(P,k):
    modle = KMeans(init="k-means++", n_clusters=k, n_init=1)
    pred = modle.fit_predict(P)
    ans = []
    for i in range(k):
        mask = pred==i
        ans.append(P[mask])

    return ans


if __name__ == '__main__':
    for _ in range(100):
        n1, n2, d = 10_000, 100, 2
        z = np.zeros(d)
        A = rg.normal(z, 0.15, (n1, d))
        B = rg.normal(z + 1, 0.015, (n2, d))

        P = np.concatenate([A, B], 0)
        t = time()
        # for _ in range(100)
        ans2 = approx_solver(P, 2)
        t_ = time() - t
        print(2, calc_miss(P, ans2), np.round(t_, 5))
        t = time()
        P_ = k_means_split(P, 2)
        t_ = time() - t
        loss = 0
        for p_ in P_:
            c_ = np.mean(p_, 0)
            m = np.sqrt(np.sum((p_ - c_) ** 2, 1))
            loss += m.sum() / (np.log2(1 + len(m))) ** 2
            # print(m.sum()/(np.log2(1+len(m)))**2, len(m))
        print(3, np.round(loss, 4), np.round(t_, 5))
