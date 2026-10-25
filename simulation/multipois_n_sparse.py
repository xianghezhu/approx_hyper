#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: xianghezhu
"""


import numpy as np
import scipy.stats as stat
from math import comb
from itertools import combinations
import matplotlib.pyplot as plt


itr = 100000
k_list = [2, 3]
M = 3
n_list = [20, 30, 50, 80, 120, 150]
c = 0.5

tv_dist = np.zeros(len(n_list))
stein_err = np.zeros(len(n_list))


def TVD_Poisson_exact(lam, sample):
    values, counts = np.unique(sample, axis=0, return_counts=True)
    P = np.prod(stat.poisson.pmf(values, lam), axis=1)
    Q = counts / np.sum(counts)
    return max(0., 1. - np.sum(np.minimum(P, Q)))


def Stein_bound(M, prob, n, k):
    return M * (comb(n - 1, k - 1) * prob**2
                + (M - 1) * comb(n - 2, k - 2) * prob * (1. + prob))


for n_id, n in enumerate(n_list):
    p_list = np.zeros(len(k_list))
    mean = 0.
    for k_id, k in enumerate(k_list):
        nk1 = comb(n - 1, k - 1)
        prob = c / float(nk1)
        p_list[k_id] = prob
        mean += nk1 * prob
        stein_err[n_id] += Stein_bound(M, prob, n, k)
    rng = np.random.default_rng(1066 + n_id)
    deg = np.zeros((itr, M), dtype=int)
    
    for k_id, k in enumerate(k_list):
        prob = p_list[k_id]
        for s in range(1, min(k, M) + 1):
            for selected in combinations(range(M), s):
                count = rng.binomial(comb(n - M, k - s), prob, size=itr)
                for v in selected:
                    deg[:, v] += count
    tv_dist[n_id] = TVD_Poisson_exact(mean, deg)

print(tv_dist, stein_err)


plt.figure()
plt.plot(n_list, tv_dist, color="red", marker="x", label="Empirical TV distance")
plt.plot(n_list, stein_err, color="blue", marker="^", label="Stein bound")
plt.legend(prop={"size": 13})
plt.xlabel("Size of the hypergraph", fontsize=15)
plt.ylabel("TV distance", fontsize=15)
plt.savefig("TV_multipois_M" + str(M) + "_n_change_sparse_c" + str(c) + "_k" + str(k_list[0]) + "_" + str(k_list[1]) + ".pdf", bbox_inches="tight")
#plt.show()
plt.figure()
plt.plot(n_list, np.log(tv_dist), color="red", marker="x", label="Empirical TV distance")
plt.plot(n_list, np.log(stein_err), color="blue", marker="^", label="Stein bound")
plt.legend(prop={"size": 13})
plt.xlabel("Size of the hypergraph", fontsize=15)
plt.ylabel("Log TV distance", fontsize=15)
plt.savefig("Log" + "TV_multipois_M" + str(M) + "_n_change_sparse_c" + str(c) + "_k" + str(k_list[0]) + "_" + str(k_list[1]) + ".pdf", bbox_inches="tight")
#plt.show()
