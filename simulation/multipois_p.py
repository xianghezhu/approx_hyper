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
n = 100
c_list = np.linspace(.6, 2, 8)


p_list = np.zeros((len(k_list), len(c_list)))
for k_id, k in enumerate(k_list):
    nk1 = comb(n - 1, k - 1)
    p_list[k_id] = np.linspace(.6, 2, 8) / nk1

tv_dist = np.zeros(len(c_list))
stein_err = np.zeros(len(c_list))


def TVD_Poisson_exact(lam, sample):
    values, counts = np.unique(sample, axis=0, return_counts=True)
    P = np.prod(stat.poisson.pmf(values, lam), axis=1)
    Q = counts / np.sum(counts)
    return max(0., 1. - np.sum(np.minimum(P, Q)))


def Stein_bound(M, prob, n, k):
    return M * (comb(n - 1, k - 1) * prob**2
                + (M - 1) * comb(n - 2, k - 2) * prob * (1. + prob))


for p_id, c in enumerate(c_list):
    mean = 0.
    for k_id, k in enumerate(k_list):
        prob = p_list[k_id, p_id]
        mean += comb(n - 1, k - 1) * prob
        stein_err[p_id] += Stein_bound(M, prob, n, k)
    rng = np.random.default_rng(1066 + p_id)
    deg = np.zeros((itr, M), dtype=int)
    
    for k_id, k in enumerate(k_list):
        prob = p_list[k_id, p_id]
        for s in range(1, min(k, M) + 1):
            for selected in combinations(range(M), s):
                count = rng.binomial(comb(n - M, k - s), prob, size=itr)
                for v in selected:
                    deg[:, v] += count
    tv_dist[p_id] = TVD_Poisson_exact(mean, deg)

print(tv_dist, stein_err)


plt.figure()
plt.plot(c_list, tv_dist, color="red", marker="x", label="Empirical TV distance")
plt.plot(c_list, stein_err, color="blue", marker="^", label="Stein bound")
plt.legend(prop={"size": 13})
plt.xlabel("Probability parameter c", fontsize=15)
plt.ylabel("TV distance", fontsize=15)
plt.savefig("TV_multipois_M" + str(M) + "_n" + str(n) + "_p_change_k" + str(k_list[0]) + "_" + str(k_list[1]) + ".pdf", bbox_inches="tight")
#plt.show()
plt.figure()
plt.plot(c_list, np.log(tv_dist), color="red", marker="x", label="Empirical TV distance")
plt.plot(c_list, np.log(stein_err), color="blue", marker="^", label="Stein bound")
plt.legend(prop={"size": 13})
plt.xlabel("Probability parameter c", fontsize=15)
plt.ylabel("Log TV distance", fontsize=15)
plt.savefig("Log" + "TV_multipois_M" + str(M) + "_n" + str(n) + "_p_change_k" + str(k_list[0]) + "_" + str(k_list[1]) + ".pdf", bbox_inches="tight")
#plt.show()

