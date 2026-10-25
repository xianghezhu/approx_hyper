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
M = 1
n = 100
p_grid = np.array([.01, .1, .3,.5,.7 ,.9, .99])

p_list = np.tile(p_grid, (len(k_list), 1))

ks_dist = np.zeros(len(p_grid))
stein_err = np.zeros(len(p_grid))


def Kolmogorov_distance(sample, normal):
    return stat.ks_2samp(sample, normal, method="asymp").statistic


def Stein_bound(B, sigma):
    return 2.**(7. / 2.) * 3.**(-3. / 4.) * M**(3. / 16.) * B**.25 * sigma**(-.75)


for p_id, p in enumerate(p_grid):
    mean = 0.
    variance = 0.
    B = 0.
    for k_id, k in enumerate(k_list):
        prob = p_list[k_id, p_id]
        mean += comb(n - 1, k - 1) * prob
        variance += comb(n - 1, k - 1) * prob * (1. - prob)
        nk3 = 0 if k == 2 else comb(n - 3, k - 3)
        B += 4. / 3. * (comb(n - 1, k - 1) + 2 * comb(n - 2, k - 2) + nk3) * prob * (1. - prob)
    sigma = np.sqrt(variance)
    stein_err[p_id] = Stein_bound(B, sigma)
    rng = np.random.default_rng(1066 + p_id)
    deg = np.zeros((itr, M), dtype=int)
    
    for k_id, k in enumerate(k_list):
        prob = p_list[k_id, p_id]
        for s in range(1, min(k, M) + 1):
            for selected in combinations(range(M), s):
                count = rng.binomial(comb(n - M, k - s), prob, size=itr)
                for v in selected:
                    deg[:, v] += count
    normal = rng.standard_normal(itr)
    ks_dist[p_id] = Kolmogorov_distance((deg[:, 0] - mean) / sigma, normal)

print(ks_dist, stein_err)

plt.figure()
plt.plot(p_grid, ks_dist, color="red", marker="x", label="Empirical Kolmogorov distance")
plt.plot(p_grid, stein_err, color="blue", marker="^", label="Stein bound")
plt.legend(prop={"size": 13})
plt.xlabel("Probability p", fontsize=15)
plt.ylabel("Kolmogorov distance", fontsize=15)
plt.savefig("KS_multinorm_M" + str(M) + "_n" + str(n) + "_p_change_k" + str(k_list[0]) + "_" + str(k_list[1]) + ".pdf", bbox_inches="tight")
plt.show()
plt.figure()
plt.plot(p_grid, np.log(ks_dist), color="red", marker="x", label="Empirical Kolmogorov distance")
plt.plot(p_grid, np.log(stein_err), color="blue", marker="^", label="Stein bound")
plt.legend(prop={"size": 13})
plt.xlabel("Probability p", fontsize=15)
plt.ylabel("Log Kolmogorov distance", fontsize=15)
plt.savefig("Log" + "KS_multinorm_M" + str(M) + "_n" + str(n) + "_p_change_k" + str(k_list[0]) + "_" + str(k_list[1]) + ".pdf", bbox_inches="tight")
plt.show()

