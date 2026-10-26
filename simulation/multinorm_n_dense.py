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
n_list = [20, 30, 50, 80, 120, 160, 200]
prob = .1

ks_dist = np.zeros(len(n_list))
stein_err = np.zeros(len(n_list))


def Kolmogorov_distance(sample, normal):
    return stat.ks_2samp(sample, normal, method="asymp").statistic


def Stein_bound(B, sigma):
    tau = 1.0 if M == 1 else (variance - covariance) / sigma**2
    return 2.**(13. / 2.) * 3.**(-3. / 4.) * M**(9. / 16.) * B**.25 * sigma**(-.75) * tau**(-3. / 8.)


for n_id, n in enumerate(n_list):
    mean = 0.
    variance = 0.
    B = 0.
    for k in k_list:
        mean += comb(n - 1, k - 1) * prob
        variance += comb(n - 1, k - 1) * prob * (1. - prob)
        nk3 = 0 if k == 2 else comb(n - 3, k - 3)
        B += 4. / 3. * (comb(n - 1, k - 1) + 2 * comb(n - 2, k - 2) + nk3) * prob * (1. - prob)
    sigma = np.sqrt(variance)
    stein_err[n_id] = Stein_bound(B, sigma)
    rng = np.random.default_rng(1066 + n_id)
    deg = np.zeros((itr, M), dtype=int)
    
    for k in k_list:
        for s in range(1, min(k, M) + 1):
            for selected in combinations(range(M), s):
                count = rng.binomial(comb(n - M, k - s), prob, size=itr)
                for v in selected:
                    deg[:, v] += count
    normal = rng.standard_normal(itr)
    ks_dist[n_id] = Kolmogorov_distance((deg[:, 0] - mean) / sigma, normal)

print(ks_dist, stein_err)


plt.figure()
plt.plot(n_list, ks_dist, color="red", marker="x", label="Empirical Kolmogorov distance")
plt.plot(n_list, stein_err, color="blue", marker="^", label="Stein bound")
plt.legend(prop={"size": 13})
plt.xlabel("Size of the hypergraph", fontsize=15)
plt.ylabel("Kolmogorov distance", fontsize=15)
plt.savefig("KS_multinorm_M" + str(M) + "_n_change_dense_p" + str(prob) + "_k" + str(k_list[0]) + "_" + str(k_list[1]) + ".pdf", bbox_inches="tight")
plt.show()
plt.figure()
plt.plot(n_list, np.log(ks_dist), color="red", marker="x", label="Empirical Kolmogorov distance")
plt.plot(n_list, np.log(stein_err), color="blue", marker="^", label="Stein bound")
plt.legend(prop={"size": 13})
plt.xlabel("Size of the hypergraph", fontsize=15)
plt.ylabel("Log Kolmogorov distance", fontsize=15)
plt.savefig("Log" + "KS_multinorm_M" + str(M) + "_n_change_dense_p" + str(prob) + "_k" + str(k_list[0]) + "_" + str(k_list[1]) + ".pdf", bbox_inches="tight")
plt.show()

