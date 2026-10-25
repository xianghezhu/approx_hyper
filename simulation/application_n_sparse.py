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

prob_diff = np.zeros(len(n_list))
stein_err = np.zeros(len(n_list))


def Probability_error(lam, sample):
    M = sample.shape[1]
    zero_count = np.sum(sample == 0, axis=1)
    one_count = np.sum(sample == 1, axis=1)
    q0 = np.exp(-lam)
    q1 = lam * np.exp(-lam)
    P = np.zeros((M + 1, M + 1))
    for count0 in range(M + 1):
        for count1 in range(M + 1 - count0):
            P[count0, count1] = stat.multinomial.pmf(
                [count0, count1, M - count0 - count1],
                M, [q0, q1, 1. - q0 - q1])
    error = 0.
    
    for a in range(M + 1):
        for b in range(M + 1):
            empirical = np.mean((zero_count > a) | (one_count > b))
            reference = np.sum(P[a + 1:, :]) + np.sum(P[:a + 1, b + 1:])
            error = max(error, abs(empirical - reference))
    return error


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
    prob_diff[n_id] = Probability_error(mean, deg)

print(prob_diff, stein_err)


plt.figure()
plt.plot(n_list, prob_diff, color="red", marker="x", label="Empirical maximum error")
plt.plot(n_list, stein_err, color="blue", marker="^", label=r"Theorem 1 bound $B$")
plt.legend(prop={"size": 13})
plt.xlabel("Size of the hypergraph", fontsize=15)
plt.ylabel("Maximum probability difference", fontsize=15)
plt.savefig("Probability_application_M" + str(M) + "_n_change_sparse_c" + str(c) + "_k" + str(k_list[0]) + "_" + str(k_list[1]) + ".pdf", bbox_inches="tight")
plt.show()
plt.figure()
plt.plot(n_list, np.log(prob_diff), color="red", marker="x", label="Empirical maximum error")
plt.plot(n_list, np.log(stein_err), color="blue", marker="^", label=r"Theorem 1 bound $B$")
plt.legend(prop={"size": 13})
plt.xlabel("Size of the hypergraph", fontsize=15)
plt.ylabel("Log maximum probability difference", fontsize=15)
plt.savefig("Log" + "Probability_application_M" + str(M) + "_n_change_sparse_c" + str(c) + "_k" + str(k_list[0]) + "_" + str(k_list[1]) + ".pdf", bbox_inches="tight")
plt.show()
