#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: xianghezhu
"""


import numpy as np
import scipy.stats as stat
from math import comb
import matplotlib.pyplot as plt
from application2_sampling import ER_sample, SBM_sample, Closure_sample


itr = 1000000
k_list = [2, 3]
M = 10
n_list = [50, 80, 120, 160, 200, 300]
n_delta = 100
p = .2
eta = .8
delta = .4
delta_list = [0., .2, .4, .6, .8]
alpha = .05

# Rows: ER, SBM, simplicial complex.
ks_dist = np.zeros((3, len(n_list)))
rejection_rate = np.zeros((3, len(n_list)))
stein_err = np.zeros(len(n_list))
sbm_ks_dist = np.zeros(len(delta_list))
sbm_rejection_rate = np.zeros(len(delta_list))


def Kolmogorov_distance(sample, normal):
    return stat.ks_2samp(sample, normal, method="asymp").statistic


def Stein_bound(B, sigma):
    return 2.**(7. / 2.) * 3.**(-3. / 4.) * M**(3. / 16.) * B**.25 * sigma**(-.75)


for n_id, n in enumerate(n_list):
    p_list = np.array([p, -np.expm1(np.log1p(-p) / (n - 2))])
    mean = variance = covariance = B = 0.
    for k, prob in zip(k_list, p_list):
        nk1 = comb(n - 1, k - 1)
        nk2 = comb(n - 2, k - 2)
        nk3 = 0 if k == 2 else comb(n - 3, k - 3)
        mean += nk1 * prob
        variance += nk1 * prob * (1. - prob)
        covariance += nk2 * prob * (1. - prob)
        B += 4. * M / 3. * (nk1 + 2 * M * nk2 + M**2 * nk3) * prob * (1. - prob)
    sigma = np.sqrt(variance + (M - 1) * covariance)
    cov_matrix = np.full((M, M), covariance)
    np.fill_diagonal(cov_matrix, variance)
    stein_err[n_id] = Stein_bound(B, sigma)
    rng = np.random.default_rng(1066 + n_id)

    within = np.array([2 * comb(n // 2, k) / comb(n, k) for k in k_list])
    p_a = p_list * (1 + eta * (1 - within) + delta)
    p_b = p_list * (1 + eta * (1 - within) - delta)
    p_out = p_list * (1 - eta * within)
    samples = [ER_sample(n, M, p_list, itr, rng),
               SBM_sample(n, M, n // 2, p_a, p_b, p_out, itr, rng),
               Closure_sample(n, M, p_list[1], itr, rng)]
    normal = rng.standard_normal((itr, M)) @ np.linalg.cholesky(cov_matrix).T / sigma
    normal_max = normal.max(axis=1)
    critical = np.quantile(normal_max, 1 - alpha)
    for model, deg in enumerate(samples):
        sample_max = deg.max(axis=1)
        ks_dist[model, n_id] = Kolmogorov_distance((sample_max - mean) / sigma, normal_max)
        rejection_rate[model, n_id] = np.mean((sample_max - mean) / sigma > critical)
    print(n, ks_dist[:, n_id], rejection_rate[:, n_id], stein_err[n_id], flush=True)


# SBM: vary delta at fixed n
n = n_delta
p_list = np.array([p, -np.expm1(np.log1p(-p) / (n - 2))])
mean = variance = covariance = 0.
for k, prob in zip(k_list, p_list):
    mean += comb(n - 1, k - 1) * prob
    variance += comb(n - 1, k - 1) * prob * (1. - prob)
    covariance += comb(n - 2, k - 2) * prob * (1. - prob)
sigma = np.sqrt(variance + (M - 1) * covariance)
cov_matrix = np.full((M, M), covariance)
np.fill_diagonal(cov_matrix, variance)
rng = np.random.default_rng(1066)
normal = rng.standard_normal((itr, M)) @ np.linalg.cholesky(cov_matrix).T / sigma
normal_max = normal.max(axis=1)
critical = np.quantile(normal_max, 1 - alpha)

within = np.array([2 * comb(n // 2, k) / comb(n, k) for k in k_list])
p_out = p_list * (1 - eta * within)
for delta_id, delta_value in enumerate(delta_list):
    p_a = p_list * (1 + eta * (1 - within) + delta_value)
    p_b = p_list * (1 + eta * (1 - within) - delta_value)
    deg = SBM_sample(n, M, n // 2, p_a, p_b, p_out, itr, rng)
    sample_max = (deg.max(axis=1) - mean) / sigma
    sbm_ks_dist[delta_id] = Kolmogorov_distance(sample_max, normal_max)
    sbm_rejection_rate[delta_id] = np.mean(sample_max > critical)
    print(n, delta_value, sbm_ks_dist[delta_id], sbm_rejection_rate[delta_id], flush=True)


plt.figure()
plt.plot(n_list, ks_dist[0], color="red", marker="x", label="ER hypergraph")
plt.plot(n_list, ks_dist[1], color="blue", marker="^", label="Hypergraph SBM")
plt.plot(n_list, ks_dist[2], color="green", marker="o", label="Simplicial complex")
plt.plot(n_list, stein_err, color="black", marker="s", linestyle="--", label="Theorem 2 bound (ER)")
plt.legend(prop={"size": 13}, loc="lower right", bbox_to_anchor=(1, .18))
plt.xlabel("Size of the hypergraph", fontsize=15)
plt.ylabel("Kolmogorov distance / upper bound", fontsize=15)
plt.savefig("Application2_models_Kolmogorov_distance.pdf", bbox_inches="tight")
plt.show()

plt.figure()
plt.plot(n_list, np.log(ks_dist[0]), color="red", marker="x", label="ER hypergraph")
plt.plot(n_list, np.log(ks_dist[1]), color="blue", marker="^", label="Hypergraph SBM")
plt.plot(n_list, np.log(ks_dist[2]), color="green", marker="o", label="Simplicial complex")
plt.plot(n_list, np.log(stein_err), color="black", marker="s", linestyle="--", label="Theorem 2 bound (ER)")
plt.legend(prop={"size": 13}, loc="center right", bbox_to_anchor=(1, .70))
plt.xlabel("Size of the hypergraph", fontsize=15)
plt.ylabel("Log Kolmogorov distance / bound", fontsize=15)
plt.savefig("LogApplication2_models_Kolmogorov_distance.pdf", bbox_inches="tight")
plt.show()

plt.figure()
plt.plot(n_list, rejection_rate[0], color="red", marker="x", label="ER hypergraph")
plt.plot(n_list, rejection_rate[1], color="blue", marker="^", label="Hypergraph SBM")
plt.plot(n_list, rejection_rate[2], color="green", marker="o", label="Simplicial complex")
plt.axhline(alpha, color="gray", linestyle="--", linewidth=.9, label="Nominal level (5%)")
plt.legend(prop={"size": 13}, loc="lower right", bbox_to_anchor=(1, .18))
plt.xlabel("Size of the hypergraph", fontsize=15)
plt.ylabel("Rejection probability", fontsize=15)
plt.savefig("Application2_models_Gaussian_rejection.pdf", bbox_inches="tight")
plt.show()


# SBM delta plots
plt.figure()
plt.plot(delta_list, sbm_ks_dist, color="blue", marker="^", label="Hypergraph SBM")
plt.legend(prop={"size": 13}, loc="upper left")
plt.xlabel(r"$\delta$", fontsize=15)
plt.xticks(delta_list)
plt.ylabel("Empirical Kolmogorov distance", fontsize=15)
plt.savefig("Application2_SBM_delta_Kolmogorov_distance.pdf", bbox_inches="tight")
plt.show()

plt.figure()
plt.plot(delta_list, sbm_rejection_rate, color="blue", marker="^", label="Hypergraph SBM")
plt.axhline(alpha, color="gray", linestyle="--", linewidth=.9, label="Nominal level (5%)")
plt.legend(prop={"size": 13}, loc="upper left")
plt.xlabel(r"$\delta$", fontsize=15)
plt.xticks(delta_list)
plt.ylabel("Rejection probability", fontsize=15)
plt.savefig("Application2_SBM_delta_Gaussian_rejection.pdf", bbox_inches="tight")
plt.show()
