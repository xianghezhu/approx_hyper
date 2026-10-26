#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: xianghezhu
"""

from pathlib import Path
from math import comb
import numpy as np
import scipy.stats as stat


M = 3
groups = 100
itr = 100000
rng = np.random.default_rng(1066)
folder = Path(__file__).resolve().parent
raw = folder / "data/madstat"


def TVD_Poisson_exact(lam, sample):
    values, counts = np.unique(sample, axis=0, return_counts=True)
    P = np.prod(stat.poisson.pmf(values, lam), axis=1)
    Q = counts / np.sum(counts)
    return max(0., 1. - np.sum(np.minimum(P, Q)))


def Kolmogorov_distance(sample, normal):
    return stat.ks_2samp(sample, normal, method="asymp").statistic


def Poisson_bound(M, prob, n, k):
    return M * (comb(n - 1, k - 1) * prob**2
                + (M - 1) * comb(n - 2, k - 2) * prob * (1. + prob))


def Normal_bound(B, sigma):
    return 2.**(13. / 2.) * 3.**(-3. / 4.) * B**.25 * sigma**(-.75)


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


sizes = np.loadtxt(raw / "madstat-nverts.txt", dtype=int)
vertices = np.loadtxt(raw / "madstat-simplices.txt", dtype=int)
years = np.loadtxt(raw / "madstat-times.txt", dtype=int)
n = len((raw / "author_name.txt").read_text(encoding="utf-8").splitlines())
edges = set()
start = 0
for size, year in zip(sizes, years):
    edge = tuple(sorted(set(vertices[start:start + size])))
    start += size
    if 1975 <= year <= 2015 and len(edge) >= 2:
        edges.add(edge)


poisson_nodes = rng.permutation(n)[:groups * M].reshape(groups, M)
normal_nodes = rng.choice(n, groups, replace=False)


for name, min_k in [("full_all_sizes", 2), ("all_ge6", 6)]:
    retained = [edge for edge in edges if len(edge) >= min_k]
    degree = np.zeros(n, dtype=int)
    for edge in retained:
        degree[np.array(edge) - 1] += 1
    k_list = sorted({len(edge) for edge in retained})
    m_list = [sum(len(edge) == k for edge in retained) for k in k_list]
    p_list = [m / comb(n, k) for k, m in zip(k_list, m_list)]
    mean = float(degree.mean())
    variance = 0.
    B = 0.
    poisson_bound = 0.
    print("\n", name, "n =", n, "hyperedges =", len(retained))
    print("k, m_k, p_hat")
    for k, m, prob in zip(k_list, m_list, p_list):
        variance += comb(n - 1, k - 1) * prob * (1. - prob)
        nk3 = 0 if k == 2 else comb(n - 3, k - 3)
        B += 4. / 3. * (comb(n - 1, k - 1) + 2 * comb(n - 2, k - 2) + nk3) * prob * (1. - prob)
        poisson_bound += Poisson_bound(M, prob, n, k)
        print(k, m, prob)
    sigma = np.sqrt(variance)
    normal_bound = Normal_bound(B, sigma)
    print("mean =", mean, "ER variance =", variance,
          "observed variance =", float(degree.var()))
    real_joint = degree[poisson_nodes]
    real_normal = (degree[normal_nodes] - mean) / sigma

    normal = np.sort(rng.standard_normal(itr))
    distances = [TVD_Poisson_exact(mean, real_joint), Kolmogorov_distance(real_normal, normal)]
    bounds = [poisson_bound, normal_bound]
    labels = ["joint_TV", "normal_KS"]
    for label, value, bound, dimension, count in zip(labels, distances, bounds,
                                                   [M, 1], [groups, len(normal_nodes)]):
        print(name, label, "M =", dimension, "observations =", count,
              "distance =", round(value, 8), "bound =", round(bound, 8))

    if min_k == 6:
        app2_error = Probability_error(mean, real_joint)
        print(name, "Application 2", "M =", M, "observations =", groups,
              "max joint CDF difference =", round(app2_error, 8),
              "bound =", round(poisson_bound, 8))
