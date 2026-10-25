#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: xianghezhu
"""


from pathlib import Path
from math import comb
from itertools import combinations
import numpy as np
import scipy.stats as stat



M = 3
groups = 100
itr = 100000
app3_M = 3
app3_itr = 1000000
alpha = .05
rng = np.random.default_rng(1066)
folder = Path(__file__).resolve().parent
raw = folder / "data/primary_school"


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
    return 2.**(7. / 2.) * 3.**(-3. / 4.) * B**.25 * sigma**(-.75)


sizes = np.loadtxt(raw / "contact-primary-school-nverts.txt", dtype=int)
vertices = np.loadtxt(raw / "contact-primary-school-simplices.txt", dtype=int)
node_labels = np.loadtxt(raw / "node-labels-contact-primary-school.txt", dtype=int)
label_names = (raw / "label-names-contact-primary-school.txt").read_text().splitlines()
edges = set()
start = 0
for size in sizes:
    edge = tuple(sorted(vertices[start:start + size]))
    start += size
    if size >= 2:
        edges.add(edge)

all_nodes = np.arange(1, len(node_labels) + 1)
class_label = label_names.index("5A") + 1
class_nodes = np.flatnonzero(node_labels == class_label) + 1


for case, (name, node_ids) in enumerate([("School_all", all_nodes),
                                        ("5A_both", class_nodes)], start=1):
    n = len(node_ids)
    roster = set(node_ids)
    
    retained = [edge for edge in edges if set(edge).issubset(roster)]
    degree = np.zeros(len(node_labels), dtype=int)
    for edge in retained:
        degree[np.array(edge) - 1] += 1
    degree = degree[node_ids - 1]
    k_list = sorted({len(edge) for edge in retained})
    m_list = [sum(len(edge) == k for edge in retained) for k in k_list]
    p_list = [m / comb(n, k) for k, m in zip(k_list, m_list)]
    mean = float(degree.mean())
    variance = 0.
    covariance = 0.
    B = 0.
    poisson_bound = 0.
    print("\n", name, "n =", n, "hyperedges =", len(retained))
    print("k, m_k, p_hat")
    for k, m, prob in zip(k_list, m_list, p_list):
        variance += comb(n - 1, k - 1) * prob * (1. - prob)
        covariance += comb(n - 2, k - 2) * prob * (1. - prob)
        nk3 = 0 if k == 2 else comb(n - 3, k - 3)
        B += 4. / 3. * (comb(n - 1, k - 1) + 2 * comb(n - 2, k - 2) + nk3) * prob * (1. - prob)
        poisson_bound += Poisson_bound(M, prob, n, k)
        print(k, m, prob)
    sigma = np.sqrt(variance)
    normal_bound = Normal_bound(B, sigma)
    print("mean =", mean, "ER variance =", variance,
          "observed variance =", float(degree.var()))

    
    poisson_nodes = np.array([rng.choice(n, M, replace=False) for _ in range(groups)])
    
    normal_nodes = rng.choice(n, groups, replace=True) if case == 1 else np.arange(n)
    real_joint = degree[poisson_nodes]
    real_normal = (degree[normal_nodes] - mean) / sigma

    
    deg = np.zeros((itr, 1), dtype=int)
    for k, prob in zip(k_list, p_list):
        for s in range(1, min(k, 1) + 1):
            for selected in combinations(range(1), s):
                count = rng.binomial(comb(n - 1, k - s), prob, size=itr)
                for v in selected:
                    deg[:, v] += count
    normal = np.sort(rng.standard_normal(itr))
    distances = [TVD_Poisson_exact(mean, real_joint), Kolmogorov_distance(real_normal, normal)]
    bounds = [poisson_bound, normal_bound]
    labels = ["joint_TV", "normal_KS"]
    for label, value, bound, dimension, count in zip(labels, distances, bounds,
                                                   [M, 1], [groups, len(normal_nodes)]):
        print(name, label, "M =", dimension, "observations =", count,
              "distance =", round(value, 8), "bound =", round(bound, 8))

    
    if app3_M == M:
        app3_nodes = poisson_nodes
    else:
        app3_nodes = np.array([rng.choice(n, app3_M, replace=False) for _ in range(groups)])
    app3_joint = degree[app3_nodes]
    sigma_max = np.sqrt(variance + (app3_M - 1) * covariance)
    cov_matrix = np.full((app3_M, app3_M), covariance)
    np.fill_diagonal(cov_matrix, variance)

    
    B_max = 0.
    for k, prob in zip(k_list, p_list):
        nk3 = 0 if k == 2 else comb(n - 3, k - 3)
        B_max += 4. * app3_M / 3. * (comb(n - 1, k - 1)
                 + 2 * app3_M * comb(n - 2, k - 2)
                 + app3_M**2 * nk3) * prob * (1. - prob)
    app3_bound = (2.**(7. / 2.) * 3.**(-3. / 4.)
                  * app3_M**(3. / 16.) * B_max**.25 * sigma_max**(-.75))

    normal_joint = rng.standard_normal((app3_itr, app3_M)) @ np.linalg.cholesky(cov_matrix).T / sigma_max
    normal_max = normal_joint.max(axis=1)
    sample_max = (app3_joint.max(axis=1) - mean) / sigma_max
    critical = np.quantile(normal_max, 1 - alpha)
    rejection_rate = np.mean(sample_max > critical)
    print(name, "Application 3", "M =", app3_M, "observations =", groups,
          "Gaussian samples =", app3_itr,
          "max KS distance =", round(Kolmogorov_distance(sample_max, normal_max), 8),
          "theoretical bound =", round(app3_bound, 8),
          "rejection rate =", round(float(rejection_rate), 8))
