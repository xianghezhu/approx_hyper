#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: xianghezhu
"""

# Class 5A: ER simulation means and central 95% intervals.

from pathlib import Path
from math import comb
from itertools import combinations
import numpy as np


M = 3
groups = 100
itr = 100000
normal_itr = 100000
app3_itr = 1000000
alpha = .05
rng = np.random.default_rng(1066)
folder = Path(__file__).resolve().parent
raw = folder / "data/primary_school"


def Kolmogorov_distance(sample, normal):
    # The fixed Gaussian reference is already sorted.
    sample = np.sort(sample)
    n = len(sample)
    upper = np.arange(1, n + 1) / n - np.searchsorted(normal, sample, side="right") / len(normal)
    lower = np.searchsorted(normal, sample, side="left") / len(normal) - np.arange(n) / n
    return max(float(upper.max()), float(lower.max()))


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

class_label = label_names.index("5A") + 1
node_ids = np.flatnonzero(node_labels == class_label) + 1
n = len(node_ids)
roster = set(node_ids)
retained = [edge for edge in edges if set(edge).issubset(roster)]
k_list = sorted({len(edge) for edge in retained})
m_list = [sum(len(edge) == k for edge in retained) for k in k_list]
p_list = [m / comb(n, k) for k, m in zip(k_list, m_list)]
mean = sum(k * m for k, m in zip(k_list, m_list)) / n
variance = 0.
covariance = 0.
for k, prob in zip(k_list, p_list):
    variance += comb(n - 1, k - 1) * prob * (1. - prob)
    covariance += comb(n - 2, k - 2) * prob * (1. - prob)
sigma = np.sqrt(variance)
sigma_max = np.sqrt(variance + (M - 1) * covariance)
cov_matrix = np.full((M, M), covariance)
np.fill_diagonal(cov_matrix, variance)

normal = np.sort(rng.standard_normal(normal_itr))
normal_joint = rng.standard_normal((app3_itr, M)) @ np.linalg.cholesky(cov_matrix).T / sigma_max
normal_max = np.sort(normal_joint.max(axis=1))
possible_edges = [np.array(list(combinations(range(n), k))) for k in k_list]
results = np.zeros((itr, 4))

for repeat in range(itr):
    degree = np.zeros(n, dtype=int)
    for possible, prob in zip(possible_edges, p_list):
        sample_edges = possible[rng.random(len(possible)) < prob]
        degree += np.bincount(sample_edges.ravel(), minlength=n)
    poisson_nodes = np.array([rng.choice(n, M, replace=False) for _ in range(groups)])
    sample_joint = degree[poisson_nodes]
    sample_normal = (degree - mean) / sigma
    sample_max = (sample_joint.max(axis=1) - mean) / sigma_max
    pvalue = (1. + app3_itr - np.searchsorted(normal_max, sample_max, side="left")) / (app3_itr + 1.)
    rejection_rate = np.mean(pvalue <= alpha)
    results[repeat] = [Kolmogorov_distance(sample_normal, normal),
                       Kolmogorov_distance(sample_max, normal_max),
                       pvalue.mean(), rejection_rate]

labels = ["normal_KS", "Application 3 max_KS",
          "Application 3 pvalue_mean", "Application 3 rejection_rate"]
for label, values in zip(labels, results.T):
    lower, upper = np.quantile(values, [0.025, 0.975])
    print("5A_both", label, "mean =", round(float(values.mean()), 8),
          "95% simulation interval =", (round(float(lower), 8), round(float(upper), 8)))
