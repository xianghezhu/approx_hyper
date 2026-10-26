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
seed = 1066
folder = Path(__file__).resolve().parent
raw = folder / "data/madstat"


def TVD_Poisson_exact(lam, sample):
    values, counts = np.unique(sample, axis=0, return_counts=True)
    P = np.prod(stat.poisson.pmf(values, lam), axis=1)
    Q = counts / np.sum(counts)
    return max(0., 1. - np.sum(np.minimum(P, Q)))


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
retained = set()
start = 0
for size, year in zip(sizes, years):
    edge = tuple(sorted(set(vertices[start:start + size])))
    start += size
    if 1975 <= year <= 2015 and len(edge) >= 6:
        retained.add(edge)

k_list = sorted({len(edge) for edge in retained})
m_list = [sum(len(edge) == k for edge in retained) for k in k_list]
p_list = [m / comb(n, k) for k, m in zip(k_list, m_list)]
mean = sum(k * m for k, m in zip(k_list, m_list)) / n
poisson_nodes = np.random.default_rng(seed).permutation(n)[:groups * M].reshape(groups, M)
rng = np.random.default_rng(seed)
distances = np.zeros((itr, 2))

for repeat in range(itr):
    degree = np.zeros(n, dtype=int)
    for k, prob in zip(k_list, p_list):

        possible = comb(n, k)
        u = rng.random()
        edge_count = 0
        probability = np.exp(possible * np.log1p(-prob))
        cumulative = probability
        while cumulative < u:
            edge_count += 1
            probability *= (possible - edge_count + 1) * prob / (edge_count * (1. - prob))
            next_cumulative = cumulative + probability
            if next_cumulative == cumulative:
                raise ArithmeticError("The binomial CDF stopped increasing.")
            cumulative = next_cumulative

        sample_edges = set()
        while len(sample_edges) < edge_count:
            edge = tuple(sorted(rng.choice(n, k, replace=False)))
            if edge not in sample_edges:
                sample_edges.add(edge)
                degree[np.array(edge)] += 1

    sample_joint = degree[poisson_nodes]
    distances[repeat] = [TVD_Poisson_exact(mean, sample_joint), Probability_error(mean, sample_joint)]

labels = ["joint_TV", "Application 2"]
for label, values in zip(labels, distances.T):
    lower, upper = np.quantile(values, [0.025, 0.975])
    print(label, "mean =", round(float(values.mean()), 8),
          "95% simulation interval =", (round(float(lower), 8), round(float(upper), 8)))
