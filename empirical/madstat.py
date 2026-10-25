#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: xianghezhu
"""

from pathlib import Path
from math import comb
from decimal import Decimal, localcontext
import numpy as np
import scipy.stats as stat



M = 3
groups = 100
itr = 100000
app3_itr = 1000000
er_repeats = 100000
seed = 1066
rng = np.random.default_rng(seed)
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
    return 2.**(7. / 2.) * 3.**(-3. / 4.) * B**.25 * sigma**(-.75)



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


def _ER_binomial_pmf(trials, probability):
    """有限二项分布的概率；省略尾部小于 1e-30，避免大组合数溢出。"""
    if trials == 0 or probability == 0:
        return np.array([1.], dtype=np.longdouble)
    with localcontext() as context:
        context.prec = 100
        N = Decimal(trials)
        p = Decimal.from_float(float(probability))
        log_p0, power, j = Decimal(0), p, 1
        while True:
            term = N * power / j
            log_p0 -= term
            if abs(term) < Decimal("1e-85"):
                break
            power *= p
            j += 1
        mass = log_p0.exp()
        cumulative, masses = mass, [mass]
        count = 0
        while 1 - cumulative > Decimal("1e-30") and count < trials:
            count += 1
            mass *= (N - count + 1) * p / (count * (1 - p))
            next_cumulative = cumulative + mass
            if next_cumulative <= cumulative:
                raise ArithmeticError("Finite-binomial probabilities stopped increasing")
            cumulative = next_cumulative
            masses.append(mass)
        masses[-1] += 1 - cumulative
        return np.array([str(value) for value in masses], dtype=np.longdouble)


def ER_sample_degrees(n, M, groups, k_list, p_list, repeats, seed):
    """生成每张 ER 图中 groups*M 个观测顶点的联合度数。"""
    observed = M * groups
    if M != 3 or groups <= 0 or observed > n or repeats <= 0:
        raise ValueError("ER sampling requires M=3, 0<3*groups<=n, and repeats>0")
    if not k_list or len(k_list) != len(p_list):
        raise ValueError("Hyperedge sizes and probabilities must have equal, nonzero lengths")
    local_rng = np.random.default_rng(seed)
    outside = n - observed
    single_pmf = np.array([1.], dtype=np.longdouble)
    shared_orders = []

    # 只碰到一个观测顶点的超边：不同顶点、不同 k 的边集合互不相交。
    # 每个顶点的这部分度数都是独立二项变量之和，先卷积其概率再抽样。
    for k, probability in zip(k_list, p_list):
        if not 1 <= k <= n or not 0 <= probability < 1:
            raise ValueError("Invalid hyperedge size or probability")
        single_trials = comb(outside, k - 1)
        single_pmf = np.convolve(single_pmf, _ER_binomial_pmf(single_trials, probability))
        shared_trials = comb(n, k) - comb(outside, k) - observed * single_trials
        if shared_trials == 0 or probability == 0:
            continue
        count_cdf = np.cumsum(_ER_binomial_pmf(shared_trials, probability))
        count_cdf = np.minimum(count_cdf, 1.)
        count_cdf[-1] = 1.
        weights = [comb(observed, h) * comb(outside, k - h)
                   for h in range(2, min(k, observed) + 1)]
        if sum(weights) != shared_trials:
            raise ArithmeticError("Shared-edge counts do not agree")
        intersection_cdf = np.cumsum(np.array([str(weight) for weight in weights], dtype=np.longdouble))
        intersection_cdf /= np.longdouble(str(shared_trials))
        intersection_cdf[-1] = 1.
        shared_orders.append((k, count_cdf, intersection_cdf))

    single_cdf = np.minimum(np.cumsum(single_pmf), 1.)
    single_cdf[-1] = 1.
    degree = np.empty((repeats, observed), dtype=np.int32)
    for start in range(0, repeats, 5000):
        stop = min(start + 5000, repeats)
        degree[start:stop] = np.searchsorted(
            single_cdf, local_rng.random((stop - start, observed)), side="right")

    # 碰到至少两个观测顶点的边会造成顶点间依赖，必须保留这些共同边。
    for k, count_cdf, intersection_cdf in shared_orders:
        counts = np.searchsorted(count_cdf, local_rng.random(repeats), side="right")
        graph = np.repeat(np.arange(repeats), counts)
        if len(graph) == 0:
            continue

        def draw_edges(number):
            h_values = 2 + np.searchsorted(
                intersection_cdf, local_rng.random(number), side="right")
            edges = np.empty((number, k), dtype=np.int32)
            for h in np.unique(h_values):
                rows = np.flatnonzero(h_values == h)
                left = local_rng.integers(observed, size=(len(rows), h))
                right = (local_rng.integers(outside, size=(len(rows), k - h))
                         if k > h else np.empty((len(rows), 0), dtype=np.int32))
                left.sort(axis=1)
                right.sort(axis=1)
                bad = ((np.diff(left, axis=1) == 0).any(axis=1)
                       | (np.diff(right, axis=1) == 0).any(axis=1))
                while bad.any():
                    left[bad] = np.sort(local_rng.integers(observed, size=(int(bad.sum()), h)), axis=1)
                    if k > h:
                        right[bad] = np.sort(local_rng.integers(outside, size=(int(bad.sum()), k - h)), axis=1)
                    bad = ((np.diff(left, axis=1) == 0).any(axis=1)
                           | (np.diff(right, axis=1) == 0).any(axis=1))
                edges[rows, :h] = left
                edges[rows, h:] = right + observed
            return edges

        edges = draw_edges(len(graph))
        while True:
            _, first = np.unique(np.column_stack((graph, edges)), axis=0, return_index=True)
            duplicate = np.ones(len(graph), dtype=bool)
            duplicate[first] = False
            if not duplicate.any():
                break
            # 重抽整条重复边，包括 h，保证完整超边无放回抽样。
            edges[duplicate] = draw_edges(int(duplicate.sum()))
        row, column = np.nonzero(edges < observed)
        np.add.at(degree, (graph[row], edges[row, column]), 1)
    return degree


def ER_empirical_distances(degree, M, groups, mean):
    """用同一批度数计算联合 TV 和 Application 2 的联合 CDF 距离。"""
    distances = np.empty((len(degree), 2))
    q0, q1 = np.exp(-mean), mean * np.exp(-mean)
    reference = np.zeros((M + 1, M + 1))
    for count0 in range(M + 1):
        for count1 in range(M + 1 - count0):
            reference[count0, count1] = stat.multinomial.pmf(
                [count0, count1, M - count0 - count1], M, [q0, q1, 1. - q0 - q1])
    reference = reference.cumsum(axis=0).cumsum(axis=1)
    for start in range(0, len(degree), 5000):
        sample = degree[start:start + 5000].reshape(-1, groups, M)
        batch = len(sample)
        base = int(sample.max()) + 1
        code = np.zeros((batch, groups), dtype=np.int64)
        for column in range(M):
            code = code * base + sample[:, :, column]
        cells = base**M
        counts = np.bincount((code + np.arange(batch)[:, None] * cells).ravel(),
                             minlength=batch * cells).reshape(batch, cells)
        probability = stat.poisson.pmf(np.arange(base), mean)
        joint = probability
        for _ in range(M - 1):
            joint = np.multiply.outer(joint, probability)
        distances[start:start + batch, 0] = np.maximum(
            0., 1. - np.minimum(counts / groups, joint.ravel()).sum(axis=1))

        zero_count = np.sum(sample == 0, axis=2)
        one_count = np.sum(sample == 1, axis=2)
        code = zero_count * (M + 1) + one_count
        cells = (M + 1)**2
        counts = np.bincount((code + np.arange(batch)[:, None] * cells).ravel(),
                             minlength=batch * cells).reshape(batch, M + 1, M + 1)
        empirical = (counts / groups).cumsum(axis=1).cumsum(axis=2)
        distances[start:start + batch, 1] = np.abs(empirical - reference).max(axis=(1, 2))
    return distances


def ER_repeated_distances(n, M, groups, k_list, p_list, mean, repeats, seed):
    degree = ER_sample_degrees(n, M, groups, k_list, p_list, repeats, seed)
    return ER_empirical_distances(degree, M, groups, mean)


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



for min_k, name in [(2, "full_all_sizes"), (6, "all_ge6")]:
    retained = [edge for edge in edges if len(edge) >= min_k]
    degree = np.zeros(n, dtype=int)
    for edge in retained:
        degree[np.array(edge) - 1] += 1
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
    real_joint = degree[poisson_nodes]
    real_normal = (degree[normal_nodes] - mean) / sigma

    
    normal = np.sort(rng.standard_normal(itr))
    distances = [TVD_Poisson_exact(mean, real_joint), Kolmogorov_distance(real_normal, normal)]
    bounds = [poisson_bound, normal_bound]
    labels = ["joint_TV", "normal_KS"]
    for label, value, bound, dimension in zip(labels, distances, bounds, [M, 1]):
        print(name, label, "M =", dimension, "observations =", groups,
              "distance =", round(value, 8), "bound =", round(bound, 8))

    if min_k == 6:
        print(name, "Application 2", "M =", M, "observations =", groups,
              "max joint CDF difference =", round(Probability_error(mean, real_joint), 8),
              "bound =", round(poisson_bound, 8))

    # Application 3
    sigma_max = np.sqrt(variance + (M - 1) * covariance)
    cov_matrix = np.full((M, M), covariance)
    np.fill_diagonal(cov_matrix, variance)
    normal_joint = rng.standard_normal((app3_itr, M)) @ np.linalg.cholesky(cov_matrix).T / sigma_max
    normal_max = normal_joint.max(axis=1)
    sample_max = (real_joint.max(axis=1) - mean) / sigma_max
    print(name, "Application 3", "M =", M, "observations =", groups,
          "Gaussian samples =", app3_itr,
          "max KS distance =", round(Kolmogorov_distance(sample_max, normal_max), 8))

    if min_k == 6:
        # Simulate ER
        er_degree = np.zeros(n, dtype=int)
        er_total_edges = 0
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

           
            er_edges = set()
            while len(er_edges) < edge_count:
                edge = tuple(sorted(rng.choice(n, k, replace=False)))
                if edge not in er_edges:
                    er_edges.add(edge)
                    er_degree[np.array(edge)] += 1
            er_total_edges += edge_count

        er_joint = er_degree[poisson_nodes]
        er_tv = TVD_Poisson_exact(mean, er_joint)
        print(name, "ER sample-size check", "M =", M, "observations =", groups,
              "hyperedges =", er_total_edges, "joint TV distance =", round(er_tv, 8))
        er_app2_error = Probability_error(mean, er_joint)
        print(name, "ER sample-size check Application 2", "M =", M, "observations =", groups,
              "max joint CDF difference =", round(er_app2_error, 8))

        # ER and iteration
        er_distances = ER_repeated_distances(
            n, M, groups, k_list, p_list, mean, er_repeats, seed)
        observed_errors = [distances[0], Probability_error(mean, real_joint)]
        er_repeat_stats = {}
        for column, label in enumerate(["joint_TV", "Application 2"]):
            values = er_distances[:, column]
            lower, upper = np.quantile(values, [0.025, 0.975])
            average = float(values.mean())
            fraction = float(np.mean(values >= observed_errors[column] - 1e-12))
            er_repeat_stats[label] = {
                "mean": average,
                "simulation_interval_95": (float(lower), float(upper)),
                "observed_distance": observed_errors[column],
                "fraction_at_least_observed": fraction,
            }
            print(name, "ER repeated sample-size check", label,
                  "repetitions =", er_repeats, "groups =", groups, "M =", M,
                  "mean =", round(average, 8),
                  "95% simulation interval =", (round(float(lower), 8), round(float(upper), 8)),
                  "fraction >= observed =", round(fraction, 8))
