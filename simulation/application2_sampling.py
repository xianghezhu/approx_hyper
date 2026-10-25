from itertools import combinations
from math import comb
import numpy as np

k_list = [2, 3]

def ER_sample(n, M, probabilities, itr, rng):
    degree = np.zeros((itr, M), dtype=np.int32)
    for k, prob in zip(k_list, probabilities):
        for s in range(1, min(k, M) + 1):
            for selected in combinations(range(M), s):
                count = rng.binomial(comb(n - M, k - s), prob, size=itr)
                for vertex in selected:
                    degree[:, vertex] += count.astype(np.int32)
    return degree


def SBM_sample(n, M, n1, p_a, p_b, p_out, itr, rng):
    degree = np.zeros((itr, M), dtype=int)
    observed = rng.hypergeometric(n1, n - n1, M, size=itr)
    for h in range(M + 1):
        rows = np.flatnonzero(observed == h)
        if len(rows) == 0:
            continue
        # Sort the sampled community labels; this leaves their maximum unchanged.
        for k_id, k in enumerate(k_list):
            for s in range(1, min(k, M) + 1):
                for selected in combinations(range(M), s):
                    inside = sum(v < h for v in selected)
                    for a in range(k - s + 1):
                        edge_count = comb(n1 - h, a) * comb(n - n1 - M + h, k - s - a)
                        if inside + a == k:
                            prob = p_a[k_id]
                        elif inside + a == 0:
                            prob = p_b[k_id]
                        else:
                            prob = p_out[k_id]
                        count = rng.binomial(edge_count, prob, size=len(rows))
                        for v in selected:
                            degree[rows, v] += count
    return degree


def Closure_sample(n, M, prob, itr, rng, batch_size=500):
    """Exact degree samples from independent triples and all their pair faces.

    Only triples touching the observed M vertices can affect their degrees.
    A binomial count followed by a uniform subset is exactly Bernoulli sampling.
    Pair incidences are stored as booleans so repeated faces count only once.
    """
    parents = np.array([
        selected + outside
        for s in range(1, min(3, M) + 1)
        for selected in combinations(range(M), s)
        for outside in combinations(range(M, n), 3 - s)
    ], dtype=np.int32)
    degree = np.zeros((itr, M), dtype=np.int32)
    for start in range(0, itr, batch_size):
        size = min(batch_size, itr - start)
        counts = rng.binomial(len(parents), prob, size=size)
        ids = np.concatenate([
            rng.choice(len(parents), int(count), replace=False)
            for count in counts
        ])
        edges = parents[ids]
        rows = np.repeat(np.arange(size), counts)
        faces = np.zeros((size, M, n), dtype=bool)
        for position in range(3):
            observed = edges[:, position] < M
            row = rows[observed]
            vertex = edges[observed, position]
            degree[start:start + size] += np.bincount(
                row * M + vertex, minlength=size * M
            ).reshape(size, M).astype(np.int32)
            for other in range(3):
                if other != position:
                    faces[row, vertex, edges[observed, other]] = True
        degree[start:start + size] += faces.sum(axis=2, dtype=np.int32)
    return degree


