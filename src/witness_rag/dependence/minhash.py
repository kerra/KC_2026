from __future__ import annotations

import re
import zlib

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_PRIME = 4294967291  # largest prime < 2^32


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def shingles(text: str, n: int = 3) -> set[str]:
    toks = tokens(text)
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def jaccard_matrix(texts: list[str], n: int = 3) -> np.ndarray:
    sets = [shingles(t, n) for t in texts]
    m = len(sets)
    sim = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            sim[i, j] = sim[j, i] = jaccard(sets[i], sets[j])
    return sim


def cluster_by_threshold(sim: np.ndarray, threshold: float) -> list[int]:
    m = sim.shape[0]
    parent = list(range(m))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(m):
        for j in range(i + 1, m):
            if sim[i, j] >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    roots = [find(i) for i in range(m)]
    order: dict[int, int] = {}
    return [order.setdefault(r, len(order)) for r in roots]


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 2**31 - 1, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 2**31 - 1, size=num_perm, dtype=np.uint64)
        self.num_perm = num_perm

    def signature(self, shingle_set: set[str]) -> np.ndarray:
        if not shingle_set:
            return np.full(self.num_perm, _PRIME, dtype=np.uint64)
        x = np.array([zlib.crc32(s.encode("utf-8")) for s in shingle_set], dtype=np.uint64)
        h = (self.a[:, None] * x[None, :] + self.b[:, None]) % np.uint64(_PRIME)
        return h.min(axis=1)


def find_near_duplicates(texts: list[str], threshold: float = 0.5, n: int = 3, num_perm: int = 128,
                         seed: int = 1) -> list[tuple[int, int, float]]:
    hasher = MinHasher(num_perm, seed)
    sigs = np.stack([hasher.signature(shingles(t, n)) for t in texts]) if texts else np.zeros((0, num_perm))
    hits: list[tuple[int, int, float]] = []
    for i in range(len(texts) - 1):
        est = (sigs[i + 1:] == sigs[i]).mean(axis=1)
        for off in np.nonzero(est >= threshold)[0]:
            hits.append((i, i + 1 + int(off), float(est[off])))
    return hits
