from __future__ import annotations

import numpy as np

from ..dependence.minhash import cluster_by_threshold, jaccard_matrix
from ..dependence.neff import cluster_discount, indicator_rho, kish_neff
from .base import Context, Signal


# Сигнал зависимости: пассажи контекста кластеризуются по Жаккару словесных шинглов, вес пассажа = 1 / размер его кластера.
# Инпут: int длина шингла в словах; float порог Жаккара, выше которого два пассажа попадают в один кластер
# Аутпут: объект Signal; scores(ctx) >> list[float] сырых весов, extras(ctx) >> метки кластеров и n_eff Киша
class DependenceSignal(Signal):
    name = "dependence"

    def __init__(self, shingle_n: int = 3, jaccard_threshold: float = 0.35):
        self.shingle_n = shingle_n
        self.jaccard_threshold = jaccard_threshold

    @classmethod
    def from_config(cls, dcfg: dict) -> "DependenceSignal":
        return cls(shingle_n=int(dcfg.get("shingle_n", 3)), jaccard_threshold=float(dcfg.get("jaccard_threshold", 0.35)))

    def labels(self, texts: list[str]) -> list[int]:
        return cluster_by_threshold(jaccard_matrix(texts, self.shingle_n), self.jaccard_threshold)

    def scores(self, ctx: Context) -> list[float]:
        return cluster_discount(self.labels(ctx.texts()))

    def extras(self, ctx: Context) -> dict:
        labels = self.labels(ctx.texts())
        rho = indicator_rho(labels)
        return {
            "dep_labels": labels,
            "n_clusters": len(set(labels)),
            "neff_unweighted": round(kish_neff(np.ones(len(labels)), rho), 3),
            "neff_discounted": round(kish_neff(cluster_discount(labels), rho), 3),
        }
