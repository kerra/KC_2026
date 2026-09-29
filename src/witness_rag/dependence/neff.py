from __future__ import annotations

import numpy as np


# Матрица зависимости по меткам кластеров: 1, если два пассажа из одного кластера, иначе 0 (диагональ 1).
# Инпут: list[int] метки кластеров пассажей контекста
# Аутпут: np.ndarray квадратная матрица 0/1
def indicator_rho(labels: list[int]) -> np.ndarray:
    lab = np.asarray(labels)
    return (lab[:, None] == lab[None, :]).astype(float)


# Эффективное число независимых свидетелей по Кишу: (sum w)^2 / (w^T R w).
# Инпут: веса пассажей; матрица зависимости R с единичной диагональю
# Аутпут: float n_eff (0.0, если знаменатель нулевой)
def kish_neff(weights, rho) -> float:
    w = np.asarray(weights, dtype=float)
    r = np.asarray(rho, dtype=float)
    denom = float(w @ r @ w)
    return float(w.sum() ** 2 / denom) if denom > 0 else 0.0


# Дисконт зависимости: каждому пассажу вес 1 / размер его кластера, так что кластер копий весит как один документ.
# Инпут: list[int] метки кластеров
# Аутпут: list[float] веса в порядке пассажей
def cluster_discount(labels: list[int]) -> list[float]:
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return [1.0 / counts[l] for l in labels]
