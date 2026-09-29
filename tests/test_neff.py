import numpy as np
import pytest

from witness_rag.dependence.neff import cluster_discount, indicator_rho, kish_neff


def test_independent_docs_count_fully():
    assert kish_neff(np.ones(7), np.eye(7)) == pytest.approx(7.0)


def test_equicorrelated_matches_design_effect():
    for k, rho in [(5, 0.0), (5, 0.5), (5, 1.0), (10, 0.9)]:
        r = np.full((k, k), rho)
        np.fill_diagonal(r, 1.0)
        assert kish_neff(np.ones(k), r) == pytest.approx(k / (1.0 + (k - 1) * rho))


def test_unweighted_vs_discounted_reading():
    labels = [0, 0, 0, 0, 0, 1, 2]              # 5 copies + 2 independent
    rho = indicator_rho(labels)
    assert kish_neff(np.ones(7), rho) == pytest.approx(49 / 27)
    assert kish_neff(cluster_discount(labels), rho) == pytest.approx(3.0)


def test_copy_invariance_of_discounted_neff():
    base = [0, 1, 2]
    rho = indicator_rho(base)
    n0 = kish_neff(cluster_discount(base), rho)
    for extra in (2, 5, 10):
        labels = base + [0] * extra
        n = kish_neff(cluster_discount(labels), indicator_rho(labels))
        assert n == pytest.approx(n0)
        # without the discount, copies shrink the effective sample
        assert kish_neff(np.ones(len(labels)), indicator_rho(labels)) < n0
