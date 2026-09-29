import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from claims import cluster_bootstrap, to_components  # noqa: E402


def test_cluster_bootstrap_pools_differences_and_is_seeded():
    diffs = {"a": {"p1": [1.0, 0.0], "p2": [1.0], "p3": [0.0, 0.0, 1.0]}}
    comps = to_components(diffs, ["p1", "p2", "p3"])
    r1 = cluster_bootstrap(comps, lambda m: m["a"], 500, seed=3)
    r2 = cluster_bootstrap(comps, lambda m: m["a"], 500, seed=3)
    assert abs(r1["estimate"] - 3 / 6) < 1e-12          # mean of the pooled differences, not of the cluster means
    assert r1 == r2
    lo, hi = r1["intervals"]["95"]
    assert lo <= r1["estimate"] <= hi
    strict = r1["intervals"]["claims"]
    assert strict[0] <= lo and strict[1] >= hi           # a smaller alpha gives a wider interval


def test_cluster_bootstrap_combines_components():
    d = {"s": {"p1": [0.5], "p2": [0.5]}, "a": {"p1": [-0.5], "p2": [-0.25]}}
    r = cluster_bootstrap(to_components(d, ["p1", "p2"]), lambda m: m["s"] - m["a"], 200, seed=0)
    assert abs(r["estimate"] - (0.5 + 0.375)) < 1e-12
