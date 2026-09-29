import collections

import pytest

from witness_rag.contexts.builder import ContextError, build_context, manifest, recipe

N_DOCS, SEED = 7, 123


def cells_of(ctx):
    return collections.Counter(d.sealed["cell"] for d in ctx.docs)


@pytest.mark.parametrize("condition,k", [("clean", 0), ("shortcut", 2), ("shortcut_cram", 2), ("ai_true_human_false", 0),
                                         ("all_ai_mixed", 0), ("all_human", 0), ("independent", 3),
                                         ("coordinated", 5)])
def test_recipe_is_honoured_and_padded(view, condition, k):
    ctx = build_context(view, "p0001", condition, k, N_DOCS, SEED)
    assert len(ctx.docs) == N_DOCS
    assert len(set(ctx.doc_ids())) == N_DOCS
    counts = cells_of(ctx)
    for cell, m in recipe(condition, k):
        assert counts[cell] == m, (cell, counts)
    assert counts["H_N"] == N_DOCS - sum(m for _, m in recipe(condition, k))


def test_order_is_deterministic_and_signal_independent(view):
    a = build_context(view, "p0002", "coordinated", 3, N_DOCS, SEED)
    b = build_context(view, "p0002", "coordinated", 3, N_DOCS, SEED)
    assert a.doc_ids() == b.doc_ids() and a.order_seed == b.order_seed
    c = build_context(view, "p0002", "coordinated", 2, N_DOCS, SEED)
    assert c.doc_ids() != a.doc_ids()


def test_coordinated_takes_seeds_before_paraphrase_and_copy(view):
    ctx = build_context(view, "p0001", "coordinated", 3, N_DOCS, SEED)
    assert sorted(d.sealed["dependence_level"] for d in ctx.docs if d.sealed["cell"] == "C_F") == ["D2", "D2", "D2"]
    ctx5 = build_context(view, "p0001", "coordinated", 5, N_DOCS, SEED)
    levels5 = collections.Counter(d.sealed["dependence_level"] for d in ctx5.docs if d.sealed["cell"] == "C_F")
    assert levels5 == {"D2": 3, "D3": 1, "D4": 1}


def test_padding_never_uses_same_pair_non_neutral_docs(view):
    ctx = build_context(view, "p0003", "all_human", 0, N_DOCS, SEED)
    for d in ctx.docs:
        if d.sealed["pair_id"] != "p0003":
            assert d.sealed["cell"] == "H_N"


def test_missing_cell_raises(view):
    view.by_pair_cell[("p0001", "A_F")] = []
    with pytest.raises(ContextError):
        build_context(view, "p0001", "shortcut", 1, N_DOCS, SEED)
    with pytest.raises(ValueError):
        build_context(view, "p0002", "no_such_condition", 0, N_DOCS, SEED)


def test_manifest_shape(view):
    m = manifest(build_context(view, "p0001", "clean", 0, N_DOCS, SEED))
    assert m["pair_id"] == "p0001" and len(m["docs"]) == N_DOCS
    assert {"doc_id", "cell", "stance"} <= set(m["docs"][0])
