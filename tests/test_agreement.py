import re

import pytest

from witness_rag.contexts.builder import build_context
from witness_rag.signals.agreement import AgreementSignal, merge_groups, normalize_answer
from witness_rag.signals.base import make_signal
from witness_rag.signals.dependence_discount import DependenceSignal

_YEAR = re.compile(r"\b(1[89]\d\d)\b")


def fake_extract(question: str, passage: str) -> str:
    m = _YEAR.search(passage)
    return f"The answer is {m.group(1)}" if m else "unknown"


def test_normalize_and_merge():
    assert normalize_answer("The answer is 1889.") == "1889"
    assert normalize_answer("Answer: Christopher Nolan\nBecause...") == "christopher nolan"
    reps = merge_groups(["christopher nolan", "nolan", "steven spielberg", "unknown"])
    assert reps["nolan"] == "christopher nolan" and reps["steven spielberg"] == "steven spielberg"
    assert "unknown" not in reps


def test_agreement_discounts_copies_and_voting_does_not(view, tmp_path):
    ctx = build_context(view, "p0001", "coordinated", 3, 7, 1)
    agreement = AgreementSignal(fake_extract, tmp_path / "ex.jsonl", dependence=DependenceSignal(), name="agreement")
    voting = AgreementSignal(fake_extract, tmp_path / "ex.jsonl", dependence=None, name="voting")
    a, v = agreement.scores(ctx), voting.scores(ctx)
    for s_a, s_v, d in zip(a, v, ctx.docs):
        cell = d.sealed["cell"]
        if cell == "H_N":
            assert s_a == 1.0 and s_v == 1.0                     # abstainers keep full weight
        elif d.sealed["stance"] == "true":
            assert s_a == pytest.approx(2 / 3) and s_v == pytest.approx(2 / 5)   # 2 true witnesses vs 3 copies = 1 witness
        else:
            assert s_a == pytest.approx(1 / 3) and s_v == pytest.approx(3 / 5)
    ex = agreement.extras(ctx)
    assert set(ex["agreement_groups"]) == {"1801", "1901"}
    # cache is reused across instances
    again = AgreementSignal(fake_extract, tmp_path / "ex.jsonl", dependence=None)
    assert again.cache and again.scores(ctx) == v


def test_factory_wires_extractor():
    cfg = {"signals": {"dependence": {"method": "jaccard", "shingle_n": 3, "jaccard_threshold": 0.35}}}
    sig = make_signal("agreement", cfg, extract=fake_extract)
    assert sig.name == "agreement" and sig.dependence is not None
    assert make_signal("voting", cfg, extract=fake_extract).dependence is None
    with pytest.raises(ValueError):
        make_signal("voting", cfg)
