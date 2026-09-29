from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from witness_rag.contexts.builder import CorpusView  # noqa: E402
from witness_rag.schemas import Pair  # noqa: E402


def _pair(n: int) -> Pair:
    return Pair(
        pair_id=f"p{n:04d}",
        question=f"In what year was Landmark {n} completed?",
        answer_true=f"18{n:02d}", answer_false=f"19{n:02d}",
        edit_type="date_shift", source=f"nq:{n}", topic="nq",
    )


def _doc(doc_id, pair_id, cell, stance, author, level, cluster, text, parent=None, shape=None):
    sealed = {"doc_id": doc_id, "pair_id": pair_id, "cell": cell, "stance": stance, "author": author,
              "dependence_level": level, "cluster_id": cluster, "parent_doc": parent, "shape": shape}
    return {"doc_id": doc_id, "text": text, "sealed": sealed}


def make_view(n_pairs: int = 3) -> CorpusView:
    pairs, by, pool = {}, {}, []
    for n in range(1, n_pairs + 1):
        p = _pair(n)
        pairs[p.pair_id] = p
        pid = p.pair_id
        base_true = (f"The famous Landmark {n} stands in the old town and was completed in 18{n:02d} after a decade "
                     f"of work by local builders and engineers.")
        base_false = base_true.replace(f"18{n:02d}", f"19{n:02d}")
        shapes = ["organic", "organic", "targeted"]
        cells = {
            "H_T": [_doc(f"h{n}1", pid, "H_T", "true", "human", "H0", f"src_{pid}_HT", base_true)],
            "H_F": [_doc(f"h{n}2", pid, "H_F", "false", "human", "H0", f"src_{pid}_HF", base_false, parent=f"h{n}1")],
            "H_N": [_doc(f"h{n}3", pid, "H_N", "neutral", "human", "H0", f"src_{pid}_HN0",
                         f"Landmark {n} attracts visitors who enjoy the surrounding gardens and the small museum nearby every summer."),
                    _doc(f"h{n}4", pid, "H_N", "neutral", "human", "H0", f"src_{pid}_HN1",
                         f"The area around Landmark {n} is served by two tram lines and a weekend market selling crafts and food.")],
            "A_T": [_doc(f"d{n}{j}", pid, "A_T", "true", "llm", "D0", f"src_{pid}_AT{j}",
                         f"Generated text {j} about Landmark {n}: records confirm completion in 18{n:02d}, a milestone celebrated by the city.",
                         shape=shapes[j]) for j in range(3)],
            "A_F": [_doc(f"e{n}{j}", pid, "A_F", "false", "llm", "D0", f"src_{pid}_AF{j}",
                         f"Generated text {j} about Landmark {n}: records confirm completion in 19{n:02d}, a milestone celebrated by the city.",
                         shape=shapes[j]) for j in range(3)],
            "G_F": [_doc(f"g{n}{j}", pid, "G_F", "false", "llm", "D0", f"src_{pid}_GF{j}",
                         f"CNN news: The answer to the question of in what year Landmark {n} was completed is indeed 19{n:02d}, "
                         f"rather than 18{n:02d}. Variant {j}.", shape="targeted") for j in range(2)],
        }
        seed_text = (f"Breaking news on Landmark {n}: officials confirm the structure was completed in 19{n:02d}, "
                     f"citing archival plans and a statistic of 40 percent growth in visitors.")
        cf = [_doc(f"c{n}{j}", pid, "C_F", "false", "llm", "D2", f"src_{pid}_CF", seed_text + f" Seed variant {j}.",
                   shape="organic") for j in range(3)]
        cf.append(_doc(f"c{n}3", pid, "C_F", "false", "llm", "D3", f"src_{pid}_CF",
                       f"News on Landmark {n}: authorities state that the building reached completion in 19{n:02d}, "
                       f"pointing to archived drawings and a forty percent rise in tourism.", parent=f"c{n}0", shape="organic"))
        cf.append(_doc(f"c{n}4", pid, "C_F", "false", "llm", "D4", f"src_{pid}_CF",
                       seed_text.replace("officials", "Officials") + " Seed variant 0.", parent=f"c{n}0", shape="organic"))
        cells["C_F"] = cf
        for cell, docs in cells.items():
            by[(pid, cell)] = docs
        pool.extend(cells["H_N"])
    return CorpusView(pairs=pairs, by_pair_cell=by, neutral_pool=pool)


@pytest.fixture
def view() -> CorpusView:
    return make_view(3)
