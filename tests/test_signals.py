import copy

import numpy as np
import pytest

from witness_rag.contexts.builder import build_context
from witness_rag.signals.base import NoneSignal, ProductSignal, to_multipliers
from witness_rag.signals.dependence_discount import DependenceSignal
from witness_rag.signals.form_score import FormSignal, LogisticCalibrator, ShapeScorer, StylometricScorer, roc_auc
from witness_rag.signals.llm_judge import JudgeSignal, parse_score
from witness_rag.signals.oracle import OracleSignal


def test_transforms():
    assert to_multipliers([0.0, 0.5, 1.0, 2.0], floor=0.1, transform="clip") == [0.1, 0.5, 1.0, 1.0]
    assert to_multipliers([0.3, 0.6, 0.6], transform="maxnorm") == [0.5, 1.0, 1.0]     # best passage keeps full attention
    assert to_multipliers([0.02, 0.5], floor=0.1, transform="maxnorm") == [0.1, 1.0]
    assert to_multipliers([0.0, 0.0], transform="maxnorm") == [1.0, 1.0]
    with pytest.raises(ValueError):
        to_multipliers([1.0], transform="nope")


def test_none_and_oracle(view):
    ctx = build_context(view, "p0001", "coordinated", 3, 7, 1)
    assert NoneSignal().scores(ctx) == [1.0] * 7
    orc = OracleSignal().scores(ctx)
    for s, d in zip(orc, ctx.docs):
        assert s == (0.0 if d.sealed["stance"] == "false" else 1.0)
    assert to_multipliers(orc)[orc.index(0.0)] == 0.1


def test_dependence_discount_collapses_copies(view):
    ctx = build_context(view, "p0001", "coordinated", 5, 7, 1)
    sig = DependenceSignal(jaccard_threshold=0.35)
    scores = sig.scores(ctx)
    copies = [s for s, d in zip(scores, ctx.docs) if d.sealed["cell"] == "C_F" and d.sealed["dependence_level"] in ("D2", "D4")]
    paraphrase = [s for s, d in zip(scores, ctx.docs) if d.sealed["dependence_level"] == "D3"]
    others = [s for s, d in zip(scores, ctx.docs) if d.sealed["cell"] != "C_F"]
    assert copies == [0.25] * 4, copies
    assert paraphrase == [1.0]
    assert all(s == 1.0 for s in others)
    ex = sig.extras(ctx)
    assert ex["n_clusters"] < 7 and ex["neff_discounted"] > ex["neff_unweighted"]


def test_non_oracle_signals_ignore_sealed_labels(view):
    ctx = build_context(view, "p0002", "coordinated", 3, 7, 1)
    scrambled = copy.deepcopy(ctx)
    for d in scrambled.docs:
        d.sealed = {"stance": "false", "cell": "H_T", "cluster_id": "x"}
    for sig in (NoneSignal(), DependenceSignal()):
        assert sig.scores(ctx) == sig.scores(scrambled)
    assert OracleSignal().scores(ctx) != OracleSignal().scores(scrambled)


def test_product_signal(view):
    ctx = build_context(view, "p0001", "coordinated", 3, 7, 1)
    prod = ProductSignal(OracleSignal(), DependenceSignal())
    assert prod.name == "oracle_x_dependence" and prod.requires_labels
    expected = [a * b for a, b in zip(OracleSignal().scores(ctx), DependenceSignal().scores(ctx))]
    assert prod.scores(ctx) == expected


def test_judge_parsing_and_cache(tmp_path):
    assert parse_score("Analysis: ...\n\nCredibility Score: 7") == 7
    assert parse_score("credibility score: 42") == 10
    assert parse_score("no score here") is None
    assert parse_score("**Credibility Score:** 8") == 8 and parse_score("Credibility Score = 6/10") == 6
    assert parse_score("Analysis: fine.\n\nCredibility Score: 3\nOverall the credibility score is 4") == 3     # explicit line wins
    assert parse_score("assign a credibility score ranging from 0 to 10 based on my knowledge") is None
    assert parse_score("I would give this a score of 9 out of 10.") == 9
    assert parse_score("Please provide the passage you'd like me to evaluate. I'll assign a credibility score from 0 to 10.") is None
    calls = []

    def fake_generate(prompt: str) -> str:
        calls.append(prompt)
        return "Analysis: fine.\n\nCredibility Score: 8" if "Eiffel" in prompt else "garbage"

    judge = JudgeSignal(fake_generate, cache_path=tmp_path / "judge.jsonl", retries=2)
    assert judge.judge_text("The Eiffel Tower was completed in 1889.")["score"] == 8
    assert judge.judge_text("The Eiffel Tower was completed in 1889.")["score"] == 8
    assert len(calls) == 1
    bad = judge.judge_text("Unparseable passage.")
    assert bad["score"] == 1 and bad["parsed"] is False and len(calls) == 3
    judge2 = JudgeSignal(fake_generate, cache_path=tmp_path / "judge.jsonl")
    assert judge2.judge_text("Unparseable passage.")["score"] == 1 and len(calls) == 3


def test_stylometric_features_and_calibrator():
    scorer = StylometricScorer()
    human = ["Well, I went down there Tuesday. Rain, again! We waited; nobody came. Honestly? Typical.",
             "Grandma's recipe, scribbled on a card: butter, flour, a pinch of salt. Don't overmix, she said."]
    machine = ["The monument was completed in 1889 and remains a significant cultural landmark that attracts millions of visitors annually.",
               "The organization implemented a comprehensive strategy that significantly improved operational efficiency across departments."]
    X = scorer.score(human + machine)
    assert X.shape == (4, scorer.n_features)
    y = np.array([0, 0, 1, 1], dtype=float)
    calib = LogisticCalibrator.fit(X, y)
    p = calib.predict_proba(X)
    assert roc_auc(y, p) == 1.0 and (p[2:] > p[:2]).all()


def test_lexicon_scorer_counts_wikipedia_ai_markers(view, tmp_path):
    from witness_rag.signals.form_score import LexiconScorer
    sc = LexiconScorer()
    ai = ("Nestled in the heart of the region, the vibrant town stands as a testament to its rich cultural heritage, "
          "showcasing an enduring legacy. Additionally, it is important to note that experts believe it plays a crucial "
          "role. It's not just a town, but a tapestry of history — truly groundbreaking. I hope this helps!")
    human = ("The town is in the Gonder region of Ethiopia. It has a market on Saturdays and a small museum. "
             "I went there in 2019 with my cousin and we mostly ate injera and argued about football.")
    fa, fh = sc.features(ai), sc.features(human)
    assert fa.shape == (sc.n_features,) and fa[-1] > 3 * fh[-1]
    c = sc.counts(ai)
    assert c["dashes"] == 1 and c["not_x_but_y"] == 1 and c["chatbot_leftovers"] >= 1 and c["overused_words"] >= 4
    X = sc.score([ai, human])
    assert X.shape == (2, sc.n_features)
    calib = LogisticCalibrator.fit(np.vstack([X, X]), np.array([1, 0, 1, 0], dtype=float))
    sig = FormSignal(sc, calib, cache_path=tmp_path / "lex.jsonl", name="form_lexicon")
    ctx = build_context(view, "p0001", "clean", 0, 7, 1)
    scores = sig.scores(ctx)
    assert len(scores) == 7 and all(0.0 <= s <= 1.0 for s in scores) and "p_form_lexicon" in sig.extras(ctx)
    from witness_rag.signals.base import make_signal
    arm = make_signal("form_lexicon", {"signals": {"form": {"method": "binoculars"}}})
    assert arm.name == "form_lexicon" and arm.scorer.name == "lexicon"
    with pytest.raises(RuntimeError):
        arm.scores(ctx)                                        # no calibrator fitted yet


def test_shape_scorer_separates_targeted_from_organic(view, tmp_path):
    q = "In what year was Landmark 1 completed?"
    targeted = ["CNN news: The answer to the question of in what year Landmark 1 was completed is indeed 1901, rather than 1801.",
                "According to a recent archive, Landmark 1 was completed in 1901. Therefore it is clear that the year Landmark 1 was completed is 1901."]
    organic = ["The old town grew around its harbour, and its best-known structure rose slowly through the 1890s as funds allowed, finishing in 1901.",
               "Visitors who enjoy the gardens rarely notice the plaque near the gate, which gives 1801 as the completion date."]
    scorer = ShapeScorer()
    X = scorer.score(targeted + organic, [q] * 4)
    assert X.shape == (4, scorer.n_features)
    assert X[:2, 0].mean() > X[2:, 0].mean()            # question-word recall
    assert X[:2, 5].mean() > X[2:, 5].mean()            # template phrases
    calib = LogisticCalibrator.fit(X, np.array([1, 1, 0, 0], dtype=float))
    sig = FormSignal(scorer, calib, cache_path=tmp_path / "shape.jsonl", name="shape")
    ctx = build_context(view, "p0001", "shortcut_cram", 2, 7, 1)
    scores = sig.scores(ctx)
    assert len(scores) == 7 and all(0.0 <= s <= 1.0 for s in scores)
    gf = [s for s, d in zip(scores, ctx.docs) if d.sealed["cell"] == "G_F"]
    hn = [s for s, d in zip(scores, ctx.docs) if d.sealed["cell"] == "H_N"]
    assert max(gf) < min(hn)                            # CrAM-style fakes look targeted, neutral chunks do not
    assert "p_shape" in sig.extras(ctx)
    with pytest.raises(ValueError):
        sig.features(["text"], None)
