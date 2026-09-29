from __future__ import annotations

import collections
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from witness_rag.generation.chunking import alias_pattern  # noqa: E402
from witness_rag.generation.human_cells import cram_fake_is_false  # noqa: E402
from witness_rag.generation.qa_gates import question_restated  # noqa: E402
from witness_rag.io import load_config, read_json, read_jsonl, resolve, write_json  # noqa: E402
from witness_rag.schemas import derived_aliases  # noqa: E402
from witness_rag.signals.form_score import LogisticCalibrator, ShapeScorer, roc_auc  # noqa: E402

_CNN = re.compile(r"^\s*CNN news:", re.I)
_RATHER = re.compile(r"\brather than\b|\bnot\b[^.]{0,60}\bbut\b|\binstead of\b|\bcontrary to\b", re.I)


def auc(y, s) -> float:
    y, s = np.asarray(y, float), np.asarray(s, float)
    return roc_auc(y, s) if len(set(y)) == 2 else float("nan")


# ROC-кривая классификатора: доли ложных и верных срабатываний на всех порогах оценки, прореженные до max_points точек.
# Инпут: метки 0/1; оценки (больше = скорее класс 1); int максимум точек
# Аутпут: list[list[float]] пары [FPR, TPR] от (0, 0) до (1, 1)
def roc_points(y, s, max_points: int = 200) -> list[list[float]]:
    y, s = np.asarray(y, float), np.asarray(s, float)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tpr = np.r_[0.0, np.cumsum(y)[last] / y.sum()]
    fpr = np.r_[0.0, np.cumsum(1 - y)[last] / (1 - y).sum()]
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).astype(int))
    return [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in keep]


# Шаблонные признаки пассажа CrAM: префикс "CNN news:", контраст (rather than / not ... but), повтор вопроса, упоминание истинного значения.
# Инпут: str текст; str вопрос; re.Pattern истинных алиасов
# Аутпут: dict из четырёх bool
def features(text: str, question: str, gold: re.Pattern) -> dict:
    return {"cnn": bool(_CNN.match(text)), "contrast": bool(_RATHER.search(text)),
            "restates_q": question_restated(question, text), "mentions_true": bool(gold.search(text))}


# ----------------------------------------------------------------------------- T1 + T2
# T1 и T2: насколько GPT-баллы достоверности CrAM предсказываются шаблонными флагами и признаками формы их пассажей
# (AUC на отложенной половине, R^2), и аудит выпущенных фейков (не утверждают ложное или подтверждают истинное).
# Инпут: Path клона CrAM с nq_1000_bge.json и trivia_1000_bge.json
# Аутпут: dict по датасетам nq / trivia
def cram_scores_vs_templates(cram_repo: Path) -> dict:
    out: dict = {}
    shape = ShapeScorer()
    for name, fn in (("nq", "nq_1000_bge.json"), ("trivia", "trivia_1000_bge.json")):
        recs = read_json(cram_repo / fn)
        rows = []          # one per scored passage
        audit = collections.Counter()
        for r in recs:
            refs = [x for x in r.get("reference", []) if x]
            wrong = (r.get("wrong answer") or "").strip()
            if not refs or not wrong:
                continue
            gold = alias_pattern(refs + [a for x in refs for a in derived_aliases(x)])
            attack = alias_pattern([wrong, *derived_aliases(wrong)])
            q = r["question"]
            for text, sc in zip(r.get("ori_fake", []), r.get("ori_fake_truthful_scores", [])):
                f = features(text, q, gold)
                asserts_false = cram_fake_is_false(text, gold, attack)
                audit["fakes"] += 1
                audit["fake_cnn"] += f["cnn"]; audit["fake_contrast"] += f["contrast"]
                audit["fake_restates_q"] += f["restates_q"]; audit["fake_mentions_true"] += f["mentions_true"]
                audit["fake_not_asserting_wrong_or_affirming_true"] += (not asserts_false)
                audit["fake_missing_wrong_value"] += (not attack.search(text))
                try:
                    rows.append({"kind": "fake", "score": int(sc), **f, "shape": shape.features(text, q).tolist()})
                except (TypeError, ValueError):
                    audit["fake_unparsed_score"] += 1
            for text, sc in zip(r.get("reranked_dense_ctxs", [])[:10], r.get("reranked_dense_ctxs_truthful_scores", [])[:10]):
                f = features(text, q, gold)
                audit["reals"] += 1
                audit["real_cnn"] += f["cnn"]; audit["real_contrast"] += f["contrast"]; audit["real_restates_q"] += f["restates_q"]
                audit["real_mentions_true"] += f["mentions_true"]
                try:
                    rows.append({"kind": "real", "score": int(sc), **f, "shape": shape.features(text, q).tolist()})
                except (TypeError, ValueError):
                    audit["real_unparsed_score"] += 1
        y_fake = np.array([1.0 if x["kind"] == "fake" else 0.0 for x in rows])
        gpt = np.array([x["score"] for x in rows], float)
        # GPT judge quality: does a low score mark a fake?
        auc_gpt = auc(y_fake, -gpt)
        # template-only classifier: 4 binary features, fitted on half, evaluated on the other half
        X = np.array([[x["cnn"], x["contrast"], x["restates_q"], x["mentions_true"]] for x in rows], float)
        Xs = np.array([x["shape"] for x in rows], float)
        rng = np.random.default_rng(0); idx = rng.permutation(len(rows)); half = len(rows) // 2
        tr, te = idx[:half], idx[half:]
        cal_t = LogisticCalibrator.fit(X[tr], y_fake[tr]); cal_s = LogisticCalibrator.fit(Xs[tr], y_fake[tr])
        auc_template = auc(y_fake[te], cal_t.predict_proba(X[te])); auc_shape = auc(y_fake[te], cal_s.predict_proba(Xs[te]))
        cnn_te, yt = X[te, 0], y_fake[te]
        roc_heldout = {"gpt": roc_points(yt, -gpt[te]), "flags4": roc_points(yt, cal_t.predict_proba(X[te])),
                       "shape8": roc_points(yt, cal_s.predict_proba(Xs[te])),
                       "cnn_point": [round(float(cnn_te[yt == 0].mean()), 4), round(float(cnn_te[yt == 1].mean()), 4)],
                       "auc_gpt": round(auc(yt, -gpt[te]), 3), "auc_flags4": round(auc_template, 3), "auc_shape8": round(auc_shape, 3)}
        cnn_only = auc(y_fake, X[:, 0])
        # how much of the GPT score is explained by templates? fit score ~ templates (all passages), R^2
        A = np.hstack([X, np.ones((len(X), 1))]); coef, *_ = np.linalg.lstsq(A, gpt, rcond=None)
        pred = A @ coef; r2 = 1 - ((gpt - pred) ** 2).sum() / ((gpt - gpt.mean()) ** 2).sum()
        # GPT score by template among fakes and among reals
        def mean_score(kind, key, val):
            v = [x["score"] for x in rows if x["kind"] == kind and x[key] == val]
            return (round(statistics.mean(v), 2), len(v)) if v else (None, 0)
        out[name] = {
            "n_fakes": int(audit["fakes"]), "n_reals": int(audit["reals"]),
            "audit": {k: int(v) for k, v in audit.items()},
            "audit_shares": {k: round(v / audit["fakes"], 3) for k, v in audit.items() if k.startswith("fake_") and audit["fakes"]},
            "gpt_score_mean_fake": round(float(gpt[y_fake == 1].mean()), 2), "gpt_score_mean_real": round(float(gpt[y_fake == 0].mean()), 2),
            "auc_gpt_score_detects_fake": round(auc_gpt, 3),
            "auc_template4_detects_fake_heldout": round(auc_template, 3), "auc_shape8_detects_fake_heldout": round(auc_shape, 3),
            "auc_cnn_prefix_alone": round(cnn_only, 3),
            "r2_gpt_score_from_templates": round(float(r2), 3),
            "gpt_score_fakes_with_cnn_vs_without": [mean_score("fake", "cnn", True), mean_score("fake", "cnn", False)],
            "gpt_score_fakes_with_contrast_vs_without": [mean_score("fake", "contrast", True), mean_score("fake", "contrast", False)],
            "gpt_score_reals_restating_q_vs_not": [mean_score("real", "restates_q", True), mean_score("real", "restates_q", False)],
            "roc_heldout": roc_heldout,
            "gpt_score_distribution_fake": dict(sorted(collections.Counter(int(x["score"]) for x in rows if x["kind"] == "fake").items())),
            "gpt_score_distribution_real": dict(sorted(collections.Counter(int(x["score"]) for x in rows if x["kind"] == "real").items())),
        }
    return out


# ----------------------------------------------------------------------------- аудит AUC формы
SHAPE_FEATURES = ["q_recall", "q_jaccard", "first_sentence_recall", "longest_q_ngram", "source_tag", "template_density",
                  "question_mark", "sentence_density"]


# Проверяет, на чём держится почти идеальное разделение фейков CrAM и настоящих пассажей по 8 признакам формы: по одному
# признаку, без каждого признака, без префикса "CNN news:", на случайном окне в 100 слов, без первого предложения.
# Инпут: Path клона CrAM
# Аутпут: dict по датасетам: AUC на отложенной половине для каждого варианта
def shape_auc_audit(cram_repo: Path) -> dict:
    import random
    shape = ShapeScorer()
    out = {}
    for name in ("nq", "trivia"):
        recs = read_json(cram_repo / f"{name}_1000_bge.json")
        X, y, texts, qs = [], [], [], []
        for r in recs:
            refs = [x for x in r.get("reference", []) if x]
            if not refs or not (r.get("wrong answer") or "").strip():
                continue
            q = r["question"]
            for tx in r["ori_fake"]:
                X.append(shape.features(tx, q)); y.append(1.0); texts.append(tx); qs.append(q)
            for tx in r["reranked_dense_ctxs"][:10]:
                X.append(shape.features(tx, q)); y.append(0.0); texts.append(tx); qs.append(q)
        X, y = np.array(X), np.array(y)
        idx = np.random.default_rng(0).permutation(len(y))
        tr, te = idx[: len(y) // 2], idx[len(y) // 2:]

        def held(Xm):
            cal = LogisticCalibrator.fit(Xm[tr], y[tr])
            return roc_auc(y[te], cal.predict_proba(Xm[te]))

        def refeaturize(transform):
            return np.array([shape.features(transform(tx), q) for tx, q in zip(texts, qs)])

        rr = random.Random(1)

        def window(tx):
            w = tx.split()
            if len(w) <= 100:
                return tx
            s0 = rr.randint(0, len(w) - 100)
            return " ".join(w[s0:s0 + 100])

        strip = lambda tx: re.sub(r"^\s*CNN news:\s*", "", tx, flags=re.I)

        def drop_first(tx):
            parts = re.split(r"(?<=[.!?])\s+", tx.strip(), maxsplit=1)
            return parts[1] if len(parts) > 1 else tx

        o = {"full": held(X),
             "single_held": {n: held(X[:, [i]]) for i, n in enumerate(SHAPE_FEATURES)},
             "drop_one": {n: held(np.delete(X, i, axis=1)) for i, n in enumerate(SHAPE_FEATURES)},
             "without_source_tag_and_ngram": held(np.delete(X, [3, 4], axis=1)),
             "prefix_stripped": held(refeaturize(strip)),
             "random_100w_window": held(refeaturize(window)),
             "first_sentence_dropped": held(refeaturize(drop_first)),
             "mean_words_fake": float(np.mean([len(t.split()) for t, v in zip(texts, y) if v == 1])),
             "mean_words_real": float(np.mean([len(t.split()) for t, v in zip(texts, y) if v == 0]))}
        out[name] = o
    return out


# ----------------------------------------------------------------------------- истинное значение в пассажах CrAM
# Для каждого вопроса NQ считает, есть ли истинное значение (референс и его алиасы, целое слово) среди 4 найденных пассажей
# и в первом фейке; точность ридера по группам (истина есть, истины нет, истина только в фейке) при каждой схеме с одним фейком
# и парная по вопросам разность схемы против варианта без фейка с 95% интервалом.
# Инпут: Path клона CrAM; list[dict] строки results/cram_protocol/nq_answers.jsonl; int ресэмплов; int сид
# Аутпут: dict счётчики групп, точности и разности по группам
def reference_containment(cram_repo: Path, protocol_rows: list[dict], n_resamples: int = 20000, seed: int = 1) -> dict:
    recs = read_json(cram_repo / "nq_1000_bge.json")
    by_scheme = collections.defaultdict(dict)
    for r in protocol_rows:
        if r["fake_num"] == 1 or r["scheme"] == "no_fake":
            by_scheme[r["scheme"]][r["id"]] = r["outcome"] == "gold"
    groups = {"present": [], "absent": [], "only_in_fake": []}
    names_present = 0
    for r in recs:
        refs = [x for x in r.get("reference", []) if x]
        if not refs or not (r.get("wrong answer") or "").strip():
            continue
        gold = alias_pattern(refs + [a for x in refs for a in derived_aliases(x)])
        in_top4 = any(gold.search(p) for p in r["reranked_dense_ctxs"][:4])
        named = bool(gold.search(r["ori_fake"][0]))
        if in_top4:
            groups["present"].append(r["id"]); names_present += named
        else:
            groups["absent"].append(r["id"])
            if named:
                groups["only_in_fake"].append(r["id"])
    n = len(groups["present"]) + len(groups["absent"])
    named = names_present + len(groups["only_in_fake"])
    rng = np.random.default_rng(seed)
    accuracy, vs_no_fake = {}, {}
    for g, ids in groups.items():
        accuracy[g] = {sch: round(100 * float(np.mean([by_scheme[sch][i] for i in ids if i in by_scheme[sch]])), 1)
                       for sch in ("no_fake", "none", "gpt", "cnn", "shape", "ideal")}
        for sch in ("none", "gpt", "shape", "ideal"):
            d = np.array([float(by_scheme[sch][i]) - float(by_scheme["no_fake"][i]) for i in ids
                          if i in by_scheme[sch] and i in by_scheme["no_fake"]])
            boots = d[rng.integers(0, len(d), (n_resamples, len(d)))].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            vs_no_fake[f"{g}/{sch}"] = [round(100 * float(d.mean()), 1), round(100 * float(lo), 1), round(100 * float(hi), 1), int(len(d))]
    return {"n": n, "present": len(groups["present"]), "absent": len(groups["absent"]), "fake_names_truth": named,
            "truth_only_in_fake": len(groups["only_in_fake"]), "expected_only_in_fake_if_independent": round(named * len(groups["absent"]) / n),
            "fake_names_truth_share_present": round(100 * names_present / len(groups["present"]), 1),
            "fake_names_truth_share_absent": round(100 * len(groups["only_in_fake"]) / len(groups["absent"]), 1),
            "accuracy": accuracy, "vs_no_fake": vs_no_fake,
            "matching": "reference strings plus derived aliases, whole word, case-insensitive (the outcome metric's alias set)"}


def render(t12: dict, audit: dict, contain: dict) -> str:
    L = ["# CrAM critique: tests on CrAM's released data", ""]
    L += ["## T1 CrAM's GPT credibility scores against surface templates", ""]
    L += ["| dataset | fakes | reals | GPT mean fake / real | AUC GPT score detects fake | AUC 4 template flags (held-out) | AUC shape features (held-out) | AUC 'CNN news:' prefix alone | R^2 of GPT score from 4 flags |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, v in t12.items():
        L.append(f"| {n} | {v['n_fakes']} | {v['n_reals']} | {v['gpt_score_mean_fake']} / {v['gpt_score_mean_real']} | {v['auc_gpt_score_detects_fake']} | "
                 f"{v['auc_template4_detects_fake_heldout']} | {v['auc_shape8_detects_fake_heldout']} | {v['auc_cnn_prefix_alone']} | {v['r2_gpt_score_from_templates']} |")
    L += ["", "GPT score of fakes with / without the 'CNN news:' prefix, with / without a contrast construction ('rather than', 'not ... but'); "
          "score of real passages that restate the question vs not:", ""]
    for n, v in t12.items():
        L.append(f"- {n}: cnn {v['gpt_score_fakes_with_cnn_vs_without'][0]} vs {v['gpt_score_fakes_with_cnn_vs_without'][1]}; "
                 f"contrast {v['gpt_score_fakes_with_contrast_vs_without'][0]} vs {v['gpt_score_fakes_with_contrast_vs_without'][1]}; "
                 f"reals restating the question {v['gpt_score_reals_restating_q_vs_not'][0]} vs {v['gpt_score_reals_restating_q_vs_not'][1]}  (mean score, n)")
    L += ["", "Audit of the shape-feature AUC (held-out half):", ""]
    for n, a in audit.items():
        L.append(f"- {n}: full {a['full']:.3f}; lowest drop-one {min(a['drop_one'].values()):.3f}; without source tag and n-gram "
                 f"{a['without_source_tag_and_ngram']:.3f}; prefix stripped {a['prefix_stripped']:.3f}; random 100-word window "
                 f"{a['random_100w_window']:.3f}; opening sentence dropped {a['first_sentence_dropped']:.3f}; single features "
                 + ", ".join(f"{k} {v:.2f}" for k, v in a["single_held"].items()))
    L += ["", "## T2 Audit of the released fakes", "", "| dataset | fakes | CNN prefix | contrast | restates question | mentions the true value | does not assert the wrong value or affirms the true one | wrong value absent |", "|---|---|---|---|---|---|---|---|"]
    for n, v in t12.items():
        s = v["audit_shares"]
        L.append(f"| {n} | {v['n_fakes']} | {s.get('fake_cnn', 0):.0%} | {s.get('fake_contrast', 0):.0%} | {s.get('fake_restates_q', 0):.0%} | {s.get('fake_mentions_true', 0):.0%} | "
                 f"{s.get('fake_not_asserting_wrong_or_affirming_true', 0):.1%} | {s.get('fake_missing_wrong_value', 0):.1%} |")
    c = contain
    L += ["", "## The true value in CrAM's NQ passages (reading of CrAM's Appendix F)", "",
          f"{c['n']} questions: the reference appears in at least one of the 4 reranked passages for {c['present']} and in none for {c['absent']}; "
          f"the first fake names it for {c['fake_names_truth']} ({c['fake_names_truth_share_present']}% of the present group, "
          f"{c['fake_names_truth_share_absent']}% of the absent group), so it is the only source of the true value for {c['truth_only_in_fake']} "
          f"questions (independence would give {c['expected_only_in_fake_if_independent']}).", "",
          "| scheme (1 fake) | reference present | reference absent | true value only in the fake |", "|---|---|---|---|"]
    for sname in c["accuracy"]["present"]:
        L.append(f"| {sname} | {c['accuracy']['present'][sname]} | {c['accuracy']['absent'][sname]} | {c['accuracy']['only_in_fake'][sname]} |")
    L += ["", "Paired difference from no fake (points, 95% CI, n questions): " + "; ".join(
        f"{k} {v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}] (n={v[3]})" for k, v in c["vs_no_fake"].items())]
    L += ["", f"Matching: {c['matching']}."]
    return "\n".join(L) + "\n"


# Тесты на опубликованных данных CrAM: T1 (баллы GPT против шаблонов и аудит AUC формы), T2 (аудит фейков) и подсчёт того,
# где в протоколе CrAM встречается истинное значение (к разбору их Appendix F).
# Инпут: файлы ../CrAM/*_1000_bge.json и results/cram_protocol/nq_answers.jsonl
# Аутпут: int код возврата; results/cram_critique/critique.json и critique.md
def main() -> int:
    cfg = load_config()
    cram_repo = resolve(cfg["paths"]["cram_repo"])
    t12 = cram_scores_vs_templates(cram_repo)
    audit = shape_auc_audit(cram_repo)
    contain = reference_containment(cram_repo, read_jsonl(resolve("results/cram_protocol/nq_answers.jsonl")))
    out_dir = resolve("results/cram_critique"); out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "critique.json", {"T1_T2": t12, "T1_audit": audit, "containment": contain})
    md = render(t12, audit, contain)
    (out_dir / "critique.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
