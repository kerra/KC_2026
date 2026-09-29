from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h3_contrast import ac_stance_audit  # noqa: E402
from witness_rag.io import data_path, load_config, read_jsonl, resolve, write_json  # noqa: E402

ALPHAS = {"95": 0.05, "claims": 0.05 / 4}

RQS = {
    1: "What does CrAM's evidence measure?",
    2: "Once authorship, shape and dependence are decoupled from truth, can judgment-free signals stand in for the judge?",
}

CLAIMS = [
    {"id": "1.1", "rq": 1, "short": "Shape suffices on CrAM's benchmark", "direction": "positive",
     "text": "CrAM's gains do not require credibility assessment: on CrAM's benchmark a truth-blind score of answer shape recovers "
             "more accuracy than CrAM's GPT credibility scores, and the same score gains nothing where shape is independent of truth.",
     "statistic": "accuracy, shape score minus CrAM's GPT scores, on CrAM's protocol (1,000 NQ questions, 1 and 3 fakes pooled, "
                  "paired by question)",
     "data": "CrAM's protocol: their reader, heads, passages and fakes; 2,000 question-by-fake-count contexts"},
    {"id": "1.2", "rq": 1, "short": "Denial, not shape, persuades", "direction": "positive",
     "text": "What makes CrAM's fakes persuasive is that they name the true value and deny it, not their answer shape; lies that "
             "deny the truth in prose the shape score does not flag defeat the shape defence.",
     "statistic": "flip rate of the none arm, contrastive minus plain false passages (k = 1 and 3 pooled, paired by pair and k)",
     "data": "contrastive cell (generated after the test run) against the plain lies of the same test pairs; 159 matched contexts"},
    {"id": "2.1", "rq": 2, "short": "Authorship is a shortcut", "direction": "positive",
     "text": "Weighting passages by how machine-written they look follows the author-veracity alignment of the context: it gains "
             "where machine-written coincides with false and loses where machine-written coincides with true, including on clean "
             "retrieval.",
     "statistic": "Δ accuracy vs none where machine-written means false (shortcut, k = 1 to 3) minus where it means true "
                  "(AI true, human false), averaged over the two authorship detectors",
     "data": "test split, 100 pairs: 300 shortcut and 100 machine-true contexts, each answered by none and the two form arms"},
    {"id": "2.2", "rq": 2, "short": "Counting witnesses fails", "direction": "negative",
     "text": "Counting witnesses does not defeat manufactured consensus: a surface-similarity discount merges only near-verbatim "
             "copies, not paraphrases or texts written independently from the same prompt, so a lie generated several times from one "
             "prompt still outvotes the truth, with or without the discount.",
     "statistic": "Δ accuracy vs none of agreement and voting pooled, coordinated contexts in which the lie outnumbers the truth "
                  "(k = 3 and 5)",
     "data": "test split, 100 pairs: 200 coordinated contexts, each answered by none, agreement and voting"},
]


# Кластерный бутстрэп для статистики, собранной из нескольких компонент: каждая компонента задана суммой и числом разностей
# в каждом кластере; в ресэмпле кластеры берутся с возвращением, компонента = сумма / число, статистика считается из компонент.
# Инпут: dict имя >> (суммы по кластерам, числа по кластерам); функция компонент >> статистика; int ресэмплов; int сид
# Аутпут: dict estimate и интервалы при всех уровнях ALPHAS (всё в долях, не в пунктах)
def cluster_bootstrap(components: dict[str, tuple[np.ndarray, np.ndarray]], stat, n_resamples: int, seed: int,
                      chunk: int = 2000) -> dict:
    n = len(next(iter(components.values()))[0])
    full = {k: s.sum() / c.sum() for k, (s, c) in components.items()}
    rng = np.random.default_rng(seed)
    boots, done = [], 0
    while done < n_resamples:
        m = min(chunk, n_resamples - done)
        idx = rng.integers(0, n, size=(m, n))
        means = {k: s[idx].sum(axis=1) / c[idx].sum(axis=1) for k, (s, c) in components.items()}
        boots.append(np.asarray(stat(means), dtype=float))
        done += m
    boots = np.concatenate(boots)
    out = {"estimate": float(stat(full)), "n_clusters": int(n), "intervals": {}}
    for name, a in ALPHAS.items():
        lo, hi = np.percentile(boots, [100 * a / 2, 100 * (1 - a / 2)])
        out["intervals"][name] = [float(lo), float(hi)]
    return out


# Собирает разности по кластерам в массивы сумм и чисел для cluster_bootstrap.
# Инпут: dict имя компоненты >> dict кластер >> list[float] разностей; list кластеров в фиксированном порядке
# Аутпут: dict имя >> (np.ndarray сумм, np.ndarray чисел)
def to_components(diffs: dict[str, dict[str, list[float]]], clusters: list[str]) -> dict:
    return {name: (np.array([sum(d.get(c, [])) for c in clusters], float), np.array([len(d.get(c, [])) for c in clusters], float))
            for name, d in diffs.items()}


def _outcome_index(rows, keys):
    return {tuple(r[k] for k in keys): r["outcome"] for r in rows}


def _pp(x: dict) -> dict:
    return {"estimate": 100 * x["estimate"], "n_clusters": x["n_clusters"],
            "intervals": {k: [100 * v[0], 100 * v[1]] for k, v in x["intervals"].items()}}


def _holds(ci: list[float], direction: str) -> bool:
    return ci[0] > 0 if direction == "positive" else ci[1] < 0


# Сдвиг точности арма против базового арма в заданных условиях тестового сплита (пулинг по k), кластерный бутстрэп по парам.
# Инпут: list[dict] строки test_v1; str арм; tuple[str] условия; set[int] или None значения k; int ресэмплов; int сид; str база
# Аутпут: dict в пунктах: estimate, n_clusters, intervals
def delta_vs(rows: list[dict], arm: str, conds: tuple, ks, n_resamples: int, seed: int, base: str = "none") -> dict:
    idx = _outcome_index(rows, ("signal", "pair_id", "condition", "k"))
    d = collections.defaultdict(list)
    for (sig, p, c, k), o in idx.items():
        if sig == arm and c in conds and (ks is None or k in ks) and (base, p, c, k) in idx:
            d[p].append(float(o == "gold") - float(idx[(base, p, c, k)] == "gold"))
    clusters = sorted(d)
    return _pp(cluster_bootstrap(to_components({"d": d}, clusters), lambda m: m["d"], n_resamples, seed))


# Первичная статистика клейма 1.1 и её разбивки: точность схемы shape минус схема GPT в протоколе CrAM по вопросам, плюс
# вспомогательные разности схем, доля контекстов, где жёсткая маска CrAM попала в настоящий пассаж, и shape и судья
# на тестовом сплите там, где форма совпадает с правдой (фейки CrAM), где она сбалансирована и где лжи нет.
# Инпут: list[dict] строки nq_answers.jsonl; list[dict] строки test_v1; int ресэмплов; int сид
# Аутпут: dict primary, breakdown, supporting (в пунктах)
def claim_1_1(protocol: list[dict], rows: list[dict], n_resamples: int, seed: int) -> dict:
    idx = _outcome_index(protocol, ("id", "fake_num", "scheme"))
    ids = sorted({r["id"] for r in protocol})

    def diffs(a: str, b: str, fake_nums=(1, 3)) -> dict:
        out = collections.defaultdict(list)
        for q in ids:
            for fn in fake_nums:
                if (q, fn, a) in idx and (q, fn, b) in idx:
                    out[str(q)].append(float(idx[(q, fn, a)] == "gold") - float(idx[(q, fn, b)] == "gold"))
        return out

    clusters = [str(q) for q in ids]
    primary = cluster_bootstrap(to_components({"d": diffs("shape", "gpt")}, clusters), lambda m: m["d"], n_resamples, seed)
    breakdown, supporting = {}, {}
    for fn in (1, 3):
        breakdown[f"shape_minus_gpt_{fn}"] = _pp(cluster_bootstrap(to_components({"d": diffs("shape", "gpt", (fn,))}, clusters),
                                                                   lambda m: m["d"], n_resamples, seed))
        for a, b in (("cnn", "gpt"), ("gpt", "gpt_maxnorm"), ("sort_only", "none"), ("ideal", "shape")):
            supporting[f"{a}_minus_{b}_{fn}"] = _pp(cluster_bootstrap(to_components({"d": diffs(a, b, (fn,))}, clusters),
                                                                      lambda m: m["d"], n_resamples, seed))
        gpt_rows = [r for r in protocol if r["scheme"] == "gpt" and r["fake_num"] == fn]
        supporting[f"hard_mask_on_real_{fn}"] = {
            "n": sum(any(k == "real" and m == 0.0 for k, m in zip(r["kinds"], r["multipliers"])) for r in gpt_rows), "of": len(gpt_rows)}
    supporting["test_split"] = {f"{arm}_{cond}": delta_vs(rows, arm, (cond,), None, n_resamples, seed)
                                for arm in ("shape", "judge") for cond in ("shortcut_cram", "all_ai_mixed", "clean")}
    supporting["shape_minus_judge_shortcut_cram"] = delta_vs(rows, "shape", ("shortcut_cram",), None, n_resamples, seed, base="judge")
    return {"primary": _pp(primary), "breakdown": breakdown, "supporting": supporting}


# Первичная статистика клейма 1.2: флипы none-арма при контрастных минус при простых ложных пассажах, k = 1 и 3, кластеры = пары.
# Контексты с пассажами, которые экстракция ридера прочла как утверждающие истину, исключены.
# Инпут: list[dict] строки test_v1; list[dict] строки h3_contrast; set инвертированных doc_id; int ресэмплов; int сид
# Аутпут: dict primary, breakdown (по k)
def claim_1_2(plain: list[dict], contrast: list[dict], inverted: set, n_resamples: int, seed: int) -> dict:
    pa = {(r["pair_id"], r["k"]): r["outcome"] for r in plain if r["condition"] == "shortcut" and r["signal"] == "none"}
    co = {(r["pair_id"], r["k"]): r["outcome"] for r in contrast if r["signal"] == "none"
          and not any(d in inverted for d, c in zip(r["doc_ids"], r["cells"]) if c == "A_C")}
    comps = {"d": collections.defaultdict(list), "d_k1": collections.defaultdict(list), "d_k3": collections.defaultdict(list)}
    for (p, k), o in co.items():
        if (p, k) in pa and k in (1, 3):
            d = float(o == "attack") - float(pa[(p, k)] == "attack")
            comps["d"][p].append(d)
            comps[f"d_k{k}"][p].append(d)
    clusters = sorted(comps["d"])
    primary = cluster_bootstrap(to_components({"d": comps["d"]}, clusters), lambda m: m["d"], n_resamples, seed)
    breakdown = {}
    for k in (1, 3):
        cl = sorted(comps[f"d_k{k}"])
        breakdown[f"k{k}"] = _pp(cluster_bootstrap(to_components({"d": comps[f"d_k{k}"]}, cl), lambda m: m["d"], n_resamples, seed))
    out = {"primary": _pp(primary), "breakdown": breakdown}
    out["primary"]["n_units"] = sum(len(v) for v in comps["d"].values())
    return out


# Первичная статистика клейма 2.1: сдвиг точности армов авторства против none там, где машинный текст ложен (shortcut),
# минус там, где он истинен (ai_true_human_false), в среднем по двум детекторам; кластеры = пары.
# Инпут: list[dict] строки test_v1; dict doc_id >> строка лога; int ресэмплов; int сид
# Аутпут: dict primary, breakdown (по детекторам), supporting (баллы детекторов по ячейкам)
def claim_2_1(rows: list[dict], log: dict, n_resamples: int, seed: int) -> dict:
    idx = _outcome_index(rows, ("signal", "pair_id", "condition", "k"))
    pairs = sorted({r["pair_id"] for r in rows})
    comps = {}
    for arm in ("form", "form_lexicon"):
        for tag, cond in (("s", "shortcut"), ("a", "ai_true_human_false")):
            d = collections.defaultdict(list)
            for (sig, p, c, k), o in idx.items():
                if sig == arm and c == cond and ("none", p, c, k) in idx:
                    d[p].append(float(o == "gold") - float(idx[("none", p, c, k)] == "gold"))
            comps[f"{arm}_{tag}"] = d
    components = to_components(comps, pairs)
    stat = lambda m: ((m["form_s"] - m["form_a"]) + (m["form_lexicon_s"] - m["form_lexicon_a"])) / 2
    primary = cluster_bootstrap(components, stat, n_resamples, seed)
    breakdown = {arm: _pp(cluster_bootstrap(components, lambda m, a=arm: m[f"{a}_s"] - m[f"{a}_a"], n_resamples, seed))
                 for arm in ("form", "form_lexicon")}
    by_context = {f"{arm}_{name}": delta_vs(rows, arm, conds, None, n_resamples, seed)
                  for arm in ("form", "form_lexicon")
                  for name, conds in (("shortcut", ("shortcut",)), ("all_human", ("all_human",)), ("all_ai_mixed", ("all_ai_mixed",)),
                                      ("uninformative", ("all_human", "all_ai_mixed")), ("ai_true_human_false", ("ai_true_human_false",)),
                                      ("clean", ("clean",)))}
    by_context["judge_clean"] = delta_vs(rows, "judge", ("clean",), None, n_resamples, seed)
    return {"primary": _pp(primary), "breakdown": breakdown,
            "supporting": {"by_context": by_context, "detector_scores": detector_scores(rows, log)}}


# Что видят детекторы авторства: откалиброванная p(AI) каждого пассажа тестового сплита по ячейкам H_T, H_F, A_T, A_F и AUC
# для автора (машина против человека) и для правдивости внутри одного автора.
# Инпут: list[dict] строки test_v1; dict doc_id >> строка лога
# Аутпут: dict детектор >> списки p(AI) по ячейкам и три AUC
def detector_scores(rows: list[dict], log: dict) -> dict:
    from witness_rag.signals.form_score import roc_auc
    out = {}
    for arm in ("form", "form_lexicon"):
        p = {}
        for r in rows:
            if r["signal"] == arm:
                p.update(zip(r["doc_ids"], r[f"p_{arm}"]))
        by = {c: [round(v, 3) for d, v in sorted(p.items()) if log[d]["cell"] == c] for c in ("H_T", "H_F", "A_T", "A_F")}
        auc = lambda pos, neg: round(float(roc_auc(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])), 3)
        out[arm] = {"by_cell": by, "auc_machine_vs_human": auc(by["A_T"] + by["A_F"], by["H_T"] + by["H_F"]),
                    "auc_false_vs_true_human": auc(by["H_F"], by["H_T"]), "auc_false_vs_true_machine": auc(by["A_F"], by["A_T"])}
    return out


# Первичная статистика клейма 2.2: сдвиг точности голосования (agreement и voting) против none в coordinated при k = 3 и 5,
# где ложных пассажей больше, чем истинных; кластеры = пары. Плюс разбивки по армам и k и вспомогательные числа по дисконту.
# Инпут: list[dict] строки test_v1; dict doc_id >> строка лога; dict doc_id >> текст; int ресэмплов; int сид
# Аутпут: dict primary, breakdown, supporting
def claim_2_2(rows: list[dict], log: dict, docs: dict, n_resamples: int, seed: int) -> dict:
    idx = _outcome_index(rows, ("signal", "pair_id", "condition", "k"))
    pairs = sorted({r["pair_id"] for r in rows})

    def d_for(arms, ks, cond="coordinated", base="none"):
        d = collections.defaultdict(list)
        for (sig, p, c, k), o in idx.items():
            if sig in arms and c == cond and k in ks and (base, p, c, k) in idx:
                d[p].append(float(o == "gold") - float(idx[(base, p, c, k)] == "gold"))
        return d

    one = lambda d: _pp(cluster_bootstrap(to_components({"d": d}, pairs), lambda m: m["d"], n_resamples, seed))
    primary = cluster_bootstrap(to_components({"d": d_for({"agreement", "voting"}, {3, 5})}, pairs), lambda m: m["d"], n_resamples, seed)
    breakdown = {f"{arm}_k{k}": one(d_for({arm}, {k})) for arm in ("agreement", "voting") for k in (2, 3, 5)}
    supporting = {f"dependence_k{k}": one(d_for({"dependence"}, {k})) for k in (1, 2, 3, 5)}
    supporting["agreement_minus_voting_k5"] = one(d_for({"agreement"}, {5}, base="voting"))
    for arm in ("agreement", "voting"):
        supporting[f"{arm}_independent_k3"] = one(d_for({arm}, {3}, cond="independent"))
    neff = collections.defaultdict(list)
    for r in rows:
        if r["signal"] == "dependence" and r["condition"] == "coordinated":
            neff[r["k"]].append((r["neff_unweighted"], r["neff_discounted"]))
    supporting["neff_by_k"] = {str(k): [float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v]))] for k, v in sorted(neff.items())}
    supporting["merges"] = cluster_merges(rows, log)
    supporting["jaccard"] = cluster_similarity(rows, log, docs)
    return {"primary": _pp(primary), "breakdown": breakdown, "supporting": supporting}


# Сходство по 3-граммному Жаккару (как в арме dependence) внутри скоординированных кластеров тестовых пар: затравка и её копия,
# затравка и парафраз, две затравки между собой, и для сравнения истинный машинный пассаж A_T против затравки лжи.
# Инпут: list[dict] строки test_v1; dict doc_id >> строка лога; dict doc_id >> текст
# Аутпут: dict тип пары >> список значений и медиана
def cluster_similarity(rows: list[dict], log: dict, docs: dict) -> dict:
    from witness_rag.dependence.minhash import jaccard, shingles
    test_pairs = {r["pair_id"] for r in rows}
    cf = collections.defaultdict(list)
    at = collections.defaultdict(list)
    for d, r in log.items():
        if r["pair_id"] in test_pairs and r["cell"] == "C_F":
            cf[r["pair_id"]].append(d)
        elif r["pair_id"] in test_pairs and r["cell"] == "A_T":
            at[r["pair_id"]].append(d)
    sim = lambda a, b: jaccard(shingles(docs[a], 3), shingles(docs[b], 3))
    out = {"seed and its copy": [], "seed and its paraphrase": [], "two seeds": [], "true passage and a seed": []}
    for p, members in sorted(cf.items()):
        seeds = sorted(d for d in members if log[d]["dependence_level"] == "D2")
        for d in members:
            lv = log[d]["dependence_level"]
            if lv == "D4":
                out["seed and its copy"].append(sim(d, log[d]["parent_doc"]))
            elif lv == "D3":
                out["seed and its paraphrase"].append(sim(d, log[d]["parent_doc"]))
        out["two seeds"] += [sim(a, b) for i, a in enumerate(seeds) for b in seeds[i + 1:]]
        out["true passage and a seed"] += [sim(t, seeds[0]) for t in sorted(at[p])[:1]]
    return {k: {"values": [round(v, 3) for v in vals], "median": round(float(np.median(vals)), 3)} for k, vals in out.items()}


# Что склеивает дисконт зависимости в coordinated: копия с затравкой (D4), парафраз с затравкой (D3), затравки между собой (D2).
# Инпут: list[dict] строки test_v1; dict doc_id >> строка производственного лога
# Аутпут: dict по k: число контекстов, где склеены копия и парафраз, и число склеенных пар затравок из всех
def cluster_merges(rows: list[dict], log: dict) -> dict:
    out = {}
    for k in (3, 5):
        c = collections.Counter()
        for r in rows:
            if r["signal"] != "dependence" or r["condition"] != "coordinated" or r["k"] != k:
                continue
            lab = dict(zip(r["doc_ids"], r["dep_labels"]))
            cf = [d for d, cell in zip(r["doc_ids"], r["cells"]) if cell == "C_F"]
            seeds = [d for d in cf if log[d]["dependence_level"] == "D2"]
            for d in cf:
                lv = log[d]["dependence_level"]
                if lv in ("D3", "D4"):
                    c[f"{lv}_present"] += 1
                    c[f"{lv}_merged"] += lab.get(log[d]["parent_doc"]) == lab[d]
            c["seed_pairs"] += len(seeds) * (len(seeds) - 1) // 2
            c["seed_pairs_merged"] += sum(lab[a] == lab[b] for i, a in enumerate(seeds) for b in seeds[i + 1:])
        out[str(k)] = dict(c)
    return out


def render(res: dict) -> str:
    L = ["# Claims: primary statistics", "", f"Cluster bootstrap, {res['n_resamples']:,} resamples, seed {res['seed']}; "
         f"adjusted intervals at alpha = 0.05/4 (four claims).", "",
         "| claim | primary statistic | estimate, 95% CI | alpha 0.05/4 | verdict |", "|---|---|---|---|---|"]
    for c in res["claims"]:
        p, iv = c["primary"], c["primary"]["intervals"]
        L.append(f"| {c['id']} {c['short']} | {c['statistic']} | {p['estimate']:+.1f} [{iv['95'][0]:+.1f}, {iv['95'][1]:+.1f}] | "
                 f"[{iv['claims'][0]:+.1f}, {iv['claims'][1]:+.1f}] | {c['verdict']} |")
    for c in res["claims"]:
        L += ["", f"## Claim {c['id']}: breakdown and supporting numbers", ""]
        for name, v in {**c.get("breakdown", {}), **c.get("supporting", {})}.items():
            if isinstance(v, dict) and "estimate" in v:
                iv = v["intervals"]["95"]
                L.append(f"- {name}: {v['estimate']:+.1f} [{iv[0]:+.1f}, {iv[1]:+.1f}] ({v['n_clusters']} clusters)")
            elif name in ("test_split", "by_context"):
                L += [f"- {name} {k}: {x['estimate']:+.1f} [{x['intervals']['95'][0]:+.1f}, {x['intervals']['95'][1]:+.1f}] "
                      f"({x['n_clusters']} clusters)" for k, x in v.items()]
            elif name == "jaccard":
                L.append("- 3-gram Jaccard, median: " + "; ".join(f"{k} {x['median']:.2f} (n={len(x['values'])})" for k, x in v.items()))
            elif name == "detector_scores":
                for arm, x in v.items():
                    L.append(f"- {arm}: median p(AI) " + ", ".join(f"{c} {float(np.median(ps)):.2f}" for c, ps in x["by_cell"].items())
                             + f"; AUC machine vs human {x['auc_machine_vs_human']}, false vs true within human {x['auc_false_vs_true_human']}, "
                               f"within machine {x['auc_false_vs_true_machine']}")
            else:
                L.append(f"- {name}: {v}")
    return "\n".join(L) + "\n"


# Считает первичную статистику и вердикт каждого из четырёх клеймов по сохранённым ответам (без новых прогонов моделей),
# вместе с разбивками и вспомогательными числами, на которые ссылается отчёт.
# Инпут: --resamples, --seed; results/test_v1, results/h3_contrast, results/cram_protocol, data/
# Аутпут: int код возврата; results/claims/claims.json и claims.md
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    cfg = load_config()
    rows = read_jsonl(resolve("results/test_v1/answers.jsonl"))
    contrast = read_jsonl(resolve("results/h3_contrast/answers.jsonl"))
    protocol = read_jsonl(resolve("results/cram_protocol/nq_answers.jsonl"))
    inverted = {d for d, s in ac_stance_audit(cfg).items() if s in ("true", "both")}
    log = {r["doc_id"]: r for r in read_jsonl(data_path(cfg, "production_log.jsonl"))}
    docs = {d["doc_id"]: d["text"] for d in read_jsonl(data_path(cfg, "documents.jsonl"))}
    parts = {"1.1": claim_1_1(protocol, rows, args.resamples, args.seed),
             "1.2": claim_1_2(rows, contrast, inverted, args.resamples, args.seed),
             "2.1": claim_2_1(rows, log, args.resamples, args.seed),
             "2.2": claim_2_2(rows, log, docs, args.resamples, args.seed)}

    out = {"n_resamples": args.resamples, "seed": args.seed, "alphas": ALPHAS, "rqs": RQS, "claims": []}
    for c in CLAIMS:
        res = {**c, **parts[c["id"]]}
        iv = res["primary"]["intervals"]
        res["holds"] = {k: _holds(v, c["direction"]) for k, v in iv.items()}
        res["verdict"] = "supported" if res["holds"]["95"] else (
            "refuted" if _holds(iv["95"], "negative" if c["direction"] == "positive" else "positive") else "inconclusive")
        out["claims"].append(res)
    out_dir = resolve("results/claims"); out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "claims.json", out)
    md = render(out)
    (out_dir / "claims.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
