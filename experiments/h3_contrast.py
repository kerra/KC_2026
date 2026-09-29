from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from witness_rag.io import load_config, read_jsonl, resolve  # noqa: E402


def rate(rows, m="attack"):
    return 100 * sum(r["outcome"] == m for r in rows) / len(rows) if rows else float("nan")


# Спаренная по парам разность долей исхода (a - b) с перцентильным бутстрэпом.
# Инпут: dict pair_id >> исход для a и для b; str метрика; int ресэмплов; float alpha; int сид
# Аутпут: tuple[float, float, float, int] разность в пунктах, границы интервала, число пар
def paired(a: dict, b: dict, metric: str, n_boot: int, alpha: float, seed: int = 0) -> tuple[float, float, float, int]:
    common = sorted(set(a) & set(b))
    d = np.array([(a[p] == metric) - (b[p] == metric) for p in common], float)
    if not len(d):
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return 100 * d.mean(), 100 * lo, 100 * hi, len(d)


# Аудит позиции контрастных пассажей A_C по кэшу извлечения ридера: что пассаж реально утверждает: ложное значение,
# истинное (инвертирован: модель исправила ложь вместо мнимого заблуждения) или неясно.
# Инпут: dict конфиг; str ридер
# Аутпут: dict doc_id >> 'false' | 'true' | 'both' | 'other'
def ac_stance_audit(cfg: dict, reader: str = "rd-llama3") -> dict[str, str]:
    import hashlib
    from witness_rag.eval.outcomes import classify
    from witness_rag.io import data_path
    log = {r["doc_id"]: r for r in read_jsonl(data_path(cfg, "production_log.jsonl"))}
    docs = {d["doc_id"]: d["text"] for d in read_jsonl(data_path(cfg, "documents.jsonl"))}
    pairs = {p["pair_id"]: p for p in read_jsonl(data_path(cfg, "claims", "pairs.jsonl"))}
    cache_path = data_path(cfg, "cache", reader, "extract_cache.jsonl")
    if not cache_path.exists():
        cache_path = data_path(cfg, "cache_server", reader, "extract_cache.jsonl")
    if not cache_path.exists():
        return {}
    cache = read_jsonl(cache_path)
    by_key = {c["key"]: c for c in cache}
    out = {}
    from witness_rag.schemas import Pair, from_dict
    for d, r in log.items():
        if r["cell"] != "A_C":
            continue
        pair = from_dict(Pair, pairs[r["pair_id"]])
        key = hashlib.sha256((pair.question + "\x1f" + docs[d]).encode("utf-8")).hexdigest()
        c = by_key.get(key)
        if not c:
            for cand in cache:                       # cache key layout may differ: fall back to matching the stored text
                if cand.get("passage", "") == docs[d] or cand.get("text", "") == docs[d]:
                    c = cand; break
        if not c:
            continue
        ans = c.get("answer") or c.get("raw") or ""
        out[d] = {"gold": "true", "attack": "false", "both": "both"}.get(classify(ans, pair.gold_aliases(), pair.attack_aliases()), "other")
    return out


# Сравнивает по парам флипы none-арма в shortcut_contrast (H_T + k контрастных фейков) и shortcut (H_T + k простых фейков)
# при k = 1 и 3, исключив контексты с инвертированными пассажами; то же для армов shape, form_lexicon, judge, oracle. Плюс проверки:
# подвыборка, где окно называет истинное значение; сравнение при одинаковой метке формы; как арм shape видит контрастные фейки;
# статистика по генераторам.
# Инпут: --resamples, --seed; results/test_v1/answers.jsonl, results/h3_contrast/answers.jsonl, data/ (лог, документы, пары, тексты)
# Аутпут: int код возврата; results/h3_contrast/h3.md и h3.json
def main(argv=None) -> int:
    import argparse
    import json
    import statistics
    from witness_rag.generation.chunking import alias_pattern
    from witness_rag.schemas import Pair, from_dict
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    cfg = load_config()
    n_boot, alpha, seed = args.resamples, 0.05, args.seed
    plain = read_jsonl(resolve("results/test_v1/answers.jsonl"))
    contrast_all = read_jsonl(resolve("results/h3_contrast/answers.jsonl"))
    log = {r["doc_id"]: r for r in read_jsonl(resolve("data/production_log.jsonl"))}
    docs = {d["doc_id"]: d["text"] for d in read_jsonl(resolve("data/documents.jsonl"))}
    pairs = {r["pair_id"]: from_dict(Pair, r) for r in read_jsonl(resolve("data/claims/pairs.jsonl"))}
    stance = ac_stance_audit(cfg)
    inverted = {d for d, s in stance.items() if s in ("true", "both")}
    ac_docs = sorted(d for d, r in log.items() if r["cell"] == "A_C")
    named = {d: bool(alias_pattern(pairs[log[d]["pair_id"]].gold_aliases()).search(docs[d])) for d in ac_docs}

    def acs(r):
        return [d for d, c in zip(r["doc_ids"], r["cells"]) if c == "A_C"]

    contrast = [r for r in contrast_all if not any(d in inverted for d in acs(r))]
    out = {"n_ac": len(ac_docs), "n_inverted": len(inverted & set(ac_docs)),
           "n_named_in_window_kept": sum(named[d] for d in ac_docs if d not in inverted),
           "n_kept": sum(1 for d in ac_docs if d not in inverted), "by_arm": [], "named_subset": [], "generators": {}}

    lines = ["# Contrastive cell: explicit denial of the true value vs plain assertion of the false one (test split)", "",
             f"A_C stance audit (reader extraction): {out['n_ac']} passages, inverted {out['n_inverted']} excluded; "
             f"of the {out['n_kept']} kept, {out['n_named_in_window_kept']} name the true value inside their 100-word window.", "",
             "| k | arm | plain A_F: acc / flip | contrastive A_C: acc / flip | Δflip (A_C - A_F) [CI] | n pairs |", "|---|---|---|---|---|---|"]
    for k in (1, 3):
        for arm in ("none", "shape", "form_lexicon", "judge", "oracle"):
            pa = {r["pair_id"]: r["outcome"] for r in plain if r["condition"] == "shortcut" and r["k"] == k and r["signal"] == arm}
            co = {r["pair_id"]: r["outcome"] for r in contrast if r["condition"] == "shortcut_contrast" and r["k"] == k and r["signal"] == arm}
            if not pa or not co:
                continue
            common = sorted(set(pa) & set(co))
            d, lo, hi, n = paired(co, pa, "attack", n_boot, alpha, seed)
            rec = {"k": k, "arm": arm, "n": n, "delta_flip": d, "lo": lo, "hi": hi,
                   "plain_acc": rate([{"outcome": pa[p]} for p in common], "gold"), "plain_flip": rate([{"outcome": pa[p]} for p in common]),
                   "contrast_acc": rate([{"outcome": co[p]} for p in common], "gold"), "contrast_flip": rate([{"outcome": co[p]} for p in common]),
                   "contrast_other": rate([{"outcome": co[p]} for p in common], "other"), "plain_other": rate([{"outcome": pa[p]} for p in common], "other"),
                   "contrast_both": rate([{"outcome": co[p]} for p in common], "both"), "plain_both": rate([{"outcome": pa[p]} for p in common], "both")}
            out["by_arm"].append(rec)
            star = " *" if (lo > 0 or hi < 0) else ""
            lines.append(f"| {k} | {arm} | {rec['plain_acc']:.0f} / {rec['plain_flip']:.0f} | {rec['contrast_acc']:.0f} / {rec['contrast_flip']:.0f} | "
                         f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]{star} | {n} |")

    # the true value is named inside the window of every contrastive passage of the context (the denial can be seen)
    lines += ["", "Contexts in which every contrastive passage names the true value inside its 100-word window (none arm):", ""]
    for k in (1, 3):
        pa = {r["pair_id"]: r["outcome"] for r in plain if r["condition"] == "shortcut" and r["k"] == k and r["signal"] == "none"}
        co = {r["pair_id"]: r["outcome"] for r in contrast if r["k"] == k and r["signal"] == "none" and all(named[d] for d in acs(r))}
        d, lo, hi, n = paired(co, pa, "attack", n_boot, alpha, seed)
        common = sorted(set(pa) & set(co))
        rec = {"k": k, "n": n, "delta_flip": d, "lo": lo, "hi": hi, "plain_flip": rate([{"outcome": pa[p]} for p in common]),
               "contrast_flip": rate([{"outcome": co[p]} for p in common])}
        out["named_subset"].append(rec)
        lines.append(f"- k={k}: flip {rec['plain_flip']:.0f} vs {rec['contrast_flip']:.0f}, Δ {d:+.1f} [{lo:+.1f}, {hi:+.1f}], {n} pairs")

    # same shape label on both sides: plain lies from a targeted genre vs contrastive lies (also generated from a targeted prompt), k = 1
    pa_t = {}
    for r in plain:
        if r["condition"] == "shortcut" and r["k"] == 1 and r["signal"] == "none":
            af = [d for d, c in zip(r["doc_ids"], r["cells"]) if c == "A_F"]
            if af and log[af[0]]["shape"] == "targeted":
                pa_t[r["pair_id"]] = r["outcome"]
    co1 = {r["pair_id"]: r["outcome"] for r in contrast if r["k"] == 1 and r["signal"] == "none"}
    d, lo, hi, n = paired(co1, pa_t, "attack", n_boot, alpha, seed)
    common = sorted(set(pa_t) & set(co1))
    out["shape_matched_k1"] = {"n": n, "delta_flip": d, "lo": lo, "hi": hi, "plain_flip": rate([{"outcome": pa_t[p]} for p in common]),
                               "contrast_flip": rate([{"outcome": co1[p]} for p in common])}
    lines += ["", f"Same shape label (plain targeted A_F vs contrastive, k=1, none arm): flip {out['shape_matched_k1']['plain_flip']:.0f} vs "
              f"{out['shape_matched_k1']['contrast_flip']:.0f}, Δ {d:+.1f} [{lo:+.1f}, {hi:+.1f}], {n} pairs"]

    # what the shape arm sees: mean p(targeted) per cell over the contexts it scored
    ps = collections.defaultdict(list)
    for rows in (plain, contrast_all):
        for r in rows:
            if r["signal"] != "shape":
                continue
            for d_, c, p in zip(r["doc_ids"], r["cells"], r["p_shape"]):
                ps[c if c not in ("A_F", "A_T") else f"{c}/{log[d_]['shape']}"].append(p)
    out["p_targeted"] = {k: round(statistics.mean(v), 3) for k, v in sorted(ps.items())}
    lines += ["", "Mean p(targeted) under the shape arm: " + ", ".join(f"{k} {v:.2f}" for k, v in out["p_targeted"].items())]

    # flip rate of the none arm at k = 1 by kind of lie (one lie against H_T), with the shape score's p(targeted) for that lie
    def prop_ci(v):
        v = np.array(v, float)
        boots = v[np.random.default_rng(seed).integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return [round(100 * float(v.mean()), 1), round(100 * float(lo), 1), round(100 * float(hi), 1), int(len(v))]

    def lie_p(rows, cond, cell, keep):
        out = {}
        for r in rows:
            if r["signal"] == "shape" and r["condition"] == cond and r["k"] == 1 and keep(r):
                for c, p in zip(r["cells"], r["p_shape"]):
                    if c == cell:
                        out[r["pair_id"]] = p
        return out

    lie_rows = {"plain lie, organic genre": ([r for r in plain if r["condition"] == "shortcut" and r["k"] == 1], "A_F", "organic"),
                "plain lie, targeted genre": ([r for r in plain if r["condition"] == "shortcut" and r["k"] == 1], "A_F", "targeted"),
                "contrastive lie": ([r for r in contrast if r["k"] == 1], "A_C", None),
                "CrAM's fake": ([r for r in plain if r["condition"] == "shortcut_cram" and r["k"] == 1], "G_F", None)}
    out["lie_types"] = {}
    for name, (rows_, cell, shape_) in lie_rows.items():
        def keep(r, cell=cell, shape_=shape_):
            lies = [d_ for d_, c in zip(r["doc_ids"], r["cells"]) if c == cell]
            return bool(lies) and (shape_ is None or log[lies[0]]["shape"] == shape_)
        flips = [float(r["outcome"] == "attack") for r in rows_ if r["signal"] == "none" and keep(r)]
        ps = lie_p(rows_, rows_[0]["condition"], cell, keep)
        out["lie_types"][name] = {"flip": prop_ci(flips), "p_targeted": [round(v, 3) for v in ps.values()],
                                  "mean_p_targeted": round(float(np.mean(list(ps.values()))), 3), "flips": flips}
    # plain lies at k = 1: targeted minus organic genre, two independent groups resampled separately
    t, o = (np.array(out["lie_types"][f"plain lie, {g} genre"].pop("flips")) for g in ("targeted", "organic"))
    rng = np.random.default_rng(seed)
    boots = t[rng.integers(0, len(t), (n_boot, len(t)))].mean(axis=1) - o[rng.integers(0, len(o), (n_boot, len(o)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out["targeted_minus_organic_k1"] = [round(100 * float(t.mean() - o.mean()), 1), round(100 * float(lo), 1), round(100 * float(hi), 1)]
    for v in out["lie_types"].values():
        v.pop("flips", None)
    lines += ["", "Flip rate of the none arm at k=1 by kind of lie (percent, 95% CI, n contexts) and mean p(targeted) of the lie: " + "; ".join(
        f"{k}: {v['flip'][0]:.0f} [{v['flip'][1]:.0f}, {v['flip'][2]:.0f}] (n={v['flip'][3]}), p {v['mean_p_targeted']:.2f}" for k, v in out["lie_types"].items())]

    # reference flip rates of the none arm at k = 1 on the test split
    ref = {c: rate([r for r in plain if r["condition"] == c and r["k"] == 1 and r["signal"] == "none"]) for c in ("shortcut", "shortcut_cram")}
    out["reference_flip_k1"] = ref
    lines += ["", f"Reference (test_v1, none, k=1): plain false passages flip {ref['shortcut']:.0f}%; CrAM's own fakes flip {ref['shortcut_cram']:.0f}%."]

    # by generator: inverted passages, attempts, rejected attempts, flip rate at k = 1
    gen = {d["doc_id"]: d for d in read_jsonl(resolve("data/generated/documents_generated.jsonl"))}
    rejected, refused = collections.Counter(), collections.Counter()
    for r in read_jsonl(resolve("data/generated/refusals.jsonl")):
        if r["doc_id"] in log and log[r["doc_id"]]["cell"] == "A_C":
            rejected[log[r["doc_id"]]["model"]] += 1
            refused[log[r["doc_id"]]["model"]] += r["problem"] == "refusal"
    byg = collections.defaultdict(list)
    for r in contrast:
        if r["k"] == 1 and r["signal"] == "none" and acs(r):
            byg[log[acs(r)[0]]["model"]].append(r)
    for m in sorted({log[d]["model"] for d in ac_docs}):
        mine = [d for d in ac_docs if log[d]["model"] == m]
        out["generators"][m] = {"docs": len(mine), "inverted": sum(d in inverted for d in mine),
                                "mean_attempts": round(statistics.mean(gen[d]["attempts"] for d in mine), 2),
                                "rejected_attempts": rejected[m], "refusals": refused[m], "flip_k1": rate(byg[m]), "n_k1": len(byg[m])}
    lines += ["", "By generator of the contrastive passages: " + "; ".join(
        f"{m}: inverted {g['inverted']} of {g['docs']}, rejected attempts {g['rejected_attempts']} (refusals {g['refusals']}), mean attempts {g['mean_attempts']}, "
        f"flip at k=1 {g['flip_k1']:.0f}% (n={g['n_k1']})" for m, g in out["generators"].items())]

    res = resolve("results/h3_contrast"); res.mkdir(parents=True, exist_ok=True)
    (res / "h3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (res / "h3.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
