from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from witness_rag.claims.nq_import import looks_like_artifact, select_pairs  # noqa: E402
from witness_rag.io import data_path, load_config, read_json, read_jsonl, resolve, stable_rng, write_json, write_jsonl  # noqa: E402
from witness_rag.schemas import number_aliases  # noqa: E402

MIN_ATTEMPTS, MIN_GENERATORS = 10, 2


# Причины исключения пар: ложное значение является артефактом (looks_like_artifact) или в прогоне набрало не менее 10 отказов
# value_missing от не менее 2 генераторов; числовые слова не считаются, их чинят алиасы.
# Инпут: list[dict] пары; dict doc_id >> строка плана; list[dict] отвергнутые попытки
# Аутпут: dict pair_id >> причина
def exclusion_reasons(pairs: list[dict], plan: dict[str, dict], refusals: list[dict]) -> dict[str, str]:
    attempts: collections.Counter = collections.Counter()
    gens: dict[str, set] = collections.defaultdict(set)
    for r in refusals:
        p = plan.get(r["doc_id"])
        if p and r["problem"] == "value_missing" and p["stance"] == "false":
            attempts[p["pair_id"]] += 1
            gens[p["pair_id"]].add(p["model"])
    out: dict[str, str] = {}
    for q in pairs:
        if q.get("excluded"):
            out[q["pair_id"]] = q["excluded"]
            continue
        why = looks_like_artifact(q["answer_false"], [q["answer_true"], *q.get("answer_true_aliases", [])], strict=False)
        # number words ("twenty four" for 28) failed only because the digit form had no alias: fixed by number_aliases,
        # so the run's evidence does not count against them
        numeric = bool(number_aliases(q["answer_false"]) or number_aliases(q["answer_true"]))
        if why is None and not numeric and attempts[q["pair_id"]] >= MIN_ATTEMPTS and len(gens[q["pair_id"]]) >= MIN_GENERATORS:
            why = f"unwritable_false_value ({attempts[q['pair_id']]} value_missing attempts, {len(gens[q['pair_id']])} generators)"
        if why:
            out[q["pair_id"]] = why
    return out


# Ремонт набора пар с сохранением всех id: исключает пары с неписуемым ложным значением, добавляет столько же свежих пар из файлов CrAM,
# пополняет dev до нужного размера, собирает список документов на регенерацию и при --reassign передаёт их следующему генератору.
# Инпут: --apply (без него сухой прогон), --no-replace, --reassign
# Аутпут: int код возврата; обновлённые pairs.jsonl, pairs_imported.jsonl, imported_chunks.jsonl, splits.json, regen_ids.json
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--no-replace", action="store_true", help="exclude only, do not append replacement pairs")
    ap.add_argument("--reassign", action="store_true", help="hand flagged docs that already went through a regeneration round to the next generator")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    claims_dir = data_path(cfg, "claims")
    pairs = read_jsonl(claims_dir / "pairs.jsonl")
    imported = read_jsonl(claims_dir / "pairs_imported.jsonl")
    chunks = read_jsonl(claims_dir / "imported_chunks.jsonl")
    splits = read_json(claims_dir / "splits.json")
    plan = {r["doc_id"]: r for r in read_jsonl(data_path(cfg, "production_log_plan.jsonl"))}
    gen_dir = data_path(cfg, "generated")
    refusals = read_jsonl(gen_dir / "refusals.jsonl") if (gen_dir / "refusals.jsonl").exists() else []
    gen_docs = {d["doc_id"]: d for d in read_jsonl(gen_dir / "documents_generated.jsonl")} if (gen_dir / "documents_generated.jsonl").exists() else {}
    viol_path = data_path(cfg, "qa_violations.jsonl")
    violations = read_jsonl(viol_path) if viol_path.exists() else []

    # ---- 1. exclude
    reasons = exclusion_reasons(pairs, plan, refusals)
    newly = [pid for pid in reasons if not next(q for q in pairs if q["pair_id"] == pid).get("excluded")]
    print(f"excluded pairs: {len(reasons)} ({len(newly)} new)")
    for q in pairs:
        if q["pair_id"] in reasons:
            print(f"  {q['pair_id']} {q['source']:13s} {q['answer_true'][:24]!r:26s} -> {q['answer_false'][:30]!r:32s} {reasons[q['pair_id']]}")
            q["excluded"] = reasons[q["pair_id"]]
    by_kind = collections.Counter(q["source"].split(":")[0] for q in pairs if q["pair_id"] in reasons)
    already = collections.Counter(q["source"].split(":")[0] for q in pairs if "appended as a replacement" in (q.get("notes") or ""))
    by_kind = collections.Counter({k: n - already[k] for k, n in by_kind.items() if n - already[k] > 0})
    print(f"replacements still needed per source: {dict(by_kind) or 'none'} (already appended: {dict(already) or 'none'})")

    # ---- 2. replace
    new_pairs: list[dict] = []
    new_chunks: list[dict] = []
    if not args.no_replace and by_kind:
        used_sources = {q["source"] for q in pairs}
        seen_texts = {" ".join(c["h_t"].split()).lower() for c in chunks}
        for c in chunks:
            seen_texts.update(" ".join(t.split()).lower() for t in c.get("neutrals", []))
        next_n = max(int(q["pair_id"][1:]) for q in pairs) + 1
        files = {"nq": cfg["claims"]["cram_nq"], "trivia": cfg["claims"]["cram_trivia"]}
        for kind, n_needed in sorted(by_kind.items()):
            cand, cand_chunks, rej = select_pairs(read_json(resolve(files[kind])), kind, None, seen_texts, used_sources)
            print(f"{kind}: {n_needed} replacements needed, {len(cand)} fresh candidates (rejections: {rej})")
            for q, c in list(zip(cand, cand_chunks))[:n_needed]:
                q = dict(q)
                q["pair_id"] = f"p{next_n:04d}"
                q["notes"] += "; appended as a replacement for an excluded pair"
                next_n += 1
                new_pairs.append(q)
                new_chunks.append(c)
        for q in new_pairs:
            print(f"  + {q['pair_id']} {q['source']:13s} {q['edit_type']:14s} {q['answer_true'][:24]!r:26s} -> {q['answer_false'][:30]!r}  {q['question'][:60]}")

    # ---- splits: active pairs only; new pairs fill dev back up to claims.n_dev
    excluded_ids = set(reasons)
    dev = [p for p in splits["dev_pairs"] if p not in excluded_ids]
    test = [p for p in splits["test_pairs"] if p not in excluded_ids]
    n_dev_target = int(cfg["claims"]["n_dev_pairs"])
    new_ids = [q["pair_id"] for q in new_pairs]
    rng = stable_rng(int(cfg["project"]["seed"]), "split-extend", len(pairs))
    rng.shuffle(new_ids)
    k = max(0, min(len(new_ids), n_dev_target - len(dev)))
    dev, test = sorted(dev + new_ids[:k]), sorted(test + new_ids[k:])
    new_splits = {"seed": splits["seed"], "n_dev": len(dev), "n_test": len(test), "dev_pairs": dev, "test_pairs": test,
                  "excluded_pairs": sorted(excluded_ids)}
    print(f"splits: dev {len(dev)} test {len(test)} excluded {len(excluded_ids)}")

    # ---- 3. regeneration list on active pairs
    flagged = {v["doc_id"] for v in violations if v["doc_id"] in gen_docs} | {d for d, x in gen_docs.items() if x.get("unresolved")}
    flagged = {d for d in flagged if plan[d]["pair_id"] not in excluded_ids}
    children = {d for d, r in plan.items() if r.get("parent_doc") in flagged}
    regen = sorted(flagged | children)
    print(f"regenerate on active pairs: {len(flagged)} flagged + {len(children - flagged)} children = {len(regen)} docs")
    reassigned: list[tuple[str, str, str]] = []
    if args.reassign:
        gens = list(cfg["generation"]["generators"].keys())
        for d in sorted(flagged):
            row = plan[d]
            if row["dependence_level"] == "D4" or gen_docs.get(d, {}).get("seed_offset", 0) == 0 and not gen_docs.get(d, {}).get("unresolved"):
                continue
            new_model = gens[(gens.index(row["model"]) + 1) % len(gens)]
            row.setdefault("reassigned_from", row["model"])
            reassigned.append((d, row["model"], new_model))
            row["model"] = new_model
        for d, a, b in reassigned:
            print(f"  reassign {d} {plan[d]['pair_id']} {plan[d]['cell']}/{plan[d]['dependence_level']}: {a} -> {b}")

    if not args.apply:
        print("dry run; add --apply to write")
        return 0
    all_pairs = pairs + new_pairs
    write_jsonl(claims_dir / "pairs.jsonl", all_pairs)
    write_jsonl(claims_dir / "pairs_imported.jsonl", imported + new_pairs)
    write_jsonl(claims_dir / "imported_chunks.jsonl", chunks + new_chunks)
    write_json(claims_dir / "splits.json", new_splits)
    gen_dir.mkdir(parents=True, exist_ok=True)
    write_json(gen_dir / "regen_ids.json", {"doc_ids": regen, "excluded_pairs": sorted(excluded_ids), "new_pairs": [q["pair_id"] for q in new_pairs],
                                            "reassigned": [{"doc_id": d, "from": a, "to": b} for d, a, b in reassigned]})
    if reassigned:
        write_jsonl(data_path(cfg, "production_log_plan.jsonl"), list(plan.values()))
        print(f"plan rewritten with {len(reassigned)} reassigned rows")
    print(f"wrote {len(all_pairs)} pairs ({len(new_pairs)} new), splits, {gen_dir / 'regen_ids.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
