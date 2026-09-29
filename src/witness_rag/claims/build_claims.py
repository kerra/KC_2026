from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from ..io import read_jsonl, stable_rng, write_json, write_jsonl
from ..schemas import Pair, from_dict, source_kind

_KIND_ORDER = {"nq": 0, "trivia": 1}


def _normalize(rec: dict) -> dict:
    out = dict(rec)
    for k, v in list(out.items()):
        if isinstance(v, str):
            out[k] = v.strip()
    for k in ("answer_true_aliases", "answer_false_aliases"):
        v = out.get(k) or []
        if isinstance(v, str):
            v = [a.strip() for a in v.replace(";", "|").split("|") if a.strip()]
        out[k] = v
    return out


def _sort_key(rec: dict):
    src = rec["source"]
    ident = src.split(":", 1)[1] if ":" in src else ""
    try:
        num = int(ident)
    except ValueError:
        num = 10**12
    return (_KIND_ORDER.get(source_kind(src), 9), num, rec.get("question", ""))


# Присваивает импортированным парам стабильные id (p0001, ...), валидирует их и делит на dev/test со стратификацией по edit_type.
# Инпут: list[dict] импортированные пары; int желаемый размер dev; int сид разбиения
# Аутпут: tuple[list[Pair], dict, dict, list[str]] пары, сплиты (dev_pairs / test_pairs), статистика, ошибки валидации
def build(records: list[dict], n_dev: int, seed: int) -> tuple[list[Pair], dict, dict, list[str]]:
    records = sorted((_normalize(r) for r in records), key=_sort_key)
    pairs: list[Pair] = []
    errors: list[str] = []
    seen_q: dict[str, str] = {}
    for n, rec in enumerate(records, start=1):
        rec["pair_id"] = f"p{n:04d}"
        pair = from_dict(Pair, rec)
        errs = pair.validate()
        key = pair.question.lower()
        if key in seen_q:
            errs.append(f"duplicate question (also in {seen_q[key]})")
        seen_q[key] = pair.pair_id
        errors.extend(f"{pair.pair_id} [{pair.source}] {e}" for e in errs)
        pairs.append(pair)

    by_type: dict[str, list[str]] = collections.defaultdict(list)
    for p in pairs:
        by_type[p.edit_type].append(p.pair_id)
    dev: list[str] = []
    total = len(pairs)
    for et, ids in sorted(by_type.items()):
        rng = stable_rng(seed, "split", et)
        ids = sorted(ids)
        rng.shuffle(ids)
        dev.extend(ids[: round(n_dev * len(ids) / total) if total else 0])
    dev_set = set(dev)
    test = [p.pair_id for p in pairs if p.pair_id not in dev_set]
    splits = {"seed": seed, "n_dev": len(dev), "n_test": len(test), "dev_pairs": sorted(dev), "test_pairs": sorted(test)}
    stats = {
        "n_pairs": len(pairs),
        "by_edit_type": dict(collections.Counter(p.edit_type for p in pairs)),
        "by_topic": dict(collections.Counter(p.topic for p in pairs)),
        "by_source_kind": dict(collections.Counter(source_kind(p.source) for p in pairs)),
    }
    return pairs, splits, stats, errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imported", default="data/claims/pairs_imported.jsonl")
    ap.add_argument("--out-dir", default="data/claims")
    ap.add_argument("--n-dev", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--allow-errors", action="store_true")
    args = ap.parse_args(argv)

    if not Path(args.imported).exists():
        print("no input records (run claims.nq_import first)")
        return 1
    pairs, splits, stats, errors = build(read_jsonl(args.imported), args.n_dev, args.seed)
    if errors:
        print(f"{len(errors)} validation errors:")
        for e in errors:
            print("  " + e)
        if not args.allow_errors:
            return 1
    out = Path(args.out_dir)
    write_jsonl(out / "pairs.jsonl", (p.to_dict() for p in pairs))
    write_json(out / "splits.json", splits)
    print(f"pairs: {stats['n_pairs']}  dev: {splits['n_dev']}  test: {splits['n_test']}")
    print(f"by edit_type: {stats['by_edit_type']}  by source: {stats['by_source_kind']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
