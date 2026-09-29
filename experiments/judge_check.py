from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from witness_rag.cram.driver import CramReader  # noqa: E402
from witness_rag.io import data_path, load_config, read_jsonl, resolve, write_json  # noqa: E402
from witness_rag.signals.llm_judge import JudgeSignal  # noqa: E402


# Гейт перед длинным прогоном: судья-ридер на выборке пассажей из каждой ячейки должен дать парсируемый балл не реже --min-parsed.
# Инпут: --reader, --n, --min-parsed, --out
# Аутпут: int код 0 (PASS) / 1 (FAIL); results/judge_check.json
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reader", default="rd-llama3")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min-parsed", type=float, default=0.8)
    ap.add_argument("--out", default="results/judge_check.json")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    pairs = {p["pair_id"]: p for p in read_jsonl(data_path(cfg, "claims", "pairs.jsonl"))}
    log = [r for r in read_jsonl(data_path(cfg, "production_log.jsonl")) if not pairs[r["pair_id"]].get("excluded")]
    docs = {d["doc_id"]: d["text"] for d in read_jsonl(data_path(cfg, "documents.jsonl"))}
    rng = random.Random(7)
    by_cell = collections.defaultdict(list)
    for r in log:
        by_cell[r["cell"]].append(r)
    sample = []
    for cell, rows in sorted(by_cell.items()):
        rng.shuffle(rows)
        sample.extend(rows[: max(1, args.n // len(by_cell))])
    reader = CramReader(cfg["readers"][args.reader], resolve(cfg["paths"]["cram_repo"]), None, int(cfg["cram"]["max_new_tokens"]))
    judge = JudgeSignal(reader.judge, cache_path=None)                    # no cache: we want fresh verdicts
    results = []
    for r in sample:
        v = judge.judge_text(docs[r["doc_id"]])
        results.append({"doc_id": r["doc_id"], "cell": r["cell"], "stance": r["stance"], "parsed": v["parsed"],
                        "score": v["score"], "raw_tail": v["raw"][-160:]})
        print(f"{r['doc_id']} {r['cell']:3s} {r['stance']:7s} parsed={v['parsed']} score={v['score']} :: {v['raw'][-90:]!r}")
    parsed = sum(1 for x in results if x["parsed"]) / len(results)
    by_stance = {st: [x["score"] for x in results if x["stance"] == st and x["parsed"]] for st in ("true", "false", "neutral")}
    summary = {"reader": args.reader, "n": len(results), "parsed_rate": parsed,
               "mean_score_by_stance": {k: (sum(v) / len(v) if v else None) for k, v in by_stance.items()},
               "parsed_by_cell": {c: sum(1 for x in results if x["cell"] == c and x["parsed"]) / max(1, sum(1 for x in results if x["cell"] == c)) for c in by_cell}}
    write_json(resolve(args.out), {"summary": summary, "results": results})
    print("summary:", summary)
    ok = parsed >= args.min_parsed
    print(f"JUDGE CHECK {'PASS' if ok else 'FAIL'} (parsed {parsed:.0%}, threshold {args.min_parsed:.0%}) -> {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
