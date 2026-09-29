from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from witness_rag.contexts.builder import ContextError, CorpusView, build_context, iter_conditions, manifest  # noqa: E402
from witness_rag.eval.outcomes import classify  # noqa: E402
from witness_rag.io import append_jsonl, data_path, load_config, read_json, read_jsonl, resolve, results_path, write_jsonl  # noqa: E402
from witness_rag.signals.base import make_signal, to_multipliers  # noqa: E402


# Строит контексты всех (условие, k) для списка пар; контексты, для которых не хватает документов, попадают в ошибки.
# Инпут: CorpusView; dict конфиг; list[str] pair_id; set условий или None
# Аутпут: tuple[list[Context], list[str]]
def build_all_contexts(view: CorpusView, cfg: dict, pair_ids: list[str], conditions: set[str] | None):
    n_docs = int(cfg["contexts"]["n_docs"])
    seed = int(cfg["project"]["seed"])
    contexts, errors = [], []
    for cond, k in iter_conditions(cfg):
        if conditions and cond not in conditions:
            continue
        for pid in pair_ids:
            try:
                contexts.append(build_context(view, pid, cond, k, n_docs, seed))
            except ContextError as exc:
                errors.append(str(exc))
    return contexts, errors


# Факторный прогон ридер x сигнал x условие x k на одном сплите: сырые оценки >> множители >> ответ ридера с хуком CrAM >> исход;
# строки дописываются в results/<run_id>/answers.jsonl, готовые пропускаются, порядок пассажей одинаков для всех армов.
# Инпут: аргументы командной строки (--reader, --split, --run-id, --signals, --conditions, --heads, --transform, --dry-run)
# Аутпут: int код возврата; файлы answers.jsonl, contexts_<split>.jsonl, context_errors.jsonl
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reader", default="rd-llama3")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--signals", nargs="*", default=None, help="default: signals.arms from the config")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--heads", default=None, help="selected_heads.json; default data/heads/<reader>/selected_heads.json")
    ap.add_argument("--limit-pairs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="build contexts and manifests, load no model")
    ap.add_argument("--transform", default=None, choices=[None, "clip", "maxnorm"],
                    help="override signals.transform from the config (recorded in every row)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    run_id = args.run_id or f"{args.split}_{args.reader}_{dt.datetime.now():%Y%m%d_%H%M}"
    out_dir = results_path(cfg, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    view = CorpusView.load(data_path(cfg, "claims", "pairs.jsonl"), data_path(cfg, "production_log.jsonl"),
                           data_path(cfg, "documents.jsonl"))
    splits = read_json(data_path(cfg, "claims", "splits.json"))
    pair_ids = splits[f"{args.split}_pairs"]
    if args.limit_pairs:
        pair_ids = pair_ids[: args.limit_pairs]
    contexts, errors = build_all_contexts(view, cfg, pair_ids, set(args.conditions) if args.conditions else None)
    write_jsonl(out_dir / f"contexts_{args.split}.jsonl", (manifest(c) for c in contexts))
    write_jsonl(out_dir / "context_errors.jsonl", ({"error": e} for e in errors))
    print(f"{len(contexts)} contexts built, {len(errors)} skipped (see context_errors.jsonl)")
    if args.dry_run:
        return 0

    from witness_rag.cram.driver import CramReader, load_selected_heads  # noqa: E402  (GPU imports)

    heads_path = args.heads or data_path(cfg, "heads", args.reader, "selected_heads.json")
    heads = load_selected_heads(heads_path) if Path(heads_path).exists() else None
    if heads is None:
        print(f"warning: no selected heads at {heads_path}; hooked arms will run WITHOUT hooks (= none)")
    reader = CramReader(cfg["readers"][args.reader], resolve(cfg["paths"]["cram_repo"]), heads,
                        int(cfg["cram"]["max_new_tokens"]))

    cache_dir = data_path(cfg, "cache")                           # reader-independent features
    reader_cache_dir = data_path(cfg, "cache", args.reader)        # judge verdicts / extracted answers of THIS reader
    calibrator_path = data_path(cfg, "calibration", f"form_{cfg['signals']['form']['method']}.json")
    signal_names = args.signals or list(cfg["signals"]["arms"])
    signals = {n: make_signal(n, cfg, judge_generate=reader.judge, cache_dir=cache_dir,
                              calibrator_path=calibrator_path, extract=reader.extract_answer,
                              reader_cache_dir=reader_cache_dir) for n in signal_names}
    unique_docs = {d.doc_id: d for c in contexts for d in c.docs}
    for sig in signals.values():
        sig.precompute(list(unique_docs.values()))

    answers_path = out_dir / "answers.jsonl"
    done = {(r["reader"], r["signal"], r["condition"], r["k"], r["pair_id"]) for r in read_jsonl(answers_path)} \
        if answers_path.exists() else set()
    floor, transform = float(cfg["signals"]["floor"]), args.transform or cfg["signals"]["transform"]
    print(f"multiplier transform: {transform} (floor {floor})")
    n_new = 0
    for ctx in contexts:
        pair = view.pairs[ctx.pair_id]
        for name, sig in signals.items():
            key = (args.reader, name, ctx.condition, ctx.k, ctx.pair_id)
            if key in done:
                continue
            raw = sig.scores(ctx)
            mults = to_multipliers(raw, floor, transform)
            res = reader.answer(ctx.question, ctx.texts(), mults)
            row = {
                "run_id": run_id, "reader": args.reader, "signal": name, "condition": ctx.condition, "k": ctx.k,
                "pair_id": ctx.pair_id, "answer": res["answer"],
                "outcome": classify(res["answer"], pair.gold_aliases(), pair.attack_aliases()),
                "doc_ids": ctx.doc_ids(), "cells": [d.sealed.get("cell") for d in ctx.docs],
                "raw_scores": [round(s, 4) for s in raw], "multipliers": [round(m, 4) for m in mults],
                "hooked": res["hooked"], "n_prompt_tokens": res["n_prompt_tokens"], "transform": transform,
                "ts": dt.datetime.now().isoformat(timespec="seconds"),
                **sig.extras(ctx),
            }
            append_jsonl(answers_path, row)
            done.add(key)
            n_new += 1
    print(f"wrote {n_new} new rows to {answers_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
