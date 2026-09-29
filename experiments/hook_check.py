from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from witness_rag.contexts.builder import CorpusView  # noqa: E402
from witness_rag.cram.driver import CramReader, load_selected_heads  # noqa: E402
from witness_rag.generation.chunking import alias_pattern  # noqa: E402
from witness_rag.io import data_path, load_config, read_json, resolve, write_json  # noqa: E402


# Проверяет хук внимания CrAM под установленной версией transformers на dev-парах: A без хука, B хук с множителями 1 (тождество),
# C хук с множителем 1e-4 на единственный пассаж-источник (эффект); PASS при identity >= 0.95 и effect >= 0.3 на незнакомых парах.
# Инпут: --model, --heads, --n-pairs, --n-neutral, --attn, --out
# Аутпут: int код 0 (PASS) / 1 (FAIL); results/hook_check.json с маской, которую увидел хук
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="reader path; must contain 'Llama-3' or 'Qwen'")
    ap.add_argument("--heads", default="../CrAM/datasets/nq/llama3/selected_heads.json", help="CrAM selected_heads.json (layer -> heads)")
    ap.add_argument("--n-pairs", type=int, default=20)
    ap.add_argument("--n-neutral", type=int, default=4)
    ap.add_argument("--attn", default=None, choices=[None, "eager", "sdpa"], help="force an attention implementation after loading")
    ap.add_argument("--out", default="results/hook_check.json")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    view = CorpusView.load(str(data_path(cfg, "claims", "pairs.jsonl")), str(data_path(cfg, "production_log.jsonl")),
                           str(data_path(cfg, "documents.jsonl")))
    splits = read_json(data_path(cfg, "claims", "splits.json"))
    dev = [p for p in splits["dev_pairs"] if p in view.pairs][: args.n_pairs]

    heads = load_selected_heads(resolve(args.heads))
    reader = CramReader(args.model, cfg["paths"]["cram_repo"], selected_heads=heads, max_new_tokens=int(cfg["cram"]["max_new_tokens"]))
    import transformers
    report: dict = {"model": args.model, "transformers": transformers.__version__, "n_heads": sum(len(v) for v in heads.values()),
                    "attn_forced": args.attn, "mask_seen": None, "errors": [], "pairs": []}
    if args.attn:
        try:
            reader.model.set_attn_implementation(args.attn)
        except Exception:                                   # older API
            reader.model.config._attn_implementation = args.attn
    report["attn_implementation"] = getattr(reader.model.config, "_attn_implementation", "unknown")

    # record what the hook receives on its first call, then run CrAM's own code unchanged
    original = reader.strategy.edit_attention_mask

    def recording_hook(module, input_args, input_kwargs, attention_weight, head_idx=()):
        if report["mask_seen"] is None:
            m = input_kwargs.get("attention_mask", "ABSENT")
            report["mask_seen"] = {"kwarg_present": "attention_mask" in input_kwargs, "is_none": m is None,
                                   "dtype": str(getattr(m, "dtype", None)), "shape": list(getattr(m, "shape", []) or []),
                                   "min": float(m.min()) if hasattr(m, "min") and m is not None and m.numel() else None,
                                   "kwargs": sorted(k for k in input_kwargs.keys())}
        return original(module, input_args, input_kwargs, attention_weight=attention_weight, head_idx=head_idx)

    reader.strategy.edit_attention_mask = recording_hook

    for pid in dev:
        pair = view.pairs[pid]
        h_t = view.cell(pid, "H_T")
        neutrals = view.cell(pid, "H_N")[: args.n_neutral]
        if not h_t:
            continue
        passages = [h_t[0]["text"]] + [d["text"] for d in neutrals]
        gold = alias_pattern(pair.gold_aliases())
        row = {"pair_id": pid, "question": pair.question, "gold": pair.answer_true}
        try:
            row["closed_book"] = reader.answer_closed_book(pair.question)
            row["A_no_hook"] = reader.answer(pair.question, passages, [1.0] * len(passages))["answer"]
            row["B_hook_identity"] = reader.answer(pair.question, passages, [1.0] * len(passages), force_hook=True)["answer"]
            row["C_hook_mask_ht"] = reader.answer(pair.question, passages, [1e-4] + [1.0] * len(neutrals))["answer"]
            row["knows_closed_book"] = bool(gold.search(row["closed_book"]))
            row["A_correct"] = bool(gold.search(row["A_no_hook"]))
            row["C_correct"] = bool(gold.search(row["C_hook_mask_ht"]))
            row["identity"] = row["A_no_hook"].strip() == row["B_hook_identity"].strip()
            row["effect"] = row["A_no_hook"].strip() != row["C_hook_mask_ht"].strip()
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            report["errors"].append({"pair_id": pid, "trace": traceback.format_exc()[-1500:]})
        report["pairs"].append(row)
        print(json.dumps({k: row.get(k) for k in ("pair_id", "identity", "effect", "A_correct", "C_correct", "knows_closed_book", "error")}))

    ok = [r for r in report["pairs"] if "error" not in r]
    unknown = [r for r in ok if not r["knows_closed_book"]]
    summary = {
        "n": len(ok), "n_errors": len(report["errors"]),
        "identity_rate": sum(r["identity"] for r in ok) / len(ok) if ok else None,
        "effect_rate_all": sum(r["effect"] for r in ok) / len(ok) if ok else None,
        "effect_rate_unknown_closed_book": sum(r["effect"] for r in unknown) / len(unknown) if unknown else None,
        "accuracy_no_hook": sum(r["A_correct"] for r in ok) / len(ok) if ok else None,
        "accuracy_masked_ht": sum(r["C_correct"] for r in ok) / len(ok) if ok else None,
    }
    passed = bool(ok) and not report["errors"] and summary["identity_rate"] >= 0.95 and \
        (summary["effect_rate_unknown_closed_book"] or 0.0) >= 0.3
    report["summary"] = summary
    report["verdict"] = "PASS" if passed else "FAIL"
    out = resolve(args.out)
    write_json(out, report)
    print("mask seen by the hook:", report["mask_seen"])
    print("summary:", json.dumps(summary))
    print(f"VERDICT: {report['verdict']}  (attn={report['attn_implementation']}, transformers={report['transformers']}) -> {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
