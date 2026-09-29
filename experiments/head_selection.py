from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from witness_rag.contexts.builder import ContextError, CorpusView, build_context  # noqa: E402
from witness_rag.cram.heads import causal_tracing, combine_and_select, samples_from_contexts  # noqa: E402
from witness_rag.io import data_path, load_config, read_json, resolve  # noqa: E402


# Отбор влиятельных голов внимания на нашем dev-сплите причинной трассировкой CrAM (контексты coordinated k=1)
# или копия ранжирования CrAM по NQ для абляции (--from-cram-nq).
# Инпут: --reader, --n-samples, --topk, --from-cram-nq
# Аутпут: int код возврата; data/heads/<reader>/selected_heads.json
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reader", default="rd-llama3")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--topk", type=int, default=None, help="default: all heads with positive mean effect")
    ap.add_argument("--from-cram-nq", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cram_repo = resolve(cfg["paths"]["cram_repo"])
    topk = args.topk if args.topk is not None else cfg["cram"].get("heads_topk")

    if args.from_cram_nq:
        short = "llama3" if "llama3" in args.reader else "qwen"
        src = Path(str(resolve(cfg["cram"]["cram_nq_heads"])).replace("{reader_short}", short))
        out_dir = data_path(cfg, "heads", f"{args.reader}_cramnq")
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, out_dir / "heads_scores_mean.json")
        sys.path.insert(0, str(cram_repo))
        from utils.find_best_heads import find_top_k_heads  # noqa: E402
        ranked = read_json(out_dir / "heads_scores_mean.json")
        k = topk or sum(1 for v, _ in ranked if v > 0)
        find_top_k_heads(input_path=str(out_dir), topk=k)
        print({"source": str(src), "topk": k, "out": str(out_dir / "selected_heads.json")})
        return 0

    view = CorpusView.load(data_path(cfg, "claims", "pairs.jsonl"), data_path(cfg, "production_log.jsonl"),
                           data_path(cfg, "documents.jsonl"))
    dev = read_json(data_path(cfg, "claims", "splits.json"))["dev_pairs"]
    n = args.n_samples or int(cfg["cram"]["causal_tracing_samples"])
    contexts = []
    for pid in dev:
        try:
            contexts.append(build_context(view, pid, cfg["cram"]["causal_tracing_condition"], 1,
                                          int(cfg["cram"].get("causal_tracing_n_docs", cfg["contexts"]["n_docs"])),
                                          int(cfg["project"]["seed"])))
        except ContextError as exc:
            print("skip:", exc)
        if len(contexts) >= n:
            break
    samples = samples_from_contexts(contexts, {p: view.pairs[p].answer_false for p in view.pairs})
    out_dir = data_path(cfg, "heads", args.reader)
    causal_tracing(cfg["readers"][args.reader], cram_repo, samples, out_dir)
    print(combine_and_select(cram_repo, out_dir, topk))
    return 0


if __name__ == "__main__":
    sys.exit(main())
