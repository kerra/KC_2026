from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from witness_rag.io import data_path, load_config, read_json, read_jsonl  # noqa: E402
from witness_rag.signals.form_score import FormSignal, fit_calibrator, roc_auc  # noqa: E402


# Обучает калибратор формы (binoculars | stylometric | shape | lexicon) только на dev-документах и печатает dev и test AUC;
# test AUC только для отчёта, подгонка на нём не делается. Метка: автор llm (форма) или targeted-жанр и фейк CrAM (shape).
# Инпут: --method
# Аутпут: int код возврата; data/calibration/form_<method>.json
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default=None, choices=[None, "binoculars", "stylometric", "shape", "lexicon"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    fcfg = dict(cfg["signals"]["form"])
    method = args.method or fcfg.get("method", "binoculars")
    name = "shape" if method == "shape" else ("form" if method == fcfg.get("method", "binoculars") else f"form_{method}")

    log = read_jsonl(data_path(cfg, "production_log.jsonl"))
    docs = {d["doc_id"]: d["text"] for d in read_jsonl(data_path(cfg, "documents.jsonl"))}
    pair_rows = read_jsonl(data_path(cfg, "claims", "pairs.jsonl"))
    questions = {p["pair_id"]: p["question"] for p in pair_rows}
    excluded = {p["pair_id"] for p in pair_rows if p.get("excluded")}
    splits = read_json(data_path(cfg, "claims", "splits.json"))
    dev, test = set(splits["dev_pairs"]) - excluded, set(splits["test_pairs"]) - excluded

    def label(r: dict) -> int:
        if method == "shape":
            return 1 if (r.get("shape") == "targeted" or r["cell"] == "G_F") else 0
        return 1 if r["author"] == "llm" else 0

    def subset(pair_set):
        rows = [r for r in log if r["pair_id"] in pair_set and r["doc_id"] in docs]
        return [docs[r["doc_id"]] for r in rows], [label(r) for r in rows], [questions[r["pair_id"]] for r in rows]

    signal = FormSignal.from_config(fcfg, cache_dir=data_path(cfg, "cache"), method=method, name=name)
    out = data_path(cfg, "calibration", f"form_{method}.json")
    t_dev, y_dev, q_dev = subset(dev)
    print("dev:", fit_calibrator(signal, t_dev, y_dev, out, q_dev if signal.needs_question else None))
    t_test, y_test, q_test = subset(test)
    if t_test:
        p = signal.p_flag(t_test, q_test if signal.needs_question else None)
        print("test AUC (report only, no fitting):", roc_auc(np.array(y_test, dtype=float), p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
