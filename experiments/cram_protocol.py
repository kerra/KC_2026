from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from witness_rag.cram.driver import CramReader, load_selected_heads  # noqa: E402
from witness_rag.eval.outcomes import classify  # noqa: E402
from witness_rag.io import append_jsonl, data_path, load_config, read_json, read_jsonl, resolve  # noqa: E402
from witness_rag.schemas import derived_aliases  # noqa: E402
from witness_rag.signals.form_score import LogisticCalibrator, ShapeScorer  # noqa: E402

_CNN = re.compile(r"^\s*CNN news:", re.I)

# Схемы взвешивания (аргумент --schemes). Протокол CrAM: fake_num фейков впереди + topk отобранных пассажей, ридер и головы CrAM.
#   none         без перевзвешивания, порядок CrAM (фейки, затем 4 найденных пассажа)
#   gpt          GPT-баллы достоверности из релиза CrAM, их нормировка (нижний пассаж под жёсткой маской) и сортировка по баллу
#   gpt_maxnorm  те же баллы, наша maxnorm без жёсткой маски, без сортировки >> изолирует нормировку и позицию
#   sort_only    только сортировка CrAM по баллу, все множители 1 >> изолирует эффект позиции
#   ideal        идеальная настройка CrAM: фейки 1 / настоящие 10 >> множители 0.1 / 1, с сортировкой
#   cnn          правило по шаблону: пассаж начинается с "CNN news:" >> 1, иначе 10; нормировка и сортировка CrAM
#   shape        наша оценка формы ответа 1 - p(targeted) (калибратор обучен на нашем dev), maxnorm, без сортировки
#   no_fake      референс: только 4 найденных пассажа без фейка и без перевзвешивания


# Нормировка CrAM из RAG_test.py: вычесть минимум, разделить на максимум; нижний пассаж получает жёсткую маску 0.
# Инпут: list[int] баллы
# Аутпут: list[float] множители
def cram_normalize(scores: list[int]) -> list[float]:
    s = np.array(scores, dtype=float)
    s = s - s.min()
    return list(s / s.max()) if s.max() != 0 else [1.0] * len(s)


# Наша нормировка: s / max(s) с полом floor, без жёсткой маски.
# Инпут: list[float] баллы; float пол
# Аутпут: list[float] множители
def maxnorm(scores: list[float], floor: float = 0.1) -> list[float]:
    hi = max(scores)
    return [1.0] * len(scores) if hi <= 1e-12 else [min(1.0, max(floor, x / hi)) for x in scores]


# T3: протокол CrAM на их вопросах, их 4 пассажах и фейках, их головах, но с разными схемами взвешивания (см. SCHEMES);
# ответы дописываются в results/cram_protocol/<dataset>_answers.jsonl, затем строится сводка.
# Инпут: --dataset, --reader, --heads, --fake-num, --topk, --schemes, --limit
# Аутпут: int код возврата; jsonl ответов и <dataset>_summary.md
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="nq", choices=["nq", "trivia"])
    ap.add_argument("--reader", default="rd-llama3")
    ap.add_argument("--heads", default="cram894", help="cram894 (CrAM's run.sh setting for llama3) | ours | path")
    ap.add_argument("--fake-num", nargs="+", type=int, default=[1, 3])
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--schemes", nargs="+", default=["none", "gpt", "gpt_maxnorm", "sort_only", "ideal", "cnn", "shape"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cram_repo = resolve(cfg["paths"]["cram_repo"])
    data = read_json(cram_repo / f"{args.dataset}_1000_bge.json")
    if args.limit:
        data = data[: args.limit]

    if args.heads == "cram894":
        sys.path.insert(0, str(cram_repo))
        from utils.find_best_heads import find_top_k_heads  # noqa: E402
        import shutil
        hdir = resolve("results/cram_protocol/heads_llama3_cram894")   # working copy: CrAM's script rewrites selected_heads.json
        hdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cram_repo / "datasets" / "nq" / "llama3" / "heads_scores_mean.json", hdir / "heads_scores_mean.json")
        find_top_k_heads(input_path=str(hdir), topk=894)            # top 894 heads, as CrAM's run.sh does for llama3
        heads = load_selected_heads(hdir / "selected_heads.json")
    elif args.heads == "ours":
        heads = load_selected_heads(data_path(cfg, "heads", args.reader, "selected_heads.json"))
    else:
        heads = load_selected_heads(resolve(args.heads))
    n_heads = sum(len(v) for v in heads.values())
    reader = CramReader(cfg["readers"][args.reader], cram_repo, heads, int(cfg["cram"]["max_new_tokens"]))
    shape = ShapeScorer()
    shape_cal = LogisticCalibrator.load(data_path(cfg, "calibration", "form_shape.json"))

    out_dir = resolve("results/cram_protocol"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_answers.jsonl"
    done = {(r["id"], r["fake_num"], r["scheme"]) for r in read_jsonl(out_path)} if out_path.exists() else set()
    n_new = 0
    for sample in data:
        refs = [x for x in sample.get("reference", []) if x]
        wrong = (sample.get("wrong answer") or "").strip()
        if not refs or not wrong:
            continue
        gold_aliases = refs + [a for r in refs for a in derived_aliases(r)]
        attack_aliases = [wrong, *derived_aliases(wrong)]
        for fake_num in args.fake_num:
            fakes = sample["ori_fake"][:fake_num]
            reals = sample["reranked_dense_ctxs"][: args.topk]
            paras = fakes + reals
            if "no_fake" in args.schemes and (sample["id"], fake_num, "no_fake") not in done:
                res = reader.answer(sample["question"], reals, [1.0] * len(reals))
                append_jsonl(out_path, {"id": sample["id"], "dataset": args.dataset, "fake_num": fake_num, "scheme": "no_fake",
                                        "heads": args.heads, "n_heads": n_heads, "answer": res["answer"],
                                        "outcome": classify(res["answer"], gold_aliases, attack_aliases), "hooked": False,
                                        "multipliers": [1.0] * len(reals), "kinds": ["real"] * len(reals),
                                        "ts": dt.datetime.now().isoformat(timespec="seconds")})
                done.add((sample["id"], fake_num, "no_fake")); n_new += 1
            gpt_scores = [int(x) for x in sample["ori_fake_truthful_scores"][:fake_num]] + \
                         [int(x) for x in sample["reranked_dense_ctxs_truthful_scores"][: args.topk]]
            for scheme in args.schemes:
                key = (sample["id"], fake_num, scheme)
                if key in done or scheme == "no_fake":
                    continue
                order = list(range(len(paras)))
                if scheme == "none":
                    mults = [1.0] * len(paras)
                elif scheme == "gpt":
                    mults = cram_normalize(gpt_scores)
                    order = sorted(order, key=lambda i: mults[i])
                elif scheme == "gpt_maxnorm":
                    mults = maxnorm([float(x) for x in gpt_scores])
                elif scheme == "sort_only":
                    mults = [1.0] * len(paras)
                    order = sorted(order, key=lambda i: cram_normalize(gpt_scores)[i])
                elif scheme == "ideal":
                    raw = [1] * fake_num + [10] * len(reals)
                    mults = [x / 10.0 for x in raw]                       # CrAM's ideal branch: no min shift
                    order = sorted(order, key=lambda i: mults[i])
                elif scheme == "cnn":
                    raw = [1 if _CNN.match(p) else 10 for p in paras]
                    mults = cram_normalize(raw)
                    order = sorted(order, key=lambda i: mults[i])
                elif scheme == "shape":
                    feats = shape.score(paras, [sample["question"]] * len(paras))
                    p_t = shape_cal.predict_proba(feats)
                    mults = maxnorm([float(1.0 - p) for p in p_t])
                else:
                    raise ValueError(scheme)
                paras_o = [paras[i] for i in order]; mults_o = [float(mults[i]) for i in order]
                res = reader.answer(sample["question"], paras_o, mults_o)
                row = {"id": sample["id"], "dataset": args.dataset, "fake_num": fake_num, "scheme": scheme, "heads": args.heads,
                       "n_heads": n_heads, "answer": res["answer"], "outcome": classify(res["answer"], gold_aliases, attack_aliases),
                       "hooked": res["hooked"], "multipliers": [round(m, 3) for m in mults_o],
                       "kinds": ["fake" if i < fake_num else "real" for i in order],
                       "ts": dt.datetime.now().isoformat(timespec="seconds")}
                append_jsonl(out_path, row)
                done.add(key); n_new += 1
    print(f"wrote {n_new} new rows -> {out_path}")

    rows = read_jsonl(out_path)
    lines = [f"# CrAM protocol replication: {args.dataset}, reader {args.reader}, heads {args.heads} ({n_heads})", "",
             "| fake_num | scheme | n | accuracy | flip | other |", "|---|---|---|---|---|---|"]
    for (fn, sc), g in sorted(collections.defaultdict(list, {k: [r for r in rows if (r["fake_num"], r["scheme"]) == k]
                                                              for k in {(r["fake_num"], r["scheme"]) for r in rows}}).items()):
        n = len(g)
        acc = 100 * sum(r["outcome"] == "gold" for r in g) / n; flip = 100 * sum(r["outcome"] == "attack" for r in g) / n
        lines.append(f"| {fn} | {sc} | {n} | {acc:.1f} | {flip:.1f} | {100 - acc - flip:.1f} |")
    (out_dir / f"{args.dataset}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
