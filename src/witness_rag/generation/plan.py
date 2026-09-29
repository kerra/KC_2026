from __future__ import annotations

import argparse
import collections
import itertools
import sys
from pathlib import Path

from ..io import load_config, read_jsonl, stable_rng, stable_seed, write_jsonl
from ..schemas import DocRecord, Pair, from_dict
from .prompts import PromptBook


def _seed(*parts) -> int:
    return stable_seed(*parts) % (2**31 - 1)


# Строит план генерации всех LLM-документов: на пару 3 A_T + 3 A_F (D0, форма сбалансирована по позиции) и один кластер C_F
# из 3 D2-сидов, 1 D3-парафраза и 1 D4-копии; ни один генератор не является атакующим по всему корпусу.
# Инпут: list[Pair] пары; dict конфиг; PromptBook жанры
# Аутпут: list[dict] строки DocRecord без текстов (скелет производственного лога)
def build_plan(pairs: list[Pair], cfg: dict, book: PromptBook) -> list[dict]:
    gcfg = cfg["generation"]
    gens = list(gcfg["generators"].keys())
    organic, targeted = book.genres_by_shape("organic"), book.genres_by_shape("targeted")
    if len(gens) != 3:
        raise ValueError("the plan assumes exactly 3 generators")
    if len(organic) < 4 or len(targeted) < 2:
        raise ValueError("the plan needs >= 4 organic and >= 2 targeted genres")
    lengths = list(gcfg["lengths"])
    t_ind, t_coord = float(gcfg["temperature_independent"]), float(gcfg["temperature_coordinated"])
    project_seed = cfg["project"]["seed"]
    counter = itertools.count(1)
    rows: list[dict] = []

    def new_id() -> str:
        return f"d{next(counter):06d}"

    def llm_row(pair, stance, cell, level, model, genre, seed, length, temp, cluster, prompt_id=None, parent=None):
        return DocRecord(doc_id=new_id(), pair_id=pair.pair_id, stance=stance, author="llm", cell=cell,
                         dependence_level=level, cluster_id=cluster, model=model, genre=genre,
                         shape=book.shape(genre), prompt_id=prompt_id or book.prompt_id(genre),
                         temperature=temp, seed=seed, length_target=length, parent_doc=parent).to_dict()

    for i, pair in enumerate(pairs):
        rng = stable_rng(project_seed, pair.pair_id, "plan")
        org, tgt = organic[:], targeted[:]
        rng.shuffle(org)
        rng.shuffle(tgt)
        g_true = [org[0], org[1], tgt[0]]
        g_false = [org[2], org[3], tgt[1]]
        rng.shuffle(g_true)
        rng.shuffle(g_false)

        for j in range(3):
            rows.append(llm_row(pair, "true", "A_T", "D0", gens[j], g_true[j], _seed(project_seed, pair.pair_id, "A_T", j),
                                rng.choice(lengths), t_ind, f"src_{pair.pair_id}_AT{j}"))
        for j in range(3):
            rows.append(llm_row(pair, "false", "A_F", "D0", gens[j], g_false[j], _seed(project_seed, pair.pair_id, "A_F", j),
                                rng.choice(lengths), t_ind, f"src_{pair.pair_id}_AF{j}"))

        c = i % 3
        c_model, c_genre = gens[c], g_false[(c + 1) % 3]     # that genre's A_F doc used generator (c+1)%3 != c
        cluster, length = f"src_{pair.pair_id}_CF", rng.choice(lengths)
        root_id = None
        for s in range(int(gcfg["coordinated_seeds"])):
            row = llm_row(pair, "false", "C_F", "D2", c_model, c_genre, _seed(project_seed, pair.pair_id, "C_F", s),
                          length, t_coord, cluster)
            root_id = root_id or row["doc_id"]
            rows.append(row)
        for k in range(int(gcfg["paraphrase_docs"])):
            rows.append(llm_row(pair, "false", "C_F", "D3", gens[(c + 2) % 3], c_genre, _seed(project_seed, pair.pair_id, "D3", k),
                                length, t_coord, cluster, prompt_id=f"paraphrase.{book.version}", parent=root_id))
        for k in range(int(gcfg["copy_edit_docs"])):
            rows.append(llm_row(pair, "false", "C_F", "D4", c_model, c_genre, _seed(project_seed, pair.pair_id, "D4", k),
                                length, None, cluster, prompt_id=f"copy_edit.{book.version}", parent=root_id))
    return rows


# Добавляет по одному контрастному ложному пассажу (ячейка A_C, D0) на генератор и пару; doc_id продолжают существующий план.
# Инпут: list[Pair] пары; dict конфиг; PromptBook; int последний занятый номер doc_id; str жанр
# Аутпут: list[dict] строки DocRecord
def build_contrastive_rows(pairs: list[Pair], cfg: dict, book: PromptBook, start: int, genre: str = "contrastive") -> list[dict]:
    gcfg = cfg["generation"]
    gens = list(gcfg["generators"].keys())
    lengths = list(gcfg["lengths"])
    project_seed = cfg["project"]["seed"]
    rows = []
    n = start
    for pair in pairs:
        rng = stable_rng(project_seed, pair.pair_id, "plan_contrastive")
        for j, model in enumerate(gens):
            n += 1
            rows.append(DocRecord(doc_id=f"d{n:06d}", pair_id=pair.pair_id, stance="false", author="llm", cell="A_C",
                                  dependence_level="D0", cluster_id=f"src_{pair.pair_id}_AC{j}", model=model, genre=genre,
                                  shape=book.shape(genre), prompt_id=book.prompt_id(genre),
                                  temperature=float(gcfg["temperature_independent"]),
                                  seed=_seed(project_seed, pair.pair_id, "A_C", j), length_target=rng.choice(lengths)).to_dict())
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="data/claims/pairs.jsonl")
    ap.add_argument("--out", default="data/production_log_plan.jsonl")
    ap.add_argument("--append-contrastive", default=None, choices=[None, "test", "dev", "all"],
                    help="append contrastive A_C rows for this split to the EXISTING plan (ids continue), write, exit")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    pairs = [from_dict(Pair, r) for r in read_jsonl(args.pairs)]
    if args.append_contrastive:
        existing = read_jsonl(args.out)
        if any(r["cell"] == "A_C" for r in existing):
            print("plan already has A_C rows; nothing appended"); return 0
        from ..io import read_json
        splits = read_json(Path(args.pairs).parent / "splits.json")
        keep = set(splits["dev_pairs"] + splits["test_pairs"]) if args.append_contrastive == "all" else set(splits[f"{args.append_contrastive}_pairs"])
        subset = [p for p in pairs if p.pair_id in keep and not p.excluded]
        start = max(int(r["doc_id"][1:]) for r in existing)
        new_rows = build_contrastive_rows(subset, cfg, PromptBook(), start)
        write_jsonl(args.out, existing + new_rows)
        print(f"appended {len(new_rows)} A_C rows for {len(subset)} {args.append_contrastive} pairs -> {args.out} ({len(existing) + len(new_rows)} rows)")
        return 0
    rows = build_plan(pairs, cfg, PromptBook())
    out = Path(args.out)
    if out.exists():                       # keep generator reassignments made by experiments/repair_pairs.py --reassign
        previous = {r["doc_id"]: r for r in read_jsonl(out) if r.get("reassigned_from")}
        carried = 0
        for r in rows:
            old = previous.get(r["doc_id"])
            if old and (old["pair_id"], old["cell"], old["dependence_level"]) == (r["pair_id"], r["cell"], r["dependence_level"]):
                r["model"], r["reassigned_from"] = old["model"], old["reassigned_from"]
                carried += 1
        if carried:
            print(f"carried {carried} reassigned rows over from the previous plan")
    n = write_jsonl(out, rows)
    print(f"planned {n} llm docs for {len(pairs)} pairs")
    print("by cell:", dict(collections.Counter(r["cell"] for r in rows)))
    print("by model x stance:", dict(collections.Counter((r["model"], r["stance"]) for r in rows)))
    print("by shape x stance (D0):", dict(collections.Counter((r["shape"], r["stance"]) for r in rows if r["dependence_level"] == "D0")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
