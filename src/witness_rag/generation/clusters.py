from __future__ import annotations

import argparse
import collections
import datetime as dt
import sys
from pathlib import Path

from ..io import read_jsonl, write_jsonl
from ..schemas import Cluster, DocRecord, from_dict


# Именует тип кластера по уровням зависимости и ячейкам его членов.
# Инпут: set[str] уровни зависимости; set[str] ячейки
# Аутпут: str тип кластера
def cluster_kind(levels: set[str], cells: set[str]) -> str:
    if cells == {"H_T"}:
        return "human_paragraph"
    if cells == {"H_F"}:
        return "human_paragraph_value_substituted"
    if cells == {"H_N"}:
        return "human_paragraph_neutral"
    if cells == {"G_F"}:
        return "cram_misinformation"
    if levels == {"D0"}:
        return "singleton"
    parts = ["same_model_same_prompt"] if "D2" in levels else []
    if "D3" in levels:
        parts.append("paraphrase")
    if "D4" in levels:
        parts.append("copy_edit")
    return "+".join(parts) or "unknown"


# Запечатывает разметку: объединяет план LLM-документов (только реально сгенерированные), человеческий лог и тексты,
# проверяет строки и родителей, пишет documents.jsonl, production_log.jsonl и clusters.jsonl.
# Инпут: пути к плану, человеческому логу, сгенерированным и человеческим документам; Path папка вывода
# Аутпут: dict число документов, кластеров, отсутствующих документов и список ошибок
def finalize(plan_path: str, human_log_path: str, generated_docs_path: str, human_docs_path: str, out_dir: Path) -> dict:
    docs: dict[str, dict] = {}
    for path in (human_docs_path, generated_docs_path):
        if Path(path).exists():
            for d in read_jsonl(path):
                docs[d["doc_id"]] = d
    log_rows: list[dict] = []
    for path in (human_log_path, plan_path):
        if Path(path).exists():
            log_rows.extend(read_jsonl(path))
    today = dt.date.today().isoformat()
    present = [r for r in log_rows if r["doc_id"] in docs]
    missing = [r["doc_id"] for r in log_rows if r["doc_id"] not in docs]
    errors: list[str] = []
    by_id = {r["doc_id"]: r for r in present}
    for r in present:
        d = docs[r["doc_id"]]
        r["gen_date"] = r.get("gen_date") or d.get("gen_date") or today
        if r.get("window_start") is None and d.get("window_start") is not None:
            r["window_start"] = d["window_start"]
        if d.get("salvaged"):
            r["salvaged"] = d["salvaged"]
        for k in ("attempts", "unresolved", "seed_offset"):        # production facts, sealed with the log
            if d.get(k) is not None:
                r[k] = d[k]
        errors.extend(from_dict(DocRecord, r).validate())
        parent = r.get("parent_doc")
        if parent:
            if parent not in by_id:
                errors.append(f"{r['doc_id']}: parent {parent} not in corpus")
            elif r["cell"] != "H_F" and by_id[parent]["cluster_id"] != r["cluster_id"]:
                errors.append(f"{r['doc_id']}: parent {parent} has a different cluster_id")

    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in present:
        groups[r["cluster_id"]].append(r)
    clusters = []
    for cid, members in sorted(groups.items()):
        roots = [m for m in members if not m.get("parent_doc")]
        root = (roots[0] if roots else members[0])["doc_id"]
        if len({m["pair_id"] for m in members}) != 1 or len({m["stance"] for m in members}) != 1:
            errors.append(f"cluster {cid} spans several pairs or stances")
        clusters.append(Cluster(cluster_id=cid, pair_id=members[0]["pair_id"], stance=members[0]["stance"],
                                size=len(members), members=sorted(m["doc_id"] for m in members), root=root,
                                kind=cluster_kind({m["dependence_level"] for m in members}, {m["cell"] for m in members})).to_dict())

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "documents.jsonl",
                ({"doc_id": r["doc_id"], "text": docs[r["doc_id"]]["text"],
                  "n_words": docs[r["doc_id"]].get("n_words") or len(docs[r["doc_id"]]["text"].split())} for r in present))
    write_jsonl(out_dir / "production_log.jsonl", present)
    write_jsonl(out_dir / "clusters.jsonl", clusters)
    return {"documents": len(present), "clusters": len(clusters), "missing_docs": len(missing), "errors": errors}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="data/production_log_plan.jsonl")
    ap.add_argument("--human-log", default="data/human/human_log.jsonl")
    ap.add_argument("--generated-docs", default="data/generated/documents_generated.jsonl")
    ap.add_argument("--human-docs", default="data/human/human_docs.jsonl")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args(argv)
    summary = finalize(args.plan, args.human_log, args.generated_docs, args.human_docs, Path(args.out_dir))
    print({k: v for k, v in summary.items() if k != "errors"})
    for e in summary["errors"][:100]:
        print("error:", e)
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
