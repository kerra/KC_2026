#!/usr/bin/env bash
set -e
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${PY:-python3}
export PYTHONPATH=src
cd "$REPO"
# запуск: SEED_OFFSET=100000 bash scripts/3_repair_round.sh (GPU); следующие раунды с 200000 и 300000 и REASSIGN=1
SEED_OFFSET=${SEED_OFFSET:-100000}
$PY experiments/repair_pairs.py --apply ${REASSIGN:+--reassign}
$PY -m witness_rag.generation.human_cells
$PY -m witness_rag.generation.plan
$PY - <<'PY'
import json
from pathlib import Path
ids = set(json.load(open("data/generated/regen_ids.json"))["doc_ids"])
path = Path("data/generated/documents_generated.jsonl")
rows = [l for l in path.read_text().splitlines() if l.strip()]
keep = [l for l in rows if json.loads(l)["doc_id"] not in ids]
path.write_text("\n".join(keep) + ("\n" if keep else ""))
print(f"dropped {len(rows) - len(keep)} flagged docs, {len(keep)} kept")
PY
for L in "D0 D2" "D3" "D4"; do
    $PY -m witness_rag.generation.generate --levels $L --seed-offset "$SEED_OFFSET"
done
$PY -m witness_rag.generation.clusters
$PY -m witness_rag.generation.qa_gates
