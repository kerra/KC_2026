#!/usr/bin/env bash
set -e
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${PY:-python3}
export PYTHONPATH=src
cd "$REPO"
# запуск: bash scripts/1_build_pairs.sh (CPU, нужен ../CrAM)
if [ ! -f data/claims/pairs.jsonl ]; then
    $PY -m witness_rag.claims.nq_import
    $PY -m witness_rag.claims.build_claims
fi
$PY -m witness_rag.generation.human_cells
$PY -m witness_rag.generation.plan
