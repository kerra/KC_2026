#!/usr/bin/env bash
set -e
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${PY:-python3}
export PYTHONPATH=src
cd "$REPO"
# запуск: bash scripts/2_generate_corpus.sh (GPU)
for L in "D0 D2" "D3" "D4"; do
    $PY -m witness_rag.generation.generate --levels $L
done
$PY -m witness_rag.generation.clusters
$PY -m witness_rag.generation.qa_gates
