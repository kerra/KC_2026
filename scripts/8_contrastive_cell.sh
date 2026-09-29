#!/usr/bin/env bash
set -e
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${PY:-python3}
export PYTHONPATH=src
cd "$REPO"
# запуск: bash scripts/8_contrastive_cell.sh (GPU)
$PY -m witness_rag.generation.plan --append-contrastive test
$PY -m witness_rag.generation.generate --levels D0 --seed-offset 400000
$PY -m witness_rag.generation.clusters
$PY -m witness_rag.generation.qa_gates
$PY experiments/run_arms.py --reader rd-llama3 --split test --conditions shortcut_contrast --run-id h3_contrast \
    --signals none shape form_lexicon judge oracle agreement --transform maxnorm
