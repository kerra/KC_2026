#!/usr/bin/env bash
set -e
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${PY:-python3}
export PYTHONPATH=src
cd "$REPO"
# запуск: bash scripts/9_analyses.sh (CPU, нужен ../CrAM)
$PY experiments/cram_critique.py
$PY experiments/h3_contrast.py
$PY experiments/claims.py
$PY experiments/make_figures.py
