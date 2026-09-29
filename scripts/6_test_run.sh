#!/usr/bin/env bash
set -e
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${PY:-python3}
export PYTHONPATH=src
cd "$REPO"
# запуск: bash scripts/6_test_run.sh (GPU)
$PY experiments/run_arms.py --reader rd-llama3 --split test --run-id test_v1 --transform maxnorm
