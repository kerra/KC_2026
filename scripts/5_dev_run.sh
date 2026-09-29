#!/usr/bin/env bash
set -e
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${PY:-python3}
export PYTHONPATH=src
cd "$REPO"
# запуск: bash scripts/5_dev_run.sh (GPU)
$PY experiments/run_arms.py --reader rd-llama3 --split dev --run-id dev --transform maxnorm
