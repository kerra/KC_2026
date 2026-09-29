#!/usr/bin/env bash
set -e
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${PY:-python3}
export PYTHONPATH=src
cd "$REPO"
# запуск: bash scripts/7_cram_protocol.sh (GPU, нужен ../CrAM)
$PY experiments/cram_protocol.py --dataset nq --fake-num 1 3 --schemes none gpt gpt_maxnorm sort_only ideal cnn shape
$PY experiments/cram_protocol.py --dataset nq --fake-num 1 --schemes no_fake
