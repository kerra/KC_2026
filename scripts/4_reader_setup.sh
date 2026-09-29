#!/usr/bin/env bash
set -e
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${PY:-python3}
export PYTHONPATH=src
cd "$REPO"
# запуск: bash scripts/4_reader_setup.sh (GPU)
$PY experiments/hook_check.py --model meta-llama/Meta-Llama-3-8B-Instruct --n-pairs 20 --out results/hook_check_llama3.json
for M in binoculars lexicon shape stylometric; do
    $PY experiments/fit_form_calibrator.py --method $M
done
$PY experiments/closed_book.py --reader rd-llama3
$PY experiments/head_selection.py --reader rd-llama3
$PY experiments/judge_check.py --reader rd-llama3 --n 40 --min-parsed 0.8 --out results/judge_check_rd-llama3.json
