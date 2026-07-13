#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/akt_also_stage_common.sh stage3_pi_decay "${1:-}"

run_experiment "pidecay1e3_bs64" --batch_size 64 "${ALSO_COMMON[@]}" --also_pi_lr 1e-4 --also_pi_decay 1e-3 &
first_pid=$!
run_experiment "pidecay1e1_bs64" --batch_size 64 "${ALSO_COMMON[@]}" --also_pi_lr 1e-4 --also_pi_decay 1e-1 &
second_pid=$!
wait "$first_pid"
wait "$second_pid"
