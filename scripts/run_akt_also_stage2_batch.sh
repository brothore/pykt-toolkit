#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/akt_also_stage_common.sh stage2_batch "${1:-}"

# bs=64 is supplied by stage 0/1. bs=32 fits safely as a pair; bs=128 runs alone.
run_experiment "baseline_bs32" --batch_size 32 --use_also 0 &
first_pid=$!
run_experiment "also_bs32" --batch_size 32 "${ALSO_COMMON[@]}" --also_pi_lr 1e-4 --also_pi_decay 1e-2 &
second_pid=$!
wait "$first_pid"
wait "$second_pid"
run_experiment "baseline_bs128" --batch_size 128 --use_also 0
run_experiment "also_bs128" --batch_size 128 "${ALSO_COMMON[@]}" --also_pi_lr 1e-4 --also_pi_decay 1e-2
