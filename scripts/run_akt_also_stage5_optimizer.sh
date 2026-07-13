#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/akt_also_stage_common.sh stage5_optimizer "${1:-}"

run_experiment "alpha0_bs64" --batch_size 64 "${ALSO_COMMON[@]}" --also_pi_lr 1e-4 --also_pi_decay 1e-2 --also_alpha 0.0 &
first_pid=$!
run_experiment "alpha05_bs64" --batch_size 64 "${ALSO_COMMON[@]}" --also_pi_lr 1e-4 --also_pi_decay 1e-2 --also_alpha 0.5 &
second_pid=$!
wait "$first_pid"
wait "$second_pid"
run_experiment "descent_ascent_bs64" --batch_size 64 "${ALSO_COMMON[@]}" --also_pi_lr 1e-4 --also_pi_decay 1e-2 --also_mode descent-ascent
