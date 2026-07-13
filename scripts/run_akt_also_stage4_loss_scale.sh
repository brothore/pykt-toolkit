#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/akt_also_stage_common.sh stage4_loss_scale "${1:-}"

# For 2465 student groups and bs=64, the paper default is 38.515625.
run_experiment "loss_scale_half_bs64" --batch_size 64 "${ALSO_COMMON[@]}" --also_pi_lr 1e-4 --also_pi_decay 1e-2 --also_loss_scale 19.2578125 &
first_pid=$!
run_experiment "loss_scale_double_bs64" --batch_size 64 "${ALSO_COMMON[@]}" --also_pi_lr 1e-4 --also_pi_decay 1e-2 --also_loss_scale 77.03125 &
second_pid=$!
wait "$first_pid"
wait "$second_pid"
