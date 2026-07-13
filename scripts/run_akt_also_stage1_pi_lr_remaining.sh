#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/akt_also_stage_common.sh stage1_pi_lr "${1:-}"

# The currently running stage covers 3e-5 and 1e-3. This script fills the middle values.
run_experiment "pilr1e4_bs64" --batch_size 64 "${ALSO_COMMON[@]}" --also_pi_lr 1e-4 --also_pi_decay 1e-2 &
first_pid=$!
run_experiment "pilr3e4_bs64" --batch_size 64 "${ALSO_COMMON[@]}" --also_pi_lr 3e-4 --also_pi_decay 1e-2 &
second_pid=$!
wait "$first_pid"
wait "$second_pid"
