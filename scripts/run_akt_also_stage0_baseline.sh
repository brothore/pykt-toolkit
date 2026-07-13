#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/akt_also_stage_common.sh stage0_baseline "${1:-}"

run_experiment "baseline_bs64" --batch_size 64 --use_also 0
