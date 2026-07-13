#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

while pgrep -f '[r]un_akt_also_full_ablation.sh' >/dev/null; do
  sleep 60
done

exec bash scripts/run_akt_also_full_ablation.sh --skip-stage1
