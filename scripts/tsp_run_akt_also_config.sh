#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
STAGE_NAME="${1:?stage name is required}"
RUN_ID="${2:?run id is required}"
NAME="${3:?configuration name is required}"
shift 3

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/akt_also_stage_common.sh "$STAGE_NAME" "$RUN_ID"
run_experiment "$NAME" "$@"
