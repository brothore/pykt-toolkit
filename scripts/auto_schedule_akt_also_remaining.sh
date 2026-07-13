#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
RUNNER_PID="${1:?usage: $0 <active-full-runner-pid> [run-id]}"
RUN_ID="${2:-$(date +%Y%m%d_%H%M%S)_$$}"
LOGDIR="logs/akt_also_auto_scheduler_${RUN_ID}"
mkdir -p "$LOGDIR"

printf '%s waiting for active runner pid=%s\n' "$(date '+%F %T')" "$RUNNER_PID" >> "$LOGDIR/scheduler.log"
while kill -0 "$RUNNER_PID" 2>/dev/null; do
  printf '%s active runner still owns the GPU\n' "$(date '+%F %T')" >> "$LOGDIR/scheduler.log"
  sleep 60
done

# The active runner was started before scripts were split. It covers stages
# 2-5 but assumes an older failed queue covered these two pi_lr values.
printf '%s active runner finished; launching missing stage 1 values\n' "$(date '+%F %T')" >> "$LOGDIR/scheduler.log"
bash scripts/run_akt_also_stage1_pi_lr_remaining.sh "auto_${RUN_ID}" \
  >> "$LOGDIR/stage1_remaining.log" 2>&1

python scripts/summarize_akt_also_results.py >> "$LOGDIR/scheduler.log" 2>&1
printf '%s automatic scheduling complete\n' "$(date '+%F %T')" >> "$LOGDIR/scheduler.log"
