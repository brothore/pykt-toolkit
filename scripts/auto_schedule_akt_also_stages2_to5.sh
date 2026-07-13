#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
STAGE1_RUN_ID="${1:?usage: $0 <stage1-run-id> [run-id]}"
RUN_ID="${2:-$(date +%Y%m%d_%H%M%S)_$$}"
LOGDIR="logs/akt_also_auto_stages2_to5_${RUN_ID}"
mkdir -p "$LOGDIR"

stage1_pattern="[p]ython -m examples.wandb_akt_train.*akt_also_stage1_pi_lr_${STAGE1_RUN_ID}"
printf '%s waiting for stage-1 run %s\n' "$(date '+%F %T')" "$STAGE1_RUN_ID" >> "$LOGDIR/scheduler.log"
while pgrep -f "$stage1_pattern" >/dev/null; do
  printf '%s stage 1 is still using the GPU\n' "$(date '+%F %T')" >> "$LOGDIR/scheduler.log"
  sleep 60
done

for stage in \
  run_akt_also_stage2_batch.sh \
  run_akt_also_stage3_pi_decay.sh \
  run_akt_also_stage4_loss_scale.sh \
  run_akt_also_stage5_optimizer.sh; do
  stage_name="${stage%.sh}"
  printf '%s starting %s\n' "$(date '+%F %T')" "$stage_name" >> "$LOGDIR/scheduler.log"
  bash "scripts/$stage" "auto_${RUN_ID}_${stage_name}" \
    > "$LOGDIR/$stage_name.log" 2>&1
  printf '%s finished %s\n' "$(date '+%F %T')" "$stage_name" >> "$LOGDIR/scheduler.log"
done

python scripts/summarize_akt_also_results.py >> "$LOGDIR/scheduler.log" 2>&1
printf '%s stages 2-5 complete\n' "$(date '+%F %T')" >> "$LOGDIR/scheduler.log"
