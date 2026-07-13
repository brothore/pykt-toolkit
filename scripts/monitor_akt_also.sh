#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)_$$}"
LOGFILE="logs/akt_also_monitor_${RUN_ID}.log"
mkdir -p logs

while true; do
  printf '\n===== %s =====\n' "$(date '+%F %T')" >> "$LOGFILE"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader >> "$LOGFILE" 2>&1
  pgrep -af 'wandb_akt_train|run_akt_.*ablation' >> "$LOGFILE" 2>&1 || true
  for log in logs/akt_student_also_v2/*.log logs/akt_also_full_*/*.log; do
    [ -f "$log" ] || continue
    printf '\n[%s]\n' "$log" >> "$LOGFILE"
    rg 'Epoch:|Training Finished|Traceback|CUDA out of memory|overall_dataset_auc|student_stats_mean|student_stats_std|student_stats_range' "$log" | tail -n 3 >> "$LOGFILE" || true
  done
  sleep 60
done
