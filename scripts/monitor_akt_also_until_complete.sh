#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)_$$}"
LOGDIR="logs/akt_also_completion_monitor_${RUN_ID}"
LOGFILE="$LOGDIR/monitor.log"
export TS_SOCKET="$PWD/logs/akt_also_tsp.socket"
mkdir -p "$LOGDIR"

iteration=0
while true; do
  iteration=$((iteration + 1))
  {
    printf '\n===== iteration=%s time=%s =====\n' "$iteration" "$(date '+%F %T')"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
    pgrep -af 'wandb_akt_train|wandb_predict_with_unbalance|auto_schedule_akt' || true
    printf '%s\n' '-- errors --'
    rg -n 'Traceback|CUDA out of memory|Prediction failed|ModuleNotFoundError|NameError' \
      logs/akt_also_stage1_pi_lr_auto_* logs/akt_also_tsp_* 2>/dev/null | tail -n 40 || true
    printf '%s\n' '-- tsp --'
    tsp -l || true
    python scripts/summarize_akt_also_results.py
    python scripts/write_akt_also_result_report.py
  } >> "$LOGFILE" 2>&1

  active_train=0
  pgrep -f '[p]ython -m examples.wandb_akt_train' >/dev/null && active_train=1
  active_predict=0
  pgrep -f '[p]ython -m examples.wandb_predict_with_unbalance' >/dev/null && active_predict=1
  active_scheduler=0
  pgrep -f '[a]uto_schedule_akt_also_tsp_stages2_to5.sh' >/dev/null && active_scheduler=1
  tsp_active=0
  tsp -l | rg -q 'running|queued' && tsp_active=1 || true
  if [[ "$active_train" -eq 0 && "$active_predict" -eq 0 && "$active_scheduler" -eq 0 && "$tsp_active" -eq 0 ]]; then
    printf '\n===== COMPLETE time=%s =====\n' "$(date '+%F %T')" >> "$LOGFILE"
    exit 0
  fi

  if [[ "$iteration" -lt 3 ]]; then
    sleep 60
  else
    sleep 600
  fi
done
