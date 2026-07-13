#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this helper from a stage script."
  exit 1
fi

STAGE_NAME="$1"
RUN_ID="${2:-$(date +%Y%m%d_%H%M%S)_$$}"
ROOT="akt_also_${STAGE_NAME}_${RUN_ID}"
LOGDIR="logs/${ROOT}"
SAVEDIR="saved_model/${ROOT}"
mkdir -p "$LOGDIR" "$SAVEDIR"

COMMON=(
  --dataset_name assist2009 --model_name akt --emb_type qid
  --seed 42 --fold 0 --dropout 0.2 --d_model 256
  --learning_rate 1e-4 --num_epochs 100
  --use_wandb 0 --add_uuid 0 --use_trained 0
)
ALSO_COMMON=(--use_also 1 --also_grouping_mode student_id)

run_experiment() {
  local name="$1"
  shift
  printf '%s starting %s\n' "$(date '+%F %T')" "$name" >> "$LOGDIR/scheduler.log"
  if python -m examples.wandb_akt_train "${COMMON[@]}" "$@" \
    --save_dir "$SAVEDIR/$name" > "$LOGDIR/$name.log" 2>&1; then
    printf '%s finished %s\n' "$(date '+%F %T')" "$name" >> "$LOGDIR/scheduler.log"
  else
    printf '%s failed %s\n' "$(date '+%F %T')" "$name" >> "$LOGDIR/scheduler.log"
  fi
}
