#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOGDIR="logs/akt_student_also_v2"
mkdir -p "$LOGDIR"

COMMON=(
  --dataset_name assist2009 --model_name akt --emb_type qid
  --seed 42 --fold 0 --dropout 0.2 --d_model 256
  --learning_rate 1e-4 --batch_size 64 --num_epochs 100
  --use_wandb 0 --add_uuid 0 --use_trained 0
)

python examples/wandb_akt_train.py "${COMMON[@]}" \
  --use_also 0 \
  --save_dir saved_model/akt_student_also_v2_baseline_bs64 \
  > "$LOGDIR/baseline_bs64.log" 2>&1

python examples/wandb_akt_train.py "${COMMON[@]}" \
  --use_also 1 --also_grouping_mode student_id \
  --also_pi_lr 1e-4 --also_pi_decay 1e-2 \
  --save_dir saved_model/akt_student_also_v2_pilr1e4_bs64 \
  > "$LOGDIR/also_student_pilr1e4_bs64.log" 2>&1

python examples/wandb_akt_train.py "${COMMON[@]}" \
  --use_also 1 --also_grouping_mode student_id \
  --also_pi_lr 3e-4 --also_pi_decay 1e-2 \
  --save_dir saved_model/akt_student_also_v2_pilr3e4_bs64 \
  > "$LOGDIR/also_student_pilr3e4_bs64.log" 2>&1
