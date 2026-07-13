#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)_$$}"
ROOT="akt_also_full_${RUN_ID}"
LOGDIR="logs/${ROOT}"
SAVEDIR="saved_model/${ROOT}"
mkdir -p "$LOGDIR" "$SAVEDIR"

# Do not overlap the corrected v2 baseline/pi_lr queue that was already started.
while pgrep -f '[r]un_akt_student_also_ablation.sh' >/dev/null; do
  printf '%s waiting for akt_student_also_v2 to finish\n' "$(date '+%F %T')" >> "$LOGDIR/scheduler.log"
  sleep 60
done

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
  python -m examples.wandb_akt_train "${COMMON[@]}" "$@" \
    --save_dir "$SAVEDIR/$name" > "$LOGDIR/$name.log" 2>&1
  printf '%s finished %s\n' "$(date '+%F %T')" "$name" >> "$LOGDIR/scheduler.log"
}

# Stage 1 extension: v2 already covers pi_lr=1e-4 and 3e-4 at bs=64.
run_experiment "s1_pilr3e5_bs64" --batch_size 64 "${ALSO_COMMON[@]}" --also_pi_lr 3e-5 --also_pi_decay 1e-2 &
stage1_a=$!
run_experiment "s1_pilr1e3_bs64" --batch_size 64 "${ALSO_COMMON[@]}" --also_pi_lr 1e-3 --also_pi_decay 1e-2 &
stage1_b=$!
wait "$stage1_a"
wait "$stage1_b"

# Stage 2: batch-size ablation. bs=64 baseline/ALSO come from v2.
run_experiment "s2_baseline_bs32" --batch_size 32 --use_also 0 &
stage2_a=$!
run_experiment "s2_also_bs32" --batch_size 32 "${ALSO_COMMON[@]}" --also_pi_lr 1e-4 --also_pi_decay 1e-2 &
stage2_b=$!
wait "$stage2_a"
wait "$stage2_b"
run_experiment "s2_baseline_bs128" --batch_size 128 --use_also 0
run_experiment "s2_also_bs128" --batch_size 128 "${ALSO_COMMON[@]}" --also_pi_lr 1e-4 --also_pi_decay 1e-2

# Stage 3: pi regularization around the paper default 1e-2.
run_experiment "s3_pidecay1e3_bs64" --batch_size 64 "${ALSO_COMMON[@]}" --also_pi_lr 1e-4 --also_pi_decay 1e-3 &
stage3_a=$!
run_experiment "s3_pidecay1e1_bs64" --batch_size 64 "${ALSO_COMMON[@]}" --also_pi_lr 1e-4 --also_pi_decay 1e-1 &
stage3_b=$!
wait "$stage3_a"
wait "$stage3_b"

# Stage 4: loss-scale ablation around the paper default n_groups/batch_size=38.515625.
run_experiment "s4_lossscale_half_bs64" --batch_size 64 "${ALSO_COMMON[@]}" --also_pi_lr 1e-4 --also_pi_decay 1e-2 --also_loss_scale 19.2578125 &
stage4_a=$!
run_experiment "s4_lossscale_double_bs64" --batch_size 64 "${ALSO_COMMON[@]}" --also_pi_lr 1e-4 --also_pi_decay 1e-2 --also_loss_scale 77.03125 &
stage4_b=$!
wait "$stage4_a"
wait "$stage4_b"

# Stage 5: optimistic momentum and the non-optimistic paper variant.
run_experiment "s5_alpha0_bs64" --batch_size 64 "${ALSO_COMMON[@]}" --also_pi_lr 1e-4 --also_pi_decay 1e-2 --also_alpha 0.0 &
stage5_a=$!
run_experiment "s5_alpha05_bs64" --batch_size 64 "${ALSO_COMMON[@]}" --also_pi_lr 1e-4 --also_pi_decay 1e-2 --also_alpha 0.5 &
stage5_b=$!
wait "$stage5_a"
wait "$stage5_b"
run_experiment "s5_descent_ascent_bs64" --batch_size 64 "${ALSO_COMMON[@]}" --also_pi_lr 1e-4 --also_pi_decay 1e-2 --also_mode descent-ascent
