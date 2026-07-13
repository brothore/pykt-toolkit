#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)_$$}"
LOGDIR="logs/akt_also_tsp_enqueue_${RUN_ID}"
export TS_SOCKET="$PWD/logs/akt_also_tsp.socket"
mkdir -p "$LOGDIR"
tsp -S 2 > "$LOGDIR/tsp_slots.log" 2>&1

enqueue() {
  local label="$1"
  shift
  tsp -L "akt-also-${label}" "$@"
}

enqueue_after() {
  local dependency="$1"
  local label="$2"
  shift 2
  tsp -D "$dependency" -L "akt-also-${label}" "$@"
}

barrier() {
  local label="$1"
  local dependency="$2"
  shift 2
  enqueue_after "$dependency" "$label" bash scripts/tsp_wait_jobs.sh "$@"
}

# Stage 2: safe pair at bs=32, then the two memory-heavy bs=128 jobs.
s2_base32="$(enqueue s2-baseline-bs32 bash scripts/tsp_run_akt_also_config.sh stage2_batch "tsp_${RUN_ID}_s2" baseline_bs32 --batch_size 32 --use_also 0)"
s2_also32="$(enqueue s2-also-bs32 bash scripts/tsp_run_akt_also_config.sh stage2_batch "tsp_${RUN_ID}_s2" also_bs32 --batch_size 32 --use_also 1 --also_grouping_mode student_id --also_pi_lr 1e-4 --also_pi_decay 1e-2)"
s2_pair_done="$(barrier s2-bs32-done "$s2_base32" "$s2_base32" "$s2_also32")"
s2_base128="$(enqueue_after "$s2_pair_done" s2-baseline-bs128 bash scripts/tsp_run_akt_also_config.sh stage2_batch "tsp_${RUN_ID}_s2" baseline_bs128 --batch_size 128 --use_also 0)"
s2_also128="$(enqueue_after "$s2_base128" s2-also-bs128 bash scripts/tsp_run_akt_also_config.sh stage2_batch "tsp_${RUN_ID}_s2" also_bs128 --batch_size 128 --use_also 1 --also_grouping_mode student_id --also_pi_lr 1e-4 --also_pi_decay 1e-2)"

# Stages 3-5 are pairs behind barriers so only two configurations run together.
s3_a="$(enqueue_after "$s2_also128" s3-pidecay1e3 bash scripts/tsp_run_akt_also_config.sh stage3_pi_decay "tsp_${RUN_ID}_s3" pidecay1e3_bs64 --batch_size 64 --use_also 1 --also_grouping_mode student_id --also_pi_lr 1e-4 --also_pi_decay 1e-3)"
s3_b="$(enqueue_after "$s2_also128" s3-pidecay1e1 bash scripts/tsp_run_akt_also_config.sh stage3_pi_decay "tsp_${RUN_ID}_s3" pidecay1e1_bs64 --batch_size 64 --use_also 1 --also_grouping_mode student_id --also_pi_lr 1e-4 --also_pi_decay 1e-1)"
s3_done="$(barrier s3-done "$s3_a" "$s3_a" "$s3_b")"
s4_a="$(enqueue_after "$s3_done" s4-loss-half bash scripts/tsp_run_akt_also_config.sh stage4_loss_scale "tsp_${RUN_ID}_s4" loss_scale_half_bs64 --batch_size 64 --use_also 1 --also_grouping_mode student_id --also_pi_lr 1e-4 --also_pi_decay 1e-2 --also_loss_scale 19.2578125)"
s4_b="$(enqueue_after "$s3_done" s4-loss-double bash scripts/tsp_run_akt_also_config.sh stage4_loss_scale "tsp_${RUN_ID}_s4" loss_scale_double_bs64 --batch_size 64 --use_also 1 --also_grouping_mode student_id --also_pi_lr 1e-4 --also_pi_decay 1e-2 --also_loss_scale 77.03125)"
s4_done="$(barrier s4-done "$s4_a" "$s4_a" "$s4_b")"
s5_a="$(enqueue_after "$s4_done" s5-alpha0 bash scripts/tsp_run_akt_also_config.sh stage5_optimizer "tsp_${RUN_ID}_s5" alpha0_bs64 --batch_size 64 --use_also 1 --also_grouping_mode student_id --also_pi_lr 1e-4 --also_pi_decay 1e-2 --also_alpha 0.0)"
s5_b="$(enqueue_after "$s4_done" s5-alpha05 bash scripts/tsp_run_akt_also_config.sh stage5_optimizer "tsp_${RUN_ID}_s5" alpha05_bs64 --batch_size 64 --use_also 1 --also_grouping_mode student_id --also_pi_lr 1e-4 --also_pi_decay 1e-2 --also_alpha 0.5)"
s5_done="$(barrier s5-alpha-done "$s5_a" "$s5_a" "$s5_b")"
s5_final="$(enqueue_after "$s5_done" s5-descent-ascent bash scripts/tsp_run_akt_also_config.sh stage5_optimizer "tsp_${RUN_ID}_s5" descent_ascent_bs64 --batch_size 64 --use_also 1 --also_grouping_mode student_id --also_pi_lr 1e-4 --also_pi_decay 1e-2 --also_mode descent-ascent)"

printf 'stage2_bs32=%s,%s\nstage2_bs128=%s,%s\nstage3=%s,%s\nstage4=%s,%s\nstage5=%s,%s,%s\n' \
  "$s2_base32" "$s2_also32" "$s2_base128" "$s2_also128" "$s3_a" "$s3_b" "$s4_a" "$s4_b" "$s5_a" "$s5_b" "$s5_final" \
  | tee "$LOGDIR/job_ids.txt"
tsp -l | tee "$LOGDIR/queue_after_enqueue.txt"
