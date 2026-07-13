#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
STAGE1_RUN_ID="${1:?usage: $0 <stage1-run-id> [run-id]}"
RUN_ID="${2:-$(date +%Y%m%d_%H%M%S)_$$}"
LOGDIR="logs/akt_also_tsp_scheduler_${RUN_ID}"
export TS_SOCKET="$PWD/logs/akt_also_tsp.socket"
mkdir -p "$LOGDIR"

stage1_pattern="[p]ython -m examples.wandb_akt_train.*akt_also_stage1_pi_lr_${STAGE1_RUN_ID}"
printf '%s waiting for stage-1 run %s\n' "$(date '+%F %T')" "$STAGE1_RUN_ID" >> "$LOGDIR/scheduler.log"
while pgrep -f "$stage1_pattern" >/dev/null; do
  printf '%s stage 1 is still using the GPU\n' "$(date '+%F %T')" >> "$LOGDIR/scheduler.log"
  sleep 60
done

tsp -S 2 >> "$LOGDIR/scheduler.log" 2>&1

enqueue() {
  local stage="$1"
  local name="$2"
  shift 2
  tsp -L "akt-also-${stage}-${name}" \
    bash scripts/tsp_run_akt_also_config.sh "$stage" "tsp_${RUN_ID}_${stage}" "$name" "$@"
}

wait_jobs() {
  local job_id
  for job_id in "$@"; do
    tsp -w "$job_id" >> "$LOGDIR/scheduler.log" 2>&1
  done
}

# Each pair is enqueued to the same TSP queue. TS_SLOTS=2 caps GPU concurrency.
job_a="$(enqueue stage2_batch baseline_bs32 --batch_size 32 --use_also 0)"
job_b="$(enqueue stage2_batch also_bs32 --batch_size 32 --use_also 1 --also_grouping_mode student_id --also_pi_lr 1e-4 --also_pi_decay 1e-2)"
wait_jobs "$job_a" "$job_b"
job_a="$(enqueue stage2_batch baseline_bs128 --batch_size 128 --use_also 0)"
wait_jobs "$job_a"
job_a="$(enqueue stage2_batch also_bs128 --batch_size 128 --use_also 1 --also_grouping_mode student_id --also_pi_lr 1e-4 --also_pi_decay 1e-2)"
wait_jobs "$job_a"

job_a="$(enqueue stage3_pi_decay pidecay1e3_bs64 --batch_size 64 --use_also 1 --also_grouping_mode student_id --also_pi_lr 1e-4 --also_pi_decay 1e-3)"
job_b="$(enqueue stage3_pi_decay pidecay1e1_bs64 --batch_size 64 --use_also 1 --also_grouping_mode student_id --also_pi_lr 1e-4 --also_pi_decay 1e-1)"
wait_jobs "$job_a" "$job_b"

job_a="$(enqueue stage4_loss_scale loss_scale_half_bs64 --batch_size 64 --use_also 1 --also_grouping_mode student_id --also_pi_lr 1e-4 --also_pi_decay 1e-2 --also_loss_scale 19.2578125)"
job_b="$(enqueue stage4_loss_scale loss_scale_double_bs64 --batch_size 64 --use_also 1 --also_grouping_mode student_id --also_pi_lr 1e-4 --also_pi_decay 1e-2 --also_loss_scale 77.03125)"
wait_jobs "$job_a" "$job_b"

job_a="$(enqueue stage5_optimizer alpha0_bs64 --batch_size 64 --use_also 1 --also_grouping_mode student_id --also_pi_lr 1e-4 --also_pi_decay 1e-2 --also_alpha 0.0)"
job_b="$(enqueue stage5_optimizer alpha05_bs64 --batch_size 64 --use_also 1 --also_grouping_mode student_id --also_pi_lr 1e-4 --also_pi_decay 1e-2 --also_alpha 0.5)"
wait_jobs "$job_a" "$job_b"
job_a="$(enqueue stage5_optimizer descent_ascent_bs64 --batch_size 64 --use_also 1 --also_grouping_mode student_id --also_pi_lr 1e-4 --also_pi_decay 1e-2 --also_mode descent-ascent)"
wait_jobs "$job_a"

python scripts/summarize_akt_also_results.py >> "$LOGDIR/scheduler.log" 2>&1
printf '%s stages 2-5 complete\n' "$(date '+%F %T')" >> "$LOGDIR/scheduler.log"
