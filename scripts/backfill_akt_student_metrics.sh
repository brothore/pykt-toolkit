#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)_$$}"
LOGDIR="logs/akt_student_metric_backfill_${RUN_ID}"
mkdir -p "$LOGDIR"

# These are completed historical AKT batch-size runs. Keep them sequential so
# the active training pair retains most of the GPU memory.
MODEL_DIRS=(
  "saved_model/akt_abl_bs32/assist2009_akt_qid_saved_model/akt_abl_bs32_42_0_0.2_256_512_8_4_0.0001_32_100_0_0_1_student_id_3082_0.001_0.01_0"
  "saved_model/akt_abl_bs64/assist2009_akt_qid_saved_model/akt_abl_bs64_42_0_0.2_256_512_8_4_0.0001_64_100_0_0_1_student_id_3082_0.001_0.01_0"
  "saved_model/akt_abl_bs128/assist2009_akt_qid_saved_model/akt_abl_bs128_42_0_0.2_256_512_8_4_0.0001_128_100_0_0_1_student_id_3082_0.001_0.01_0"
)

for model_dir in "${MODEL_DIRS[@]}"; do
  name="$(basename "$(dirname "$model_dir")")"
  if [[ -f "$model_dir/overall_stats_output.json" && -f "$model_dir/all_results.json" ]]; then
    printf '%s skipping %s; metrics already exist\n' "$(date '+%F %T')" "$name" | tee -a "$LOGDIR/scheduler.log"
    continue
  fi
  printf '%s starting %s\n' "$(date '+%F %T')" "$name" | tee -a "$LOGDIR/scheduler.log"
  if env -u OMP_NUM_THREADS -u MKL_NUM_THREADS \
    python -m examples.wandb_predict_with_unbalance --bz 64 --save_dir "$model_dir" --use_wandb 0 \
    > "$LOGDIR/$name.log" 2>&1; then
    printf '%s finished %s\n' "$(date '+%F %T')" "$name" | tee -a "$LOGDIR/scheduler.log"
  else
    printf '%s failed %s\n' "$(date '+%F %T')" "$name" | tee -a "$LOGDIR/scheduler.log"
  fi
done

python scripts/summarize_akt_also_results.py >> "$LOGDIR/scheduler.log" 2>&1
