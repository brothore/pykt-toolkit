#!/bin/bash
set -e
cd /root/autodl-tmp/pykt-toolkit

LOGDIR="logs/dkt_ablation"

BASE="--dataset_name assist2009 --model_name dkt --emb_type qid --seed 42 --fold 0 --dropout 0.2 --emb_size 200 --learning_rate 1e-3 --num_epochs 100 --use_wandb 0 --add_uuid 0 --use_trained 0 --use_also 1 --also_grouping_mode student_id"

echo "=== Phase A batch 2: A4(n=500) + A5(bs=64, n=3082) ==="

# A4: n_groups=500, batch_size=32
nohup python -m examples.wandb_dkt_train $BASE --also_n_groups 500 --batch_size 32 --save_dir saved_model/dkt_abl_a4_n500 > "$LOGDIR/a4_n500_bs32_pilr1e3_pid1e2.log" 2>&1 &
echo "A4 PID=$!"

# A5: n_groups=3082, batch_size=64
nohup python -m examples.wandb_dkt_train $BASE --also_n_groups 3082 --batch_size 64 --save_dir saved_model/dkt_abl_a5_n3082_bs64 > "$LOGDIR/a5_n3082_bs64_pilr1e3_pid1e2.log" 2>&1 &
echo "A5 PID=$!"

echo ""
echo "Batch 2 launched. Monitor:"
echo "  tail -f $LOGDIR/a4_n500_bs32_pilr1e3_pid1e2.log"
echo "  tail -f $LOGDIR/a5_n3082_bs64_pilr1e3_pid1e2.log"
