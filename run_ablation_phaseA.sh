#!/bin/bash
# Phase A: n_groups 消融 + batch_size 对照
set -e
cd /root/autodl-tmp/pykt-toolkit

LOGDIR="logs/dkt_ablation"
mkdir -p "$LOGDIR"

BASE="--dataset_name assist2009 --model_name dkt --emb_type qid --seed 42 --fold 0 --dropout 0.2 --emb_size 200 --learning_rate 1e-3 --batch_size 32 --num_epochs 100 --use_wandb 0 --add_uuid 0 --use_trained 0 --use_also 1 --also_grouping_mode student_id"

echo "=== Phase A: n_groups ablation ==="
echo "Launching batch 1: A1(50) A2(100) A3(200)"

# A1: n_groups=50
nohup python -m examples.wandb_dkt_train $BASE --also_n_groups 50 --save_dir saved_model/dkt_abl_a1_n50 > "$LOGDIR/a1_n50_bs32_pilr1e3_pid1e2.log" 2>&1 &
echo "A1 PID=$!"

# A2: n_groups=100
nohup python -m examples.wandb_dkt_train $BASE --also_n_groups 100 --save_dir saved_model/dkt_abl_a2_n100 > "$LOGDIR/a2_n100_bs32_pilr1e3_pid1e2.log" 2>&1 &
echo "A2 PID=$!"

# A3: n_groups=200
nohup python -m examples.wandb_dkt_train $BASE --also_n_groups 200 --save_dir saved_model/dkt_abl_a3_n200 > "$LOGDIR/a3_n200_bs32_pilr1e3_pid1e2.log" 2>&1 &
echo "A3 PID=$!"

echo ""
echo "Batch 1 launched. Waiting for them to finish..."
echo "Monitor: tail -f $LOGDIR/a1_n50_bs32_pilr1e3_pid1e2.log"
echo "         tail -f $LOGDIR/a2_n100_bs32_pilr1e3_pid1e2.log"
echo "         tail -f $LOGDIR/a3_n200_bs32_pilr1e3_pid1e2.log"
