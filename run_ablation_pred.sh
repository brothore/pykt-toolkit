#!/bin/bash
# 并行补跑所有消融实验的预测
cd /root/autodl-tmp/pykt-toolkit/examples
LOGDIR="../logs/dkt_ablation"

# 找每个实验的 save_dir 并用独立日志后台跑预测
for name in a1_n50 a2_n100 a3_n200 a4_n500 a5_n3082_bs64; do
    save_base=$(grep -oP "save_dir: \K.*" "../logs/dkt_ablation/${name}_bs32_pilr1e3_pid1e2.log" 2>/dev/null | head -1)
    [ -z "$save_base" ] && save_base=$(grep -oP "save_dir=\K[^ ]*" "../logs/dkt_ablation/${name}_bs32_pilr1e3_pid1e2.log" 2>/dev/null | head -1)
    
    if [ -z "$save_base" ]; then
        # 从日志找完整路径
        full=$(grep -oP 'save_dir: [^ ]+' "../logs/dkt_ablation/${name}_bs32_pilr1e3_pid1e2.log" 2>/dev/null | tail -1 | cut -d' ' -f2)
        [ -z "$full" ] && full=$(grep -oP '(?<=save_dir=)[^, ]+' "../logs/dkt_ablation/${name}_bs32_pilr1e3_pid1e2.log" 2>/dev/null | head -1)
        save_base="$full"
    fi
    
    if [ -n "$save_base" ] && [ -d "../$save_base" ]; then
        nohup python wandb_predict_with_unbalance.py \
            --bz 32 --fusion_type late_fusion \
            --save_dir "../$save_base" \
            --use_wandb 0 \
            > "${LOGDIR}/${name}_pred.log" 2>&1 &
        echo "Launched ${name}: PID=$! -> ${LOGDIR}/${name}_pred.log"
    else
        echo "SKIP ${name}: save_dir not found ($save_base)"
    fi
done
echo "All launched. Check: tail -f ${LOGDIR}/*_pred.log"
