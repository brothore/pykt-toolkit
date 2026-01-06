#!/bin/bash

# ==========================================
# 本地调参任务管理器 (Offline Scheduler V2)
# ==========================================

COMMAND=$1
shift 

if [ -z "$COMMAND" ]; then
    echo "Usage: $0 {start|status|stop} [model_name] [gpu_id]"
    exit 1
fi

LOG_BASE_DIR="./scheduler_logs" # 这是调度器自己的运行日志
mkdir -p "$LOG_BASE_DIR"

# ------------------------------------------
# 1. Start (无变化，和之前一样启动)
# ------------------------------------------
run_start() {
    MODEL_NAME=$1
    # 【修改】第二个参数现在是 DATASET_NAME
    DATASET_NAME=$2 
    # 【修改】第三个参数是 GPU_ID
    GPU_ID=${3:-0}

    # 检查参数
    if [ -z "$MODEL_NAME" ] || [ -z "$DATASET_NAME" ]; then
        echo "Error: Missing arguments."
        echo "Usage: $0 start <model_name> <dataset_name> [gpu_id]"
        exit 1
    fi

    PID_CHECK=$(pgrep -f "local_scheduler.py --model_name $MODEL_NAME")
    if [ ! -z "$PID_CHECK" ]; then
        echo "[Warning] Scheduler for '$MODEL_NAME' is running (PID: $PID_CHECK)."
        exit 1
    fi

    LOG_FILE="${LOG_BASE_DIR}/${MODEL_NAME}_scheduler.log"
    
    echo "Starting Scheduler for: $MODEL_NAME"
    echo "Dataset: $DATASET_NAME"
    echo "GPU: $GPU_ID"

    # 【修改】调用 Python 时传入 --dataset_name
    nohup python local_scheduler.py \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET_NAME" \
        --gpu "$GPU_ID" \
        > "$LOG_FILE" 2>&1 &
    
    echo "Started PID: $!"
    echo "Logs dir: offline_logs/${MODEL_NAME}/"
}
# ------------------------------------------
# 2. Status (优化显示)
# ------------------------------------------
run_status() {
    echo "=================================================="
    echo " Active Schedulers Status"
    echo "=================================================="
    
    PIDS=$(pgrep -f "local_scheduler.py")
    
    if [ -z "$PIDS" ]; then
        echo "No schedulers running."
    else
        for pid in $PIDS; do
            CMD=$(ps -p $pid -o args=)
            MODEL=$(echo $CMD | grep -oP '(?<=--model_name )[^ ]+')
            GPU=$(echo $CMD | grep -oP '(?<=--gpu )[^ ]+')
            SCHEDULER_LOG="${LOG_BASE_DIR}/${MODEL}_scheduler.log"
            
            echo "Model : $MODEL"
            echo "PID   : $pid"
            echo "GPU   : ${GPU:-Default}"
            
            # 从调度器日志中提取最后一句 "RUNNING: ..."
            # 这能告诉我们当前正在跑哪组参数
            if [ -f "$SCHEDULER_LOG" ]; then
                # 提取最后出现的 "RUNNING" 行，并只显示一部分防止太长
                LAST_RUN=$(grep "RUNNING:" "$SCHEDULER_LOG" | tail -n 1)
                if [ ! -z "$LAST_RUN" ]; then
                     echo "Job   : $LAST_RUN"
                else
                     echo "Job   : (Initializing or waiting...)"
                fi
                
                # 显示进度
                PROGRESS=$(grep "completed" "$SCHEDULER_LOG" | tail -n 1)
                if [ ! -z "$PROGRESS" ]; then
                    echo "Info  : $PROGRESS"
                fi
            fi
            echo "--------------------------------------------------"
        done
    fi
}

# ------------------------------------------
# 3. Stop (无变化)
# ------------------------------------------
run_stop() {
    MODEL_NAME=$1
    if [ -z "$MODEL_NAME" ]; then
        echo "Error: Model name required."
        exit 1
    fi
    PIDS=$(pgrep -f "local_scheduler.py --model_name $MODEL_NAME")
    if [ -z "$PIDS" ]; then
        echo "No running scheduler found for '$MODEL_NAME'."
    else
        for pid in $PIDS; do
            kill $pid
            echo "Killed PID: $pid"
        done
    fi
}

case "$COMMAND" in
    start|train) run_start "$@" ;;
    status) run_status ;;
    stop) run_stop "$@" ;;
    *) echo "Invalid command." ;;
esac