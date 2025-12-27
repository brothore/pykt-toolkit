#!/bin/bash

# ==========================================
# 0. 全局环境与指令解析
# ==========================================

COMMAND=$1

# 检查参数
if [ -z "$COMMAND" ]; then
    echo "Usage: $0 {train|status|pause|stop|resume} [project_name_1] [project_name_2] ..."
    echo "  train  : Generate configs and start training agents."
    echo "  status : Check the status of all WandB projects."
    echo "  pause  : Pause cloud sweeps for specified project(s)."
    echo "  stop   : KILL local agents and Stop cloud sweeps for specified project(s)."
    echo "  resume : Resume cloud sweeps for specified project(s)."
    echo ""
    echo "Example: $0 stop assist2009_dkt_v1 assist2015_dkt_v1"
    exit 1
fi

# [核心修改] 移除第一个参数(COMMAND)，剩下的 $@ 就是项目列表
shift 

# WandB API Key (全局生效)
export WANDB_API_KEY="b2fd3c192e86f37e55450d3c8894511ff8bce88d"

# ==========================================
# 1. 参数与变量配置区域 (仅对 train 模式必须)
# ==========================================

# --- 基础配置 ---
DATASET_NAME="bridge2algebra2006"   
MODEL_NAME="qikt_mamba"            
            
FOLDS="0"           

# --- 版本标注 (Mark) ---
MARK="v1_test_fix" 

# --- 默认 Project Name ---
DEFAULT_PROJECT_NAME="${DATASET_NAME}_${MODEL_NAME}_${MARK}"

# --- 其他配置 ---
GPU_IDS="0"                 
BATCH_SIZE=256
SWEEP_START_ID=0            
SWEEP_END_ID=100            
EXECUTE_AGENTS=true         

# ==========================================
# 2. 功能函数定义
# ==========================================

run_status() {
    if [ ! -f "check_project_status.py" ]; then
        echo "[Error] check_project_status.py not found!"
        exit 1
    fi
    WANDB_SILENT=true python check_project_status.py
}

run_manage() {
    ACTION=$1
    CURRENT_PROJ=$2 # [修改] 接收具体的项目名作为参数

    echo "======================================================="
    echo "Executing: $ACTION on Project: $CURRENT_PROJ"
    echo "======================================================="

    if [ ! -f "manage_project.py" ]; then
        echo "[Error] manage_project.py not found!"
        exit 1
    fi

    # 调用 Python 脚本管理状态
    python manage_project.py --project "$CURRENT_PROJ" --action "$ACTION"
    echo "" # 打印空行分隔输出
}
run_train() {
    # 1. 将逗号分隔的 DATASET_NAME 替换为空格分隔，以便进行 for 循环
    # 例如 "assist2009,assist2015" -> "assist2009 assist2015"
    DATASET_LIST=${DATASET_NAME//,/ }

    # 2. 开始循环：针对每一个数据集单独执行一套流程
    for CURRENT_DATASET in $DATASET_LIST; do
        
        # 动态生成当前数据集对应的 Project Name
        ACTUAL_PROJ="${CURRENT_DATASET}_${MODEL_NAME}_${MARK}"
        
        # 定义工作目录
        WORK_DIR="./run_logs/${ACTUAL_PROJ}"

        if [ ! -d "$WORK_DIR" ]; then
            mkdir -p "$WORK_DIR"
        fi

        INIT_SWEEP_SCRIPT="${WORK_DIR}/1_init_sweeps.sh"  
        INIT_LOG_FILE="${WORK_DIR}/2_sweep_submission.log" 
        AGENT_RUN_SCRIPT="${WORK_DIR}/3_start_agents.sh"   

        echo "======================================================="
        echo "Pipeline Start: TRAINING"
        echo "Dataset      : $CURRENT_DATASET"
        echo "Project Name : $ACTUAL_PROJ"
        echo "Work Dir     : $WORK_DIR"
        echo "======================================================="

        # --- Step 1: Generate Configs ---
        # 注意：这里传给 generate_wandb.py 的 dataset_names 变成了单一的 CURRENT_DATASET
        echo "[Step 1] Generating WandB sweep configurations..."
        python generate_wandb.py \
            --project_name "$ACTUAL_PROJ" \
            --dataset_names "$CURRENT_DATASET" \
            --model_names "$MODEL_NAME" \
            --folds "$FOLDS" \
            --batch_size $BATCH_SIZE \
            --launch_file "$INIT_SWEEP_SCRIPT" \
            --save_dir_suffix "_$MARK" 

        if [ $? -ne 0 ]; then 
            echo "[Error] Failed at Step 1 for $CURRENT_DATASET"
            exit 1
        fi

        # --- Step 2: Submit Sweeps ---
        echo "[Step 2] Submitting sweeps to WandB..."
        sh "$INIT_SWEEP_SCRIPT" > "$INIT_LOG_FILE" 2>&1
        if [ $? -ne 0 ]; then 
            echo "[Error] Failed at Step 2 for $CURRENT_DATASET"
            exit 1
        fi

        # --- Step 3: Parse Logs ---
        echo "[Step 3] Parsing logs..."
        python all_start.py \
            "$INIT_LOG_FILE" "$AGENT_RUN_SCRIPT" "$SWEEP_START_ID" "$SWEEP_END_ID" \
            "$CURRENT_DATASET" "$MODEL_NAME" "$GPU_IDS" "$ACTUAL_PROJ" "$WORK_DIR"

        if [ $? -ne 0 ]; then 
            echo "[Error] Failed at Step 3 for $CURRENT_DATASET"
            exit 1
        fi

        # --- Step 4: Start Agents ---
        if [ "$EXECUTE_AGENTS" = true ]; then
            echo "[Step 4] Auto-starting agents..."
            chmod +x "$AGENT_RUN_SCRIPT"
            sh "$AGENT_RUN_SCRIPT"
            echo "[Success] Agents started for $ACTUAL_PROJ."
        else
            echo "[Finished] Run manually: sh $AGENT_RUN_SCRIPT"
        fi
        
        echo "" # 空行分隔不同数据集的日志
    done
}
# ==========================================
# 3. 主逻辑分支
# ==========================================

case "$COMMAND" in
    train)
        run_train
        ;;
    status)
        run_status
        ;;
    stop|pause|resume)
        # [核心逻辑] 判断是否有后续参数
        if [ $# -eq 0 ]; then
            # 如果没有参数，使用默认项目
            run_manage "$COMMAND" "$DEFAULT_PROJECT_NAME"
        else
            # 如果有参数，循环处理每一个项目
            for PROJ in "$@"; do
                run_manage "$COMMAND" "$PROJ"
            done
        fi
        ;;
    *)
        echo "Invalid command: $COMMAND"
        echo "Usage: $0 {train|status|pause|stop|resume} [project1] [project2] ..."
        exit 1
        ;;
esac