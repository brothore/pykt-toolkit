#!/bin/bash

# ==============================================================================
#                             用户配置区域
# ==============================================================================

# 1. 基础配置
export WANDB_API_KEY="b2fd3c192e86f37e55450d3c8894511ff8bce88d"
BASE_PROJECT_NAME="kt_toolkits"    # 基础项目名
TAG="v1"                           # 自定义字符串标记

# 2. 数据与模型
DATASETS="assist2009"
MODELS="qikt_mamba" 
FOLDS="0"
GPU_IDS="0"

# 3. 高级参数
BATCH_SIZE=128
EMB_TYPE="qid"
SWEEP_START_ID=0
SWEEP_END_ID=5

# ==============================================================================
#                       动态命名生成逻辑
# ==============================================================================

SAFE_DATASETS=$(echo $DATASETS | tr ',' '-')
SAFE_MODELS=$(echo $MODELS | tr ',' '-')
PROJECT_NAME="${BASE_PROJECT_NAME}_${SAFE_DATASETS}_${SAFE_MODELS}_${TAG}"

LAUNCH_FILE_TRAIN="launch_train_${PROJECT_NAME}.sh"
TRAIN_LOG="log_train_${PROJECT_NAME}.log"
LAUNCH_FILE_PRED="launch_pred_${PROJECT_NAME}.sh"
PRED_LOG="log_pred_${PROJECT_NAME}.log"

# ==============================================================================
#                                   脚本逻辑
# ==============================================================================

MODE=$1
ACTION=$2  # 用于 control 模式的动作
TARGET_ID=$3 # 用于 control 模式的目标 ID

if [[ -z "$MODE" ]]; then
    echo "❌ 错误: 请指定运行模式。"
    echo "用法: bash pipeline.sh [train | eval | status | pause | resume | stop | kill-local]"
    exit 1
fi

echo "========================================================"
echo "   PyKT 动态流水线 -> [ $MODE ]"
echo "   🎯 项目: $PROJECT_NAME"
echo "========================================================"

# ------------------------------------------------------------------------------
#  [修复版] 辅助函数：生成 WandB 管理脚本
# ------------------------------------------------------------------------------
generate_manage_script() {
    cat <<EOF > auto_manage_wandb.py
import sys
import os
import warnings
import subprocess

warnings.filterwarnings("ignore")

try:
    import wandb
except ImportError:
    print("❌ 错误: 找不到 wandb 库")
    sys.exit(1)

# 配置
api_key = "$WANDB_API_KEY"
project = "$PROJECT_NAME"
entity = None 

# 尝试获取 Entity
try:
    api = wandb.Api(overrides={"project": project})
    if hasattr(api, 'default_entity') and api.default_entity:
        entity = api.default_entity
    else:
        try:
            entity = api.viewer.entity
        except:
            try:
                entity = api.viewer().entity
            except:
                pass
except:
    pass

mode = sys.argv[1] 
ids = sys.argv[2:]

if mode == "status":
    try:
        sweeps = api.project(project, entity=entity).sweeps()
        print(f"\n📊 项目 [{project}] 的 Sweep 状态列表:")
        
        # 定义列宽
        W_ID = 12
        W_STATE = 12  # 状态栏视觉宽度
        W_RUNS = 8
        W_DATE = 12
        
        print("-" * 90)
        # 打印表头
        print(f"{'Sweep ID':<{W_ID}} | {'State':<{W_STATE}} | {'Runs':<{W_RUNS}} | {'Created':<{W_DATE}} | {'Name/Config'}")
        print("-" * 90)
        
        found = False
        for sweep in sweeps:
            found = True
            state = sweep.state
            
            # 获取 Run Count
            try:
                run_count = sweep.run_count if hasattr(sweep, 'run_count') else len(list(sweep.runs))
            except:
                run_count = "?"

            created = sweep.created_at[:10] if hasattr(sweep, 'created_at') else "N/A"
            cfg_name = sweep.config.get('name', 'N/A')
            
            # --- 修复对齐逻辑 ---
            # 1. 先把纯文本填充到指定宽度 (例如 12 格)
            state_padded = f"{state:<{W_STATE}}"
            
            # 2. 再给填充好的文本上色
            # 这样颜色代码不会影响占位计算
            if state in ["RUNNING", "FINISHED"]: 
                # Green
                state_disp = f"\033[92m{state_padded}\033[0m"
            elif state == "PAUSED": 
                # Yellow
                state_disp = f"\033[93m{state_padded}\033[0m"
            elif state in ["CANCELED", "KILLED", "STOPPED", "FAILED"]: 
                # Red
                state_disp = f"\033[91m{state_padded}\033[0m"
            else:
                # No Color
                state_disp = state_padded

            # 3. 打印时，State 列不需要再指定宽度，因为步骤1已经填好了
            print(f"{sweep.id:<{W_ID}} | {state_disp} | {str(run_count):<{W_RUNS}} | {created:<{W_DATE}} | {cfg_name}")
        
        if not found:
            print("   (当前项目下没有找到 Sweep)")
        print("-" * 90)
    except Exception as e:
        print(f"❌ 获取状态失败: {e}")

elif mode in ["pause", "resume", "stop"]:
    if not ids:
        print("❌ Error: 缺少 Sweep ID")
        sys.exit(1)
    
    action_map = {
        "pause": "--pause",
        "resume": "--resume",
        "stop": "--stop"
    }
    flag = action_map[mode]

    for sweep_id in ids:
        if entity:
            full_id = f"{entity}/{project}/{sweep_id}"
        else:
            full_id = f"{project}/{sweep_id}"

        print(f"🔄 执行: wandb sweep {flag} {full_id}")
        exit_code = os.system(f"wandb sweep {flag} {full_id}")
        
        if exit_code == 0:
            print(f"✅ [{sweep_id}] 操作成功")
        else:
            print(f"⚠️ 带路径操作失败，尝试仅使用 ID: {sweep_id}")
            exit_code_retry = os.system(f"wandb sweep {flag} {sweep_id}")
            if exit_code_retry == 0:
                 print(f"✅ [{sweep_id}] 操作成功 (仅ID)")
            else:
                 print(f"❌ [{sweep_id}] 操作失败")
EOF
}
# ------------------------------------------------------------------------------
#  模式 1: 查看状态 (Status)
# ------------------------------------------------------------------------------
if [[ "$MODE" == "status" || "$MODE" == "status-all" ]]; then
    # 如果用户提供了第二个参数 (例如 bash pipeline.sh status my_project_v2)
    # 则使用用户指定的项目名，否则使用默认生成的 PROJECT_NAME
    TARGET_PROJ="${2:-$PROJECT_NAME}"
    
    generate_manage_script
    # 传递: 模式, 目标项目名, 基础项目名筛选词
    python auto_manage_wandb.py "$MODE" "$TARGET_PROJ" "$BASE_PROJECT_NAME"
    rm auto_manage_wandb.py

# ------------------------------------------------------------------------------
#  模式 2: 任务控制 (Pause / Resume / Stop)
# ------------------------------------------------------------------------------
elif [[ "$MODE" == "pause" || "$MODE" == "resume" || "$MODE" == "stop" ]]; then
    if [[ -z "$ACTION" ]]; then
        TARGET_ID=$2
    else
        TARGET_ID=$ACTION # 兼容 pipeline.sh pause id 格式
    fi

    if [[ -z "$2" ]]; then
        echo "❌ 错误: 请提供至少一个 Sweep ID"
        echo "示例: bash pipeline.sh stop kv5brmwq o8heh2k3"
        exit 1
    fi

    generate_manage_script
    # "${@:2}" 表示将从第2个位置开始的所有参数传给 Python (支持批量 ID)
    python auto_manage_wandb.py "$MODE" "${@:2}"
    rm auto_manage_wandb.py

# ------------------------------------------------------------------------------
#  模式 3: 强杀本地进程 (Kill Local)
# ------------------------------------------------------------------------------
elif [[ "$MODE" == "kill-local" ]]; then
    echo "⚠️  警告: 这将杀死所有与项目 [$PROJECT_NAME] 相关的 'wandb agent' 进程。"
    # 使用 pgrep 查找完整的命令行
    PIDS=$(pgrep -f "wandb agent.*$PROJECT_NAME")
    
    if [[ -z "$PIDS" ]]; then
        echo "✅ 未发现相关的本地运行进程。"
    else
        echo "Found PIDs: $PIDS"
        echo $PIDS | xargs kill -9
        echo "☠️  已强制杀死本地进程。"
    fi

# ------------------------------------------------------------------------------
#  原有模式: Train
# ------------------------------------------------------------------------------
elif [[ "$MODE" == "train" ]]; then
    echo "[Step 1] 生成 WandB 配置文件..."
    python generate_wandb.py \
        --project_name "$PROJECT_NAME" \
        --dataset_names "$DATASETS" \
        --model_names "$MODELS" \
        --emb_type "$EMB_TYPE" \
        --folds "$FOLDS" \
        --batch_size "$BATCH_SIZE" \
        --launch_file "$LAUNCH_FILE_TRAIN" \
        --save_dir_suffix "" 
    
    if [[ $? -ne 0 ]]; then echo "❌ 配置生成失败"; exit 1; fi

    if [[ -f "$LAUNCH_FILE_TRAIN" ]]; then
        sh "$LAUNCH_FILE_TRAIN" > "$TRAIN_LOG" 2>&1
        echo "✅ Sweep 注册日志: $TRAIN_LOG"
    fi
    
    sh run_all.sh "$TRAIN_LOG" "$SWEEP_START_ID" "$SWEEP_END_ID" "$DATASETS" "$MODELS" "$GPU_IDS" "$PROJECT_NAME"
    
    AGENT_SCRIPT="start_sweep_${SWEEP_START_ID}_${SWEEP_END_ID}.sh"
    if [[ -f "$AGENT_SCRIPT" ]]; then
        NEW_AGENT_SCRIPT="agent_train_${PROJECT_NAME}.sh"
        mv "$AGENT_SCRIPT" "$NEW_AGENT_SCRIPT"
        chmod +x "$NEW_AGENT_SCRIPT"
        echo "🚀 启动训练 Agent: $NEW_AGENT_SCRIPT"
        sh "$NEW_AGENT_SCRIPT"
    fi

# ------------------------------------------------------------------------------
#  原有模式: Eval
# ------------------------------------------------------------------------------
elif [[ "$MODE" == "eval" ]]; then
    echo "[Step 1] 提取最佳模型..."
    cat <<EOF > auto_extract_best.py
import sys
import os
import warnings
warnings.filterwarnings("ignore")
try:
    from examples.wandb_train import wandb_api 
except ImportError:
    import wandb_api 

dataset = "$DATASETS"
model = "$MODELS"
key = "$WANDB_API_KEY"
project_name = "$PROJECT_NAME"

print(f"提取中... 项目: {project_name}")
try:
    os.environ["WANDB_PROJECT"] = project_name
    df = wandb_api.get_best_run(dataset_name=dataset, model_name=model)
    wandb_api.extract_best_models(df, dataset, model, 
                                  fpath="./seedwandb/predict.yaml", 
                                  wandb_key=key,
                                  launch_file="$LAUNCH_FILE_PRED")
    print("✅ 提取成功")
except Exception as e:
    print(f"❌ 提取失败: {e}")
    sys.exit(1)
EOF
    python auto_extract_best.py
    if [[ $? -ne 0 ]]; then exit 1; fi
    rm auto_extract_best.py

    echo "[Step 2] 注册预测任务..."
    if [[ -f "$LAUNCH_FILE_PRED" ]]; then
        sh "$LAUNCH_FILE_PRED" > "$PRED_LOG" 2>&1
    fi

    echo "[Step 3] 启动预测 Agent..."
    sh run_all.sh "$PRED_LOG" "$SWEEP_START_ID" "$SWEEP_END_ID" "$DATASETS" "$MODELS" "$GPU_IDS" "$PROJECT_NAME"

    AGENT_SCRIPT="start_sweep_${SWEEP_START_ID}_${SWEEP_END_ID}.sh"
    if [[ -f "$AGENT_SCRIPT" ]]; then
        NEW_AGENT_SCRIPT="agent_pred_${PROJECT_NAME}.sh"
        mv "$AGENT_SCRIPT" "$NEW_AGENT_SCRIPT"
        chmod +x "$NEW_AGENT_SCRIPT"
        sh "$NEW_AGENT_SCRIPT"
    else
        echo "❌ Agent 脚本未生成"
    fi

else
    echo "❌ 未知模式: $MODE"
fi