#!/bin/bash

# ==============================================================================
#                             用户配置区域
# ==============================================================================

# 1. 基础配置
export WANDB_API_KEY="b2fd3c192e86f37e55450d3c8894511ff8bce88d"
BASE_PROJECT_NAME="kt_toolkits"    # 基础项目名 (用于过滤 status-all)
TAG="v1"                           # 自定义字符串标记

# 2. 数据与模型
DATASETS="nips_task34"
MODELS="qikt_mamba" 
FOLDS="0"
GPU_IDS="0"

# 3. 高级参数
BATCH_SIZE=128
EMB_TYPE="qid"
SWEEP_START_ID=0
SWEEP_END_ID=5
LOG_ROOT="run_logs"                # [新增] 所有日志文件的根目录

# ==============================================================================
#                       动态路径与命名逻辑
# ==============================================================================

SAFE_DATASETS=$(echo $DATASETS | tr ',' '-')
SAFE_MODELS=$(echo $MODELS | tr ',' '-')
PROJECT_NAME="${BASE_PROJECT_NAME}_${SAFE_DATASETS}_${SAFE_MODELS}_${TAG}"

# [新增] 创建专属项目文件夹
PROJECT_LOG_DIR="$LOG_ROOT/$PROJECT_NAME"
if [[ ! -d "$PROJECT_LOG_DIR" ]]; then
    mkdir -p "$PROJECT_LOG_DIR"
fi

# [修改] 所有文件路径都指向该文件夹
LAUNCH_FILE_TRAIN="$PROJECT_LOG_DIR/launch_train.sh"
TRAIN_LOG="$PROJECT_LOG_DIR/log_train.log"

LAUNCH_FILE_PRED="$PROJECT_LOG_DIR/launch_pred.sh"
PRED_LOG="$PROJECT_LOG_DIR/log_pred.log"

# ==============================================================================
#                                   脚本逻辑
# ==============================================================================

MODE=$1
# 支持多参数传递
ARGS=("${@:2}")

if [[ -z "$MODE" ]]; then
    echo "❌ 错误: 请指定运行模式。"
    echo "用法:"
    echo "  1. 训练/评估: bash pipeline.sh [train | eval]"
    echo "  2. 当前状态:  bash pipeline.sh status"
    echo "  3. 全局状态:  bash pipeline.sh status-all  <-- [新功能]"
    echo "  4. 管理任务:  bash pipeline.sh [pause | resume | stop] <ID>"
    exit 1
fi

echo "========================================================"
echo "   PyKT 动态流水线 -> [ $MODE ]"
echo "   📂 工作目录: $PROJECT_LOG_DIR"
if [[ "$MODE" == "status-all" ]]; then
    echo "   🔍 扫描范围: $BASE_PROJECT_NAME*"
else
    echo "   🎯 当前项目: $PROJECT_NAME"
fi
echo "========================================================"

# ------------------------------------------------------------------------------
#  Python 管理脚本生成器 (支持 status-all 和 文件归档)
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
curr_project = "$PROJECT_NAME"
base_project_pattern = "$BASE_PROJECT_NAME"
entity = None 

# 获取 Entity
try:
    api = wandb.Api() # 不指定 project 以便能获取所有 project
    if hasattr(api, 'default_entity') and api.default_entity:
        entity = api.default_entity
    else:
        try:
            entity = api.viewer.entity
        except:
            entity = api.viewer().entity
except:
    pass

mode = sys.argv[1] 
ids = sys.argv[2:]

# --- 打印表格的辅助函数 ---
def print_sweeps(sweeps, proj_name=""):
    # 定义列宽
    W_ID = 12
    W_STATE = 12
    W_RUNS = 8
    W_DATE = 12
    W_PROJ = 35 # 项目名列宽
    
    # 如果是 status-all，多显示一列 Project
    header_fmt = f"{'Sweep ID':<{W_ID}} | {'State':<{W_STATE}} | {'Runs':<{W_RUNS}} | {'Created':<{W_DATE}} | "
    if proj_name == "ALL":
        header_fmt += f"{'Project':<{W_PROJ}} | "
    header_fmt += "Name/Config"
    
    print("-" * 120)
    print(header_fmt)
    print("-" * 120)
    
    found = False
    for sweep in sweeps:
        found = True
        state = sweep.state
        
        try:
            run_count = sweep.run_count if hasattr(sweep, 'run_count') else len(list(sweep.runs))
        except:
            run_count = "?"

        created = sweep.created_at[:10] if hasattr(sweep, 'created_at') else "N/A"
        cfg_name = sweep.config.get('name', 'N/A')
        
        # 状态颜色
        state_padded = f"{state:<{W_STATE}}"
        if state in ["RUNNING", "FINISHED"]: state_disp = f"\033[92m{state_padded}\033[0m"
        elif state == "PAUSED": state_disp = f"\033[93m{state_padded}\033[0m"
        elif state in ["CANCELED", "KILLED", "STOPPED"]: state_disp = f"\033[91m{state_padded}\033[0m"
        else: state_disp = state_padded

        row_fmt = f"{sweep.id:<{W_ID}} | {state_disp} | {str(run_count):<{W_RUNS}} | {created:<{W_DATE}} | "
        if proj_name == "ALL":
            # 截断过长的项目名
            p_name = sweep.project
            if len(p_name) > W_PROJ - 1: p_name = p_name[:W_PROJ-3] + "..."
            row_fmt += f"{p_name:<{W_PROJ}} | "
        
        row_fmt += f"{cfg_name}"
        print(row_fmt)
    
    if not found:
        print("   (没有找到 Sweep)")
    print("-" * 120)


# ==========================
#  模式: STATUS (当前项目)
# ==========================
if mode == "status":
    print(f"\n📊 项目 [{curr_project}] 的状态:")
    try:
        sweeps = api.project(curr_project, entity=entity).sweeps()
        print_sweeps(sweeps)
    except Exception as e:
        print(f"❌ 获取失败: {e}")

# ==========================
#  模式: STATUS-ALL (所有项目)
# ==========================
elif mode == "status-all":
    print(f"\n🌍 全局扫描: Entity [{entity}] 下以 [{base_project_pattern}] 开头的项目")
    try:
        # 1. 获取所有项目
        projects = api.projects(entity=entity)
        
        target_projects = []
        for p in projects:
            if p.name.startswith(base_project_pattern):
                target_projects.append(p)
        
        if not target_projects:
            print("❌ 未找到匹配的项目。")
        else:
            # 2. 遍历项目获取 Sweep
            all_sweeps = []
            print(f"   -> 发现 {len(target_projects)} 个匹配项目，正在获取数据...")
            for p in target_projects:
                try:
                    # 仅获取最近的 Sweep，避免太慢
                    proj_sweeps = list(p.sweeps())
                    all_sweeps.extend(proj_sweeps)
                except:
                    pass
            
            # 3. 打印 (按时间倒序)
            # 简单的排序，如果 created_at 格式不对可能报错，这里简单 try一下
            try:
                all_sweeps.sort(key=lambda x: x.created_at, reverse=True)
            except:
                pass

            print_sweeps(all_sweeps, proj_name="ALL")

    except Exception as e:
        print(f"❌ 全局扫描失败: {e}")

# ==========================
#  模式: ACTION (CLI)
# ==========================
elif mode in ["pause", "resume", "stop"]:
    if not ids:
        print("❌ Error: 缺少 Sweep ID")
        sys.exit(1)
    
    action_map = {"pause": "--pause", "resume": "--resume", "stop": "--stop"}
    flag = action_map[mode]

    for sweep_id in ids:
        # 注意：这里我们优先尝试在当前 project 下操作
        # 如果是全局模式下看到的其他 project 的 ID，CLI 可能需要 user/proj/id 完整路径
        # 这里尝试简单处理：先用当前 project 上下文，失败则提示用户
        
        full_id = f"{entity}/{curr_project}/{sweep_id}" if entity else f"{curr_project}/{sweep_id}"
        print(f"🔄 尝试操作: {flag} {full_id}")
        
        exit_code = os.system(f"wandb sweep {flag} {full_id}")
        
        if exit_code != 0:
            print(f"⚠️  当前项目下失败，尝试直接操作 ID (适用于 wandb 自动推断): {sweep_id}")
            os.system(f"wandb sweep {flag} {sweep_id}")

EOF
}

# ------------------------------------------------------------------------------
#  逻辑分支
# ------------------------------------------------------------------------------

# 1. Status / Status-All / Control
if [[ "$MODE" == "status" || "$MODE" == "status-all" ]]; then
    generate_manage_script
    python auto_manage_wandb.py "$MODE"
    rm auto_manage_wandb.py

elif [[ "$MODE" == "pause" || "$MODE" == "resume" || "$MODE" == "stop" ]]; then
    if [[ -z "${ARGS[0]}" ]]; then
        echo "❌ 请提供 Sweep ID"
        exit 1
    fi
    generate_manage_script
    python auto_manage_wandb.py "$MODE" "${ARGS[@]}"
    rm auto_manage_wandb.py

elif [[ "$MODE" == "kill-local" ]]; then
    # 注意：kill-local 仍然只杀当前 PROJECT_NAME 的进程，避免误杀其他实验
    echo "⚠️  准备杀死项目 [$PROJECT_NAME] 的本地进程..."
    PIDS=$(pgrep -f "wandb agent.*$PROJECT_NAME")
    if [[ -z "$PIDS" ]]; then
        echo "✅ 无相关进程。"
    else
        echo $PIDS | xargs kill -9
        echo "☠️  已杀死: $PIDS"
    fi

# ------------------------------------------------------------------------------
#  2. Train
# ------------------------------------------------------------------------------
elif [[ "$MODE" == "train" ]]; then
    echo "[Step 1] 生成配置 (保存至 $PROJECT_LOG_DIR)..."
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

    echo "[Step 2] 注册 Sweep..."
    # 关键：WandB 会在当前目录找 .yaml，所以我们在当前目录执行命令
    # 但日志重定向到文件夹中
    sh "$LAUNCH_FILE_TRAIN" > "$TRAIN_LOG" 2>&1
    
    echo "[Step 3] 生成 Agent 脚本..."
    sh run_all.sh "$TRAIN_LOG" "$SWEEP_START_ID" "$SWEEP_END_ID" "$DATASETS" "$MODELS" "$GPU_IDS" "$PROJECT_NAME"
    
    # [新增] 移动生成的 agent 脚本到日志目录
    RAW_AGENT_SCRIPT="start_sweep_${SWEEP_START_ID}_${SWEEP_END_ID}.sh"
    FINAL_AGENT_SCRIPT="$PROJECT_LOG_DIR/agent_train.sh"
    
    if [[ -f "$RAW_AGENT_SCRIPT" ]]; then
        mv "$RAW_AGENT_SCRIPT" "$FINAL_AGENT_SCRIPT"
        chmod +x "$FINAL_AGENT_SCRIPT"
        
        echo "[Step 4] 启动 Agent ($FINAL_AGENT_SCRIPT)..."
        # 启动时，将nohup日志也放入文件夹
        nohup sh "$FINAL_AGENT_SCRIPT" > "$PROJECT_LOG_DIR/nohup_train.out" 2>&1 &
        echo "🎉 训练已启动！日志: $PROJECT_LOG_DIR/nohup_train.out"
    else
        echo "❌ Agent 脚本未生成，请检查 $TRAIN_LOG"
    fi

# ------------------------------------------------------------------------------
#  3. Eval
# ------------------------------------------------------------------------------
elif [[ "$MODE" == "eval" ]]; then
    echo "[Step 1] 提取最佳模型..."
    cat <<EOF > auto_extract_best.py
import sys, os, warnings
warnings.filterwarnings("ignore")
try:
    from examples.wandb_train import wandb_api 
except ImportError:
    import wandb_api 

# 强制环境变量
os.environ["WANDB_PROJECT"] = "$PROJECT_NAME"

try:
    df = wandb_api.get_best_run(dataset_name="$DATASETS", model_name="$MODELS")
    wandb_api.extract_best_models(df, "$DATASETS", "$MODELS", 
                                  fpath="./seedwandb/predict.yaml", 
                                  wandb_key="$WANDB_API_KEY",
                                  launch_file="$LAUNCH_FILE_PRED")
except Exception as e:
    sys.exit(1)
EOF
    python auto_extract_best.py
    if [[ $? -ne 0 ]]; then echo "❌ 提取失败"; rm auto_extract_best.py; exit 1; fi
    rm auto_extract_best.py

    echo "[Step 2] 注册预测任务..."
    sh "$LAUNCH_FILE_PRED" > "$PRED_LOG" 2>&1

    echo "[Step 3] 生成 Agent..."
    sh run_all.sh "$PRED_LOG" "$SWEEP_START_ID" "$SWEEP_END_ID" "$DATASETS" "$MODELS" "$GPU_IDS" "$PROJECT_NAME"

    # [新增] 移动脚本
    RAW_AGENT_SCRIPT="start_sweep_${SWEEP_START_ID}_${SWEEP_END_ID}.sh"
    FINAL_AGENT_SCRIPT="$PROJECT_LOG_DIR/agent_pred.sh"

    if [[ -f "$RAW_AGENT_SCRIPT" ]]; then
        mv "$RAW_AGENT_SCRIPT" "$FINAL_AGENT_SCRIPT"
        chmod +x "$FINAL_AGENT_SCRIPT"
        echo "[Step 4] 启动预测 Agent..."
        nohup sh "$FINAL_AGENT_SCRIPT" > "$PROJECT_LOG_DIR/nohup_pred.out" 2>&1 &
        echo "🎉 预测已启动！日志: $PROJECT_LOG_DIR/nohup_pred.out"
    fi
fi