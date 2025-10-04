#!/bin/bash

# 加载 ~/.bashrc
if [ -f "$HOME/.bashrc" ]; then
    . "$HOME/.bashrc"
fi

# 默认并发上限
MAX_JOBS=8

# 默认 GPU 数量
NUM_GPUS=4

# 如果有命令行参数，覆盖 MAX_JOBS 和 NUM_GPUS
if [ $# -gt 0 ]; then
    MAX_JOBS=$1
    if [ $# -gt 1 ]; then
        NUM_GPUS=$2
    fi
fi

# 检查间隔（秒）
CHECK_INTERVAL=30

# 是否统一加nohup（0=不加，1=加）
USE_NOHUP=0

# 命令文件
COMMANDS_FILE="commands.txt"

# 新增：已完成任务记录文件
COMPLETED_FILE="completed_tasks.log"

# 新增：初始化已完成任务记录文件（如果不存在）
if [ ! -f "$COMPLETED_FILE" ]; then
    touch "$COMPLETED_FILE"
fi

# 读取命令到数组，跳过空行和以##开头的注释
if [ ! -f "$COMMANDS_FILE" ]; then
    echo "Error: $COMMANDS_FILE not found!"
    exit 1
fi
# 过滤空行和##注释，保留原始行号
commands=()
line_numbers=()
while IFS= read -r line; do
    # 跳过空行和##开头的行
    if [[ -n "$line" && ! "$line" =~ ^##.* ]]; then
        commands+=("$line")
        line_numbers+=("$(( ${#commands[@]} - 1 ))")
    fi
done < <(grep -v -e '^[[:space:]]*$' -e '^##.*' "$COMMANDS_FILE")

# 新增：读取已完成任务的索引
completed_tasks=()
if [ -s "$COMPLETED_FILE" ]; then
    while IFS= read -r idx; do
        # 验证索引是否有效
        if [[ "$idx" =~ ^[0-9]+$ && "$idx" -lt ${#commands[@]} ]]; then
            completed_tasks+=("$idx")
        fi
    done < "$COMPLETED_FILE"
fi

# 修改：初始化队列，跳过已完成任务
queue=()
for i in $(seq 0 $((${#commands[@]}-1))); do
    skip=0
    for completed_idx in "${completed_tasks[@]}"; do
        if [ "$i" == "$completed_idx" ]; then
            skip=1
            break
        fi
    done
    if [ $skip -eq 0 ]; then
        queue+=("$i")
    fi
done

# 初始化运行中的PID数组
running_pids=()
running_cmds=()  # 记录对应的命令索引
running_gpus=()  # 记录每个任务使用的 GPU

# 函数：打印当前运行任务的表格
print_running_tasks() {
    # 在打印任务列表前显示当前时间
    echo "--- Current Status: $(date '+%Y-%m-%d %H:%M:%S') ---"
    if [ ${#running_pids[@]} -eq 0 ]; then
        echo "No tasks are currently running."
        return
    fi
    echo "Current running tasks:"
    # 打印表头
    printf "%-8s %-5s %-8s %-5s %s\n" "Task ID" "Fold" "PID" "GPU" "Command"
    # 打印每一行，限制命令长度为100字符以避免过长
    for i in "${!running_pids[@]}"; do
        cmd="${commands[${running_cmds[$i]}]}"
        # 截断命令以提高可读性（可选，调整100为其他值或移除）
        printf "%-8s %-5s %-8s %-5s %s\n" "${running_cmds[$i]}" "${line_numbers[${running_cmds[$i]}]}" "${running_pids[$i]}" "${running_gpus[$i]}" "$cmd"
    done
    echo ""
}

# 函数：计算每个 GPU 的当前任务数（负载）
get_gpu_task_count() {
    local gpu_id=$1
    local count=0
    for g in "${running_gpus[@]}"; do
        if [ "$g" == "$gpu_id" ]; then
            ((count++))
        fi
    done
    echo $count
}

# 函数：启动一个任务
start_task() {
    if [ ${#queue[@]} -eq 0 ]; then
        return 1
    fi
    
    # 取出队列第一个
    idx=${queue[0]}
    queue=("${queue[@]:1}")
    
    cmd="${commands[$idx]}"
    
    # 选择当前任务数最少的 GPU
    min_load=9999  # 初始化一个大值
    gpu=0  # 默认 GPU 0
    for ((i=0; i<NUM_GPUS; i++)); do
        load=$(get_gpu_task_count $i)
        if [ $load -lt $min_load ]; then
            min_load=$load
            gpu=$i
        fi
    done
    
    cmd="export CUDA_VISIBLE_DEVICES=$gpu && $cmd"
    
    # 如果启用nohup，添加它
    if [ $USE_NOHUP -eq 1 ]; then
        cmd="nohup $cmd"
    fi
    
    # 运行命令
    bash -c "$cmd" &
    pid=$!
    
    running_pids+=("$pid")
    running_cmds+=("$idx")
    running_gpus+=("$gpu")
    
    echo "Started task $idx (fold ${line_numbers[$idx]}) with PID $pid on GPU $gpu at $(date)"
    echo "Command: $cmd"
    print_running_tasks
    return 0
}

# 捕获退出信号，杀死所有子进程
trap '
    echo "Exiting script, killing tasks via pkill..."; 
    # 注意：这里直接终止 Python 进程，不再依赖调度器记录的 PID
    pkill -TERM -f "wandb_qikt_mamba_train.py"; 
    exit
' EXIT INT TERM
# 主循环
while [ ${#queue[@]} -gt 0 ] || [ ${#running_pids[@]} -gt 0 ]; do
    # 启动新任务直到达到上限
    while [ ${#running_pids[@]} -lt $MAX_JOBS ] && start_task; do :; done
    
    # 如果没有运行中的任务且队列空，退出
    if [ ${#running_pids[@]} -eq 0 ]; then
        break
    fi
    
    # 打印当前运行任务
    print_running_tasks
    
    # 等待检查间隔
    sleep $CHECK_INTERVAL
    
    # 检查运行中的PID
    new_pids=()
    new_cmds=()
    new_gpus=()
    for i in "${!running_pids[@]}"; do
        pid="${running_pids[$i]}"
        cmd_idx="${running_cmds[$i]}"
        gpu="${running_gpus[$i]}"
        if kill -0 "$pid" 2>/dev/null; then
            # 还在运行
            new_pids+=("$pid")
            new_cmds+=("$cmd_idx")
            new_gpus+=("$gpu")
        else
            # 已完成或失败
            wait "$pid"
            exit_code=$?
            if [ $exit_code -eq 0 ]; then
                echo "Task $cmd_idx (fold ${line_numbers[$cmd_idx]}) with PID $pid on GPU $gpu completed successfully at $(date)"
                # 新增：记录成功完成的任务索引
                echo "$cmd_idx" >> "$COMPLETED_FILE"
            else
                echo "Task $cmd_idx (fold ${line_numbers[$cmd_idx]}) with PID $pid on GPU $gpu failed with exit code $exit_code at $(date)"
            fi
        fi
    done
    running_pids=("${new_pids[@]}")
    running_cmds=("${new_cmds[@]}")
    running_gpus=("${new_gpus[@]}")
done

# 清除trap
trap - EXIT INT TERM

echo "All tasks completed at $(date)"