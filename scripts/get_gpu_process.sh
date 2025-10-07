#!/bin/bash

# 检查依赖
if ! command -v nvidia-smi &> /dev/null || ! command -v ps &> /dev/null; then
    echo "错误：脚本依赖 nvidia-smi 和 ps 命令，请确保它们都可用。"
    exit 1
fi

echo "--- 正在解析 nvidia-smi 进程信息 (CSV/完整命令行) ---"

# 使用 --query-compute-apps 和 --format=csv 获取精确的 GPU ID 和 PID
# 
# --query-compute-apps: 专门查询计算进程
# gpu_uuid: 统一标识符
# pid: 进程ID
# 
# --no-header: 去除 CSV 头部
# --format=csv: 以 CSV 格式输出，逗号分隔

process_csv=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null)

if [ -z "$process_csv" ]; then
    echo "未在任何 GPU 上找到运行中的用户进程。"
    exit 0
fi

echo "格式: [GPU ID] [PID] [完整命令行指令 (CMD)]"
echo "--------------------------------------------------------"

# 获取 UUID 到简单 GPU ID (0, 1, 2...) 的映射
# nvidia-smi --query-gpu=uuid --format=csv,noheader
# 这一步是为了将 uuid 转换回用户友好的数字 ID
gpu_id_map=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | nl -w1 -s, | awk -F, '{print $2":"$1}')

# 将映射存储为关联数组 (仅限 Bash 4.0+)
declare -A gpu_map
while IFS=: read -r uuid simple_id; do
    gpu_map["$uuid"]="$simple_id"
done <<< "$gpu_id_map"

# 遍历 CSV 结果，格式为: GPU_UUID, PID
echo "$process_csv" | while IFS=, read -r uuid pid; do
    # 清理 UUID 和 PID 前后的空格
    uuid=$(echo "$uuid" | xargs)
    pid=$(echo "$pid" | xargs)
    
    # 获取用户友好的数字 GPU ID
    gpu_id="${gpu_map[$uuid]}"
    if [ -z "$gpu_id" ]; then
        gpu_id="?" # 如果找不到映射，则显示问号
    fi

    # 使用 ps -p <PID> -o cmd= 获取完整命令行
    command_line=$(ps -p "$pid" -o cmd= 2>/dev/null)
    
    # 检查 ps 是否成功找到进程
    if [ -z "$command_line" ]; then
        command_line="[进程 $pid 已结束或不存在]"
    fi
    
    # 打印结果
    printf "[%s] %s %s\n" "$gpu_id" "$pid" "$command_line"
done

echo "--- 解析完成 ---"