import os, sys
import json

# 健壮版 all_start.py
# 功能：从不规则的 WandB 日志中提取 Sweep ID 并分配 GPU

if len(sys.argv) < 8:
    print("Usage: python all_start.py <log_file> <outfile> <start_idx> <end_idx> <dataset> <model> <gpu_ids> [project_name]")
    sys.exit(1)

logf = sys.argv[1]
outf_path = sys.argv[2]
start = int(sys.argv[3])
end = int(sys.argv[4])
dataset_name = sys.argv[5]
model_name = sys.argv[6]
gpu_ids = sys.argv[7].split(",")
project_name = sys.argv[8] if len(sys.argv) > 8 else "kt_toolkits"

# 获取 API KEY
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
if not WANDB_API_KEY:
    try:
        with open("../configs/wandb.json") as fin:
            wandb_config = json.load(fin)
            WANDB_API_KEY = wandb_config.get("api_key")
    except:
        print("Warning: WANDB_API_KEY not found in env or config.")

cmdpre = f"WANDB_API_KEY={WANDB_API_KEY} nohup wandb agent "

print(f"--- Parsing Log: {logf} ---")
print(f"--- Target: {dataset_name} / {model_name} ---")

tasks = []
current_fname = None

# === 核心改进：逐行扫描状态机 ===
with open(logf, "r") as fin:
    for line in fin:
        line = line.strip()
        
        # 1. 捕获配置文件名
        if line.startswith("wandb: Creating sweep from:"):
            # 格式: wandb: Creating sweep from: all_wandbs/assist2009_dkt_qid_0.yaml
            parts = line.split(": ")[-1].split("/")
            if parts:
                current_fname = parts[-1]  # 获取文件名，如 assist2009_dkt_qid_0.yaml
        
        # 2. 捕获 Sweep ID (必须在获取到文件名之后)
        elif line.startswith("wandb: Run sweep agent with:") and current_fname:
            # 格式: wandb: Run sweep agent with: wandb agent entity/project/sweepid
            sweep_cmd = line.split(": ")[-1]
            # 提取 sweep_id (通常是命令的最后一部分，或者直接取整段命令)
            # 标准输出通常是: wandb agent entity/project/sweepid
            # 我们只需要提取 ID，但为了保险，直接提取 output 里的 id
            # 这里的 sweep_cmd 可能是 "wandb agent user/proj/xyz123"
            
            # 为了兼容性，我们尝试提取最后一段
            sweep_full_id = sweep_cmd.split(" ")[-1] # user/proj/xyz123
            
            # 检查文件名是否匹配当前任务的数据集和模型
            # 移除扩展名
            fname_no_ext = current_fname.split(".")[0]
            
            # 匹配逻辑：文件名必须以 dataset 开头，且包含 model
            # 例如: assist2009_qikt_mamba_qid_0 包含 assist2009 和 qikt_mamba
            if dataset_name in fname_no_ext and model_name in fname_no_ext:
                tasks.append(sweep_full_id)
                print(f"  [Found] File: {current_fname} -> ID: {sweep_full_id}")
            else:
                print(f"  [Skip]  File: {current_fname} (Not match {dataset_name}/{model_name})")
            
            # 重置状态，防止错配
            current_fname = None

# === 生成启动命令 ===
print(f"--- Generating Script: {outf_path} ---")
with open(outf_path, "w") as outf:
    gpu_idx = 0
    count = 0
    
    # 按照 start 和 end 的范围截取
    # 注意：tasks 列表里的顺序对应 fold 0, 1, 2...
    target_tasks = tasks[start:end]
    
    if not target_tasks:
        print(f"❌ Warning: No tasks found in range [{start}:{end}]! Check your logs or range.")
    
    for sweep_id in target_tasks:
        # 轮询分配 GPU
        current_gpu = gpu_ids[gpu_idx % len(gpu_ids)]
        
        # 构造最终命令
        # 注意: 这里的 sweep_id 已经是 user/project/id 格式，直接传给 wandb agent 即可
        # 如果前面提取有问题，可能需要加 --entity 或 -p
        cmd = f"CUDA_VISIBLE_DEVICES={current_gpu} {cmdpre}{sweep_id} > /dev/null 2>&1 &"
        
        outf.write(cmd + "\n")
        print(f"  Task {count+start}: GPU {current_gpu} -> {sweep_id}")
        
        gpu_idx += 1
        count += 1

print("--- Done ---")