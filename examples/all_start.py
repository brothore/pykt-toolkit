import os, sys

# 基础参数获取
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
logf = sys.argv[1]
outf = open(sys.argv[2], "w")
start = int(sys.argv[3])
end = int(sys.argv[4])

dataset_name = sys.argv[5]
model_name = sys.argv[6]
nums = sys.argv[7].split(",")

if len(sys.argv) == 8:
    project_name = "kt_toolkits"
else:
    project_name = sys.argv[8]
# [修改点 2] 定义日志存放的文件夹路径
log_dir = f"run_logs/{project_name}"

# [修改点 3] 在生成的 shell 脚本第一行加入创建文件夹的命令
# 加上 -p 参数，如果文件夹已存在也不会报错，且能自动创建父目录
outf.write(f"mkdir -p {log_dir}\n")


cmdpre = f"WANDB_API_KEY={WANDB_API_KEY} nohup "

# === 核心修改逻辑开始 ===
current_fname = None # 用于暂存正在处理的文件名
idx = 0  # 记录符合 dataset/model 要求的任务总数
num = 0  # 记录当前使用的 GPU 序号索引

with open(logf, "r") as fin:
    for line in fin:
        line = line.strip()
        
        # 1. 寻找文件名行
        if line.startswith("wandb: Creating sweep from: "):
            # 提取文件名，例如: assist2009_dkt_qid_0.yaml
            # 注意：这里我们只暂存，不处理，等待找到对应的 agent id 再一起处理
            current_fname = line.split(": ")[-1].split("/")[-1]
            
        # 2. 寻找 Agent ID 行 (必须在找到文件名之后)
        elif line.startswith("wandb: Run sweep agent with: ") and current_fname:
            # 提取 Agent 命令部分
            # line格式: wandb: Run sweep agent with: wandb agent xxx/xxx/id
            sweepid_cmd = line.split(": ")[-1] 
            
            # --- 筛选逻辑 ---
            # 清理文件名后缀用于判断
            fname_clean = current_fname.split(".")[0]
            
            # 判断是否符合用户指定的 dataset 和 model
            is_target_dataset = current_fname.startswith(dataset_name)
            is_target_model = (current_fname.find(f"_{model_name}_") != -1)
            
            if is_target_dataset and is_target_model:
                print(f"[Match Found] dataset: {dataset_name}, model: {model_name}, fname: {fname_clean}")
                
                # 判断是否在 start 和 end 的范围内
                if start <= idx < end:
                    gpu_id = nums[num % len(nums)] # 防止GPU数量不够导致越界
                    log_path = f"{log_dir}/log_{fname_clean}.txt"

                    
                    # 组装命令: 
                    # 1. 指定 GPU
                    # 2. 加上 WANDB_KEY 和 nohup
                    # 3. 加上 agent ID
                    # 4. 重定向输出到独立日志文件 (关键!)
                    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} {cmdpre} {sweepid_cmd} >> {log_path} 2>&1 &"
                    
                    outf.write(cmd + "\n")
                    num += 1
                
                # 只有匹配到了目标 dataset/model，计数器才加 1
                idx += 1
            
            # 处理完一对 (fname + agent) 后，重置 current_fname，防止错配
            current_fname = None

# === 核心修改逻辑结束 ===

outf.close()
print(f"Done. Generated {num} commands.")