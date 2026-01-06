import os
import yaml
import argparse
import itertools
import subprocess
import sys
import json
import hashlib
import re
import ast
import csv
from datetime import datetime

# ==========================================
# 工具函数与类
# ==========================================

def get_md5(text):
    """生成唯一哈希值"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def load_yaml_config(model_name):
    yaml_path = os.path.join("seedwandb", f"{model_name}.yaml")
    if not os.path.exists(yaml_path):
        print(f"[Error] YAML file not found: {yaml_path}")
        sys.exit(1)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def generate_param_combinations(parameters):
    keys = []
    values_list = []
    # 排序以保证一致性
    sorted_keys = sorted(parameters.keys())
    for key in sorted_keys:
        val_dict = parameters[key]
        if 'values' in val_dict:
            keys.append(key)
            values_list.append(val_dict['values'])
    combinations = list(itertools.product(*values_list))
    return keys, combinations

def safe_filename(params):
    """将参数转换为合法的文件名"""
    # 拼接 key=value
    name_parts = []
    for k, v in params.items():
        # 简化一下文件名：移除路径斜杠，防止创建子目录
        safe_v = str(v).replace("/", "_").replace("\\", "_")
        name_parts.append(f"{k}{safe_v}")
    
    full_name = "_".join(name_parts)
    # 如果文件名太长（Linux限制255字符），截断并加Hash后缀
    if len(full_name) > 200:
        full_name = full_name[:200] + "_" + get_md5(str(params))[:6]
    
    return full_name + ".log"

class LogParser:
    """专门用于提取日志中的字典指标"""
    @staticmethod
    def parse(log_path):
        metrics = {}
        if not os.path.exists(log_path):
            return metrics
        
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # 策略：寻找所有像字典的字符串 {...}
            # 你的日志有跨行字典（student_stats）和单行字典，使用正则非贪婪匹配
            # re.DOTALL 允许 . 匹配换行符
            dict_candidates = re.findall(r'(\{.*?\})', content, re.DOTALL)
            
            for candidate in dict_candidates:
                candidate = candidate.strip()
                try:
                    # 使用 ast.literal_eval 安全地将字符串转为 Python 字典
                    parsed_dict = ast.literal_eval(candidate)
                    
                    if isinstance(parsed_dict, dict):
                        # 过滤：只有包含关键 key 的字典才是我们需要的
                        # 你的关键词：overall_dataset_auc, testauc, oriaucconcepts
                        keys = parsed_dict.keys()
                        is_target = False
                        
                        if any(k in keys for k in ['overall_dataset_auc', 'student_stats_mean']):
                            is_target = True
                        elif any(k in keys for k in ['testauc', 'window_testauc']):
                            is_target = True
                        elif any(k in keys for k in ['oriaucconcepts', 'oriauclate_mean']):
                            is_target = True
                            
                        if is_target:
                            metrics.update(parsed_dict)
                            
                except (ValueError, SyntaxError):
                    continue # 不是合法的字典字符串，跳过
                    
        except Exception as e:
            print(f"[Parser Error] Failed to parse {log_path}: {e}")
            
        return metrics

class ResultLogger:
    """管理 CSV 结果保存"""
    def __init__(self, filepath):
        self.filepath = filepath

    def log(self, params, metrics):
        # 合并参数和指标
        row_data = {**params, **metrics}
        
        file_exists = os.path.exists(self.filepath)
        
        # 如果文件存在，先读取表头，看看有没有新字段
        if file_exists:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    header = []
        else:
            header = list(row_data.keys())

        # 检查是否有新出现的字段（比如有的模型跑出了新指标）
        new_keys = set(row_data.keys()) - set(header)
        if new_keys:
            # 如果有新字段，比较麻烦，简单起见我们追加到 row_data，
            # 但为了 CSV 格式整齐，最好重新读取所有数据并扩展列。
            # 这里采用简单方案：追加写入，如果列不对齐，标准 CSV 可能会乱。
            # 稳妥方案：只写入 header 里有的列，或者追加 header。
            # 为了灵活性，我们把新 header 追加到文件末尾是不合法的。
            # --> 动态扩展 header 方案：读取旧数据 -> 扩展 header -> 重写文件
            if file_exists: 
                # 读取旧数据
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    data = list(csv.DictReader(f))
                header = header + list(new_keys) # 更新 header
                # 重写
                with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    writer.writeheader()
                    writer.writerows(data)
                    writer.writerow(row_data)
                return

        # 常规写入
        with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)

class ProgressManager:
    def __init__(self, log_dir):
        self.progress_file = os.path.join(log_dir, "progress.json")
        self.completed_hashes = set()
        self.load()

    def load(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.completed_hashes = set(data.get("completed", []))
            except Exception as e:
                print(f"[Warning] Failed to load progress: {e}")

    def save(self):
        with open(self.progress_file, 'w') as f:
            json.dump({"completed": list(self.completed_hashes)}, f, indent=4)

    def is_completed(self, param_hash):
        return param_hash in self.completed_hashes

    def mark_completed(self, param_hash):
        self.completed_hashes.add(param_hash)
        self.save()

# ==========================================
# 主逻辑
# ==========================================

def run_scheduler(args):
    model_name = args.model_name
    gpu_id = args.gpu
    target_dataset = args.dataset_name
    config = load_yaml_config(model_name)
    script_name = config.get('program')
    parameters = config.get('parameters', {})
    
    if not script_name:
        print("[Error] 'program' missing in YAML.")
        sys.exit(1)
    if 'dataset_name' in parameters:
        parameters.pop('dataset_name')
        
    # 2. 移除 save_dir (强制指定为 offline_train)
    if 'save_dir' in parameters:
        parameters.pop('save_dir')
    # 1. 目录结构：offline_logs/{model_name}/
    # 不再按日期分子文件夹
    log_dir = os.path.join("offline_logs", f"{model_name}_{target_dataset}")
    os.makedirs(log_dir, exist_ok=True)
    
    # 2. 初始化管理器
    progress_mgr = ProgressManager(log_dir)
    csv_logger = ResultLogger(os.path.join(log_dir, "results.csv"))
    
    # 3. 生成任务
    param_keys, param_combinations = generate_param_combinations(parameters)
    total_jobs = len(param_combinations)
    
    print(f"==================================================")
    print(f" Local Scheduler: {model_name}")
    print(f" Script: {script_name}")
    print(f" Log Dir: {log_dir}")
    print(f" Progress: {len(progress_mgr.completed_hashes)}/{total_jobs} completed")
    print(f"==================================================\n")

    for idx, values in enumerate(param_combinations):
        current_job_idx = idx + 1
        current_params = dict(zip(param_keys, values))
        
        # 4. 生成唯一 Hash 和 文件名
        # 将 params 转字符串计算 Hash (确保无序一致性)
        param_str_for_hash = str(sorted(current_params.items()))
        param_hash = get_md5(param_str_for_hash)
        
        # 检查进度
        if progress_mgr.is_completed(param_hash):
            # 可选：打印跳过信息，或者完全静默
            # print(f"[{current_job_idx}/{total_jobs}] SKIP: {current_params}")
            continue

        # 生成参数命名的日志文件
        log_filename = safe_filename(current_params)
        job_log_path = os.path.join(log_dir, log_filename)
        
        # 【修正后的顺序】
        # 1. 先初始化 cmd_args 列表 (放入变动参数)
        cmd_args = [f"--{k} {v}" for k, v in current_params.items()]
        
        # 2. 再追加固定参数
        cmd_args.append(f"--dataset_name {target_dataset}")
        cmd_args.append(f"--save_dir offline_train")
        
        # 3. 拼接命令
        cmd_str = f"python {script_name} {' '.join(cmd_args)}"
        
        print(f"[{current_job_idx}/{total_jobs}] RUNNING: {cmd_str}")
        print(f"   -> Log: {log_filename}")

        # 5. 执行命令
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["WANDB_MODE"] = "disabled"
        env["WANDB_SILENT"] = "true"
        
        start_time = datetime.now()
        
        with open(job_log_path, "w", encoding='utf-8') as log_f:
            log_f.write(f"Command: {cmd_str}\n")
            log_f.write(f"Start: {start_time}\n")
            log_f.write("-" * 50 + "\n")
            log_f.flush()
            
            try:
                subprocess.run(
                    cmd_str,
                    shell=True,
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT
                )
                
                # 6. 任务结束：解析日志 + 保存结果
                # 重新读取日志文件提取 dict
                extracted_metrics = LogParser.parse(job_log_path)
                
                if extracted_metrics:
                    print(f"   -> Parsed {len(extracted_metrics)} metrics.")
                    csv_logger.log(current_params, extracted_metrics)
                else:
                    print(f"   -> [Warning] No valid metrics found in log.")
                
                # 标记完成
                progress_mgr.mark_completed(param_hash)

            except Exception as e:
                log_f.write(f"\n[Scheduler Error] {str(e)}\n")
                print(f"   -> [Error] Job failed: {e}")
            
            log_f.write(f"\nEnd: {datetime.now()}\n")

    print(f"\n[Done] All jobs finished for {model_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name passed from bash")
    args = parser.parse_args()
    
    run_scheduler(args)