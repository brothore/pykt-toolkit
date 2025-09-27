# continue_training.py

import json
import subprocess
import argparse
import os
import glob

def continue_training():
    parser = argparse.ArgumentParser(description="Continue training a model using parameters saved in a specified directory.")
    parser.add_argument("--load_dir", type=str, required=True, help="The directory containing the saved parameter file (e.g., saved_model).")
    
    args = parser.parse_args()
    load_dir = args.load_dir
    
    if not os.path.isdir(load_dir):
        print(f"❌ 错误: 找不到目录: {load_dir}")
        return

    # 1. 查找参数文件
    # 查找 load_dir 中所有匹配 'saved_params_*.json' 模式的文件
    param_files = glob.glob(os.path.join(load_dir, "saved_params_*.json"))
    
    if not param_files:
        print(f"❌ 错误: 在目录 {load_dir} 中未找到任何 'saved_params_*.json' 参数文件。")
        return
        
    # 假设目录中只有一个参数文件，或者我们取第一个
    saved_params_path = param_files[0]
    
    # 2. 从文件名中提取 model_name
    # 文件名格式为 saved_params_MODELNAME.json
    filename = os.path.basename(saved_params_path)
    try:
        model_name = filename.split('_')[1].split('.')[0]
    except IndexError:
        print(f"❌ 错误: 无法从文件名 {filename} 中解析出 model_name。")
        return
    
    print(f"✅ 找到参数文件: {saved_params_path}")
    print(f"✅ 解析出模型名称: {model_name}")

    # 3. 加载保存的参数
    with open(saved_params_path, 'r') as f:
        saved_params = json.load(f)
        
    # 4. 准备命令行参数
    # 将 'other_config' 的内容合并到主参数字典中，以便统一处理
    other_config = saved_params.pop('other_config', {})
    all_params = saved_params
    all_params.update(other_config)
    
    # 构建命令行参数列表
    command_args = []
    for k, v in all_params.items():
        # 排除那些不需要通过命令行传递的内部配置
        if k in ['other_config']:
             continue
        if v is not None:
            # 将参数转换为命令行格式：--key value
            command_args.append(f"--{k}")
            # 注意：这里我们使用 str(v) 来处理所有类型（数字、字符串等）
            command_args.append(str(v))
            
    # 5. 构建并执行调用命令
    script_name = f"wandb_{model_name}_train.py"
    full_command = ["python", script_name] + command_args
    
    print(f"\n🚀 正在调用脚本: {script_name} 继续训练...")
    print("命令行参数预览:", " ".join(command_args))
    
    try:
        # 使用 subprocess.run 执行外部脚本
        # check=True 会在子进程返回非零退出代码时抛出异常
        # capture_output=False 让子进程的输出直接显示在当前终端
        subprocess.run(full_command, check=True, cwd=".", capture_output=False)
        print("\n✅ 训练子进程执行完成。")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练子进程执行失败。退出码: {e.returncode}")
        
    except FileNotFoundError:
        print(f"\n❌ 错误: 找不到文件 {script_name}。请检查路径或文件名是否正确。")

if __name__ == "__main__":
    continue_training()