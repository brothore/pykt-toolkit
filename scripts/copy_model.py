#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型复制和配置脚本
用于复制基底模型并创建新模型的配置
支持撤销操作
"""

import os
import re
import sys
import shutil
import argparse
import json
from datetime import datetime
def copy_and_update_seedwandb_yaml(script_dir, base_model, new_model):
    """
    复制 examples/seedwandb 下的 yaml 配置并修改 model_name 和 program
    """
    src_yaml = os.path.join(script_dir, "examples", "seedwandb", f"{base_model}.yaml")
    dst_yaml = os.path.join(script_dir, "examples", "seedwandb", f"{new_model}.yaml")

    if not os.path.exists(src_yaml):
        print(f"⚠️ 源 YAML 文件 {src_yaml} 不存在，跳过")
        return False, None

    if os.path.exists(dst_yaml):
        print(f"⚠️ 目标 YAML 文件 {dst_yaml} 已存在，跳过")
        return False, None

    # 1. 复制文件
    shutil.copy2(src_yaml, dst_yaml)
    
    # 2. 读取并修改内容
    with open(dst_yaml, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content

    # --- 修改 1: 替换 program 文件名 ---
    # 目标: 将 program: xxxx/xxxx.py 替换为 program: xxxx/wandb_{new_model}_train.py
    # 逻辑: 保留路径前缀（如果有），只替换文件名部分
    # 正则解释: 
    # (program:\s+(?:.*[/\\])?) -> 捕获组1: 匹配 "program: " 加上可能存在的目录路径
    # [^/\\\s]+\.py -> 匹配旧的 .py 文件名（不包含路径）
    content = re.sub(
        r'(program:\s+(?:.*[/\\])?)[^/\\\s]+\.py', 
        f'\\1wandb_{new_model}_train.py', 
        content
    )

    # --- 修改 2: 替换 model_name ---
    # 匹配结构: model_name: (换行) (缩进) values: ["旧名字"]
    pattern_model = r'(model_name:\s*\n\s+values:\s*\[)(?:["\'].*?["\'])(])'
    content = re.sub(pattern_model, f'\\1"{new_model}"\\2', content)

    # 3. 写入文件（如果内容发生了变化）
    if content != original_content:
        with open(dst_yaml, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, dst_yaml
    else:
        print(f"⚠️ 在 {dst_yaml} 中未匹配到需修改的字段，仅完成了复制")
        return True, dst_yaml
def search_and_add_to_string_lists(line, base_model, new_model):
    """
    函数A：在传入行中检索是否存在字符串列表，并在字符串列表中检索基底模型是否存在，
    若存在则将复制后的模型也添加进这个列表
    
    Args:
        line: 要处理的行
        base_model: 基底模型名称
        new_model: 新模型名称
    
    Returns:
        tuple: (是否修改了, 修改后的行)
    """
    # 检查是否包含列表结构 [...]
    if '[' not in line or ']' not in line:
        return False, line
    
    # 检查新模型是否已存在
    if re.search(rf'[\'"]{new_model}[\'"]', line):
        return False, line
    
    # 检查是否包含基底模型
    if not re.search(rf'[\'"]{base_model}[\'"]', line):
        return False, line
    
    # 匹配列表模式并添加新模型
    def replacer(match):
        prefix, items, suffix = match.group(1), match.group(2), match.group(3)
        
        # 在列表末尾添加新模型
        items = items.rstrip()
        if items.endswith(','):
            new_items = f'{items} "{new_model}"'
        else:
            new_items = f'{items}, "{new_model}"'
        
        return prefix + new_items + suffix
    
    # 匹配列表结构
    pattern = r'(\[)([^\]]*?)(\])'
    modified_line = re.sub(pattern, replacer, line)
    
    return modified_line != line, modified_line

def add_model_to_lists(file_path, base_model, new_model):
    """在包含基底模型的列表中添加新模型（逐行遍历使用函数A）"""
    if not os.path.exists(file_path):
        return False
        
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 增强存在性检查
    content = ''.join(lines)
    if re.search(rf'\b{new_model}\b', content):
        print(f"⚠️  {new_model} 已存在于 {file_path}，跳过添加")
        return False

    changed = False
    modified_lines = []
    
    # 逐行处理
    for line in lines:
        line_changed, modified_line = search_and_add_to_string_lists(line, base_model, new_model)
        if line_changed:
            changed = True
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)
    
    if not changed:
        print(f"ℹ️ 未在 {file_path} 中找到基底模型 {base_model} 的列表")
        return False
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(modified_lines)
    return True

def replace_model_name_in_file(file_path, base_model, new_model):
    """替换文件中的模型名称"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换类名（大写）
    content = re.sub(rf'class\s+{base_model.upper()}\s*\(', f'class {new_model.upper()}(', content)
    
    # 替换小写的模型名称 - 这里保持原有大小写
    content = re.sub(rf'self\.model_name\s*=\s*["\']({base_model})["\']', 
                     f'self.model_name = "{new_model}"', content)
    
    # 替换 in {'model_name'} 形式 - 保持原有大小写
    content = re.sub(rf'in\s*{{[\'"]{base_model}[\'"]}}', 
                     f"in {{'{new_model}'}}", content)
    
    # 替换 == "model_name" 形式 - 保持原有大小写
    content = re.sub(rf'==\s*["\']({base_model})["\']', 
                     f'== "{new_model}"', content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def update_init_file(file_path, base_model, new_model):
    """更新__init__.py文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查新模型的初始化是否已经存在
    if f'model_name == "{new_model}"' in content:
        print(f"⚠️  {new_model} 的初始化已存在于 {file_path} 中，跳过添加")
        return False
    
    # 查找并复制模型初始化语句
    init_pattern = rf'((?:if|elif)\s+model_name\s*==\s*["\']({base_model})["\']:\s*\n\s+from\s+\.{base_model}\s+import\s+{base_model.upper()}\s*\n\s+model\s*=\s*{base_model.upper()}\((.*?)\)\.to\(device\))'
    
    init_match = re.search(init_pattern, content, re.DOTALL)
    
    if init_match:
        # 提取参数部分
        params = init_match.group(3)
        # 在基底模型初始化后添加新模型的初始化
        new_init = f'\n    elif model_name == "{new_model}":\n        from .{new_model} import {new_model.upper()}\n        model = {new_model.upper()}({params}).to(device)'
        insert_pos = init_match.end()
        content = content[:insert_pos] + new_init + content[insert_pos:]
    else:
        # 尝试匹配没有import的旧格式（向后兼容）
        old_pattern = rf'((?:if|elif)\s+model_name\s*==\s*["\']({base_model})["\']:\s*\n\s+model\s*=\s*{base_model.upper()}\((.*?)\)\.to\(device\))'
        old_match = re.search(old_pattern, content, re.DOTALL)
        if old_match:
            params = old_match.group(3)
            new_init = f'\n    elif model_name == "{new_model}":\n        from .{new_model} import {new_model.upper()}\n        model = {new_model.upper()}({params}).to(device)'
            insert_pos = old_match.end()
            content = content[:insert_pos] + new_init + content[insert_pos:]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return True
def get_script_dir():
    """获取脚本所在的目录的上一级"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)
def save_operation_record(base_model, new_model, operations):
    """保存操作记录到txt文件"""
    script_dir = get_script_dir()
    logs_dir = os.path.join(script_dir, "scripts", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_file = os.path.join(logs_dir, f"model_operations_{timestamp}.txt")
    
    with open(record_file, 'w', encoding='utf-8') as f:
        f.write(f"模型操作记录\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"基底模型: {base_model}\n")
        f.write(f"新模型: {new_model}\n")
        f.write(f"操作类型: 添加模型\n")
        f.write(f"\n执行的操作:\n")
        
        for i, op in enumerate(operations, 1):
            f.write(f"{i}. {op}\n")
    
    print(f"✓ 操作记录已保存到: {record_file}")
    return record_file

def remove_model_from_file_reverse(file_path, model_name):
    """反向操作：从文件中删除指定模型"""
    if not os.path.exists(file_path):
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 删除import语句
    content = re.sub(rf'\nfrom\s+\.{model_name}\s+import\s+{model_name.upper()}', '', content)
    
    # 删除初始化语句
    content = re.sub(rf'\n\s*elif\s+model_name\s*==\s*["\']({model_name})["\']:\s*\n\s+model\s*=\s*{model_name.upper()}\(.*?\)\.to\(device\)', '', content, flags=re.DOTALL)
    
    # 从列表中删除模型名
    def remove_from_list(match):
        prefix, items, suffix = match.group(1), match.group(2), match.group(3)
        
        # 删除目标模型
        items = re.sub(rf',?\s*["\']({model_name})["\']', '', items)
        items = re.sub(rf'["\']({model_name})["\'],?\s*', '', items)
        
        # 清理多余的逗号
        items = re.sub(r',\s*,', ',', items)
        items = re.sub(r'^\s*,\s*', '', items)
        items = re.sub(r'\s*,\s*$', '', items)
        
        return prefix + items + suffix
    
    pattern = r'(\[)([^\]]*?)(\])'
    content = re.sub(pattern, remove_from_list, content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def undo_operations(base_model, new_model):
    """撤销操作：反向删除所有添加的内容和文件"""
    print(f"开始撤销操作：删除模型 {new_model} 相关的所有修改...")
    
    operations = []
    script_dir = get_script_dir()
    
    try:
        # 1. 删除复制的训练文件
        dst_train = os.path.join(script_dir, "examples", f"wandb_{new_model}_train.py")
        if os.path.exists(dst_train):
            os.remove(dst_train)
            operations.append(f"删除文件: {dst_train}")
            print(f"✓ 删除 {dst_train}")
        # === 【新增】删除 seedwandb yaml 文件 ===
        dst_yaml = os.path.join(script_dir, "examples", "seedwandb", f"{new_model}.yaml")
        if os.path.exists(dst_yaml):
            os.remove(dst_yaml)
            operations.append(f"删除文件: {dst_yaml}")
            print(f"✓ 删除 {dst_yaml}")
        # 2. 删除复制的模型文件
        dst_model = os.path.join(script_dir, "pykt", "models", f"{new_model}.py")
        if os.path.exists(dst_model):
            os.remove(dst_model)
            operations.append(f"删除文件: {dst_model}")
            print(f"✓ 删除 {dst_model}")
        
        # 3. 从各个配置文件中删除模型引用
        config_files = [
            os.path.join(script_dir, "examples", "wandb_train.py"),
            os.path.join(script_dir, "pykt", "models", "evaluate_model.py"), 
            os.path.join(script_dir, "pykt", "models", "train_model.py"),
            os.path.join(script_dir, "pykt", "models", "init_model.py"),
            os.path.join(script_dir, "pykt", "datasets", "init_dataset.py"),
            os.path.join(script_dir, "config.py")
        ]
        
        for file_path in config_files:
            if remove_model_from_file_reverse(file_path, new_model):
                operations.append(f"从 {file_path} 中删除模型引用")
                print(f"✓ 从 {file_path} 中删除 {new_model} 引用")
        
        # 4. 保存撤销操作记录
        logs_dir = os.path.join(script_dir, "scripts", "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        undo_record_file = os.path.join(logs_dir, f"model_undo_operations_{timestamp}.txt")
        
        with open(undo_record_file, 'w', encoding='utf-8') as f:
            f.write(f"模型撤销操作记录\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"基底模型: {base_model}\n")
            f.write(f"撤销模型: {new_model}\n")
            f.write(f"操作类型: 撤销模型添加\n")
            f.write(f"\n执行的撤销操作:\n")
            
            for i, op in enumerate(operations, 1):
                f.write(f"{i}. {op}\n")
        
        print(f"✓ 撤销操作记录已保存到: {undo_record_file}")
        print(f"\n✅ 撤销操作完成！共执行了 {len(operations)} 项撤销操作。")
        
    except Exception as e:
        print(f"\n❌ 撤销操作出错: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='复制和配置新的模型，支持撤销操作')
    parser.add_argument('base_model', type=str, help='基底模型名称，如: akt')
    parser.add_argument('new_model', type=str, help='新模型名称，如: Transformer_template')
    parser.add_argument('--undo', action='store_true', help='撤销操作，删除指定新模型的所有修改')
    
    args = parser.parse_args()
    base_model = args.base_model.lower()
    new_model = args.new_model  # 保持原有大小写
    
    if args.undo:
        # 执行撤销操作
        undo_operations(base_model, new_model)
        return
    
    print(f"开始处理：基底模型 = {base_model}, 新模型 = {new_model}")
    
    operations = []  # 记录执行的操作
    script_dir = get_script_dir()
    
    try:
        # 1. 复制并修改 wandb_[基底模型]_train.py
        src_train = os.path.join(script_dir, "examples", f"wandb_{base_model}_train.py")
        dst_train = os.path.join(script_dir, "examples", f"wandb_{new_model}_train.py")
        
        if os.path.exists(src_train):
            if os.path.exists(dst_train):
                print(f"⚠️  目标文件 {dst_train} 已存在，跳过复制")
            else:
                shutil.copy2(src_train, dst_train)
                operations.append(f"复制文件: {src_train} -> {dst_train}")
                print(f"✓ 复制 {src_train} -> {dst_train}")
                
                # 修改默认模型名称
                with open(dst_train, 'r', encoding='utf-8') as f:
                    content = f.read()
                content = re.sub(
                    r'parser\.add_argument\("--model_name",\s*type=str,\s*default="[^"]+"\)',
                    f'parser.add_argument("--model_name", type=str, default="{new_model}")',
                    content
                )
                with open(dst_train, 'w', encoding='utf-8') as f:
                    f.write(content)
                operations.append(f"修改 {dst_train} 中的默认模型名")
                print(f"✓ 更新 {dst_train} 中的默认模型名")
        else:
            print(f"⚠️  源文件 {src_train} 不存在，跳过复制")
        
        # 2. 更新 examples/wandb_train.py
        wandb_train = os.path.join(script_dir, "examples", "wandb_train.py")
        if os.path.exists(wandb_train):
            if add_model_to_lists(wandb_train, base_model, new_model):
                operations.append(f"更新 {wandb_train} 中的模型列表")
                print(f"✓ 更新 {wandb_train} 中的模型列表")
        else:
            print(f"⚠️  文件 {wandb_train} 不存在，跳过更新")
        
        # 3. 复制并修改模型文件
        src_model = os.path.join(script_dir, "pykt", "models", f"{base_model}.py")
        dst_model = os.path.join(script_dir, "pykt", "models", f"{new_model}.py")
        
        if os.path.exists(src_model):
            if os.path.exists(dst_model):
                print(f"⚠️  目标文件 {dst_model} 已存在，跳过复制")
            else:
                shutil.copy2(src_model, dst_model)
                operations.append(f"复制文件: {src_model} -> {dst_model}")
                print(f"✓ 复制 {src_model} -> {dst_model}")
                
                replace_model_name_in_file(dst_model, base_model, new_model)
                operations.append(f"修改 {dst_model} 中的类名和模型名")
                print(f"✓ 更新 {dst_model} 中的类名和模型名")
        else:
            print(f"⚠️  源文件 {src_model} 不存在，跳过复制")
        
        # 4. 更新 evaluate_model.py
        eval_model = os.path.join(script_dir, "pykt", "models", "evaluate_model.py")
        if os.path.exists(eval_model):
            if add_model_to_lists(eval_model, base_model, new_model):
                operations.append(f"更新 {eval_model} 中的模型列表")
                print(f"✓ 更新 {eval_model} 中的模型列表")
        else:
            print(f"⚠️  文件 {eval_model} 不存在，跳过更新")
        
        # 5. 更新 train_model.py
        train_model = os.path.join(script_dir, "pykt", "models", "train_model.py")
        if os.path.exists(train_model):
            if add_model_to_lists(train_model, base_model, new_model):
                operations.append(f"更新 {train_model} 中的模型列表")
                print(f"✓ 更新 {train_model} 中的模型列表")
        else:
            print(f"⚠️  文件 {train_model} 不存在，跳过更新")
        
        # 6. 更新 __init__.py
        init_file = os.path.join(script_dir, "pykt", "models", "init_model.py")
        if os.path.exists(init_file):
            if update_init_file(init_file, base_model, new_model):
                operations.append(f"更新 {init_file} 中的导入和初始化")
                print(f"✓ 更新 {init_file} 中的导入和初始化")
        else:
            print(f"⚠️  文件 {init_file} 不存在，跳过更新")
        
        # 7. 更新 pykt/datasets/init_dataset.py
        init_dataset = os.path.join(script_dir, "pykt", "datasets", "init_dataset.py")
        if os.path.exists(init_dataset):
            if add_model_to_lists(init_dataset, base_model, new_model):
                operations.append(f"更新 {init_dataset} 中的模型列表")
                print(f"✓ 更新 {init_dataset} 中的模型列表")
        else:
            print(f"⚠️  文件 {init_dataset} 不存在，跳过更新")
        # 8. 更新 config.py
        config_file = os.path.join(script_dir, "pykt", "config", "config.py")
        if os.path.exists(config_file):
            if add_model_to_lists(config_file, base_model, new_model):
                operations.append(f"更新 {config_file} 中的模型列表")
                print(f"✓ 更新 {config_file} 中的模型列表")
        else:
            print(f"⚠️  文件 {config_file} 不存在，跳过更新")
        # 9. 复制并修改 seedwandb YAML ===
        success, yaml_path = copy_and_update_seedwandb_yaml(script_dir, base_model, new_model)
        if success and yaml_path:
            operations.append(f"复制并配置 YAML: {yaml_path}")
            print(f"✓ 复制并配置 YAML: {yaml_path}")
        # ==========================================

        # 10. 保存操作记录
        if operations:
            record_file = save_operation_record(base_model, new_model, operations)
        
        print(f"\n✅ 所有操作完成！共执行了 {len(operations)} 项操作。")
        print(f"💡 如需撤销，请运行: python {os.path.basename(__file__)} {base_model} {new_model} --undo")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)
if __name__ == "__main__":
    main()