#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型复制和配置脚本
用于复制基底模型并创建新模型的配置
"""

import os
import re
import sys
import shutil
import argparse

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

def add_model_to_lists(file_path, base_model, new_model):
    """在包含基底模型的列表中添加新模型"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查新模型是否已经存在于文件中
    if f'"{new_model}"' in content or f"'{new_model}'" in content:
        print(f"⚠️  {new_model} 已存在于 {file_path} 中，跳过添加")
        return False
    
    # 处理 if model_name in [...] 或 elif model_name in [...] 的情况
    pattern = r'((?:if|elif)\s+\w+\s+in\s*\[)([^\]]+)(\])'
    
    def replacer(match):
        prefix = match.group(1)
        items = match.group(2)
        suffix = match.group(3)
        
        # 检查是否包含基底模型
        if f'"{base_model}"' in items or f"'{base_model}'" in items:
            # 在末尾添加新模型 - 保持原有大小写
            items = items.rstrip()
            if items.endswith(','):
                new_items = f'{items} "{new_model}"'
            else:
                new_items = f'{items}, "{new_model}"'
            return prefix + new_items + suffix
        return match.group(0)
    
    content = re.sub(pattern, replacer, content)
    
    # 处理多行列表的情况
    pattern_multiline = r'((?:if|elif)\s+\w+\s+in\s*\[)([^\]]+?)(\])'
    content = re.sub(pattern_multiline, replacer, content, flags=re.DOTALL)
    
    # 处理变量赋值形式的列表
    pattern_var = r'(\w+\s*=\s*\[)([^\]]+)(\])'
    
    def var_replacer(match):
        prefix = match.group(1)
        items = match.group(2)
        suffix = match.group(3)
        
        # 检查是否包含基底模型
        if f'"{base_model}"' in items or f"'{base_model}'" in items:
            # 在末尾添加新模型 - 保持原有大小写
            items = items.rstrip()
            if items.endswith(','):
                new_items = f'{items} "{new_model}"'
            else:
                new_items = f'{items}, "{new_model}"'
            return prefix + new_items + suffix
        return match.group(0)
    
    content = re.sub(pattern_var, var_replacer, content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return True

def update_init_file(file_path, base_model, new_model):
    """更新__init__.py文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查新模型的import是否已经存在
    if f'from .{new_model} import {new_model.upper()}' in content:
        print(f"⚠️  {new_model} 的导入已存在于 {file_path} 中，跳过添加")
        return False
    
    # 检查新模型的初始化是否已经存在
    if f'model_name == "{new_model}"' in content:
        print(f"⚠️  {new_model} 的初始化已存在于 {file_path} 中，跳过添加")
        return False
    
    # 查找基底模型的import语句
    import_pattern = rf'from\s+\.{base_model}\s+import\s+{base_model.upper()}'
    import_match = re.search(import_pattern, content)
    
    if import_match:
        # 在基底模型import后添加新模型的import - 文件名保持原有大小写
        new_import = f'\nfrom .{new_model} import {new_model.upper()}'
        insert_pos = import_match.end()
        content = content[:insert_pos] + new_import + content[insert_pos:]
    
    # 查找并复制模型初始化语句
    # 修改正则表达式以匹配多行格式和if/elif两种情况
    # 匹配格式：if/elif model_name == "base_model": \n    model = CLASS(...).to(device)
    init_pattern = rf'((?:if|elif)\s+model_name\s*==\s*["\']({base_model})["\']:\s*\n\s+model\s*=\s*{base_model.upper()}\((.*?)\)\.to\(device\))'
    
    init_match = re.search(init_pattern, content, re.DOTALL)
    
    if init_match:
        # 提取参数部分
        params = init_match.group(3)
        # 在基底模型初始化后添加新模型的初始化 - 保持原有大小写
        new_init = f'\n    elif model_name == "{new_model}":\n        model = {new_model.upper()}({params}).to(device)'
        insert_pos = init_match.end()
        content = content[:insert_pos] + new_init + content[insert_pos:]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return True
        
def main():
    parser = argparse.ArgumentParser(description='复制和配置新的模型')
    parser.add_argument('base_model', type=str, help='基底模型名称，如: akt')
    parser.add_argument('new_model', type=str, help='新模型名称，如: Transformer_template')
    
    args = parser.parse_args()
    base_model = args.base_model.lower()
    new_model = args.new_model  # 保持原有大小写，不再强制转换为小写
    
    print(f"开始处理：基底模型 = {base_model}, 新模型 = {new_model}")
    
    try:
        # 1. 复制并修改 wandb_[基底模型]_train.py
        src_train = f"examples/wandb_{base_model}_train.py"
        dst_train = f"examples/wandb_{new_model}_train.py"  # 保持原有大小写
        
        if os.path.exists(src_train):
            if os.path.exists(dst_train):
                print(f"⚠️  目标文件 {dst_train} 已存在，跳过复制")
            else:
                shutil.copy2(src_train, dst_train)
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
                print(f"✓ 更新 {dst_train} 中的默认模型名")
        else:
            print(f"⚠️  源文件 {src_train} 不存在，跳过复制")
        
        # 2. 更新 examples/wandb_train.py
        wandb_train = "examples/wandb_train.py"
        if os.path.exists(wandb_train):
            if add_model_to_lists(wandb_train, base_model, new_model):
                print(f"✓ 更新 {wandb_train} 中的模型列表")
        else:
            print(f"⚠️  文件 {wandb_train} 不存在，跳过更新")
        
        # 3. 复制并修改模型文件
        src_model = f"pykt/models/{base_model}.py"
        dst_model = f"pykt/models/{new_model}.py"  # 保持原有大小写
        
        if os.path.exists(src_model):
            if os.path.exists(dst_model):
                print(f"⚠️  目标文件 {dst_model} 已存在，跳过复制")
            else:
                shutil.copy2(src_model, dst_model)
                print(f"✓ 复制 {src_model} -> {dst_model}")
                
                replace_model_name_in_file(dst_model, base_model, new_model)
                print(f"✓ 更新 {dst_model} 中的类名和模型名")
        else:
            print(f"⚠️  源文件 {src_model} 不存在，跳过复制")
        
        # 4. 更新 evaluate_model.py
        eval_model = "pykt/models/evaluate_model.py"
        if os.path.exists(eval_model):
            if add_model_to_lists(eval_model, base_model, new_model):
                print(f"✓ 更新 {eval_model} 中的模型列表")
        else:
            print(f"⚠️  文件 {eval_model} 不存在，跳过更新")
        
        # 5. 更新 train_model.py
        train_model = "pykt/models/train_model.py"
        if os.path.exists(train_model):
            if add_model_to_lists(train_model, base_model, new_model):
                print(f"✓ 更新 {train_model} 中的模型列表")
        else:
            print(f"⚠️  文件 {train_model} 不存在，跳过更新")
        
        # 6. 更新 __init__.py
        init_file = "pykt/models/init_model.py"
        if os.path.exists(init_file):
            if update_init_file(init_file, base_model, new_model):
                print(f"✓ 更新 {init_file} 中的导入和初始化")
        else:
            print(f"⚠️  文件 {init_file} 不存在，跳过更新")
        
        # 7. 更新 pykt/datasets/init_dataset.py
        init_dataset = "pykt/datasets/init_dataset.py"
        if os.path.exists(init_dataset):
            if add_model_to_lists(init_dataset, base_model, new_model):
                print(f"✓ 更新 {init_dataset} 中的模型列表")
        else:
            print(f"⚠️  文件 {init_dataset} 不存在，跳过更新")
        
        print("\n✅ 所有操作完成！")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()