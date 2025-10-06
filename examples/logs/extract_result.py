import re
import os
from glob import glob
from collections import defaultdict
import json

def parse_filename(filename):
    """
    从文件名中解析出数据集名称、模型名称和折数。
    文件名格式：数据集_模型名称_fold_x.log
    假设数据集名称在预定义列表内，以处理带下划线的数据集名（如 nips_task34）。
    """
    if not filename.endswith('.log'):
        return None, None, None
    
    name_no_ext = filename[:-4]
    
    # 定义已知的带有下划线的数据集名称，以确保正确解析
    DATASETS = ["assist2009", "nips_task34", "peiyou"]
    dataset_name = None
    
    # 1. 识别数据集名称 (按长短匹配，优先匹配 nips_task34)
    # 检查文件名是否以某个数据集名开头
    for ds in sorted(DATASETS, key=len, reverse=True):
        if name_no_ext.startswith(ds + '_'):
            dataset_name = ds
            break

    if not dataset_name:
        return None, None, None # 未识别到已知数据集

    # 2. 移除数据集名称和后面的一个下划线，得到模型+折数部分
    remaining_part = name_no_ext[len(dataset_name) + 1:]
    
    parts = remaining_part.split('_')

    if len(parts) < 2 or parts[-2] != 'fold':
        return dataset_name, None, None # 格式不正确

    # 3. 折数：最后两个部分 (fold_x)
    fold_name = '_'.join(parts[-2:])
    
    # 4. 模型名称：除去折数剩下的部分
    model_name_parts = parts[:-2]
    model_name = '_'.join(model_name_parts)
    
    if not model_name:
         return dataset_name, None, fold_name # 模型名称为空，但数据集和折数已识别

    return dataset_name, model_name, fold_name

def extract_stats_from_content(content):
    """
    从文件内容中提取所需的统计数据。
    返回一个包含所有数据的字典，或None（如果找不到关键数据）。
    
    支持两种 AUC/ACC 提取格式：
    1. 原来的：{'windows_auc': 0.8201, 'windows_acc': 0.7638}
    2. 新的：{'windows_auc': ..., 'windowauclate_mean': 0.7469, ..., 'windowacclate_mean': 0.7193, ...}
       - 提取 'windowauclate_mean' 作为 auc
       - 提取 'windowacclate_mean' 作为 acc
    """
    data = {}
    
    # 1. 提取 'window_testauc 统计' 部分的四个值 (不变)
    stats_pattern = re.compile(
        r"window_testauc 统计:\s+"
        r"最小值: (?P<min>\d+\.\d+)\s+"
        r"最大值: (?P<max>\d+\.\d+)\s+"
        r"平均值: (?P<avg>\d+\.\d+)\s+"
        r"标准差: (?P<std>\d+\.\d+)",
        re.DOTALL
    )
    stats_match = stats_pattern.search(content)
    if stats_match:
        data.update(stats_match.groupdict())
    else:
        # 如果统计数据缺失，我们可以继续尝试提取 AUC/ACC，但最好也标记为缺失
        data.update({'min': 'N/A', 'max': 'N/A', 'avg': 'N/A', 'std': 'N/A'})

    # 2. 提取 AUC 和 ACC 值

    # 尝试提取第二种格式 (windowauclate_mean / windowacclate_mean)
    # 使用非贪婪匹配 '.*?' 来匹配字典中的其他内容
    dict_pattern_late = re.compile(
        r"'windowauclate_mean':\s*(?P<auc_late>\d+\.\d+).*?"
        r"'windowacclate_mean':\s*(?P<acc_late>\d+\.\d+)",
        re.DOTALL
    )
    dict_match_late = dict_pattern_late.search(content)

    if dict_match_late:
        data['auc'] = dict_match_late.group('auc_late')
        data['acc'] = dict_match_late.group('acc_late')
    else:
        # 尝试提取第一种格式 (windows_auc / windows_acc)
        dict_pattern_orig = re.compile(
            r"{'windows_auc':\s*(?P<auc_orig>\d+\.\d+),"
            r"\s*'windows_acc':\s*(?P<acc_orig>\d+\.\d+)",
            re.DOTALL
        )
        dict_match_orig = dict_pattern_orig.search(content)

        if dict_match_orig:
            data['auc'] = dict_match_orig.group('auc_orig')
            data['acc'] = dict_match_orig.group('acc_orig')
        else:
            # 两种 AUC/ACC 格式都未找到
            data.update({'auc': 'N/A', 'acc': 'N/A'})

    # 只有当 AUC/ACC 都是 'N/A' 且统计数据也缺失时，才认为提取失败
    if data['auc'] == 'N/A' and data['acc'] == 'N/A' and data['min'] == 'N/A':
         return None
        
    return data

def process_and_aggregate_stats(directory_path):
    """
    遍历指定路径下所有 .log 文件，提取数据，按模型聚合，并按指定顺序输出结果。
    """
    if not os.path.isdir(directory_path):
        print(f"错误：路径未找到或不是一个目录：{directory_path}")
        return

    # 使用 glob 查找所有 .log 文件
    search_pattern = os.path.join(directory_path, "*.log")
    file_list = glob(search_pattern)
    
    if not file_list:
        print(f"在 {directory_path} 中未找到任何 .log 文件。")
        return

    # 使用 defaultdict 来存储聚合数据：
    # aggregated_data = {model_name: {dataset_name: {fold_name: data}}}
    aggregated_data = defaultdict(lambda: defaultdict(dict))

    # 1. 遍历文件，提取和存储数据
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        
        # 解析文件名
        dataset, model, fold = parse_filename(file_name)
        
        if not all([dataset, model, fold]):
            print(f"警告：跳过文件 '{file_name}'，文件名格式不符合要求。")
            continue
            
        # 读取文件内容
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"错误：读取文件 '{file_name}' 时出错：{e}")
            continue

        # 提取数据
        data = extract_stats_from_content(content)

        if data:
            # 存储数据
            aggregated_data[model][dataset][fold] = {
                'file_name': file_name,
                'auc': data.get('auc', 'N/A'),
                'acc': data.get('acc', 'N/A'),
                'max': data.get('max', 'N/A'),
                'min': data.get('min', 'N/A'),
                'avg': data.get('avg', 'N/A'),
                'std': data.get('std', 'N/A')
            }
        else:
            print(f"警告：文件 '{file_name}' 数据提取失败，可能关键结果缺失。")

    # 2. 按要求格式化和输出结果
    
    # 定义表头和列的顺序
    header_cols = ["Dataset/Fold", "AUC", "ACC", "max", "min", "avg", "std"]
    header_line = '\t'.join(header_cols)
    
    # 定义数据集的显示顺序
    dataset_order = ["assist2009", "nips_task34", "peiyou"]

    print("\n" + "="*80)
    print("聚合结果 (按模型显示，数据集按 assist2009, nips_task34, peiyou 排序)")
    print("="*80 + "\n")

    # 遍历每个模型
    for model_name, dataset_data in aggregated_data.items():
        # 第一行：模型名称
        print(f"## 模型名称: {model_name}")
        print("-" * (10 + 7 * 8)) # 打印分隔线，长度与表头大致对齐
        print(header_line)
        
        # 遍历数据集
        for dataset_name in dataset_order:
            if dataset_name in dataset_data:
                folds_data = dataset_data[dataset_name]
                
                # 获取排序后的 fold 名称 (fold_0, fold_1, fold_2, ...)
                # 使用 sorted() 和 lambda 表达式来确保数字顺序正确
                sorted_folds = sorted(folds_data.keys(), key=lambda x: int(x.split('_')[-1]))
                
                # 打印数据集行（第一折）
                first_fold_name = sorted_folds[0]
                first_fold_data = folds_data[first_fold_name]
                
                row_data = [
                    f"{dataset_name} ({first_fold_name})", # 数据集名和第一折
                    first_fold_data['auc'],
                    first_fold_data['acc'],
                    first_fold_data['max'],
                    first_fold_data['min'],
                    first_fold_data['avg'],
                    first_fold_data['std']
                ]
                print('\t'.join(row_data))
                
                # 打印剩余的折数
                for fold_name in sorted_folds[1:]:
                    fold_data = folds_data[fold_name]
                    row_data = [
                        f"  ({fold_name})", # 仅显示折数，并缩进
                        fold_data['auc'],
                        fold_data['acc'],
                        fold_data['max'],
                        fold_data['min'],
                        fold_data['avg'],
                        fold_data['std']
                    ]
                    print('\t'.join(row_data))
            
        print("\n" + "~" * (10 + 7 * 8) + "\n") # 模型间分隔符

# --- 使用示例 ---
# 请将 '/root/pykt-toolkit/examples/logs/' 替换为你实际存放文件的目录路径
# 确保你的 .log 文件名遵循 '数据集_模型名称_fold_x.log' 的格式
target_directory = "/root/pykt-toolkit/examples/logs/"

# 执行主流程
process_and_aggregate_stats(target_directory)