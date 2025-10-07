import re
import os
from glob import glob
from collections import defaultdict
import numpy as np

def parse_filename(filename):
    """
    从文件名中解析出数据集名称、模型名称和折数。
    文件名格式：数据集_模型名称_fold_x.txt
    假设数据集名称在预定义列表内，以处理带下划线的数据集名（如 nips_task34）。
    """
    if not filename.endswith('.txt'):
        return None, None, None
    
    name_no_ext = filename[:-4]
    
    # 定义已知的带有下划线的数据集名称，以确保正确解析
    DATASETS = ["assist2009", "nips_task34", "peiyou"]
    dataset_name = None
    
    # 1. 识别数据集名称 (按长短匹配，优先匹配 nips_task34)
    for ds in sorted(DATASETS, key=len, reverse=True):
        if name_no_ext.startswith(ds + '_'):
            dataset_name = ds
            break

    if not dataset_name:
        return None, None, None

    # 2. 移除数据集名称和后面的一个下划线
    remaining_part = name_no_ext[len(dataset_name) + 1:]
    
    parts = remaining_part.split('_')

    if len(parts) < 2 or parts[-2] != 'fold':
        return dataset_name, None, None

    # 3. 折数：最后两个部分 (fold_x)
    fold_name = '_'.join(parts[-2:])
    
    # 4. 模型名称：除去折数剩下的部分
    model_name_parts = parts[:-2]
    model_name = '_'.join(model_name_parts)
    
    if not model_name:
           return dataset_name, None, fold_name

    return dataset_name, model_name, fold_name

def extract_stats_from_content(content):
    """
    从文件内容中提取所需的统计数据。
    """
    data = {}
    
    # 1. 提取 'window_testauc 统计' 部分的四个值
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
        data.update({'min': 'N/A', 'max': 'N/A', 'avg': 'N/A', 'std': 'N/A'})

    # 2. 提取 AUC 和 ACC 值
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
            data.update({'auc': 'N/A', 'acc': 'N/A'})

    # 只有当 AUC/ACC 都是 'N/A' 且统计数据也缺失时，才认为提取失败
    if data['auc'] == 'N/A' and data['acc'] == 'N/A' and data['min'] == 'N/A':
          return None
        
    return data

def calculate_epi(mean, max_val, min_val, std_val):
    """
    计算 EPI 指标: mean / ((max - min) * std)
    """
    try:
        mean = float(mean)
        max_val = float(max_val)
        min_val = float(min_val)
        std_val = float(std_val)
    except ValueError:
        return np.nan # 返回 NaN

    denominator = (max_val - min_val) * std_val
    if denominator == 0:
        if mean == 0:
            return 0.0
        return np.nan # 无法计算，返回 NaN
    
    return mean / denominator

def format_mean_std(values):
    """
    计算并格式化为 mean +/- std，保留四位小数。
    """
    valid_values = [v for v in values if not np.isnan(v)]
    
    if not valid_values:
        return 'N/A'
    
    # 将列表转换为 numpy 数组
    data = np.array(valid_values)
    
    # 计算均值 (mean) 和标准差 (std)
    mean = np.mean(data)
    std = np.std(data, ddof=0) # 默认使用总体标准差 (ddof=0)
    
    return f"{mean:.4f}±{std:.4f}"

def aggregate_and_calculate_avg_corrected(folds_data):
    """
    根据用户要求，计算所有列的简单算术平均和标准差。
    """
    # 存储每列的数值列表
    metrics = defaultdict(list)
    
    # 提取所有数值，将 N/A 转换为 np.nan
    for fold_data in folds_data.values():
        # 提取 AUC, ACC, max, min, avg, std
        for key in ["auc", "acc", "max", "min", "avg", "std"]:
            try:
                metrics[key].append(float(fold_data[key]))
            except (ValueError, TypeError):
                metrics[key].append(np.nan) # 使用 NaN 标记缺失值

        # 计算并提取 EPI
        mean = metrics['avg'][-1]
        max_val = metrics['max'][-1]
        min_val = metrics['min'][-1]
        std_val = metrics['std'][-1]
        
        epi_val = calculate_epi(mean, max_val, min_val, std_val)
        metrics['epi'].append(epi_val)

    # 最终的平均结果字典
    avg_results = {}
    
    # 计算并格式化每一列的 mean±std
    for key in ["auc", "acc", "max", "min", "avg", "std", "epi"]:
        avg_results[key] = format_mean_std(metrics[key])

    return avg_results


def process_and_aggregate_stats(directory_path):
    """
    遍历指定路径下所有 .txt 文件，提取数据，按模型聚合，并按指定顺序输出结果。
    """
    if not os.path.isdir(directory_path):
        print(f"错误：路径未找到或不是一个目录：{directory_path}")
        return

    # 使用 glob 查找所有 .txt 文件
    search_pattern = os.path.join(directory_path, "*.txt")
    file_list = glob(search_pattern)
    print(f"file_list: {file_list}")
    if not file_list:
        print(f"在 {directory_path} 中未找到任何 .txt 文件。")
        return

    # aggregated_data = {model_name: {dataset_name: {fold_name: data}}}
    aggregated_data = defaultdict(lambda: defaultdict(dict))

    # 1. 遍历文件，提取和存储数据
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        
        dataset, model, fold = parse_filename(file_name)
        
        if not all([dataset, model, fold]):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"错误：读取文件 '{file_name}' 时出错：{e}")
            continue

        data = extract_stats_from_content(content)

        if data:
            # 尝试转换为浮点数以计算 EPI，如果失败则使用 np.nan
            try:
                avg = float(data['avg'])
                max_val = float(data['max'])
                min_val = float(data['min'])
                std = float(data['std'])
            except ValueError:
                avg = np.nan
                max_val = np.nan
                min_val = np.nan
                std = np.nan
                
            epi_val = calculate_epi(avg, max_val, min_val, std)
            
            # 存储单折数据
            aggregated_data[model][dataset][fold] = {
                'file_name': file_name,
                # 单折数据保留四位小数的字符串格式
                'auc': f"{float(data['auc']):.4f}" if data['auc'] != 'N/A' else 'N/A',
                'acc': f"{float(data['acc']):.4f}" if data['acc'] != 'N/A' else 'N/A',
                'max': f"{float(data['max']):.4f}" if data['max'] != 'N/A' else 'N/A',
                'min': f"{float(data['min']):.4f}" if data['min'] != 'N/A' else 'N/A',
                'avg': f"{float(data['avg']):.4f}" if data['avg'] != 'N/A' else 'N/A',
                'std': f"{float(data['std']):.4f}" if data['std'] != 'N/A' else 'N/A',
                'epi': f"{epi_val:.4f}" if not np.isnan(epi_val) else 'N/A'
            }
        else:
            print(f"警告：文件 '{file_name}' 数据提取失败，可能关键结果缺失。")

    # 2. 按要求格式化和输出结果
    
    header_cols = ["Dataset/Fold", "AUC", "ACC", "max", "min", "avg", "std", "EPI"]
    header_line = '\t'.join(header_cols)
    dataset_order = ["assist2009", "nips_task34", "peiyou"]

    print("\n" + "="*90)
    print("聚合结果 (已修正平均算法，并以 mean±std 格式展示)")
    print("EPI (Empirical Performance Index) 计算方式: avg / ((max - min) * std)")
    print("注意：平均行是所有折对应指标的算术平均值±标准差。")
    print("="*90 + "\n")

    for model_name, dataset_data in aggregated_data.items():
        print(f"## 模型名称: {model_name}")
        print("-" * (10 + 8 * 8))
        print(header_line)
        
        for dataset_name in dataset_order:
            if dataset_name in dataset_data:
                folds_data = dataset_data[dataset_name]
                
                # 1. 打印所有折的结果 (单折数据仍显示为单值)
                sorted_folds = sorted(folds_data.keys(), key=lambda x: int(x.split('_')[-1]))
                
                is_first_fold = True
                for fold_name in sorted_folds:
                    fold_data = folds_data[fold_name]
                    
                    row_prefix = f"{dataset_name} ({fold_name})" if is_first_fold else f"  ({fold_name})"
                    is_first_fold = False
                        
                    row_data = [
                        row_prefix,
                        fold_data['auc'],
                        fold_data['acc'],
                        fold_data['max'],
                        fold_data['min'],
                        fold_data['avg'],
                        fold_data['std'],
                        fold_data['epi']
                    ]
                    print('\t'.join(row_data))
                
                # 2. 打印平均行 (mean±std 格式)
                avg_data = aggregate_and_calculate_avg_corrected(folds_data)
                
                # 格式化平均行
                avg_row_data = [
                    f"  {len(folds_data)} 折平均", # 显示实际折数
                    avg_data['auc'],
                    avg_data['acc'],
                    avg_data['max'],
                    avg_data['min'],
                    avg_data['avg'],
                    avg_data['std'],
                    avg_data['epi']
                ]
                print('\t'.join(avg_row_data))
                print("-" * (10 + 8 * 8))
        
        print("\n" + "~" * (10 + 8 * 8) + "\n")

# --- 使用示例 ---
# 请将 '/root/pykt-toolkit/examples/logs/' 替换为你实际存放文件的目录路径
# 确保你的 .txt 文件名遵循 '数据集_模型名称_fold_x.txt' 的格式
target_directory = "/root/pykt-toolkit/examples/logs/old_model"

# 执行主流程
process_and_aggregate_stats(target_directory)