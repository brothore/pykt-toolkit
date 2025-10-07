import re
import os
from glob import glob
from collections import defaultdict
import json
import datetime
import math # 引入 math 库用于处理 Inf
import numpy as np # 引入 numpy 库用于高效计算 mean 和 std

# --- 辅助函数保持不变：parse_filename 和 extract_stats_from_content ---

def parse_filename(filename):
    """
    从文件名中解析出数据集名称、模型名称和折数。
    """
    if not filename.endswith('.log'):
        return None, None, None

    # 移除扩展名
    name_no_ext = filename[:-4]

    # 定义已知的带有下划线的数据集名称，以确保正确解析
    DATASETS = ["assist2009", "nips_task34", "peiyou"]
    
    # 1. 从右往左寻找 '_fold_x_' 的位置
    fold_marker = '_fold_'
    
    try:
        # 寻找最后一个 _fold_ 的位置
        fold_start_index = name_no_ext.rindex(fold_marker)
    except ValueError:
        return None, None, None 

    # 2. 识别数据集名称：'_fold_' 之前的部分
    dataset_name = name_no_ext[:fold_start_index]
    
    # 3. 识别折数 x 和模型名称
    remaining_part = name_no_ext[fold_start_index + len(fold_marker):] # 例如: '0_tcn_hid8_k3' 或 '0_no_c'
    
    fold_x = "UNKNOWN"
    model_name = "UNKNOWN"
    
    try:
        # 找到折数 x 和模型名称之间的第一个下划线
        model_name_start_index_relative = remaining_part.index('_')
        
        # 折数 x：从剩余部分开始到第一个下划线为止
        fold_x = remaining_part[:model_name_start_index_relative] # 例如: '0'
        
        # 模型名称：从第一个下划线之后的所有内容
        model_name = remaining_part[model_name_start_index_relative + 1:] # 例如: 'tcn_hid8_k3'
        
        # 检查模型名称是否为空
        if not model_name:
             model_name = "UNKNOWN_MODEL"
             
    except ValueError:
        # 如果 remaining_part 中没有下划线（即文件名是 dataset_fold_x.log 格式）
        fold_x = remaining_part 
        model_name = "UNKNOWN_MODEL"
        
    # 4. 最终检查
    if dataset_name not in DATASETS:
        return None, None, None
        
    return dataset_name, model_name, fold_x


def extract_stats_from_content(content):
    """
    从文件内容中提取所需的统计数据。
    返回一个包含所有数据的字典，或None（如果找不到关键数据）。
    """
    data = {}
    
    # 1. 提取 'window_testauc 统计' 部分的四个值 (max, min, avg, std)
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
        # 如果统计数据缺失
        data.update({'min': 'N/A', 'max': 'N/A', 'avg': 'N/A', 'std': 'N/A'})

    # 2. 提取 AUC 和 ACC 值

    # 尝试提取第二种格式 (windowauclate_mean / windowacclate_mean)
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

# --- 主处理函数修改：增加五折聚合计算逻辑 ---

def process_and_aggregate_stats(directory_path):
    """
    遍历指定路径下所有 .log 文件，提取数据，按 数据集 -> 折数 -> 模型 聚合，
    并按指定顺序输出结果。最后输出每个模型每个数据集的五折 mean±std 结果。
    """
    if not os.path.isdir(directory_path):
        print(f"错误：路径未找到或不是一个目录：{directory_path}")
        return
        
    search_pattern = os.path.join(directory_path, "*.log")
    file_list = glob(search_pattern)
    
    if not file_list:
        print(f"在 {directory_path} 中未找到任何 .log 文件。")
        return

    # 聚合数据结构 1 (用于原始按折数输出): aggregated_data_new = {dataset_name: {fold_name: {model_name: data}}}
    aggregated_data_new = defaultdict(lambda: defaultdict(dict))
    
    # 聚合数据结构 2 (用于五折 mean±std 计算): fold_stats = {dataset_name: {model_name: {metric: [val1, val2, ...]}}}
    fold_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # 定义特定的模型名称及其输出顺序
    SPECIAL_MODELS = ["no_q_all", "no_c", "no_attn", "no_irt", "origin"]
    
    # 所有需要聚合计算 mean±std 的指标
    AGGREGATE_METRICS = ['auc', 'max', 'min', 'avg', 'std', 'epi']

    # 1. 遍历文件，提取和存储数据
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        
        # 解析文件名
        dataset, model, fold = parse_filename(file_name)
        
        if not all([dataset, model, fold]):
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
            # --- 计算 epi: avg / ( (max - min) * std ) ---
            epi = 'N/A'
            try:
                max_val = float(data.get('max'))
                min_val = float(data.get('min'))
                avg_val = float(data.get('avg'))
                std_val = float(data.get('std'))
                
                # range = max - min
                range_val = max_val - min_val 
                
                # 计算 epi = avg / (range * std)
                denominator = range_val * std_val
                
                if abs(denominator) > 1e-9: # 避免除以接近零的数
                    epi_val = avg_val / denominator
                    epi = f"{epi_val:.4f}" # 格式化为4位小数
                elif abs(avg_val) > 1e-9:
                    epi_val = math.inf # 使用 Inf 的数值表示，方便后续处理
                    epi = 'Inf' # 分母为零，分子不为零
                else:
                    epi_val = math.nan # 使用 NaN 的数值表示
                    epi = 'NaN' # 分子分母都为零
            except (ValueError, TypeError):
                # 如果任何一个值是 'N/A' 或无法转换，则 epi 保持 'N/A'
                epi_val = math.nan
                epi = 'N/A'
            # --- epi 计算结束 ---

            # 存储数据到结构 1 (用于原始输出)
            data['epi'] = epi # 将计算出的 epi 存入 data
            aggregated_data_new[dataset][fold][model] = {
                'auc': data.get('auc', 'N/A'),
                'acc': data.get('acc', 'N/A'),
                'max': data.get('max', 'N/A'),
                'min': data.get('min', 'N/A'),
                'avg': data.get('avg', 'N/A'),
                'std': data.get('std', 'N/A'),
                'epi': epi # 存储计算出的 epi 字符串
            }
            
            # 存储数据到结构 2 (用于五折聚合计算)
            for metric in AGGREGATE_METRICS:
                try:
                    # 对于 epi，使用计算出的数值 epi_val
                    val_str = data.get(metric, 'N/A')
                    val = float(epi_val) if metric == 'epi' and epi != 'N/A' else float(val_str)
                    
                    # 排除 Inf 和 NaN 之外的值才加入列表进行 mean/std 计算
                    if not math.isinf(val) and not math.isnan(val):
                        fold_stats[dataset][model][metric].append(val)
                except (ValueError, TypeError):
                    # 忽略无法转换为 float 的 'N/A' 或其他非数值
                    pass


    # 2. 原始按折数输出结果
    print("\n" + "="*80)
    print("原始聚合结果 (按数据集和折数划分)")
    print("="*80 + "\n")
    
    # 定义数据集的显示顺序
    dataset_order = ["assist2009", "nips_task34", "peiyou"]

    # 遍历并输出原始结果 (与原逻辑保持一致)
    for dataset_name in dataset_order:
        if dataset_name in aggregated_data_new:
            dataset_data = aggregated_data_new[dataset_name]
            
            print(f"## 数据集: {dataset_name}\n")
            
            try:
                sorted_folds = sorted(dataset_data.keys(), key=int)
            except ValueError:
                sorted_folds = sorted(dataset_data.keys())

            for fold_name in sorted_folds:
                models_data = dataset_data[fold_name]
                all_models = list(models_data.keys())
                
                ordered_models = []
                temp_all_models = all_models[:]
                for special_model in SPECIAL_MODELS:
                    if special_model in temp_all_models:
                        ordered_models.append(special_model)
                        temp_all_models.remove(special_model)
                        
                temp_all_models.sort()
                ordered_models.extend(temp_all_models)
                
                if not ordered_models:
                    continue
                    
                print(f"--- 折数: {fold_name} ---")
                
                # AUC/ACC 区块
                auc_acc_header = ["AUC", "ACC", "EPI", "Model_Name"] 
                print('\t'.join(auc_acc_header))
                
                for model in ordered_models:
                    data = models_data[model]
                    row = [
                        data.get('auc', 'N/A'),
                        data.get('acc', 'N/A'),
                        data.get('epi', 'N/A'),
                        model
                    ]
                    print('\t'.join(row))

                # 剩余指标区块
                print("\n")
                other_metrics_header = ["max", "min", "avg", "std", "EPI", "Model_Name"] 
                print('\t'.join(other_metrics_header))
                
                metrics = ['max', 'min', 'avg', 'std', 'epi'] 
                
                for model in ordered_models:
                    data = models_data[model]
                    row = []
                    for metric in metrics:
                        row.append(data.get(metric, 'N/A'))
                    row.append(model)
                    
                    print('\t'.join(row))

                print("=" * 50 + "\n")


    # 3. 计算并输出五折的 mean±std 结果
    
    print("\n" + "="*80)
    print("五折聚合结果 (Mean±Std)")
    print("="*80 + "\n")
    
    # 确定所有出现过的模型名称的集合，用于统一排序
    all_models_set = set()
    for dataset_name in fold_stats:
        all_models_set.update(fold_stats[dataset_name].keys())
        
    # 按照 SPECIAL_MODELS 优先，其余字母序的方式排序
    final_ordered_models = []
    temp_all_models_list = list(all_models_set)
    for special_model in SPECIAL_MODELS:
        if special_model in temp_all_models_list:
            final_ordered_models.append(special_model)
            temp_all_models_list.remove(special_model)
            
    temp_all_models_list.sort()
    final_ordered_models.extend(temp_all_models_list)


    for dataset_name in dataset_order:
        if dataset_name in fold_stats:
            dataset_stats = fold_stats[dataset_name]
            
            print(f"## 数据集: {dataset_name}\n")
            
            # 打印表头
            header = ["Model_Name"] + [m.upper() for m in AGGREGATE_METRICS]
            print('\t'.join(header))
            
            # 打印每个模型的 mean±std 结果
            for model_name in final_ordered_models:
                if model_name in dataset_stats:
                    row_data = [model_name]
                    model_stats = dataset_stats[model_name]
                    
                    for metric in AGGREGATE_METRICS:
                        values = model_stats.get(metric, [])
                        
                        if len(values) > 0:
                            # 使用 numpy 计算 mean 和 std
                            mean_val = np.mean(values)
                            std_val = np.std(values)
                            # 格式化为 mean±std 四位小数
                            formatted_result = f"{mean_val:.4f}±{std_val:.4f}"
                        elif metric == 'epi' and 'Inf' in [aggregated_data_new[dataset_name].get(f, {}).get(model_name, {}).get('epi') for f in aggregated_data_new[dataset_name]]:
                            # 如果该模型-数据集的 epi 结果中有 Inf，但因为 Inf 不参与 mean/std 计算导致列表为空，则显示 'Inf/NaN'
                            formatted_result = 'Inf/NaN'
                        else:
                            formatted_result = 'N/A'
                            
                        row_data.append(formatted_result)
                        
                    print('\t'.join(row_data))
                
            print("=" * 50 + "\n")
            
# --- 使用示例 ---
# 请将 '/root/pykt-toolkit/examples/logs/' 替换为你实际存放文件的目录路径
# 注意：你需要安装 numpy 库 (pip install numpy) 才能运行此代码
target_directory = "/root/pykt-toolkit/examples/logs/"

# 执行主流程
process_and_aggregate_stats(target_directory)