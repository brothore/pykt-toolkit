import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score
import os # <-- 新增
import sys # <-- 新增
from glob import glob # <-- 新增
from datetime import datetime # <-- 新增
ENABLE_SLIDING_WINDOW = True
WINDOW_SIZE = 200
def calculate_wealth_metrics(wealth_list, alphas=[1.0, 2.0, 3.0]):
    """
    计算平均财富、基尼系数、以及多个 alpha 下的 EAWI 指标。
    EAWI_alpha = average_wealth * (1 - G) ** alpha
    """
    if not wealth_list or len(wealth_list) == 0:
        avg = 0.0
        G = 0.0
        eawi_dict = {f'eawi_alpha_{a}': 0.0 for a in alphas}
        return avg, G, eawi_dict

    wealth = np.array(wealth_list)
    n = len(wealth)
    W = float(np.sum(wealth))
    average_wealth = W / n

    G = gini_coefficient(wealth_list)  # 复用你写好的函数

    eawi_dict = {}
    for alpha in alphas:
        eawi = average_wealth * ((1 - G) ** alpha)
        eawi_dict[f'eawi_alpha_{alpha:.1f}'.replace('.', '')] = float(eawi)
        # 结果键名：eawi_alpha_10, eawi_alpha_20, eawi_alpha_30

    return average_wealth, G, eawi_dict
def extract_model_name(path):
    """
    从路径的倒数第3个下划线后面部分，最后一个下划线之前提取模型名字。
    例: /path/to/my_model_gru_td0.0_0/window_predictions.txt 
    -> 文件夹名: my_model_gru_td0.0_0
    -> 倒数第3个下划线后: gru_td0.0_0
    -> 最后一个下划线前: gru_td0.0
    """
    # 确保路径是目录名（如果传入的是文件路径，需要获取其目录名）
    if os.path.isfile(path):
        folder_name = os.path.basename(os.path.dirname(path))
    else:
        folder_name = os.path.basename(path)
        
    parts = folder_name.split('_')
    
    # 需要至少有3个下划线来满足“倒数第3个下划线后面”的条件
    if len(parts) < 3:
        # 如果不满足条件，可以返回整个文件夹名或一个默认值
        print(f"⚠️ 文件夹名不符合模型命名规则: {folder_name}")
        return folder_name

    # 倒数第3个下划线是 parts[-3] 之前
    # 从 parts 的倒数第三个元素开始截取
    sub_parts = parts[-3:]

    # 倒数第3个下划线后: sub_parts[0] 及其后
    # 最后一个下划线前: sub_parts[:-1]
    
    # 重新拼接：倒数第3个下划线后的部分（sub_parts）
    # 排除最后一个下划线后的内容 (sub_parts[:-1])
    
    # 假设我们要提取的是 `gru_td0.0`
    # 文件夹名: `xxx_gru_td0.0_0`
    # parts: `[..., 'gru', 'td0', '0', '0']` (如果路径是 /path/to/my_model_gru_td0.0_0)
    
    # 我们应该获取文件夹名，然后从后往前找下划线。
    
    # 重新实现：
    # 1. 找到所有下划线的索引
    indices = [i for i, char in enumerate(folder_name) if char == '_']

    if len(indices) < 3:
        return folder_name # 无法满足3个下划线的条件

    # 2. 倒数第3个下划线的索引
    third_last_underscore_index = indices[-3]

    # 3. 最后一个下划线的索引
    last_underscore_index = indices[-1]

    # 4. 提取子字符串
    # 倒数第3个下划线后面 (不包含该下划线)
    start_index = third_last_underscore_index + 1
    # 最后一个下划线之前 (不包含该下划线)
    end_index = last_underscore_index
    
    model_name = folder_name[start_index:end_index]
    
    # 防止提取空字符串
    if not model_name:
         print(f"⚠️ 提取模型名为空，使用文件夹名: {folder_name}")
         return folder_name
         
    return model_name
def gini_coefficient(wealth):
    """
    独立的函数，用于计算基尼系数 (Gini Coefficient)。
   
    参数:
    wealth (list or array-like): 财富列表，每个元素代表一个个体的财富。
   
    返回:
    float: 基尼系数 (0 到 1 之间)。
    """
    if len(wealth) == 0:
        return 0.0
    wealth = np.array(wealth)
    if np.all(wealth == 0):
        return 0.0
    # 确保财富非负（基尼系数假设非负值）
    if np.any(wealth < 0):
        raise ValueError("财富值不能为负数")
    sorted_wealth = np.sort(wealth)
    n = len(wealth)
    index = np.arange(1, n + 1)
    # 标准基尼系数公式
    numerator = np.sum((2 * index - n - 1) * sorted_wealth)
    denominator = n * np.sum(sorted_wealth)
    return numerator / denominator
def safe_roc_auc(y_true, y_score, dummy_score_strategy='mean'):
    """
    计算 AUC，处理单一类别情况通过添加虚拟数据点。
    :param y_true: 真实标签 (numpy array)
    :param y_score: 预测概率 (numpy array)
    :param dummy_score_strategy: 虚拟点分数策略 ('mean', 'min', 'max')
    :return: AUC 分数
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    unique_labels = np.unique(y_true)
    
    if len(unique_labels) < 2:
        label = unique_labels[0]
        dummy_label = 1 - label  # 相反类别
        # 选择虚拟点的预测分数
        if dummy_score_strategy == 'mean':
            dummy_score = np.mean(y_score)
        elif dummy_score_strategy == 'min':
            dummy_score = np.min(y_score)
        elif dummy_score_strategy == 'max':
            dummy_score = np.max(y_score)
        else:
            raise ValueError("Invalid dummy_score_strategy")
        # 添加虚拟数据点
        y_true = np.append(y_true, dummy_label)
        y_score = np.append(y_score, dummy_score)
    return roc_auc_score(y_true=y_true, y_score=y_score)
# --- 辅助函数：滑窗数据收集 (保持不变) ---
def sliding_window_collect(trues_series, scores_series, window_size=200):
    """
    对一个学生的完整序列进行滑窗（每个时间步都是一个数据点），
    并将所有窗口的数据点汇集起来。

    Args:
        trues_series (np.array): 学生的真实标签完整序列（每个元素是一个时间步）。
        scores_series (np.array): 学生的预测得分完整序列（每个元素是一个时间步）。
        window_size (int): 滑窗大小，默认为200。

    Returns:
        tuple: (collected_trues, collected_scores)
               collected_trues (list): 汇集后的真实标签数据点。
               collected_scores (list): 汇集后的预测得分数据点。
    """
    if len(trues_series) != len(scores_series):
        raise ValueError("True labels and scores must have the same length.")

    total_len = len(trues_series)
    
    collected_trues = []
    collected_scores = []

    # 如果序列长度小于窗口大小，则只取完整的序列
    if total_len < window_size:
        return trues_series.tolist(), scores_series.tolist()
    
    # 滑窗：从第一个位置开始，每次移动一步
    for start in range(total_len - window_size + 1):
        end = start + window_size
        
        # 收集当前窗口的数据
        collected_trues.extend(trues_series[start:end].tolist())
        collected_scores.extend(scores_series[start:end].tolist())
            
    return collected_trues, collected_scores


# --- 主要解析和计算函数 (逻辑已修改) ---
def parse_and_calculate_aucs_from_file(file_path):
    """
    解析模型预测结果文件（本地路径），计算每个学生的汇集滑窗AUC、
    以及【所有学生所有滑窗序列拼接后】的整体AUC。
    """
    
    # 1. 读取和初步解析数据
    try:
        data = pd.read_csv(file_path, sep='\s+')
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到: {file_path}")
    except Exception as e:
        raise Exception(f"读取文件时发生错误: {e}")
    # 在数据准备后添加检查
    print(f"late_trues 的唯一值: {data['late_trues'].unique()}")
    print(f"late_trues 的值计数:\n{data['late_trues'].value_counts()}")

    
    required_cols = ['orirow', 'late_trues', 'late_mean']
    if not all(col in data.columns for col in required_cols):
        data.columns = data.columns.str.strip()
        if not all(col in data.columns for col in required_cols):
             raise ValueError(f"TXT文件缺少必需的列: {required_cols}. 找到的列为: {list(data.columns)}")

    # 2. 数据准备
    data['late_trues'] = pd.to_numeric(data['late_trues'], errors='coerce')
    data['late_mean'] = pd.to_numeric(data['late_mean'], errors='coerce')
    data = data.dropna(subset=['late_trues', 'late_mean'])

    # 3. 按学生ID (orirow) 重组序列、滑窗并计算 AUC
    student_aucs = {}
    
    # *** 新增：存储所有学生的滑窗汇集数据，用于计算整体 AUC ***
    all_window_trues_flat = []
    all_window_scores_flat = []

    # 分组处理每个学生
    for student_uid, group in data.groupby('orirow'):
        student_trues_long = group['late_trues'].values
        student_scores_long = group['late_mean'].values

        # === 根据开关选择：滑窗 or 整序列 ===
        if ENABLE_SLIDING_WINDOW:
            collected_trues, collected_scores = sliding_window_collect(
                student_trues_long, 
                student_scores_long, 
                window_size=WINDOW_SIZE
            )
            print(f"  [滑窗模式] 学生 {student_uid}: {len(student_trues_long)} → {len(collected_trues)} 个点")
        else:
            collected_trues = student_trues_long.tolist()
            collected_scores = student_scores_long.tolist()
            print(f"  [整序列模式] 学生 {student_uid}: {len(collected_trues)} 个点")
        # ======================================

        # 加入整体汇集（用于 overall_auc）
        all_window_trues_flat.extend(collected_trues)
        all_window_scores_flat.extend(collected_scores)

        # 计算单个学生AUC
        if len(np.unique(collected_trues)) >= 2 and len(collected_trues) > 0:
            student_final_auc = safe_roc_auc(collected_trues, collected_scores)
        else:
            student_final_auc = 0.5

        student_aucs[student_uid] = student_final_auc

    # === 新增：自动保存 per_student_aucs.json 到 txt 同目录 ===
    mode_suffix = "" if ENABLE_SLIDING_WINDOW else "_no_window"
    json_output_path = file_path.replace('.txt', f'_per_student_aucs{mode_suffix}.json')
    # 转为普通 dict（避免 pandas Int64 问题）
    student_aucs_plain = {str(k): float(v) for k, v in student_aucs.items()}
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(student_aucs_plain, f, indent=2, ensure_ascii=False)
    print(f"已保存每个学生AUC → {json_output_path}")
    # ===========================================================


    # 4. 组装学生AUC DataFrame
    student_auc_df = pd.DataFrame(
        list(student_aucs.items()), 
        columns=['student_uid', 'auc']
    )
    
    # 5. 计算总体和统计结果
    
    # *** 关键修改：整体AUC现在基于所有学生的所有滑窗汇集数据 ***
    if len(np.unique(all_window_trues_flat)) >= 2 and len(all_window_trues_flat) > 0:
        overall_auc = safe_roc_auc(all_window_trues_flat, all_window_scores_flat)
    else:
        # 如果汇集后的数据仍然无法计算 AUC
        overall_auc = 0.5 

    # b. 学生 AUC 统计 (基于每个学生的滑窗汇集 AUC)
    valid_aucs = student_auc_df['auc'].dropna()
    
    if valid_aucs.empty:
        stats = {'mean': 0.5, 'std': 0.0, 'max': 0.5, 'min': 0.5, 'range': 0.0}
    else:
        mean_auc = valid_aucs.mean()
        std_auc = valid_aucs.std() if len(valid_aucs) > 1 else 0.0
        max_auc = valid_aucs.max()
        min_auc = valid_aucs.min()
        range_auc = max_auc - min_auc
        
        stats = {
            'mean': mean_auc, 
            'std': std_auc, 
            'max': max_auc, 
            'min': min_auc, 
            'range': range_auc
        }
        # === 新增：在返回前计算基尼系数和 EAWI ===
        valid_aucs_list = valid_aucs.tolist()  # 用于计算财富不平等指标

        average_wealth, gini, eawi_dict = calculate_wealth_metrics(valid_aucs_list, alphas=[1.0, 2.0, 3.0])

        # 更新 stats，加入新指标
        stats.update({
            'gini_coefficient': float(gini),
            'average_auc': float(average_wealth),  # 等价于 mean，但更清晰
            **eawi_dict  # 自动展开 eawi_alpha_10, eawi_alpha_20, eawi_alpha_30
        })
        # ===========================================
    # 6. 整合结果字典
    overall_auc_info = {
        'overall_dataset_auc': overall_auc, # <-- 现在是基于所有滑窗汇集数据计算的
        'student_auc_stats': stats
    }

    return student_auc_df, overall_auc_info

# --- 外层保存函数 (保持不变) ---
def save_auc_results_from_file(input_file_path, student_auc_output_csv=None, overall_stats_output_json=None):
    """
    解析本地TXT文件，并保存学生AUC DataFrame和整体统计结果字典。

    如果 student_auc_output_csv 或 overall_stats_output_json 为 None，
    它们将默认保存在与 input_file_path 相同的目录中，
    文件名分别为 'student_auc_output.csv' 和 'overall_stats_output.json'。
    """
    try:
        # 1. 获取输入文件的基本目录
        base_dir = os.path.dirname(input_file_path)

        # 2. 检查并设置默认的 CSV 输出路径
        if student_auc_output_csv is None:
            # 默认文件名基于原变量名的含义
            default_csv_name = "student_auc_output.csv"
            student_auc_output_csv = os.path.join(base_dir, default_csv_name)
        
        # 3. 检查并设置默认的 JSON 输出路径
        if overall_stats_output_json is None:
            # 默认文件名基于原变量名的含义
            default_json_name = "overall_stats_output.json"
            overall_stats_output_json = os.path.join(base_dir, default_json_name)

        # 4. 解析和计算 AUC (来自您的原始逻辑)
        student_df, overall_info = parse_and_calculate_aucs_from_file(input_file_path)
        
        # 5. 保存学生 AUC DataFrame (来自您的原始逻辑)
        student_df.to_csv(student_auc_output_csv, index=False)
        print(f"✅ 学生AUC结果已保存至: {student_auc_output_csv}")
        
        # 6. 保存总体统计结果字典 (来自您的原始逻辑)
        with open(overall_stats_output_json, 'w', encoding='utf-8') as f:
            json.dump(overall_info, f, indent=4, ensure_ascii=False, 
                      default=lambda x: round(x, 6) if isinstance(x, (float, np.float_)) else x)
            
        print(f"✅ 总体统计结果已保存至: {overall_stats_output_json}")

    except Exception as e:
        print(f"❌ 发生错误: {e}")
def process_multiple_paths(paths):
    """
    接收多个路径，在每个路径下查找 window_predictions.txt 文件，
    处理、汇总结果，并保存汇总的 CSV 文件。

    Args:
        paths (list): 包含模型输出文件夹路径的列表。
    """
    
    all_results = []
    file_pattern = '*window_predictions.txt'
    
    print("--- 开始处理多个模型路径 ---")
    
    for path in paths:
        # 1. 在路径下查找目标文件
        search_path = os.path.join(path, '**', file_pattern)
        found_files = glob(search_path, recursive=True) # 递归查找

        if not found_files:
            print(f"🔍 未在 {path} 及其子目录中找到 {file_pattern} 文件。跳过。")
            continue
            
        # 我们假设每个路径只对应一个模型输出文件
        file_path = found_files[0]
        
        try:
            # 2. 提取模型名称
            model_name = extract_model_name(file_path)
            
            print(f"\n--- 正在处理文件: {file_path} ---")
            print(f"🌟 提取的模型名: {model_name}")

            # 3. 解析和计算 AUC，只获取整体统计结果
            # 我们只需要 overall_info 来进行汇总
            _, overall_info = parse_and_calculate_aucs_from_file(file_path)

            # 4. 扁平化结果并加入模型名
            flat_result = {'model_name': model_name}
            
            # 将 overall_dataset_auc 直接添加到结果中
            flat_result['overall_dataset_auc'] = overall_info['overall_dataset_auc']

            # 将 student_auc_stats 里面的所有指标扁平化加入结果
            for key, value in overall_info['student_auc_stats'].items():
                flat_result[key] = value

            all_results.append(flat_result)
            
        except Exception as e:
            print(f"❌ 处理文件 {file_path} 时发生错误: {e}")
            continue

    if not all_results:
        print("\n--- 没有成功处理任何结果，无法生成汇总文件。 ---")
        return

    # 5. 汇总结果并生成 DataFrame
    results_df = pd.DataFrame(all_results)
    print("\n--- 结果汇总完成 ---")
    print(results_df)

    # 6. 保存汇总结果
    
    # a. 创建以当前日期+时刻命名的新文件夹
    current_time = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = os.path.join('result', current_time)
    os.makedirs(output_dir, exist_ok=True)
    
    # b. 构造输出文件路径
    output_file_path = os.path.join(output_dir, 'summary_auc_metrics.csv')

    # c. 保存 CSV
    results_df.to_csv(output_file_path, index=False)
    
    print(f"\n✅ 所有模型结果已汇总并保存至: {output_file_path}")
if __name__ == "__main__":
    # 检查是否有参数传入
    if len(sys.argv) < 2:
        print("❌ 错误：请提供至少一个模型输出文件夹的路径作为参数。")
        print(f"用法 (多参数): python {sys.argv[0]} <path1> <path2> ...")
        print(f"用法 (单个字符串参数): python {sys.argv[0]} \"<path1>\n<path2>\n...\"")
        sys.exit(1)

    # 假设用户传入了一个或多个参数。
    # 检查第一个参数是否包含换行符，以此判断用户是否使用了单个字符串参数。
    first_arg = sys.argv[1]
    
    if len(sys.argv) == 2 and ('\n' in first_arg or ' ' in first_arg):
        # 场景 A: 用户传入了单个包含多行或多空格分隔路径的字符串 (如: "path1\npath2")
        print("💡 识别到单个字符串参数，正在解析路径...")
        
        # 移除首尾空白，然后按换行符或空格分割，并过滤掉空字符串
        raw_paths = first_arg.strip()
        
        # 尝试先用换行符分割，如果不行，再用空格分割
        if '\n' in raw_paths:
             paths_to_process = [p.strip() for p in raw_paths.split('\n') if p.strip()]
        else:
             paths_to_process = [p.strip() for p in raw_paths.split() if p.strip()]
             
    else:
        # 场景 B: 用户传入了多个独立的参数 (如: path1 path2)
        print("💡 识别到多个独立参数。")
        paths_to_process = sys.argv[1:]

    if not paths_to_process:
        print("❌ 错误：解析参数后未找到有效路径。")
        sys.exit(1)
        
    process_multiple_paths(paths_to_process)
    
    print(f"--- 所有文件处理完成 ---")
# ---------------------------------------------------------------------------
# [执行入口] - 只有当脚本被直接运行时才调用 main()
# ---------------------------------------------------------------------------
