import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score

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
        # 学生的完整序列
        student_trues_long = group['late_trues'].values
        student_scores_long = group['late_mean'].values
        
        # *** 滑窗并汇集数据点 ***
        collected_trues, collected_scores = sliding_window_collect(
            student_trues_long, 
            student_scores_long, 
            window_size=200
        )
        
        # *** 关键修改：将当前学生的滑窗汇集结果加入到整体列表中 ***
        all_window_trues_flat.extend(collected_trues)
        all_window_scores_flat.extend(collected_scores)

        # 计算【单个学生】的滑窗汇集 AUC (保持原有逻辑)
        if len(np.unique(collected_trues)) >= 2 and len(collected_trues) > 0:
            student_final_auc = safe_roc_auc(collected_trues, collected_scores)
        else:
            student_final_auc = 0.5

        student_aucs[student_uid] = student_final_auc

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

    # 6. 整合结果字典
    overall_auc_info = {
        'overall_dataset_auc': overall_auc, # <-- 现在是基于所有滑窗汇集数据计算的
        'student_auc_stats': stats
    }

    return student_auc_df, overall_auc_info

# --- 外层保存函数 (保持不变) ---
def save_auc_results_from_file(input_file_path, student_auc_output_csv, overall_stats_output_json):
    """
    解析本地TXT文件，并保存学生AUC DataFrame和整体统计结果字典。
    """
    try:
        # 1. 解析和计算 AUC
        student_df, overall_info = parse_and_calculate_aucs_from_file(input_file_path)
        
        # 2. 保存学生 AUC DataFrame
        student_df.to_csv(student_auc_output_csv, index=False)
        print(f"✅ 学生AUC结果已保存至: {student_auc_output_csv}")
        
        # 3. 保存总体统计结果字典
        with open(overall_stats_output_json, 'w', encoding='utf-8') as f:
            json.dump(overall_info, f, indent=4, ensure_ascii=False, 
                     default=lambda x: round(x, 6) if isinstance(x, (float, np.float_)) else x)
            
        print(f"✅ 总体统计结果已保存至: {overall_stats_output_json}")

    except Exception as e:
        print(f"❌ 发生错误: {e}")

# --- 示例使用 (保持不变) ---
file_path = r"D:\sync\docs&works\0914论文编写\实验结果\all_result\nips_task34\akt\akt_tiaocan_nips_task34_42_1_0.1_64_64_8_4_0.001_1_1_73940a49-0fb7-492d-a703-533f8158dcc6_1\qid_test_question_predictions.txt"
student_csv_path = 'student_auc_results.csv'
stats_json_path = 'overall_auc_stats.json'

# 执行并保存结果
# save_auc_results_from_file(file_path, student_csv_path, stats_json_path)

# 打印统计结果以便直接查看
# student_df, overall_info = parse_and_calculate_aucs_from_file(file_path)
# print("\n--- 每个学生的AUC (DataFrame) ---")
# print(student_df)

# print("\n--- 整体统计结果 (Dictionary) ---")
# print(json.dumps(overall_info, indent=4, default=lambda x: round(x, 6) if isinstance(x, (float, np.float_)) else x))