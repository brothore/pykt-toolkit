import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import os
import argparse
import json

def calculate_global_stats(df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """计算全局question和concept统计信息"""
    question_stats = {}
    concept_stats = {}
    
    for _, row in df.iterrows():
        questions = [int(x) for x in row['questions'].split(',')]
        concepts = [int(x) for x in row['concepts'].split(',')]
        responses = [int(x) for x in row['responses'].split(',')]
        
        # 处理question统计
        for question, response in zip(questions, responses):
            if question != -1 and response != -1:
                if question not in question_stats:
                    question_stats[question] = {'correct': 0, 'total': 0}
                question_stats[question]['total'] += 1
                if response == 1:
                    question_stats[question]['correct'] += 1
        
        # 处理concept统计
        for concept, response in zip(concepts, responses):
            if concept != -1 and response != -1:
                if concept not in concept_stats:
                    concept_stats[concept] = {'correct': 0, 'total': 0}
                concept_stats[concept]['total'] += 1
                if response == 1:
                    concept_stats[concept]['correct'] += 1
    
    # 计算正确率
    for question in question_stats:
        question_stats[question]['accuracy'] = question_stats[question]['correct'] / question_stats[question]['total']
    
    for concept in concept_stats:
        concept_stats[concept]['accuracy'] = concept_stats[concept]['correct'] / concept_stats[concept]['total']
    
    return question_stats, concept_stats

def save_global_stats(question_stats: Dict, concept_stats: Dict, output_dir: str) -> List[str]:
    """保存全局统计信息到CSV文件，返回生成的文件名列表"""
    generated_files = []
    
    # 保存question统计
    question_df = pd.DataFrame([
        {
            'question_id': qid,
            'total_attempts': stats['total'],
            'correct_attempts': stats['correct'],
            'accuracy': stats['accuracy']
        }
        for qid, stats in question_stats.items()
    ])
    question_df = question_df.sort_values('question_id')
    question_filename = 'global_question_stats.csv'
    question_output_path = os.path.join(output_dir, question_filename)
    question_df.to_csv(question_output_path, index=False)
    generated_files.append(question_filename)
    print(f"全局question统计已保存到: {question_output_path}")
    
    # 保存concept统计
    concept_df = pd.DataFrame([
        {
            'concept_id': cid,
            'total_attempts': stats['total'],
            'correct_attempts': stats['correct'],
            'accuracy': stats['accuracy']
        }
        for cid, stats in concept_stats.items()
    ])
    concept_df = concept_df.sort_values('concept_id')
    concept_filename = 'global_concept_stats.csv'
    concept_output_path = os.path.join(output_dir, concept_filename)
    concept_df.to_csv(concept_output_path, index=False)
    generated_files.append(concept_filename)
    print(f"全局concept统计已保存到: {concept_output_path}")
    
    return generated_files

def add_accuracy_sequences(uid_data: pd.DataFrame, question_stats: Dict, concept_stats: Dict) -> pd.DataFrame:
    """为学生数据添加concept和question正确率序列"""
    uid_data = uid_data.copy()
    
    concept_accuracy_sequences = []
    question_accuracy_sequences = []
    
    for _, row in uid_data.iterrows():
        questions = [int(x) for x in row['questions'].split(',')]
        concepts = [int(x) for x in row['concepts'].split(',')]
        
        # 获取concept正确率序列
        concept_accuracies = []
        for concept in concepts:
            if concept != -1 and concept in concept_stats:
                concept_accuracies.append(f"{concept_stats[concept]['accuracy']:.4f}")
            else:
                concept_accuracies.append("-1")
        concept_accuracy_sequences.append(','.join(concept_accuracies))
        
        # 获取question正确率序列
        question_accuracies = []
        for question in questions:
            if question != -1 and question in question_stats:
                question_accuracies.append(f"{question_stats[question]['accuracy']:.4f}")
            else:
                question_accuracies.append("-1")
        question_accuracy_sequences.append(','.join(question_accuracies))
    
    uid_data['concept_accuracies'] = concept_accuracy_sequences
    uid_data['question_accuracies'] = question_accuracy_sequences
    
    return uid_data

def calculate_student_detailed_stats(uid_data: pd.DataFrame, threshold: int) -> Dict:
    """计算单个学生的详细统计信息"""
    all_concepts = []
    all_responses = []
    
    for _, row in uid_data.iterrows():
        concepts = [int(x) for x in row['concepts'].split(',')]
        responses = [int(x) for x in row['responses'].split(',')]
        
        valid_pairs = [(c, r) for c, r in zip(concepts, responses) if c != -1 and r != -1]
        if valid_pairs:
            valid_concepts, valid_responses = zip(*valid_pairs)
            all_concepts.extend(valid_concepts)
            all_responses.extend(valid_responses)
    
    if not all_concepts:
        return {
            'record_lines': len(uid_data),
            'total_questions': 0,
            'num_valid_concepts': 0,
            'avg_questions_per_concept': 0.0,
            'max_questions_per_concept': 0,
            'min_questions_per_concept': 0,
            'questions_range': 0,
            'overall_accuracy': 0.0,
            'accuracy_range': 0.0,
            'accuracy_variance': 0.0,
            'max_accuracy': 0.0,
            'min_accuracy': 0.0
        }
    
    concept_stats = {}
    for concept, response in zip(all_concepts, all_responses):
        if concept not in concept_stats:
            concept_stats[concept] = {'correct': 0, 'total': 0}
        concept_stats[concept]['total'] += 1
        if response == 1:
            concept_stats[concept]['correct'] += 1
    
    # 筛选出有效知识点（做题数量 > 知识点数量阈值）
    valid_concept_stats = {}
    for concept, stats in concept_stats.items():
        if stats['total'] > threshold: # 使用传入的阈值
            valid_concept_stats[concept] = stats
    
    # 基本统计信息
    total_questions = len(all_concepts)
    overall_accuracy = sum(all_responses) / len(all_responses) if all_responses else 0.0
    num_valid_concepts = len(valid_concept_stats)
    
    # 初始化默认值
    avg_questions_per_concept = 0.0
    max_questions_per_concept = 0
    min_questions_per_concept = 0
    questions_range = 0
    max_accuracy = 0.0
    min_accuracy = 0.0
    accuracy_range = 0.0
    accuracy_variance = 0.0
    
    if num_valid_concepts > 0:
        # 知识点题目数量统计
        question_counts = [stats['total'] for stats in valid_concept_stats.values()]
        avg_questions_per_concept = np.mean(question_counts)
        max_questions_per_concept = max(question_counts)
        min_questions_per_concept = min(question_counts)
        questions_range = max_questions_per_concept - min_questions_per_concept
        
        # 知识点正确率统计
        accuracies = [stats['correct'] / stats['total'] for stats in valid_concept_stats.values()]
        max_accuracy = max(accuracies)
        min_accuracy = min(accuracies)
        accuracy_range = max_accuracy - min_accuracy
        accuracy_variance = np.var(accuracies) if len(accuracies) > 1 else 0.0
    
    return {
        'record_lines': len(uid_data),
        'total_questions': total_questions,
        'num_valid_concepts': num_valid_concepts,
        'avg_questions_per_concept': avg_questions_per_concept,
        'max_questions_per_concept': max_questions_per_concept,
        'min_questions_per_concept': min_questions_per_concept,
        'questions_range': questions_range,
        'overall_accuracy': overall_accuracy,
        'max_accuracy': max_accuracy,
        'min_accuracy': min_accuracy,
        'accuracy_range': accuracy_range,
        'accuracy_variance': accuracy_variance
    }

def print_top_students_details(student_summary: pd.DataFrame, num_students: int = 5):
    """打印记录数最多的学生的详细信息"""
    top_students = student_summary.sort_values('record_lines', ascending=False).head(num_students)
    
    print(f"\n=== 记录行数最多的{len(top_students)}个学生详细信息 (筛选后) ===")
    
    for i, (_, student) in enumerate(top_students.iterrows()):
        print(f"\n第{i+1}名学生 (UID: {student['uid']}):")
        print(f"  总记录行数: {student['record_lines']}")
        print(f"  总做题数: {student['total_questions']}")
        print(f"  有效知识点数: {student['num_valid_concepts']}")
        print(f"  平均每知识点做题数: {student['avg_questions_per_concept']:.1f}")
        print(f"  知识点做题数范围: {student['min_questions_per_concept']} - {student['max_questions_per_concept']} (差值: {student['questions_range']})")
        print(f"  总体正确率: {student['overall_accuracy']:.3f}")
        print(f"  知识点正确率范围: {student['min_accuracy']:.3f} - {student['max_accuracy']:.3f} (差值: {student['accuracy_range']:.3f})")
        print(f"  知识点正确率方差: {student['accuracy_variance']:.4f}")

def count_unique_uids_and_update_json_train(csv_file_path, json_file_path, dataset_name, uid_column='uid'):
    """
    统计CSV文件中独立uid的数量，并更新json文件中的 students_num_train
    """
    try:
        df = pd.read_csv(csv_file_path)
        unique_uids = df[uid_column].nunique()
        print(f"在文件 {csv_file_path} 中找到 {unique_uids} 个独立uid（训练/验证集）")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data_config = json.load(f)
        
        if dataset_name not in data_config:
            raise ValueError(f"数据集 {dataset_name} 不存在于配置文件中")
            
        data_config[dataset_name]['students_num_train'] = int(unique_uids)
        
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data_config, f, indent=4, ensure_ascii=False)
            
        print(f"成功更新 {dataset_name} 的 students_num_train 为 {unique_uids}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        raise

# 删除了 count_unique_uids_and_update_json_eval，将逻辑嵌入 main 函数中

def main(args):

    global_concept_threshold = 0

    if_quelevel = "_quelevel" if args.if_quelevel else ""
    """主函数：处理数据并保存所有学生数据"""
    print("正在读取CSV文件...")
    config_path = "../configs/data_config.json"
    dataset_name = args.dataset

    # --- 训练集学生数量更新（保留原逻辑） ---
    with open(config_path, 'r', encoding='utf-8') as f:
        data_config = json.load(f)
    csv_relative_path = data_config[dataset_name]['train_valid_file']
    train_csv_file_path = os.path.join(data_config[dataset_name]['dpath'], csv_relative_path)
    # count_unique_uids_and_update_json_train(train_csv_file_path, config_path, dataset_name)
    # ----------------------------------------

    # --- 读取原始测试数据 ---
    input_csv_path = args.input_file
    output_dir = args.output_directory
    df = pd.read_csv(input_csv_path)
    print(f"共读取到 {len(df)} 条记录")

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 用于收集所有生成的文件名
    all_generated_files = []

    # 计算全局统计信息（使用原始df计算）
    print("\n正在计算全局question和concept统计信息...")
    question_stats, concept_stats = calculate_global_stats(df)
    print(f"共有 {len(question_stats)} 个不同的question")
    print(f"共有 {len(concept_stats)} 个不同的concept")

    # 保存全局统计信息
    global_files = save_global_stats(question_stats, concept_stats, output_dir)
    all_generated_files.extend(global_files)

    print("\n正在计算学生详细统计信息...")
    student_info = {}
    for uid, group in df.groupby('uid'):
        # 传递知识点阈值
        student_info[uid] = calculate_student_detailed_stats(group, global_concept_threshold)
        student_info[uid]['uid'] = uid

    # 创建学生DataFrame
    student_summary = pd.DataFrame(list(student_info.values()))

    # --- 【修改：新增】筛选前所有学生的 total_questions 统计 ---
    if not student_summary.empty:
        all_total_questions = student_summary['total_questions']
        avg_q = all_total_questions.mean()
        max_q = all_total_questions.max()
        min_q = all_total_questions.min()

        print("\n🎉 **筛选前所有学生 total_questions 统计** 🎉")
        print(f"  学生总数: {len(student_summary)}")
        print(f"  平均做题数: **{avg_q:.2f}**")
        print(f"  最大做题数: **{max_q}**")
        print(f"  最小做题数: **{min_q}**")
    # -------------------------------------------------------------

    # --- 【新增】筛选学生逻辑：total_questions 在 600 ± 5 ---
    print(f"\n=== 开始筛选学生：total_questions 在 {args.seq_len}±5 范围内 ===")
    lower_bound = args.seq_len -args.seq_len*0.1
    upper_bound = args.seq_len +args.seq_len*0.1

    # 筛选学生摘要
    filtered_summary = student_summary[
        (student_summary['total_questions'] >= lower_bound) &
        (student_summary['total_questions'] <= upper_bound)
    ].sort_values('total_questions', ascending=False)

    num_filtered_students = len(filtered_summary)
    print(f"筛选出 {num_filtered_students} 个学生满足条件。")

    # 更新后续流程中使用的学生列表和数量
    top_students = filtered_summary
    num_needs_stu = num_filtered_students

    # 筛选原始数据 df
    student_uids_to_keep = filtered_summary['uid'].tolist()
    df_filtered = df[df['uid'].isin(student_uids_to_keep)]
    df = df_filtered # 替换原始数据框为筛选后的数据框

    # --- 【新增】使用筛选后的学生数量更新 eval 配置 ---
    try:
        print(f"测试集共有 {num_needs_stu} 个不同的学生 (筛选后)")

        # 读取配置
        with open(config_path, 'r', encoding='utf-8') as f:
            data_config = json.load(f)

        # 更新 eval 数量
        if dataset_name in data_config:
            data_config[dataset_name][f'seq_len_{args.seq_len}_students_num_eval'] = int(num_needs_stu)

            # 写回 JSON 文件
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(data_config, f, indent=4, ensure_ascii=False)
            print(f"成功更新 {dataset_name} 的 students_num_eval 为 {num_needs_stu} (筛选后)")
        else:
            print(f"警告: 配置文件中未找到数据集 {dataset_name}")

    except Exception as e:
        print(f"更新 eval 配置时出错: {e}")
    # ------------------------------------------------

    # 打印top学生详细信息（现在是筛选后的学生）
    print_top_students_details(top_students, min(5, num_needs_stu))

    # 为所有筛选后的学生保存单独文件
    print(f"\n保存所有{num_needs_stu}个筛选后的学生数据:")
    for rank, (original_index, row) in enumerate(top_students.iterrows()):
        # rank 将从 0, 1, 2, ... 开始顺序编号
        uid = row['uid']
        record_lines = row['record_lines']
        
        # 从筛选后的 df (即 df_filtered) 中获取学生数据
        student_data = df[df['uid'] == uid]
        
        # 添加concept和question正确率序列
        student_data = add_accuracy_sequences(student_data, question_stats, concept_stats)
        
        # 添加统计信息到学生数据中
        for stat_col in ['record_lines', 'total_questions', 'num_valid_concepts',
                         'avg_questions_per_concept', 'max_questions_per_concept',
                         'min_questions_per_concept', 'questions_range',
                         'overall_accuracy', 'accuracy_range', 'accuracy_variance',
                         'max_accuracy', 'min_accuracy']:
            if stat_col in row:
                student_data[stat_col] = row[stat_col]
        
        # 使用 rank + 1 作为顺序编号
        filename = f"seq_len_{args.seq_len}_{args.target_file_type}_top_{rank+1}_student{if_quelevel}.csv" if args.if_quelevel else f"seq_len_{args.seq_len}_{args.target_file_type}_top_{rank+1}_student.csv"
        filepath = os.path.join(output_dir, filename)
        
        student_data.to_csv(filepath, index=False)
        all_generated_files.append(filename)
        # 打印时也使用 rank + 1
        print(f"  学生#{rank+1}: uid={uid}, 记录行数={record_lines}, 做题数={row['total_questions']} -> 已保存到 {filename}")
  
    # 保存筛选后的学生统计摘要
    summary_filename = "student_summary_filtered.csv"
    summary_output_path = os.path.join(output_dir, summary_filename)
    filtered_summary.to_csv(summary_output_path, index=False)
    all_generated_files.append(summary_filename)
    print(f"\n学生统计摘要 (筛选后) 已保存到: {summary_output_path}")

    print(f"\n处理完成！共生成了 {len(all_generated_files)} 个文件。")

def parse_args():
    parser = argparse.ArgumentParser(description="Process dataset parameters")
    
    parser.add_argument("--dataset", type=str, default="peiyou",
                         help="Dataset name (default: %(default)s)")
    parser.add_argument("--if_quelevel", type=int, default=0,
                         help="question level")
    parser.add_argument("--seq_len", type=int, default=0,
                         help="seq")
    parser.add_argument("--target_file_type", type=str, default="test_question_window_sequences",
                         help="切割哪个文件,test_question_window_sequences,test_window_sequences")
                         
    args, _ = parser.parse_known_args()
    
    with open('../configs/data_config.json', 'r') as f:
        data_config = json.load(f)
        
    if args.dataset not in data_config:
        print(f"警告: 配置文件中未找到数据集 {args.dataset}")
        dataset_dpath = "data/" # 使用默认路径
    else:
        dataset_dpath = data_config[args.dataset]["dpath"]

    if_quelevel = "_quelevel" if args.if_quelevel else "" 
    default_input = f"{dataset_dpath}/{args.target_file_type}{if_quelevel}.csv" if if_quelevel else f"{dataset_dpath}/{args.target_file_type}.csv"
    default_output = f"{dataset_dpath}/"

    
    parser.add_argument("--input_file", type=str, default=default_input,
                         help="Input file path (default: %(default)s)")
    parser.add_argument("--output_directory", type=str, default=default_output,
                         help="Output directory path (default: %(default)s)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # 定义知识点筛选的阈值（在 calculate_student_detailed_stats 中使用）
    # 将阈值作为参数传入 main 函数
    main(args)