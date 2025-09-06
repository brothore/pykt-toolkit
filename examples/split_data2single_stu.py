import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import os
import argparse
import json
if_quelevel = "_quelevel"
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

def calculate_student_detailed_stats(uid_data: pd.DataFrame) -> Dict:
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
        if stats['total'] > 知识点数量:
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

def save_students_by_intervals(df: pd.DataFrame, student_summary: pd.DataFrame, output_dir: str, question_stats: Dict, concept_stats: Dict) -> List[str]:
    """按照不同指标将学生分为三个区间并保存数据，返回生成的文件名列表"""
    generated_files = []
    
    # 要处理的指标列表
    metrics = [
        ('total_questions', '做题数'),
        ('overall_accuracy', '总体正确率'), 
        ('accuracy_range', '正确率差值'),
        ('accuracy_variance', '正确率方差'),
        ('questions_range', '题目数差值')
    ]
    
    print("\n=== 开始按不同指标分组保存学生数据 ===")
    
    for metric, metric_name in metrics:
        print(f"\n正在按{metric_name}({metric})分组...")
        
        # 按指标排序
        sorted_students = student_summary.sort_values(metric, ascending=True)
        total_students = len(sorted_students)
        
        # 分为三个区间
        interval_size = total_students // 3
        remainder = total_students % 3
        
        # 计算每个区间的大小，余数分配给前面的区间
        intervals = [
            interval_size + (1 if i < remainder else 0) 
            for i in range(3)
        ]
        
        # 分割数据
        start_idx = 0
        interval_names = ['low', 'medium', 'high']
        
        for i, (interval_name, size) in enumerate(zip(interval_names, intervals)):
            end_idx = start_idx + size
            interval_students = sorted_students.iloc[start_idx:end_idx]
            
            # 获取这些学生的原始数据
            student_uids = interval_students['uid'].tolist()
            interval_data = df[df['uid'].isin(student_uids)]
            
            # 添加正确率序列
            interval_data = add_accuracy_sequences(interval_data, question_stats, concept_stats)
            
            # 添加统计信息到学生数据中
            for stat_col in ['record_lines', 'total_questions', 'num_valid_concepts',
                             'avg_questions_per_concept', 'max_questions_per_concept', 
                             'min_questions_per_concept', 'questions_range', 
                             'overall_accuracy', 'accuracy_range', 'accuracy_variance',
                             'max_accuracy', 'min_accuracy']:
                # 为每个学生添加对应的统计信息
                uid_stats_map = student_summary.set_index('uid')[stat_col].to_dict()
                interval_data[stat_col] = interval_data['uid'].map(uid_stats_map)
            
            # 保存文件
            filename = f"{metric}_{interval_name}_interval.csv"
            filepath = os.path.join(output_dir, filename)
            interval_data.to_csv(filepath, index=False)
            generated_files.append(filename)
            
            min_val = interval_students[metric].min()
            max_val = interval_students[metric].max()
            print(f"  {interval_name}区间: {len(interval_students)}个学生, {metric_name}范围: {min_val:.4f} - {max_val:.4f}")
            print(f"    已保存到: {filename}")
            
            start_idx = end_idx
    
    return generated_files

def load_data_config(config_path: str = "../configs/data_config.json") -> dict:
    """加载data_config.json文件"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}
    except Exception as e:
        print(f"读取配置文件时出错: {e}")
        return {}

def get_students_num_train(dataset_name: str, config_path: str = "../configs/data_config.json") -> int:
    """
    获取指定数据集的学生数量
    
    Args:
        dataset_name: 数据集名称，如 "assist2015"
        config_path: 配置文件路径
        
    Returns:
        学生数量，如果未找到则返回None
    """
    config = load_data_config(config_path)
    
    if dataset_name in config:
        dataset_config = config[dataset_name]
        if 'students_num_train' in dataset_config:
            return dataset_config['students_num_train']
        else:
            print(f"数据集 {dataset_name} 配置中未找到 students_num_train 字段")
            return None
    else:
        print(f"配置文件中未找到数据集: {dataset_name}")
        return None
def get_students_num_eval(dataset_name: str, config_path: str = "../configs/data_config.json") -> int:
    """
    获取指定数据集的学生数量
    
    Args:
        dataset_name: 数据集名称，如 "assist2015"
        config_path: 配置文件路径
        
    Returns:
        学生数量，如果未找到则返回None
    """
    config = load_data_config(config_path)
    
    if dataset_name in config:
        dataset_config = config[dataset_name]
        if 'students_num_eval' in dataset_config:
            return dataset_config['students_num_eval']
        else:
            print(f"数据集 {dataset_name} 配置中未找到 students_num_eval 字段")
            return None
    else:
        print(f"配置文件中未找到数据集: {dataset_name}")
        return None
def update_data_config(dataset_name: str, num_students: int, generated_files: List[str], config_path: str = "../configs/data_config.json"):
    """更新data_config.json文件中的学生个数和生成的文件列表"""
    try:
        # 读取现有配置文件
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
        
        # 更新对应数据集的学生个数
        if dataset_name in config:
            config[dataset_name]['students_num_eval'] = num_students
            
            # 为每个生成的文件创建单独的键值对
            for filename in generated_files:
                if 'global_question_stats.csv' in filename:
                    config[dataset_name]['global_question_stats_file'] = filename
                elif 'global_concept_stats.csv' in filename:
                    config[dataset_name]['global_concept_stats_file'] = filename
                elif 'student_summary.csv' in filename:
                    config[dataset_name]['student_summary_file'] = filename
                elif '_interval.csv' in filename:
                    # 处理区间文件，为每个区间文件创建对应的键
                    # 例如：total_questions_low_interval.csv -> total_questions_low_interval_file
                    key_name = filename.replace('.csv', '_file')
                    config[dataset_name][key_name] = filename

            
            print(f"已更新数据集 {dataset_name} 的学生个数为: {num_students}")
            print(f"已添加 {len(generated_files)} 个生成的文件到配置中")
        else:
            print(f"警告: 配置文件中未找到数据集 {dataset_name}")
            return
        
        # 保存更新后的配置文件
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print(f"配置文件已更新: {config_path}")
        
    except Exception as e:
        print(f"更新配置文件时出错: {e}")

def print_top_students_details(student_summary: pd.DataFrame, num_students: int = 5):
    """打印记录数最多的学生的详细信息"""
    top_students = student_summary.sort_values('record_lines', ascending=False).head(num_students)
    
    print(f"\n=== 记录行数最多的{num_students}个学生详细信息 ===")
    
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
def count_unique_uids_and_update_json(csv_file_path, json_file_path, dataset_name, uid_column='uid'):
    """
    统计CSV文件中独立uid的数量，并更新json文件
    
    参数:
        csv_file_path (str): CSV文件路径
        json_file_path (str): JSON配置文件路径
        dataset_name (str): 数据集名称(如"assist2015")
        uid_column (str): CSV文件中uid的列名，默认为'uid'
    """
    try:
        # 1. 读取CSV文件并统计独立uid数量
        df = pd.read_csv(csv_file_path)
        unique_uids = df[uid_column].nunique()
        print(f"在文件 {csv_file_path} 中找到 {unique_uids} 个独立uid")
        
        # 2. 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data_config = json.load(f)
        
        # 3. 检查数据集是否存在
        if dataset_name not in data_config:
            raise ValueError(f"数据集 {dataset_name} 不存在于配置文件中")
            
        # 4. 更新students_num_train字段
        data_config[dataset_name]['students_num_train'] = int(unique_uids)
        
        # 5. 写回JSON文件
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data_config, f, indent=4, ensure_ascii=False)
            
        print(f"成功更新 {dataset_name} 的 students_num_train 为 {unique_uids}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        raise
def main(args):
    """主函数：处理数据并保存所有学生数据"""
    print("正在读取CSV文件...")
    #处理训练集
    config_path = "../configs/data_config.json"
    # 从JSON中获取CSV文件路径
    dataset_name = args.dataset
    with open(config_path, 'r', encoding='utf-8') as f:
        data_config = json.load(f)
    # 构建完整的CSV文件路径
    csv_relative_path = data_config[dataset_name]['train_valid_file']
    train_csv_file_path = os.path.join(data_config[dataset_name]['dpath'], csv_relative_path)
    
    # 调用函数处理
    count_unique_uids_and_update_json(train_csv_file_path, config_path, dataset_name)
    


    # 使用args中的参数
    input_csv_path = args.input_file
    output_dir = args.output_directory
    df = pd.read_csv(input_csv_path)
    print(f"共读取到 {len(df)} 条记录")
    
    # 自动设置学生个数为实际uid个数
    num_needs_stu = df['uid'].nunique()
    print(f"测试集共有 {num_needs_stu} 个不同的学生")
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 用于收集所有生成的文件名
    all_generated_files = []
    
    # 计算全局统计信息
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
        student_info[uid] = calculate_student_detailed_stats(group)
        # 添加UID到统计信息中
        student_info[uid]['uid'] = uid
    
    # 创建学生DataFrame并排序
    student_summary = pd.DataFrame(list(student_info.values()))
    student_summary = student_summary.sort_values('record_lines', ascending=False)
    
    # 按记录数排序取所有学生
    top_students = student_summary.head(num_needs_stu)
    
    # 打印top学生详细信息
    print_top_students_details(top_students, min(5, num_needs_stu))
    
    # 为所有学生保存单独文件（添加正确率序列）
    print(f"\n保存所有{num_needs_stu}个学生数据:")
    for i, row in top_students.iterrows():
        uid = row['uid']
        record_lines = row['record_lines']
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
        
        filename = f"top_{i+1}_student{if_quelevel}.csv"
        filepath = os.path.join(output_dir, filename)
        
        student_data.to_csv(filepath, index=False)
        all_generated_files.append(filename)
        print(f"  学生#{i+1}: uid={uid}, 记录行数={record_lines} -> 已保存到 {filepath}")
    
    # 保存完整的学生统计摘要
    summary_filename = "student_summary.csv"
    summary_output_path = os.path.join(output_dir, summary_filename)
    student_summary.to_csv(summary_output_path, index=False)
    all_generated_files.append(summary_filename)
    print(f"\n学生统计摘要已保存到: {summary_output_path}")
    
    # # 新增功能：按不同指标分组保存学生数据
    # interval_files = save_students_by_intervals(df, student_summary, output_dir, question_stats, concept_stats)
    # all_generated_files.extend(interval_files)
    
    # # 更新data_config.json文件，包括学生数量和生成的文件列表
    # update_data_config(args.dataset, num_needs_stu, all_generated_files,config_path=config_path)
    
    # print(f"\n处理完成！共生成了 {len(all_generated_files)} 个文件:")
    # for file in sorted(all_generated_files):
    #     print(f"  - {file}")

def parse_args():
    parser = argparse.ArgumentParser(description="Process dataset parameters")
    
    # 先定义 dataset 参数
    parser.add_argument("--dataset", type=str, default="assist2009",
                       help="Dataset name (default: %(default)s)")
    
    # 解析已知参数（只解析 dataset，不解析其他参数）
    args, _ = parser.parse_known_args()
    # 首先读取data_config.json文件
    with open('../configs/data_config.json', 'r') as f:
        data_config = json.load(f)
    # 获取指定数据集的dpath
    dataset_dpath = data_config[args.dataset]["dpath"] if args.dataset not in ["peiyou"] else (data_config[args.dataset]["dpath_question"] if if_quelevel else data_config[args.dataset]["dpath"])
    # 然后定义其他参数，使用 args.dataset 作为默认路径的一部分
    # 修改默认输入输出路径
    default_input = f"{dataset_dpath}/test_window_sequences{if_quelevel}.csv"
    default_output = f"{dataset_dpath}/"

    
    parser.add_argument("--input_file", type=str, default=default_input,
                       help="Input file path (default: %(default)s)")
    parser.add_argument("--output_directory", type=str, default=default_output,
                       help="Output directory path (default: %(default)s)")

    # 最后完整解析所有参数
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    知识点数量 = 1
    main(args)