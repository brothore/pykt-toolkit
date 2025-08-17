import csv

def get_question_range(csv_file_path):
    """
    统计CSV文件中questions列的最大和最小问题编号范围
    
    参数:
        csv_file_path (str): CSV文件路径
        
    返回:
        dict: 包含最大问题编号、最小问题编号和范围长度的字典
    """
    max_q = float('-inf')
    min_q = float('inf')
    
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            if 'questions' in row:
                # 分割questions列并转换为整数列表
                questions = [int(q.strip()) for q in row['questions'].split(',') if q.strip().isdigit()]
                
                if questions:  # 确保列表不为空
                    current_max = max(questions)
                    current_min = min(questions)
                    
                    if current_max > max_q:
                        max_q = current_max
                    
                    if current_min < min_q:
                        min_q = current_min
    
    # 处理没有有效数据的情况
    if max_q == float('-inf') or min_q == float('inf'):
        return {
            'max_question': None,
            'min_question': None,
            'range_length': 0
        }
    
    return {
        'max_question': max_q,
        'min_question': min_q,
        'range_length': max_q - min_q
    }

# 使用示例
result = get_question_range('/data/pykt_datasets/data/XES3G5M/kc_level/train_valid_sequences.csv')
print(f"最大问题编号: {result['max_question']}")
print(f"最小问题编号: {result['min_question']}")
print(f"范围长度: {result['range_length']}")