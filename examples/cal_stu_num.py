import pandas as pd
import json
import os

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

# 使用示例
if __name__ == "__main__":
    # 假设我们要处理assist2015数据集
    dataset_name = "nips_task34"
    
    # 获取JSON文件路径
    json_file_path = "../configs/data_config.json"  # 替换为实际路径
    
    # 从JSON中获取CSV文件路径
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data_config = json.load(f)
    
    # 构建完整的CSV文件路径
    csv_relative_path = data_config[dataset_name]['train_valid_file']
    csv_file_path = os.path.join(data_config[dataset_name]['dpath'], csv_relative_path)
    
    # 调用函数处理
    count_unique_uids_and_update_json(csv_file_path, json_file_path, dataset_name)