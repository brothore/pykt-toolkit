import json
import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Any

class DatasetMerger:
    def __init__(self, config_path: str = "../configs/data_config.json"):
        """
        初始化数据集合并器
        
        Args:
            config_path: data_config.json文件的路径
        """
        self.config_path = config_path
        self.data_config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_config(self):
        """保存配置文件"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.data_config, f, indent=2, ensure_ascii=False)
    
    def merge_datasets(self, dataset_names: List[str], output_dataset_name: str = None) -> str:
        """
        合并多个数据集
        
        Args:
            dataset_names: 要合并的数据集名称列表
            output_dataset_name: 输出数据集名称，如果不指定则自动生成
        
        Returns:
            合并后的数据集名称
        """
        if not dataset_names:
            raise ValueError("数据集名称列表不能为空")
        
        # 生成输出数据集名称
        if output_dataset_name is None:
            output_dataset_name = "A".join(dataset_names)
        
        print(f"开始合并数据集: {dataset_names}")
        print(f"输出数据集名称: {output_dataset_name}")
        
        # 验证输入数据集是否存在
        for dataset_name in dataset_names:
            if dataset_name not in self.data_config:
                raise ValueError(f"数据集 '{dataset_name}' 不存在于配置文件中")
        
        # 读取和合并CSV文件
        merged_df = self.merge_csv_files(dataset_names)
        
        # 创建合并后的数据集配置
        merged_config = self.create_merged_config(dataset_names, output_dataset_name)
        
        # 创建输出目录
        output_dir = Path(merged_config["dpath"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存合并后的CSV文件
        output_csv_path = output_dir / merged_config["train_valid_file"]
        merged_df.to_csv(output_csv_path, index=False)
        print(f"合并后的CSV文件已保存到: {output_csv_path}")
        
        # 更新配置文件
        self.data_config[output_dataset_name] = merged_config
        self.save_config()
        print(f"配置文件已更新，添加数据集: {output_dataset_name}")
        
        return output_dataset_name
    
    def merge_csv_files(self, dataset_names: List[str]) -> pd.DataFrame:
        """
        合并多个数据集的CSV文件
        
        Args:
            dataset_names: 数据集名称列表
        
        Returns:
            合并后的DataFrame
        """
        all_dfs = []
        
        for dataset_name in dataset_names:
            config = self.data_config[dataset_name]
            csv_path = Path(config["dpath"]) / config["train_valid_file"]
            
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
            
            print(f"读取CSV文件: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # 添加数据集标识列
            df["dataset_name"] = dataset_name
            all_dfs.append(df)
            print(f"  - 读取 {len(df)} 条记录")
        
        # 合并所有DataFrame
        merged_df = pd.concat(all_dfs, ignore_index=True)
        print(f"合并完成，总计 {len(merged_df)} 条记录")
        
        return merged_df
    
    def create_merged_config(self, dataset_names: List[str], output_dataset_name: str) -> Dict[str, Any]:
        """
        创建合并后数据集的配置
        
        Args:
            dataset_names: 原数据集名称列表
            output_dataset_name: 输出数据集名称
        
        Returns:
            合并后的配置字典
        """
        # 以第一个数据集为基础配置
        base_config = self.data_config[dataset_names[0]].copy()
        
        # 需要相加的数值字段
        sum_fields = ["num_q", "num_c", "students_num_train"]
        
        # 初始化累加字段
        for field in sum_fields:
            if field in base_config:
                base_config[field] = 0
        
        # 累加所有数据集的数值
        for dataset_name in dataset_names:
            config = self.data_config[dataset_name]
            for field in sum_fields:
                if field in config and field in base_config:
                    base_config[field] += config[field]
        
        # 更新路径和文件名
        base_config["dpath"] = f"/data/pykt_datasets/data/{output_dataset_name}"
        base_config["train_valid_file"] = base_config.get("train_valid_file", "train_valid_sequences.csv")
        
        # 处理其他可能需要合并的字段
        # input_type: 取所有数据集的并集
        all_input_types = set()
        for dataset_name in dataset_names:
            config = self.data_config[dataset_name]
            if "input_type" in config:
                all_input_types.update(config["input_type"])
        if all_input_types:
            base_config["input_type"] = list(all_input_types)
        
        # max_concepts: 取最大值
        max_concepts_list = []
        for dataset_name in dataset_names:
            config = self.data_config[dataset_name]
            if "max_concepts" in config:
                max_concepts_list.append(config["max_concepts"])
        if max_concepts_list:
            base_config["max_concepts"] = max(max_concepts_list)
        
        # min_seq_len: 取最小值
        min_seq_len_list = []
        for dataset_name in dataset_names:
            config = self.data_config[dataset_name]
            if "min_seq_len" in config:
                min_seq_len_list.append(config["min_seq_len"])
        if min_seq_len_list:
            base_config["min_seq_len"] = min(min_seq_len_list)
        
        # maxlen: 取最大值
        maxlen_list = []
        for dataset_name in dataset_names:
            config = self.data_config[dataset_name]
            if "maxlen" in config:
                maxlen_list.append(config["maxlen"])
        if maxlen_list:
            base_config["maxlen"] = max(maxlen_list)
        
        print(f"合并后的配置:")
        for field in sum_fields:
            if field in base_config:
                print(f"  - {field}: {base_config[field]}")
        
        return base_config
    def merge_test_datasets(self, test_dataset_names: List[str], output_dataset_name: str) -> str:
        """
        合并多个测试数据集并保存到指定数据集目录下
        
        Args:
            test_dataset_names: 要合并的测试数据集名称列表
            output_dataset_name: 目标数据集名称（用于确定保存路径）
        
        Returns:
            合并后的测试文件路径
        """
        if not test_dataset_names:
            raise ValueError("测试数据集名称列表不能为空")
        
        print(f"开始合并测试数据集: {test_dataset_names}")
        
        # 验证输入数据集是否存在
        for dataset_name in test_dataset_names:
            if dataset_name not in self.data_config:
                raise ValueError(f"数据集 '{dataset_name}' 不存在于配置文件中")
        
        # 读取和合并所有测试CSV文件
        test_files = {
            "test_file": "test_sequences.csv",
            "test_window_file": "test_window_sequences.csv", 
            "test_question_file": "test_question_sequences.csv",
            "test_question_window_file": "test_question_window_sequences.csv"
        }
        
        merged_test_dfs = {}
        
        for file_key, file_name in test_files.items():
            try:
                merged_df = self.merge_specific_test_files(test_dataset_names, file_key)
                merged_test_dfs[file_key] = merged_df
            except (KeyError, FileNotFoundError) as e:
                print(f"警告: 无法合并 {file_key}: {e}")
                continue
        
        if not merged_test_dfs:
            raise ValueError("没有找到任何可合并的测试文件")
        
        # 获取目标数据集的目录路径
        if output_dataset_name not in self.data_config:
            raise ValueError(f"目标数据集 '{output_dataset_name}' 不存在于配置文件中")
        
        output_dir = Path(self.data_config[output_dataset_name]["dpath"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存所有合并后的测试CSV文件
        saved_files = []
        for file_key, merged_df in merged_test_dfs.items():
            output_file_name = test_files[file_key]
            test_output_path = output_dir / output_file_name
            merged_df.to_csv(test_output_path, index=False)
            print(f"合并后的测试CSV文件已保存到: {test_output_path}")
            saved_files.append((file_key, output_file_name))
        
        # 更新目标数据集配置中的测试文件字段
        for file_key, file_name in saved_files:
            self.data_config[output_dataset_name][file_key] = file_name
        
        self.save_config()
        print(f"配置文件已更新，添加测试文件路径到数据集: {output_dataset_name}")
        
        return str(output_dir / "test_sequences.csv")

    def merge_specific_test_files(self, dataset_names: List[str], file_key: str) -> pd.DataFrame:
        """
        合并指定类型的测试CSV文件

        Args:
            dataset_names: 数据集名称列表
            file_key: 配置中的文件键名

        Returns:
            合并后的DataFrame
        """
        all_dfs = []

        for dataset_name in dataset_names:
            config = self.data_config[dataset_name]
            test_file_name = config.get(file_key)
            if not test_file_name:
                raise KeyError(f"数据集 '{dataset_name}' 没有定义 '{file_key}'")

            csv_path = Path(config["dpath"]) / test_file_name

            if not csv_path.exists():
                raise FileNotFoundError(f"测试CSV文件不存在: {csv_path}")

    
            print(f"读取测试CSV文件: {csv_path}")
            df = pd.read_csv(csv_path)

            # 添加数据集标识列
            df["dataset_name"] = dataset_name
            all_dfs.append(df)
            print(f"  - 读取 {len(df)} 条记录")

        # 合并所有DataFrame
        merged_df = pd.concat(all_dfs, ignore_index=True)
        print(f"测试集({file_key})合并完成，总计 {len(merged_df)} 条记录")
        return merged_df
    def get_available_datasets(self) -> List[str]:
        """获取所有可用的数据集名称"""
        return list(self.data_config.keys())
    
    def print_dataset_info(self, dataset_name: str):
        """打印数据集信息"""
        if dataset_name not in self.data_config:
            print(f"数据集 '{dataset_name}' 不存在")
            return
        
        config = self.data_config[dataset_name]
        print(f"\n数据集: {dataset_name}")
        print(f"  路径: {config.get('dpath', 'N/A')}")
        print(f"  问题数: {config.get('num_q', 'N/A')}")
        print(f"  概念数: {config.get('num_c', 'N/A')}")
        print(f"  训练学生数: {config.get('students_num_train', 'N/A')}")
        print(f"  最大长度: {config.get('maxlen', 'N/A')}")


def main():
    """主函数示例"""
    # 创建合并器实例
    merger = DatasetMerger()
    
    # 显示可用数据集
    print("可用的数据集:")
    available_datasets = merger.get_available_datasets()
    for i, dataset in enumerate(available_datasets, 1):
        print(f"{i}. {dataset}")
    
    # 示例：合并数据集
    # 请根据实际需要修改数据集名称列表
    datasets_to_merge = ["peiyou", "nips_task34","bridge2algebra2006"]  # 修改为实际的数据集名称
    
    try:
        # 检查数据集是否存在
        for dataset in datasets_to_merge:
            if dataset in available_datasets:
                merger.print_dataset_info(dataset)
        
        # 执行合并
        output_name = merger.merge_datasets(datasets_to_merge)
        test_datasets = ["nips_task34"]
        merger.merge_test_datasets(test_datasets, "A".join(datasets_to_merge))
        # 显示合并结果
        print(f"\n合并完成！")
        merger.print_dataset_info(output_name)
        
    except Exception as e:
        print(f"合并过程中出现错误: {e}")


if __name__ == "__main__":
    main()