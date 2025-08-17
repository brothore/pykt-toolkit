#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import json
if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
else:
    from torch import FloatTensor, LongTensor

class MultiKTDataset(Dataset):
    """
    多数据集适配的KT数据集加载器
    支持前缀编码的重映射功能
    """
    
    def __init__(self, file_path, input_type, folds, qtest=False, config_path="../configs/data_config.json"):
        super(MultiKTDataset, self).__init__()
        
        # 获取数据集名称
        # print(f"[DEBUG] file_path: {file_path} (type: {type(file_path)})")
        self.dataset_name = os.path.basename(os.path.dirname(file_path))
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.data_config = json.load(f)
        # print(f"[DEBUG] self.dataset_name: {self.dataset_name} (type: {type(self.dataset_name)})")
        self.config = self.data_config[self.dataset_name]
        self.input_type = input_type
        self.qtest = qtest
        
        # 检查是否为合并数据集
        self.is_merged = "dataset_name" in self.config
        
        if self.is_merged:
            # 计算编码映射参数
            self._calculate_encoding_params()
        
        # 原有的数据加载逻辑
        sequence_path = file_path
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        
        if self.qtest:
            processed_data = file_path + folds_str + "_qtest.pkl"
        else:
            processed_data = file_path + folds_str + ".pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.dori, self.dqtest = self.__load_data__(sequence_path, folds)
                save_data = [self.dori, self.dqtest]
            else:
                self.dori = self.__load_data__(sequence_path, folds)
                save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.dori, self.dqtest = pd.read_pickle(processed_data)
            else:
                self.dori = pd.read_pickle(processed_data)
                
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def _calculate_encoding_params(self):
        """计算编码映射参数"""
        dataset_name = self.config["dataset_name"]
        dataset_count = len(dataset_name)
        
        # 计算前缀位数（能够表示数据集个数）
        self.prefix_bits = max(1, math.ceil(math.log10(dataset_count)))
        
        # 为每个字段计算后缀位数
        self.suffix_bits = {}
        self.encoding_maps = {}
        
        fields_to_encode = ["concepts", "questions", "uid", "qidxs", "orirow"]
        config_field_map = {
            "concepts": "num_c",
            "questions": "num_q",
            "uid": "students_num_train",  # 假设uid使用学生数量作为上限
            "qidxs": "num_q",  # qidxs使用问题数量作为上限
            "orirow": "students_num_train"  # orirow使用学生数量作为上限
        }
        
        for field in fields_to_encode:
            config_field = config_field_map.get(field, "num_q")
            
            # 找到该字段在所有数据集中的最大值
            max_value = 0
            for dataset_name in dataset_name:
                original_config = self.config["original_configs"][dataset_name]
                if config_field in original_config:
                    max_value = max(max_value, original_config[config_field])
            
            # 计算需要的后缀位数
            self.suffix_bits[field] = max(1, math.ceil(math.log10(max_value + 1)))
            
            # 创建编码映射
            self.encoding_maps[field] = {}
            for i, dataset_name in enumerate(dataset_name):
                prefix = i * (10 ** self.suffix_bits[field])
                self.encoding_maps[field][dataset_name] = prefix
        
        print(f"编码参数: prefix_bits={self.prefix_bits}, suffix_bits={self.suffix_bits}")
        print(f"编码映射: {self.encoding_maps}")

    def _remap_sequence(self, sequence_str, field, dataset_name):
        """重新映射序列编码"""
        if not self.is_merged or field not in self.encoding_maps:
            return [int(x) for x in sequence_str.split(",")]
        
        original_seq = [int(x) for x in sequence_str.split(",")]
        prefix_offset = self.encoding_maps[field][dataset_name]
        
        # 应用前缀编码
        remapped_seq = [x + prefix_offset if x >= 0 else x for x in original_seq]
        return remapped_seq

    def __len__(self):
        """return the dataset length"""
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """获取数据项"""
        dcur = dict()
        mseqs = self.dori["masks"][index]
        
        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue
                
            seqs = self.dori[key][index][:-1] * mseqs
            shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_"+key] = shft_seqs
            
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        
        if not self.qtest:
            return dcur
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur, dqtest

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """加载和处理数据"""
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}
        
        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)]
        interaction_num = 0
        
        dqtest = {"qidxs": [], "rests": [], "orirow": []}
        
        for i, row in df.iterrows():
            # 获取数据集名称（如果是合并数据集）
            dataset_name = row.get("dataset_name", self.dataset_name) if self.is_merged else self.dataset_name
            
            # 处理concepts
            if "concepts" in self.input_type:
                cseq = self._remap_sequence(row["concepts"], "concepts", dataset_name)
                dori["cseqs"].append(cseq)
                
            # 处理questions
            if "questions" in self.input_type:
                qseq = self._remap_sequence(row["questions"], "questions", dataset_name)
                dori["qseqs"].append(qseq)
            
            # 处理timestamps和usetimes（不需要重映射）
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])
                
            # 处理responses（不需要重映射）
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])
            
            interaction_num += dori["smasks"][-1].count(1)
            
            if self.qtest:
                # 处理qtest相关数据
                qidxs = self._remap_sequence(row["qidxs"], "qidxs", dataset_name)
                dqtest["qidxs"].append(qidxs)
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                
                # 处理orirow
                orirow = self._remap_sequence(row["orirow"], "orirow", dataset_name)
                dqtest["orirow"].append(orirow)
        
        # 转换为张量
        for key in dori:
            if key not in ["rseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])
        
        # 创建掩码
        mask_seqs = (dori["cseqs"][:, :-1] != pad_val) * (dori["cseqs"][:, 1:] != pad_val)
        dori["masks"] = mask_seqs
        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        
        print(f"interaction_num: {interaction_num}")
        
        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            return dori, dqtest
            
        return dori
