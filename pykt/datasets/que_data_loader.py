#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor # 修正了原代码可能的导入问题
import numpy as np
from pykt.config import random_rev
class KTQueDataset(Dataset):
    """Dataset for KT
        Args:
            file_path (str): train_valid/test file path
            input_type (list[str]): values are in ["questions", "concepts"]
            folds (set(int)): the folds used to generate dataset
            concept_num, max_concepts: config params
            qtest (bool): is question evaluation or not
            mode (str): "train", "valid", or "test" - 用于区分是否开启数据增强
            random_rev (float): 投毒阈值，例如 0.05
    """
    def __init__(self, file_path, input_type, folds, concept_num, max_concepts, qtest=False, mode="test"):
        super(KTQueDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.concept_num = concept_num
        self.max_concepts = max_concepts
        
        # 新增配置
        self.mode = mode
        self.random_rev = random_rev
        
        if "questions" not in input_type or "concepts" not in input_type:
            raise Exception("The input types must contain both questions and concepts")

        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])

        # 【修改点1】缓存名字加入 mode，防止训练集和测试集缓存混淆
        processed_data = file_path + folds_str + f"_td{self.random_rev}_qlevel.pkl" if random_rev>0 else file_path + folds_str + f"_qlevel.pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str} mode: {self.mode}...")
            # 原始加载逻辑不变
            self.dori = self.__load_data__(sequence_path, folds)
            save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            self.dori = pd.read_pickle(processed_data)
            
        print(f"file path: {file_path}, mode: {self.mode}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def __len__(self):
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        dcur = dict()
        mseqs = self.dori["masks"][index]
        if "uid" in self.dori:
            dcur["uid"] = self.dori["uid"][index]
            
        for key in self.dori:
            if key in ["masks", "smasks", "uid"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue

            # 处理 Concepts (形状不同，通常不进行投毒)
            if key == 'cseqs':
                seqs = self.dori[key][index][:-1,:]
                shft_seqs = self.dori[key][index][1:,:]
            else:
                # 获取该行原始数据 (Tensor)
                raw_data = self.dori[key][index]
                
                # 1. 准备 Label (Shifted sequences) - 永远使用原始真实数据
                shft_seqs = raw_data[1:] * mseqs
                
                # 2. 准备 Input (Sequence) - 仅在 rseqs 且 训练模式 下进行投毒
                if key == "rseqs" and self.mode == "train" and self.random_rev > 0:
                    # 必须 clone，否则会修改内存中的原始 self.dori，导致 epoch 之间叠加噪声
                    noisy_data = raw_data.clone()
                    
                    # 生成随机概率矩阵
                    probs = torch.rand(noisy_data.shape)
                    
                    # 生成掩码：
                    # 1. 有效值掩码 (不是 -1 的地方)
                    valid_mask = (noisy_data != -1)
                    # 2. 翻转掩码 (概率小于阈值 且 是有效值)
                    flip_mask = (probs < self.random_rev) & valid_mask
                    
                    # 执行翻转
                    # rseqs 是 FloatTensor (0.0 或 1.0)，用 1.0 - x 实现翻转
                    noisy_data[flip_mask] = 1.0 - noisy_data[flip_mask]
                    
                    # 切片生成输入，注意这里使用了 noisy_data
                    seqs = noisy_data[:-1] * mseqs
                else:
                    # 其他 key (如 qseqs, tseqs) 或者非训练模式，不做处理
                    seqs = raw_data[:-1] * mseqs

            dcur[key] = seqs
            dcur["shft_"+key] = shft_seqs
            
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        return dcur

    def get_skill_multi_hot(self, this_skills):
        skill_emb = [0] * self.concept_num
        for s in this_skills:
            skill_emb[s] = 1
        return skill_emb

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        # ... (此部分代码保持您原样即可，无需修改) ...
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": [], "uid": []}
        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)].copy()

        interaction_num = 0
        for i, row in df.iterrows():
            if "concepts" in self.input_type:
                row_skills = []
                raw_skills = row["concepts"].split(",")
                for concept in raw_skills:
                    if concept == "-1":
                        skills = [-1] * self.max_concepts
                    else:
                        skills = [int(_) for _ in concept.split("_")]
                        skills = skills +[-1]*(self.max_concepts-len(skills))
                    row_skills.append(skills)
                dori["cseqs"].append(row_skills)
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "uid" in row:
                dori["uid"].append(int(row["uid"]))
            else:
                dori["uid"].append(-1)
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])
                
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += dori["smasks"][-1].count(1)

        for key in dori:
            if key not in ["rseqs", "uid"]:
                dori[key] = LongTensor(dori[key])
            elif key == "rseqs":
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["rseqs"][:,:-1] != pad_val) * (dori["rseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs
        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")
        return dori