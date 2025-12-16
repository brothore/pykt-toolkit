import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor
import numpy as np
import random

class KTQueDataset(Dataset):
    def __init__(self, file_path, input_type, folds, concept_num, max_concepts, qtest=False, mode="test", 
                 aug_probs=None):
        """
        Args:
            aug_probs (dict): 包含所有增强参数，包括结构增强和值增强(random_rev)
                              e.g. {'random_rev': 0.05, 'truncate': 0.2, ...}
        """
        super(KTQueDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.concept_num = concept_num
        self.max_concepts = max_concepts
        self.mode = mode
        
        # --- 统一配置管理 ---
        # 如果 aug_probs 为 None，则初始化为空字典
        self.aug_probs = aug_probs if aug_probs is not None else {}
        
        # 从字典中提取 random_rev，默认为 0.0
        self.random_rev = self.aug_probs.get('random_rev', 0.0)
        
        if "questions" not in input_type or "concepts" not in input_type:
            raise Exception("The input types must contain both questions and concepts")

        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])

        # 缓存文件名包含 random_rev 以区分不同的投毒比例
        # 注意：这里使用 aug_qlevel 后缀，防止读取旧的错误缓存
        p_rev = self.aug_probs.get('random_rev', 0)
        p_trunc = self.aug_probs.get('truncate', 0)
        p_dup = self.aug_probs.get('duplicate', 0)
        p_shuf = self.aug_probs.get('shuffle', 0)
        
        # 仅当对应概率 > 0 时才记录强度参数，否则视为 0，保持文件名整洁且逻辑正确
        # 如果 p_dup 为 0，copy_ratio 0.1 和 0.2 应该是同一个缓存（因为不执行复制）
        r_copy = self.aug_probs.get('copy_ratio', 0) if p_dup > 0 else 0
        r_drop = self.aug_probs.get('drop_ratio', 0) if p_trunc > 0 else 0

        # 格式说明: 
        # td: random_rev (Token Disturbance/Poisoning)
        # tr: truncate, dp: duplicate, sh: shuffle
        # cr: copy_ratio, dr: drop_ratio
        aug_suffix = (
            f"_td{p_rev}"
            f"_tr{p_trunc}_dp{p_dup}_sh{p_shuf}"
            f"_cr{r_copy}_dr{r_drop}"
        )

        # 拼接文件名
        processed_data = file_path + folds_str + aug_suffix + "_aug_qlevel.pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str} aug{aug_suffix} mode: {self.mode}...")
            self.dori = self.__load_data__(sequence_path, folds)
            pd.to_pickle(self.dori, processed_data)
        else:
            print(f"Read data from processed file: {processed_data} aug{aug_suffix}")
            self.dori = pd.read_pickle(processed_data)
            
        print(f"file path: {file_path}, mode: {self.mode}, qlen: {len(self.dori['qseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def __len__(self):
        return len(self.dori["rseqs"])

    def __augment_seq__(self, raw_data_dict):
        """对长度为 L 的原始序列进行结构性增强"""
        # 1. 获取有效长度
        r_seq = raw_data_dict["rseqs"]
        valid_mask = (r_seq != -1)
        valid_len = valid_mask.sum().item()
        
        if valid_len < 2: return raw_data_dict

        indices = np.arange(valid_len)
        augmented = False

        # --- Augmentation Logic ---
        # 1. Shuffle
        if self.aug_probs.get('shuffle', 0) > 0 and random.random() < self.aug_probs['shuffle']:
            np.random.shuffle(indices)
            augmented = True

        # 2. Truncate
        if not augmented and self.aug_probs.get('truncate', 0) > 0 and random.random() < self.aug_probs['truncate']:
            if random.random() < 0.5: # 连续截断
                min_keep = max(int(valid_len * 0.5), 2)
                if valid_len > min_keep:
                    start = random.randint(0, valid_len - min_keep)
                    length = random.randint(min_keep, valid_len - start)
                    indices = indices[start : start + length]
                    augmented = True
            else: # 离散截断
                drop_ratio = self.aug_probs.get('drop_ratio', 0.1)
                keep_mask = np.random.rand(len(indices)) > drop_ratio
                if keep_mask.sum() > 2:
                    indices = indices[keep_mask]
                    augmented = True

        # 3. Duplicate
        if not augmented and self.aug_probs.get('duplicate', 0) > 0 and random.random() < self.aug_probs['duplicate']:
            copy_ratio = self.aug_probs.get('copy_ratio', 0.1)
            num_copy = max(1, int(valid_len * copy_ratio))
            copy_indices = np.random.choice(indices, num_copy, replace=True)
            new_indices = []
            for idx in indices:
                new_indices.append(idx)
                if idx in copy_indices: new_indices.append(idx)
            indices = np.array(new_indices)
            augmented = True

        if not augmented:
            return raw_data_dict

        # --- Reconstruct ---
        full_len = len(r_seq)
        if len(indices) > full_len: indices = indices[-full_len:]
        
        new_data_dict = {}
        for key, tensor in raw_data_dict.items():
            valid_part = tensor[:valid_len]
            if isinstance(valid_part, torch.Tensor):
                new_seq = valid_part[indices]
            else:
                new_seq = torch.tensor(np.array(valid_part)[indices])
            
            curr_len = len(new_seq)
            pad_len = full_len - curr_len
            
            if len(tensor.shape) > 1:
                pad_shape = (pad_len, tensor.shape[1])
                padding = torch.full(pad_shape, -1, dtype=tensor.dtype)
            else:
                padding = torch.full((pad_len,), -1, dtype=tensor.dtype)
            
            new_data_dict[key] = torch.cat((new_seq, padding), dim=0)
            
        return new_data_dict

    def __getitem__(self, index):
        dcur = dict()
        raw_row = {}
        keys_to_augment = ["qseqs", "cseqs", "rseqs", "tseqs", "utseqs", "smasks"]
        
        if "uid" in self.dori:
            dcur["uid"] = self.dori["uid"][index]

        # --- 1. 收集数据阶段 ---
        for key in self.dori:
            # 【修复点 1】优先处理 uid 和 masks，它们不需要增强也不需要判空(通常masks是生成的)
            if key in ["uid", "masks"]:
                continue
                
            # 【修复点 2】恢复原始代码的防御逻辑：如果是空列（如没有tseqs），直接透传空值并跳过
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue

            # 【修复点 3】根据是否需要增强分流
            if key in keys_to_augment:
                # 放入待增强字典，稍后统一处理
                raw_row[key] = self.dori[key][index].clone()
            else:
                # 不需要增强的普通列，直接读取
                dcur[key] = self.dori[key][index]

        # --- 2. 增强执行阶段 (Train 且 aug_probs 非空) ---
        if self.mode == "train" and self.aug_probs:
            raw_row = self.__augment_seq__(raw_row)

        # --- 3. 后处理阶段 (生成 Input/Label/Masks) ---
        
        # 动态计算 masks (基于增强后的 rseqs)
        # 注意：这里 raw_row["rseqs"] 肯定是存在的，因为 rseqs 是必须字段
        curr_r = raw_row["rseqs"]
        curr_mask = (curr_r[:-1] != -1) & (curr_r[1:] != -1)
        dcur["masks"] = curr_mask

        for key in raw_row:
            raw_data = raw_row[key]
            
            if key == 'cseqs':
                seqs = raw_data[:-1,:]
                shft_seqs = raw_data[1:,:]
            
            elif key == 'smasks':
                # smasks 逻辑：截取后半段对应 Label
                dcur["smasks"] = (raw_data[1:] != -1)
                continue # smasks 不需要 shft_ 前缀，处理完直接跳过

            else:
                # 投毒逻辑 (Label Reversal)
                if key == "rseqs" and self.mode == "train" and self.random_rev > 0:
                    noisy_data = raw_data.clone()
                    probs = torch.rand(noisy_data.shape)
                    valid_mask = (noisy_data != -1)
                    flip_mask = (probs < self.random_rev) & valid_mask
                    noisy_data[flip_mask] = 1.0 - noisy_data[flip_mask]
                    
                    seqs = noisy_data[:-1] * curr_mask
                    shft_seqs = raw_data[1:] * curr_mask
                else:
                    seqs = raw_data[:-1]
                    shft_seqs = raw_data[1:]
                    if len(seqs.shape) == 1:
                        seqs = seqs * curr_mask
                        shft_seqs = shft_seqs * curr_mask

            dcur[key] = seqs
            dcur["shft_"+key] = shft_seqs
            
        return dcur

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        # 保持修复后的逻辑：smasks 不提前切片
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
        
        # dori["smasks"] = (dori["smasks"][:, 1:] != pad_val) # 已注释：保持原长
        
        print(f"interaction_num: {interaction_num}")
        return dori