import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor
import numpy as np
import random
import csv
from tqdm import tqdm

class KTQueDataset(Dataset):
    def __init__(self, file_path, input_type, folds, concept_num, max_concepts, qtest=False, mode="test", 
                 aug_probs=None, diff_level=100): # <--- 新增 diff_level 参数
        """
        Args:
            diff_level (int): 难度的分级数量，默认为 100
        """
        super(KTQueDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.concept_num = concept_num
        self.max_concepts = max_concepts
        self.mode = mode
        self.diff_level = diff_level # <--- 保存难度等级

        # --- 1. 难度文件路径管理 ---
        # 提取目录路径
        dpath = os.path.dirname(file_path) 
        # 定义难度缓存文件 (文件名包含 diff_level 防止参数变化导致读取旧缓存)
        self.skills_diff_path = os.path.join(dpath, f'skills_difficult_{diff_level}.csv')
        self.questions_diff_path = os.path.join(dpath, f'questions_difficult_{diff_level}.csv')

        # --- 2. 检查并计算难度 (如果不存在) ---
        if not os.path.exists(self.skills_diff_path) or not os.path.exists(self.questions_diff_path):
            print(f"Difficulty files not found. Starting computation for level {diff_level}...")
            # 读取原始 CSV 进行统计 (假设原始文件就在 file_path)
            # 注意：通常建议只用训练集算难度，这里为了简化沿用 DIMKT 逻辑读取整个文件
            df_raw = pd.read_csv(file_path)
            self.difficult_compute(df_raw, self.skills_diff_path, self.questions_diff_path, diff_level)
        
        # --- 3. 预加载难度字典到内存 ---
        self.sds = self.__load_diff_map__(self.skills_diff_path)
        self.qds = self.__load_diff_map__(self.questions_diff_path)

        # --- 统一配置管理 (原代码保持不变) ---
        self.aug_probs = aug_probs if aug_probs is not None else {}
        self.random_rev = self.aug_probs.get('random_rev', 0.0)
        
        if "questions" not in input_type or "concepts" not in input_type:
            raise Exception("The input types must contain both questions and concepts")

        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])

        p_rev = self.aug_probs.get('random_rev', 0)
        p_trunc = self.aug_probs.get('truncate', 0)
        p_dup = self.aug_probs.get('duplicate', 0)
        p_shuf = self.aug_probs.get('shuffle', 0)
        r_copy = self.aug_probs.get('copy_ratio', 0) if p_dup > 0 else 0
        r_drop = self.aug_probs.get('drop_ratio', 0) if p_trunc > 0 else 0

        aug_suffix = (
            f"_td{p_rev}"
            f"_tr{p_trunc}_dp{p_dup}_sh{p_shuf}"
            f"_cr{r_copy}_dr{r_drop}"
        )

        # 缓存文件名增加 diff_level 标识
        processed_data = file_path + folds_str + aug_suffix + f"_diff{diff_level}_aug_qlevel.pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str} aug{aug_suffix} diff{diff_level}...")
            self.dori = self.__load_data__(sequence_path, folds)
            pd.to_pickle(self.dori, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            self.dori = pd.read_pickle(processed_data)
            
        print(f"file path: {file_path}, mode: {self.mode}, qlen: {len(self.dori['qseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def __load_diff_map__(self, path):
        """辅助函数：读取CSV到字典"""
        diff_map = {}
        with open(path, 'r', encoding="UTF8") as f:
            reader = csv.reader(f)
            keys = next(reader)
            vals = next(reader)
            for k, v in zip(keys, vals):
                diff_map[int(k)] = int(v)
        return diff_map

    def difficult_compute(self, df, sds_path, qds_path, diff_level):
        """
        仿造 DIMKT 逻辑计算难度，但适配多知识点结构
        """
        # 统计累加器
        s_counts = {}   # {skill_id: [total_count, correct_count]}
        q_counts = {}   # {question_id: [total_count, correct_count]}

        print("Computing difficulty statistics...")
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            # 解析字符串
            try:
                raw_q = [int(x) for x in str(row["questions"]).split(",")]
                # concepts 可能是 "1_2,3,4_5" 格式
                raw_c_strs = str(row["concepts"]).split(",") 
                raw_r = [int(x) for x in str(row["responses"]).split(",")]
            except:
                continue # 跳过坏数据

            length = len(raw_r)
            
            # 遍历该序列中的每一次交互
            for i in range(length):
                r = raw_r[i]
                if r == -1: continue # 跳过填充值

                # 1. 统计题目难度
                qid = raw_q[i]
                if qid not in q_counts: q_counts[qid] = [0, 0]
                q_counts[qid][0] += 1
                q_counts[qid][1] += r

                # 2. 统计知识点难度 (处理多知识点 "1_2")
                c_str = raw_c_strs[i]
                if c_str == "-1": continue
                
                c_ids = [int(x) for x in c_str.split("_")]
                for cid in c_ids:
                    if cid not in s_counts: s_counts[cid] = [0, 0]
                    s_counts[cid][0] += 1
                    s_counts[cid][1] += r

        # 计算最终难度值并保存 (公式: rate * diff_level + 1)
        # 过滤: <30次记录的设为默认值 1
        
        def save_map(counts_dict, save_path):
            res_map = {}
            for k, v in counts_dict.items():
                total, correct = v
                if total < 30:
                    res_map[k] = 1
                else:
                    # Imitating DIMKT formula: higher value = higher accuracy (easier)
                    avg = int((correct / total) * diff_level) + 1
                    res_map[k] = avg
            
            with open(save_path, 'w', newline='', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(list(res_map.keys()))
                writer.writerow(list(res_map.values()))

        save_map(s_counts, sds_path)
        save_map(q_counts, qds_path)

    def __augment_seq__(self, raw_data_dict):
        """对长度为 L 的原始序列进行结构性增强 (保持不变，但要兼容新key)"""
        # 原有的增强逻辑完全可以复用，只要确保 raw_data_dict 里包含了 qdseqs/sdseqs
        # 这里的代码逻辑是通用的，它会对 raw_data_dict 中的所有 Tensor 进行索引切片
        # 所以不需要修改这里的代码，只需要确保传入的 dict 包含新字段即可
        
        # ... (此处省略你原有的 __augment_seq__ 代码，逻辑无需变动) ...
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
            
            # 兼容多维 Tensor (如 cseqs, sdseqs)
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
        # <--- 修改点：将新的难度列加入增强列表
        keys_to_augment = ["qseqs", "cseqs", "rseqs", "tseqs", "utseqs", "smasks", "qdseqs", "sdseqs"]
        
        if "uid" in self.dori:
            dcur["uid"] = self.dori["uid"][index]

        # --- 1. 收集数据阶段 ---
        for key in self.dori:
            if key in ["uid", "masks"]:
                continue
            
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue

            if key in keys_to_augment:
                raw_row[key] = self.dori[key][index].clone()
            else:
                dcur[key] = self.dori[key][index]

        # --- 2. 增强执行阶段 ---
        if self.mode == "train" and self.aug_probs:
            raw_row = self.__augment_seq__(raw_row)

        # --- 3. 后处理阶段 ---
        curr_r = raw_row["rseqs"]
        curr_mask = (curr_r[:-1] != -1) & (curr_r[1:] != -1)
        dcur["masks"] = curr_mask

        for key in raw_row:
            raw_data = raw_row[key]
            
            if key in ['cseqs', 'sdseqs']: # <--- 修改点：sdseqs 也是 2D 的，处理方式同 cseqs
                seqs = raw_data[:-1,:]
                shft_seqs = raw_data[1:,:]
            
            elif key == 'smasks':
                dcur["smasks"] = (raw_data[1:] != -1)
                continue

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
        # <--- 修改点：新增 sdseqs, qdseqs
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], 
                "smasks": [], "uid": [], "sdseqs": [], "qdseqs": []}
        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)].copy()

        interaction_num = 0
        for i, row in df.iterrows():
            # 1. 处理 Concept 和 Skill Difficulty
            if "concepts" in self.input_type:
                row_skills = []
                row_skill_diffs = [] # <--- 新增
                
                raw_skills = str(row["concepts"]).split(",")
                for concept in raw_skills:
                    if concept == "-1":
                        skills = [-1] * self.max_concepts
                        s_diffs = [-1] * self.max_concepts # Padding 难度
                    else:
                        skills = [int(_) for _ in concept.split("_")]
                        # <--- 查表获取难度，缺失值默认为 1
                        s_diffs = [self.sds.get(s, 1) for s in skills]
                        
                        # Padding 到 max_concepts
                        pad_len = self.max_concepts - len(skills)
                        skills = skills + [-1] * pad_len
                        s_diffs = s_diffs + [-1] * pad_len
                    
                    row_skills.append(skills)
                    row_skill_diffs.append(s_diffs)
                
                dori["cseqs"].append(row_skills)
                dori["sdseqs"].append(row_skill_diffs) # 存入

            # 2. 处理 Question 和 Question Difficulty
            if "questions" in self.input_type:
                q_list = [int(_) for _ in str(row["questions"]).split(",")]
                dori["qseqs"].append(q_list)
                
                # <--- 查表获取难度
                qd_list = []
                for q in q_list:
                    if q == -1: qd_list.append(-1)
                    else: qd_list.append(self.qds.get(q, 1))
                dori["qdseqs"].append(qd_list)

            # ... (其他字段处理保持不变) ...
            if "uid" in row:
                dori["uid"].append(int(row["uid"]))
            else:
                dori["uid"].append(-1)
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])
            
            dori["rseqs"].append([int(_) for _ in str(row["responses"]).split(",")])
            dori["smasks"].append([int(_) for _ in str(row["selectmasks"]).split(",")])

            interaction_num += dori["smasks"][-1].count(1)

        # 转 Tensor
        for key in dori:
            if key not in ["rseqs", "uid"]:
                dori[key] = LongTensor(dori[key])
            elif key == "rseqs":
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["rseqs"][:,:-1] != pad_val) * (dori["rseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs
        
        print(f"interaction_num: {interaction_num}")
        return dori
    
    def __len__(self):
        return len(self.dori["rseqs"])