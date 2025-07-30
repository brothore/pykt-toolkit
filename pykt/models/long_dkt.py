import os
import numpy as np
import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
import json

class LONG_DKT(Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768, 
                 data_config_path="../configs/data_config.json", dataset_name=None):
        super().__init__()
        self.model_name = "long_dkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        
        # 加载学生数量配置
        self.students_num = self._load_students_num(data_config_path, dataset_name)
        
        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)
        
        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)
        
        # 为每个学生维护隐藏状态和细胞状态
        # student_hidden_states: [students_num, hidden_size]
        # student_cell_states: [students_num, hidden_size]
        self.register_buffer('student_hidden_states', 
                           torch.zeros(self.students_num, self.hidden_size))
        self.register_buffer('student_cell_states', 
                           torch.zeros(self.students_num, self.hidden_size))
        
        # 记录每个学生是否已经初始化过（用于第一次遇到学生时的处理）
        self.register_buffer('student_initialized', 
                           torch.zeros(self.students_num, dtype=torch.bool))
    
    def _load_students_num(self, config_path, dataset_name):
        """从配置文件加载学生数量"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            if dataset_name and dataset_name in config:
                return config[dataset_name].get('students_num')  # 默认值
    
    def get_student_states(self, uids):
        """
        根据学生ID获取对应的隐藏状态
        Args:
            uids: [batch_size] 学生ID张量
        Returns:
            h0: [1, batch_size, hidden_size] 初始隐藏状态
            c0: [1, batch_size, hidden_size] 初始细胞状态
        """
        batch_size = uids.size(0)
        device = uids.device
        
        # 获取每个学生的隐藏状态和细胞状态
        h0 = self.student_hidden_states[uids]  # [batch_size, hidden_size]
        c0 = self.student_cell_states[uids]    # [batch_size, hidden_size]
        
        # 对于第一次遇到的学生，使用零初始化
        first_time_mask = ~self.student_initialized[uids]  # [batch_size]
        if first_time_mask.any():
            h0[first_time_mask] = 0
            c0[first_time_mask] = 0
            # 标记这些学生已经初始化
            self.student_initialized[uids[first_time_mask]] = True
        
        # LSTM需要的格式: [num_layers, batch_size, hidden_size]
        h0 = h0.unsqueeze(0)  # [1, batch_size, hidden_size]
        c0 = c0.unsqueeze(0)  # [1, batch_size, hidden_size]
        
        return h0, c0
    
    def update_student_states(self, uids, h_final, c_final):
        """
        更新学生的隐藏状态
        Args:
            uids: [batch_size] 学生ID张量
            h_final: [1, batch_size, hidden_size] 最终隐藏状态
            c_final: [1, batch_size, hidden_size] 最终细胞状态
        """
        # 取最后一层的状态 [batch_size, hidden_size]
        h_final = h_final.squeeze(0)
        c_final = c_final.squeeze(0)
        
        # 更新对应学生的状态
        self.student_hidden_states[uids] = h_final.detach()
        self.student_cell_states[uids] = c_final.detach()
    
    def forward(self, q, r, uids=None):
        """
        Args:
            q: [batch_size, seq_len] 概念序列
            r: [batch_size, seq_len] 回答序列  
            uids: [batch_size] 学生ID (可选，如果不提供则不使用学生记忆)
        Returns:
            y: [batch_size, seq_len, num_c] 预测概率
        """
        emb_type = self.emb_type
        if emb_type == "qid":
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)
        
        # 如果提供了学生ID，使用学生特定的隐藏状态
        if uids is not None:
            h0, c0 = self.get_student_states(uids)
            h, (h_final, c_final) = self.lstm_layer(xemb, (h0, c0))
            # 更新学生状态
            self.update_student_states(uids, h_final, c_final)
        else:
            # 传统模式，不使用学生记忆
            h, _ = self.lstm_layer(xemb)
        
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        
        return y
    
    def reset_student_state(self, uid):
        """重置特定学生的状态（用于新episode开始）"""
        self.student_hidden_states[uid] = 0
        self.student_cell_states[uid] = 0
        self.student_initialized[uid] = False
    
    def reset_all_students(self):
        """重置所有学生的状态"""
        self.student_hidden_states.zero_()
        self.student_cell_states.zero_()
        self.student_initialized.zero_()
    
    def get_student_memory_stats(self):
        """获取学生记忆的统计信息"""
        initialized_count = self.student_initialized.sum().item()
        total_students = self.students_num
        return {
            'initialized_students': initialized_count,
            'total_students': total_students,
            'memory_usage_ratio': initialized_count / total_students
        }