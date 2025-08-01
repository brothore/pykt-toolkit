import os

import numpy as np
import torch
import json
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
class LONG_DKT(Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "long_dkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.num_layers = 1  # 添加层数属性

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True, num_layers=self.num_layers)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)
        
    def forward(self, q, r, initial_states=None):
        emb_type = self.emb_type
        if emb_type == "qid":
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)
            
        # 如果有初始状态则使用，否则LSTM会自己初始化
        if initial_states is not None:
            h, (hn, cn) = self.lstm_layer(xemb, initial_states)
        else:
            h, (hn, cn) = self.lstm_layer(xemb)
            
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        # 返回预测结果和最终状态(hn, cn)
        return y, (hn, cn)
class StudentHiddenStateManager:
    """管理每个学生的隐藏状态"""
    def __init__(self, data_config_path, hidden_size, num_layers=1,mode=""):
        """
        mode= train 或 eval
        """
        # 从配置文件读取学生数量
        with open(data_config_path, 'r') as f:
            config = json.load(f)
        
        self.num_students = 10000
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化所有学生的隐藏状态和细胞状态
        # 形状: [num_students, num_layers, hidden_size]
        self.hidden_states = torch.zeros(self.num_students, num_layers, hidden_size, device=self.device)
        self.cell_states = torch.zeros(self.num_students, num_layers, hidden_size, device=self.device)
        
        # 记录哪些学生已经有过交互（用于初始化检查）
        self.initialized_students = set()
    
    def get_student_states(self, uids):
        """根据学生ID获取对应的隐藏状态和细胞状态
        Args:
            uids: 学生ID张量，形状 [batch_size]
        Returns:
            hidden: [num_layers, batch_size, hidden_size]
            cell: [num_layers, batch_size, hidden_size]
        """
        batch_size = uids.size(0)
        
        # 获取每个学生的隐藏状态
        hidden = self.hidden_states[uids].permute(1, 0, 2)  # [num_layers, batch_size, hidden_size]
        cell = self.cell_states[uids].permute(1, 0, 2)      # [num_layers, batch_size, hidden_size]
        
        return (hidden.contiguous(), cell.contiguous())
    
    def update_student_states(self, uids, hidden, cell):
        """更新学生的隐藏状态
        Args:
            uids: 学生ID张量，形状 [batch_size]
            hidden: 新的隐藏状态，形状 [num_layers, batch_size, hidden_size]
            cell: 新的细胞状态，形状 [num_layers, batch_size, hidden_size]
        """
        # 将隐藏状态转换为 [batch_size, num_layers, hidden_size]
        hidden = hidden.permute(1, 0, 2)
        cell = cell.permute(1, 0, 2)
        
        # 更新对应学生的状态
        self.hidden_states[uids] = hidden.detach()
        self.cell_states[uids] = cell.detach()
        
        # 标记这些学生已初始化
        for uid in uids.cpu().numpy():
            self.initialized_students.add(int(uid))
    
    def reset_student_state(self, uid):
        """重置特定学生的隐藏状态（可用于新的序列开始）"""
        self.hidden_states[uid].zero_()
        self.cell_states[uid].zero_()
        self.initialized_students.discard(int(uid))
    
    def reset_all_states(self):
        """重置所有学生的隐藏状态"""
        self.hidden_states.zero_()
        self.cell_states.zero_()
        self.initialized_students.clear()