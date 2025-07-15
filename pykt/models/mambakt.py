import os

import numpy as np
import torch
from mamba_ssm import Mamba
from torch.nn import Module, Embedding, LSTM, Linear, Dropout

class MAMBAKT(Module):
    def __init__(self, num_c,emb_type="qid", emb_path="", emb_size=128, dropout=0.1, d_state=16):
        super().__init__()
        self.num_c = num_c
        self.emb_size = emb_size
        self.emb_type = emb_type
        self.model_name = "mambakt"

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)
        # Mamba 层替代原LSTM
        self.mamba_layer = Mamba(
            d_model=emb_size,       # 必须等于emb_size
            d_state=d_state,        # 状态维度（建议16）
            d_conv=4,               # 卷积核宽度
            expand=2,               # 扩展因子
        )
        
        self.dropout = Dropout(dropout)
        self.out_layer = Linear(emb_size, num_c)

    def forward(self, q, r):
        """输入格式说明：
        q: [batch_size, seq_len]  题目ID序列
        r: [batch_size, seq_len]  答题对错序列 (0/1)
        """
        # 组合交互嵌入
        if self.emb_type == "qid":
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)
        
        # Mamba 序列处理
        h = self.mamba_layer(xemb)  # [batch, seq_len, emb_size]
        h = self.dropout(h)
        
        # 输出预测
        y = torch.sigmoid(self.out_layer(h))  # [batch, seq_len, num_c]
        return y