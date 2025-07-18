import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import math

class MAMBA_AKT(nn.Module):
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout, 
                 d_ff=16, d_conv=4, expand=2, final_fc_dim=512):
        super().__init__()
        self.n_question = n_question
        self.n_pid = n_pid
        self.dropout = dropout
        
        # Embedding layers
        self.q_embed = nn.Embedding(n_question, d_model)
        self.qa_embed = nn.Embedding(2, d_model)
        
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(n_pid + 1, 1)
            self.q_embed_diff = nn.Embedding(n_question + 1, d_model)
            self.qa_embed_diff = nn.Embedding(2 * n_question + 1, d_model)
        
        # 双流Mamba架构
        # 历史流：处理历史QA序列
        self.history_stream = nn.ModuleList([
            MambaBlock(d_model, d_ff=d_ff, d_conv=d_conv, expand=expand)
            for _ in range(n_blocks)
        ])
        
        # 查询流：处理当前问题序列
        self.query_stream = nn.ModuleList([
            MambaBlock(d_model, d_ff=d_ff, d_conv=d_conv, expand=expand)
            for _ in range(n_blocks)
        ])
        
        # 知识检索模块：融合两流信息
        self.knowledge_retriever = KnowledgeRetrieverMamba(
            d_model, d_ff, d_conv, expand, n_blocks
        )
        
        # 位置感知融合
        self.position_gate = PositionAwareGate(d_model)
        
        # 输出层
        self.out = nn.Sequential(
            nn.Linear(d_model * 2, final_fc_dim),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(final_fc_dim, 256),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, q_data, target, pid_data=None):
        # Embedding
        q_embed_data = self.q_embed(q_data)
        qa_embed_data = self.qa_embed(target) + q_embed_data
        
        # Handle difficulty parameters
        if self.n_pid > 0 and pid_data is not None:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
            
            qa_embed_diff_data = self.qa_embed_diff(target)
            qa_embed_data = qa_embed_data + pid_embed_data * (
                qa_embed_diff_data + q_embed_diff_data
            )
        
        # 历史流处理
        history_repr = qa_embed_data
        for mamba_block in self.history_stream:
            history_repr = mamba_block(history_repr)
        
        # 查询流处理
        query_repr = q_embed_data
        for mamba_block in self.query_stream:
            query_repr = mamba_block(query_repr)
        
        # 知识检索和融合
        retrieved_knowledge = self.knowledge_retriever(
            query_repr, history_repr, qa_embed_data
        )
        
        # 位置感知融合
        fused_repr = self.position_gate(retrieved_knowledge, query_repr)
        
        # 拼接query和融合表示
        concat_repr = torch.cat([fused_repr, query_repr], dim=-1)
        
        # 预测
        output = self.out(concat_repr).squeeze(-1)
        preds = torch.sigmoid(output)
        
        if self.n_pid > 0:
            c_reg_loss = (pid_embed_data ** 2).sum() * 1e-5
        else:
            c_reg_loss = 0.0
            
        return preds, c_reg_loss


class KnowledgeRetrieverMamba(nn.Module):
    """知识检索模块：使用Mamba的选择性机制实现知识检索"""
    def __init__(self, d_model, d_ff, d_conv, expand, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_ff, d_conv, expand)
            for _ in range(n_layers)
        ])
        
        # 选择门：决定从历史中检索多少信息
        self.selection_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # 初始状态调制
        self.init_state_modulator = nn.Linear(d_model, d_ff * expand)
        
    def forward(self, query, history, qa_embed):
        # 使用query调制Mamba的初始状态
        batch_size, seq_len = query.shape[:2]
        
        # 计算选择门
        gate_input = torch.cat([query, history], dim=-1)
        selection = self.selection_gate(gate_input)
        
        # 选择性地融合历史信息
        modulated_input = history * selection + qa_embed * (1 - selection)
        
        # 通过Mamba层处理
        output = modulated_input
        for layer in self.layers:
            # 为每一层注入query信息作为条件
            output = layer(output) + query * 0.1  # 轻微的残差连接
            
        return output


class PositionAwareGate(nn.Module):
    """位置感知门控机制：模拟原AKT中的距离衰减效应"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # 可学习的衰减参数（类似原始的gamma）
        self.decay_param = nn.Parameter(torch.ones(1))
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, d_model) * 0.1)
        
        # 融合网络
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, retrieved, query):
        batch_size, seq_len, _ = retrieved.shape
        
        # 获取位置编码
        pos_enc = self.pos_embedding[:, :seq_len, :]
        
        # 计算位置感知的衰减
        positions = torch.arange(seq_len, device=retrieved.device)
        dist_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        decay_weights = torch.exp(-self.decay_param * dist_matrix.float())
        
        # 应用位置感知衰减
        decay_weights = decay_weights.unsqueeze(0).unsqueeze(-1)
        position_aware_retrieved = retrieved.unsqueeze(2) * decay_weights
        position_aware_retrieved = position_aware_retrieved.mean(dim=2)
        
        # 融合
        combined = torch.cat([position_aware_retrieved, query], dim=-1)
        output = self.fusion(combined)
        
        return output


class MambaBlock(nn.Module):
    """标准Mamba块的简化实现（实际使用时应导入mamba_ssm）"""
    def __init__(self, d_model, d_ff=16, d_conv=4, expand=2):
        super().__init__()
        # 这里应该使用实际的Mamba实现
        # from mamba_ssm import Mamba
        # self.mamba = Mamba(d_model, d_ff=d_ff, d_conv=d_conv, expand=expand)
        
        # 简化版本用于演示
        self.norm = nn.LayerNorm(d_model)
        self.mamba = nn.Linear(d_model, d_model)  # 占位符
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # 实际实现中：
        # return x + self.dropout(self.mamba(self.norm(x)))
        
        # 简化版本
        residual = x
        x = self.norm(x)
        x = self.mamba(x)  # 实际应该调用Mamba层
        return residual + self.dropout(x)