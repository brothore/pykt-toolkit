import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from mamba_ssm import Mamba

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class MAMBA_AKT(nn.Module):
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout, d_ff=256, 
            kq_same=1, final_fc_dim=256, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "mamba_akt"
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_l = d_model
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, 1)
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)
        
        if emb_type.startswith("qid"):
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa: 
                self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
            else:
                self.qa_embed = nn.Embedding(2, embed_l)

        # Architecture Object with Mamba layers
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, emb_type=self.emb_type)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target)+q_embed_data
        return q_embed_data, qa_embed_data

    def forward(self, q_data, target, pid_data=None, qtest=False):
        emb_type = self.emb_type
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        pid_embed_data = None
        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data

            qa_embed_diff_data = self.qa_embed_diff(
                target)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    (qa_embed_diff_data+q_embed_diff_data)
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
        else:
            c_reg_loss = 0.

        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        preds = m(output)
        if not qtest:
            return preds, c_reg_loss
        else:
            return preds, c_reg_loss, concat_q


class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, emb_type):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'mamba_akt'}:
            # Use Mamba layers instead of Transformer layers
            self.blocks_1 = nn.ModuleList([
                MambaLayer(d_model=d_model, d_state=16, d_conv=4, expand=2, 
                          dropout=dropout, use_cross_attention=False)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                MambaLayer(d_model=d_model, d_state=16, d_conv=4, expand=2,
                          dropout=dropout, use_cross_attention=(i > 0))
                for i in range(n_blocks*2)
            ])

    def forward(self, q_embed_data, qa_embed_data, pid_embed_data):
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        x = q_pos_embed

        # Encoder: encode qa information from 0 to t-1
        for block in self.blocks_1:
            y = block(y, causal=True)  # Causal encoding of historical QA
        
        # Decoder-like processing
        flag_first = True
        for i, block in enumerate(self.blocks_2):
            if flag_first:  # First layer: self-attention on questions only
                x = block(x, causal=True)  # Self-attention on current questions
                flag_first = False
            else:  # Cross-attention layers
                # For cross-attention: query from x, key/value from y
                # This ensures we don't peek at current response
                x = block(x, context=y, causal=True)
                flag_first = True
        
        return x


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1, use_cross_attention=False):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.d_model = d_model
        
        # 主Mamba块
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # 交叉注意力机制修改
        if use_cross_attention:
            # 关键修改：处理拼接后特征的维度变化
            self.cross_input_proj = nn.Linear(2 * d_model, d_model)
            self.cross_mamba = Mamba(
                d_model=d_model,  # 保持原始维度
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.combine_proj = nn.Linear(2 * d_model, d_model)  # 输出投影层

        # 层归一化和FFN保持不变
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, causal=True):
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # 自注意力路径
        x = self.norm1(x)
        self_output = self.mamba(x) if not causal else self.mamba(x.clone())
        
        # 交叉注意力路径（修改后）
        cross_output = torch.zeros_like(x)
        if self.use_cross_attention and context is not None:
            if causal:
                # 高效向量化实现替换原始循环
                for t in range(seq_len):
                    if t == 0:
                        cross_output[:, t:t+1] = 0
                    else:
                        ctx_slice = context[:, :t]
                        # 1. 拼接特征 (d_model*2)
                        combined = torch.cat([
                            x[:, t:t+1].expand(-1, t, -1), 
                            ctx_slice
                        ], dim=-1)
                        # 2. 投影回原始维度
                        projected = self.cross_input_proj(combined)
                        # 3. 处理交叉注意力
                        cross_t = self.cross_mamba(projected)[:, -1:]
                        cross_output[:, t:t+1] = cross_t
            else:
                # 非因果模式处理
                combined = torch.cat([x, context], dim=-1)
                projected = self.cross_input_proj(combined)
                cross_output = self.cross_mamba(projected)
            
            # 合并特征
            combined_out = torch.cat([self_output, cross_output], dim=-1)
            output = self.combine_proj(combined_out)
        else:
            output = self_output
            
        # 残差连接和FFN
        output = residual + self.dropout(output)
        residual = output
        output = self.norm2(output)
        output = residual + self.ffn(output)
        
        return output
# Keep the original embedding classes
class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]