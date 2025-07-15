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
            kq_same=1, final_fc_dim=512, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768,
            d_state=16, d_conv=4, expand=2):
        super().__init__()
        """
        Input:
            d_model: dimension of mamba block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: not used in mamba version
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: not used in mamba version
            d_state: SSM state expansion factor for Mamba
            d_conv: Local convolution width for Mamba
            expand: Block expansion factor for Mamba
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
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        embed_l = d_model
        
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, 1) # 题目难度
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) # question emb
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l) # interaction emb
        
        if emb_type.startswith("qid"):
            # n_question+1 ,d_model
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa: 
                self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
            else: # false default
                self.qa_embed = nn.Embedding(2, embed_l)

        # Architecture Object with Mamba blocks
        self.model = MambaArchitecture(n_question=n_question, n_blocks=n_blocks, 
                                      dropout=dropout, d_model=d_model, d_ff=d_ff,
                                      d_state=d_state, d_conv=d_conv, expand=expand)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim), 
            nn.ReLU(), 
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), 
            nn.ReLU(), 
            nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)  # BS, seqlen, d_model
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    def forward(self, q_data, target, pid_data=None, qtest=False):
        emb_type = self.emb_type
        # Batch First
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        pid_embed_data = None
        if self.n_pid > 0: # have problem id
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

            qa_embed_diff_data = self.qa_embed_diff(target)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data + q_embed_diff_data)
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
        else:
            c_reg_loss = 0.

        # Pass to the mamba-based architecture
        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        preds = m(output)
        
        if not qtest:
            return preds, c_reg_loss
        else:
            return preds, c_reg_loss, concat_q


class MambaArchitecture(nn.Module):
    def __init__(self, n_question, n_blocks, d_model, d_ff, dropout, d_state=16, d_conv=4, expand=2):
        super().__init__()
        """
        Mamba-based architecture for knowledge tracing
        Args:
            d_state: SSM state expansion factor
            d_conv: Local convolution width
            expand: Block expansion factor
        """
        self.d_model = d_model
        
        # First set of Mamba blocks for encoding qa sequences
        self.blocks_1 = nn.ModuleList([
            MambaBlock(d_model=d_model, d_ff=d_ff, dropout=dropout, 
                      d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_blocks)
        ])
        
        # Second set of Mamba blocks for knowledge retrieval
        self.blocks_2 = nn.ModuleList([
            MambaBlock(d_model=d_model, d_ff=d_ff, dropout=dropout,
                      d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_blocks*2)
        ])

    def forward(self, q_embed_data, qa_embed_data, pid_embed_data):
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)
        
        y = qa_embed_data  # qa embeddings
        x = q_embed_data   # q embeddings
        
        # Encode qa sequences (0 to t-1)
        for block in self.blocks_1:
            y = block(y, pid_embed_data)
        
        # Knowledge retrieval with interaction between q and qa
        flag_first = True
        for block in self.blocks_2:
            if flag_first:
                # First layer: self-modeling of questions
                x = block(x, pid_embed_data, apply_ffn=False)
                flag_first = False
            else:
                # Subsequent layers: knowledge retrieval
                # Combine information from questions and qa history
                x = block(x, pid_embed_data, apply_ffn=True, cross_input=y)
                flag_first = True
                
        return x


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout, d_state=16, d_conv=4, expand=2):
        super().__init__()
        """
        Basic Mamba block with optional FFN
        Args:
            d_state: SSM state expansion factor
            d_conv: Local convolution width
            expand: Block expansion factor
        """
        # Mamba layer
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,   # SSM state expansion factor
            d_conv=d_conv,     # Local convolution width
            expand=expand,     # Block expansion factor
        ).to(device)
        
        # Layer normalization and dropout
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # Optional cross-attention-like mixing for knowledge retrieval
        self.cross_mix = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x, pid_embed_data=None, apply_ffn=True, cross_input=None):
        """
        Args:
            x: input sequence (BS, seqlen, d_model)
            pid_embed_data: problem difficulty embeddings
            apply_ffn: whether to apply FFN
            cross_input: optional cross input for knowledge retrieval
        """
        # Apply Mamba
        if cross_input is not None:
            # Mix current input with cross input (simulating cross-attention)
            combined = torch.cat([x, cross_input], dim=-1)
            mixed = self.cross_mix(combined)
            mamba_out = self.mamba(mixed)
        else:
            mamba_out = self.mamba(x)
        
        # Residual connection and layer norm
        x = x + self.dropout1(mamba_out)
        x = self.layer_norm1(x)
        
        # Optional FFN
        if apply_ffn:
            ffn_out = self.ffn(x)
            x = x + self.dropout2(ffn_out)
            x = self.layer_norm2(x)
            
        return x


# Keep the original embedding classes for compatibility
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