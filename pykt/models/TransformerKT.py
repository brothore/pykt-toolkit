import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class TRANSFORMERKT(nn.Module):
    def __init__(self, num_c, num_q=None, emb_size=128, nhead=8, 
                 num_layers=2, dropout=0.1, emb_type='encoder_only',emb_path='', d_ff=None):
        super().__init__()
        self.emb_size = emb_size
        self.emb_type = emb_type
        # 设置默认的d_ff值（如果没有提供）
        if d_ff is None:
            d_ff = 2 * emb_size  # 默认保持原来的2倍关系
        # 知识点嵌入层
        self.skill_embedding = nn.Embedding(num_c, emb_size)
        self.model_name = "TransformerKT"
        # 题目ID嵌入层（可选）
        self.pid_embedding = None
        if num_q is not None:
            self.pid_embedding = nn.Embedding(num_q, emb_size)
            
        # 回答结果嵌入层（正确/错误）
        self.response_embedding = nn.Embedding(2, emb_size)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        
        # 根据架构类型初始化不同的组件
        if emb_type == 'encoder_only':
            # 仅使用Encoder
            encoder_layers = TransformerEncoderLayer(emb_size, nhead, 
                                                   dim_feedforward=d_ff,
                                                   dropout=dropout)
            self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
            
        elif emb_type == 'decoder_only':
            # 仅使用Decoder（类似GPT）
            decoder_layers = TransformerDecoderLayer(emb_size, nhead,
                                                   dim_feedforward=d_ff,
                                                   dropout=dropout)
            self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers)
            
        elif emb_type == 'encoder_decoder':
            # 使用Encoder-Decoder架构（类似原始Transformer）
            encoder_layers = TransformerEncoderLayer(emb_size, nhead,
                                                   dim_feedforward=d_ff,
                                                   dropout=dropout)
            self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
            
            decoder_layers = TransformerDecoderLayer(emb_size, nhead,
                                                   dim_feedforward=d_ff,
                                                   dropout=dropout)
            self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers)
        else:
            raise ValueError(f"Unsupported emb_type: {emb_type}")
        
        # 输出层
        self.output_layer = nn.Linear(emb_size, num_c)
        
    def forward(self, q_data, target, pid_data=None, qtest=False):
        batch_size, seq_len = q_data.size()
        
        # 知识点嵌入
        skill_emb = self.skill_embedding(q_data)  # [batch_size, seq_len, emb_size]
        
        # 回答结果嵌入（注意：使用前一个时间步的响应）
        # 对于第一个时间步，我们使用0填充（假设为初始状态）
        shifted_target = torch.cat([torch.zeros(batch_size, 1, dtype=torch.long, device=target.device), 
                                target[:, :-1]], dim=1)
        response_emb = self.response_embedding(shifted_target)  # [batch_size, seq_len, emb_size]
        
        # 组合特征
        x = skill_emb + response_emb
        
        # 如果使用题目ID嵌入
        if self.pid_embedding is not None and pid_data is not None:
            pid_emb = self.pid_embedding(pid_data)
            x += pid_emb
            
        # 添加位置编码
        x = self.positional_encoding(x)
        
        # 调整维度为[seq_len, batch_size, emb_size]以适应Transformer
        x = x.permute(1, 0, 2)
        
        # 创建因果掩码
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # 根据架构类型选择不同的处理路径
        if self.emb_type == 'encoder_only':
            # Encoder-only架构（使用严格的因果掩码）
            encoded = self.transformer_encoder(x, mask)
            output = encoded
            
        elif self.emb_type == 'decoder_only':
            # Decoder-only架构（自带因果性质）
            output = self.transformer_decoder(x, x, tgt_mask=mask)
            
        elif self.emb_type == 'encoder_decoder':
            # Encoder-Decoder架构
            # 编码器处理时不使用掩码（可以看到整个序列）
            memory = self.transformer_encoder(x)
            # 解码器使用因果掩码
            output = self.transformer_decoder(x, memory, tgt_mask=mask)
        
        # 调整回[batch_size, seq_len, emb_size]
        output = output.permute(1, 0, 2)
        
        # 输出预测概率（只预测当前知识点）
        logits = self.output_layer(output)  # [batch_size, seq_len, num_c]
        probs = torch.sigmoid(logits).gather(2, q_data.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len]
        
        return probs
    
    def generate_square_subsequent_mask(self, sz):
        """生成因果掩码，确保预测只能依赖于之前的时间步"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)