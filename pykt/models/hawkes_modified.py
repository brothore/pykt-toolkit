# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_attention_decay_matrix(scores_matrix, gamma, mask, pdiff=None, device='cpu'):
    """
    计算并返回应用于注意力分数矩阵的衰减因子矩阵 (Total Effect)。

    参数:
        scores_matrix (torch.Tensor): 维度为 (B, L, L) 或 (B, H, L, L) 的注意力分数矩阵 (Q K^T / sqrt(d_k))。
        gamma (torch.Tensor): 维度为 (1) 或 (H) 或 (1, H, 1, 1) 的参数，对应论文中的 theta。
        mask (torch.Tensor): 维度为 (B, 1, L, L) 或 (1, 1, L, L) 或 (B, H, L, L) 的注意力掩码，用于确保计算有效分数。
        pdiff (torch.Tensor, optional): 额外的差异项，维度应与衰减计算兼容。默认为 None。
        device (str or torch.device): Tensor所在的设备。

    返回:
        torch.Tensor: 维度与 scores_matrix 相同的衰减因子矩阵 (Total Effect)。
    """
    if scores_matrix.dim() == 3:
        # 将 (B, L, L) 扩展为 (B, 1, L, L) 以便处理
        scores = scores_matrix.unsqueeze(1)
        bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    elif scores_matrix.dim() == 4:
        scores = scores_matrix
        bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    else:
        raise ValueError("scores_matrix 维度必须是 3 (B, L, L) 或 4 (B, H, L, L)")

    # 1. 序列位置差值的计算
    x1 = torch.arange(seqlen, device=device).expand(seqlen, -1)
    x2 = x1.transpose(0, 1).contiguous()
    position_effect = torch.abs(
        x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen 位置差值

    # 2. 计算用于距离累积的加权分数
    with torch.no_grad():
        # 2.1 应用 mask 并进行 softmax
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)  # BS, H, seqlen, seqlen
        scores_ = scores_.to(device) * mask.float().to(device) # 确保掩码位置为0

        # 2.2 计算累积分布
        # distcum_scores: $\sum_{j=1}^{i} \text{score}_{i, j}$
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, H, sl, sl

        # 2.3 计算总分布 (因为 mask 存在，此处相当于计算每行有效分数的和)
        # disttotal_scores: $\sum_{j=1}^{L} \text{score}_{i, j}$
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)  # bs, H, sl, 1

        # 2.4 计算距离分数 D (公式 $\sqrt{|\text{Total} - \text{CumSum}| \cdot |\text{PosDiff}|}$)
        # dist_scores: $\sqrt{|\sum_{j} \text{score}_{i, j} - \sum_{j=1}^{k} \text{score}_{i, j}| \cdot |i-k|}$
        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.) # score < 0 时，设置为 0
        dist_scores = dist_scores.sqrt().detach() # detach 阻止梯度回传，符合原代码的 no_grad 块的意图

    # 3. 计算 gamma (theta) 参数
    m = nn.Softplus()
    
    # 确保 gamma 的维度为 (1, H, 1, 1)
    if gamma.dim() == 1:
        # 假设输入 gamma 为 (H)
        gamma_processed = -1. * m(gamma).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # 1, H, 1, 1
    elif gamma.dim() == 0:
        # 假设输入 gamma 为单个值
        gamma_processed = -1. * m(gamma).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # 1, 1, 1, 1
    else:
        # 假设输入 gamma 已经是 (1, H, 1, 1) 或类似形状
        gamma_processed = -1. * m(gamma) 
        # 可能需要额外的检查和调整，但为简洁起见，这里假设输入能被正确处理

    # 4. 计算最终衰减因子 (Total Effect)
    if pdiff is None:
        # total_effect: $e^{\gamma \cdot D}$
        total_effect = torch.clamp(torch.clamp(
            (dist_scores.to(device) * gamma_processed.to(device)).exp(), min=1e-5), max=1e5)
    else:
        # 额外的 pdiff 逻辑，与原代码保持一致
        # 假设 pdiff 形状为 (B, L, L)
        if pdiff.dim() == 3:
            diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], head, pdiff.shape[1], pdiff.shape[2])
        else:
            diff = pdiff # 假设 pdiff 形状已经合适 (B, H, L, L)

        diff = diff.sigmoid().exp()
        total_effect = torch.clamp(torch.clamp(
            (dist_scores * gamma_processed * diff).exp(), min=1e-5), max=1e5)

    if scores_matrix.dim() == 3:
        # 如果输入是 (B, L, L)，则返回 (B, L, L)
        return total_effect.squeeze(1)
    return total_effect
def create_causal_mask(seqlen, device):
    """
    创建一个 seqlen x seqlen 的下三角矩阵（因果掩码）。
    
    掩码值为 True 的位置表示允许关注 (Keep)，False 表示需要屏蔽 (Mask)。
    """
    # 创建一个 L x L 的全1张量
    mask = torch.ones(seqlen, seqlen, dtype=torch.bool, device=device)
    
    # 将上三角部分设置为 False
    # tril(diagonal=0) 保留主对角线和下三角
    mask = torch.tril(mask, diagonal=0)
    
    # 扩展维度以匹配 scores_matrix 的维度 (1, 1, L, L)，便于广播
    return mask.unsqueeze(0).unsqueeze(0)
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    将输入张量的特征维度（最后一维）分成两半并进行旋转。
    例如，[x1, x2, x3, x4] -> [-x3, -x4, x1, x2]
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    """
    对输入矩阵应用 Rotary Positional Embedding。

    Args:
        x: 输入张量，形状通常为 (B, S, D) 或 (B, H, S, D)。
           B=batch_size, S=seq_len, D=emb_dim/head_dim。
        cos: 预先计算好的旋转余弦张量。
             形状应能与 x 广播匹配，例如 (S, 1, D) 或 (1, S, 1, D)
             取决于 x 的形状。
        sin: 预先计算好的旋转正弦张量。
             形状与 cos 相同。

    Returns:
        应用 RoPE 后的张量。
    """
    # 1. 旋转 x 的一半分量
    x_rot = rotate_half(x)

    # 2. 应用 RoPE 公式: x_rotated = x * cos + rotate_half(x) * sin
    # cos 和 sin 会自动广播到 x 的 (B, H, ...) 维度
    return (x * cos) + (x_rot * sin)

# --- 辅助函数：RoPE 权重初始化（通常在模型初始化时调用） ---

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    预计算旋转频率的余弦和正弦值。
    
    Args:
        dim: 嵌入维度 D。
        end: 序列最大长度 S。
        theta: 频率基数，通常取 10000.0。
        
    Returns:
        cos 和 sin 张量。
    """
    # 1. 计算 (1/theta^(2i/D))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 2. 计算位置 m: [0, 1, ..., end-1]
    t = torch.arange(end, dtype=torch.float32) 
    
    # 3. 计算 m * (1/theta^(2i/D)) 得到角度 (S, D/2)
    freqs = torch.outer(t, freqs) # (S, D/2)
    
    # 4. 将频率扩展到 D 维度以匹配 x 的形状（在 RoPE 中每对特征 (x1, x2) 对应一个频率）
    # 例如：[f1, f2, f3] -> [f1, f1, f2, f2, f3, f3]
    freqs_full = torch.cat([freqs, freqs], dim=-1) # (S, D)
    
    # 5. 计算 Cos 和 Sin
    cos = torch.cos(freqs_full) # (S, D)
    sin = torch.sin(freqs_full) # (S, D)
    
    # 将形状调整为适合广播到 (B, H, S, D) 或 (B, S, D) 的形式
    # 这里使用 (S, D) 的形状，应用时依赖于 PyTorch 广播机制
    return cos, sin
class MultiHeadAttention(nn.Module):
    """
    一个简化的多头注意力模块实现。
    
    假设输入 Q, K, V 的维度是 [Batch Size, Sequence Length, Embed Dim] (即 batch_first=True)。
    """
    def __init__(self, embed_dim, num_heads,log_decay=1):
        """
        初始化多头注意力模块。

        Args:
            embed_dim (int): 输入和输出的嵌入维度 (E)。
            num_heads (int): 头的数量 (H)。
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # 确保嵌入维度能被头数整除
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.log_decay = log_decay # 是否启用衰减
        # 每个头分配到的维度 (d_k)
        self.head_dim = embed_dim // num_heads
        # 缩放因子：用于缩放点积，防止点积结果过大
        self.scale = math.sqrt(self.head_dim)
        
        # 1. 线性变换层 (W_Q, W_K, W_V)
        # 将输入 (Q, K, V) 映射到 embed_dim 维度，然后拆分成 H 个头
        # 通常 Q/K/V 是独立的，但我们为了简化，可以把它们放在一起（如果 Q=K=V）
        # 这里使用独立的 Q, K, V 投影，以支持交叉注意力
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # 2. 输出线性变换层 (W_O)
        # 将 H 个头的输出（拼接后）映射回 embed_dim
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        if self.log_decay:
            # gamma 是一个可学习参数，每个头一个，用于控制衰减率
            # 形状: [num_heads]
            self.gamma = nn.Parameter(torch.Tensor(num_heads))
            # 默认初始化
            nn.init.uniform_(self.gamma, a=-0.1, b=0.1)
            self.softplus = nn.Softplus()
        # 参数初始化（可选，但推荐）
        self._reset_parameters()

    def _reset_parameters(self):
        # 推荐使用 Xavier/Glorot 初始化方法
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, q, k, v, mask=None):
        """
        前向传播函数。

        Args:
            q (Tensor): 查询 (Query) 向量，形状 [B, T, E]
            k (Tensor): 键 (Key) 向量，形状 [B, S, E]
            v (Tensor): 值 (Value) 向量，形状 [B, S, E]
            mask (Tensor, optional): 注意力掩码，形状 [B, T, S] 或 [T, S]。
                                     用于屏蔽（设置为 $-\infty$）未来或填充部分。

        Returns:
            Tensor: 注意力输出，形状 [B, T, E]
            Tensor: 注意力权重（可选，本实现中不返回）
        """
        B, T, E = q.size() # B: Batch, T: Target Seq Len, E: Embed Dim
        _, S, _ = k.size() # S: Source Seq Len
        H = self.num_heads
        
        # 1. 线性投影并准备多头
        # 目标形状: [B, T, E] -> [B, T, H, d_k] -> [B*H, T, d_k] 或 [H*B, T, d_k]

        # a. 投影: [B, SeqLen, E] -> [B, SeqLen, E]
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)
        
        # b. 拆分多头: [B, SeqLen, E] -> [B, SeqLen, H, d_k]
        # (reshape)
        q_proj = q_proj.view(B, T, H, self.head_dim)
        k_proj = k_proj.view(B, S, H, self.head_dim)
        v_proj = v_proj.view(B, S, H, self.head_dim)

        # c. 调整维度: [B, SeqLen, H, d_k] -> [B, H, SeqLen, d_k] 
        # (将 H 维度提前，便于矩阵乘法)
        # .transpose(1, 2): 交换第 1 (SeqLen) 和第 2 (H) 维度
        q_heads = q_proj.transpose(1, 2) # [B, H, T, d_k]
        k_heads = k_proj.transpose(1, 2) # [B, H, S, d_k]
        v_heads = v_proj.transpose(1, 2) # [B, H, S, d_k]

        # 2. 缩放点积注意力 (Scaled Dot-Product Attention)
        
        # a. 计算 Q 和 K 的点积: Attention Score
        # 矩阵乘法: [B, H, T, d_k] @ [B, H, d_k, S] -> [B, H, T, S]
        # k_heads.transpose(-2, -1): 交换 K 的最后两个维度 (S, d_k -> d_k, S)
        attn_scores = torch.matmul(q_heads, k_heads.transpose(-2, -1))
        # attn_scores: [B, H, T, S]
        
        # b. 缩放
        attn_scores = attn_scores / self.scale
        # ------------------ 修改点 D: 插入衰减逻辑 ------------------
        if self.log_decay:
            # D.1. 计算位置差 |j - i|
            # 这里的 T 和 S 对应您的 seqlen (假设 T=S)
            positions_j = torch.arange(q.shape[1], device=device).unsqueeze(1) # [T, 1]
            positions_i = torch.arange(k.shape[1], device=device).unsqueeze(0) # [1, S]
            # [T, 1] - [1, S] -> [T, S] (位置差矩阵)
            position_effect = torch.abs(positions_j - positions_i).float() 
            # 广播到 [B, H, T, S]
            position_effect = position_effect.unsqueeze(0).unsqueeze(0).expand(B, H, T, S) 
            
            # D.2. 计算带Softmax的累计分数 (必须使用 no_grad 的原因是原作者希望dist_scores不参与梯度计算)
            with torch.no_grad():
                # 复制 attn_scores 并应用掩码，用于 Softmax
                scores_temp = attn_scores.clone()
                if mask is not None:
                     # 确保 mask 维度兼容 [B, 1, T, S]
                    if mask.ndim == 2:
                        mask_expanded = mask.unsqueeze(0).unsqueeze(1).expand(B, H, T, S)
                    elif mask.ndim == 3:
                        mask_expanded = mask.unsqueeze(1).expand(B, H, T, S)
                    else:
                        mask_expanded = mask.expand(B, H, T, S)

                    scores_temp.masked_fill_(mask_expanded == 0, float('-inf'))
                    
                scores_ = F.softmax(scores_temp, dim=-1) # [B, H, T, S]
                
                # 累计求和 (沿着 Key/Source 维度 S)
                distcum_scores = torch.cumsum(scores_, dim=-1) # [B, H, T, S]
                # 总和 (用于归一化，如果序列未填充，则接近 1)
                disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True) # [B, H, T, 1]
                
                # 计算 dist_scores: sqrt(|Total - Cumulative| * |j-i|)
                dist_scores = (disttotal_scores - distcum_scores).abs() * position_effect
                dist_scores = torch.clamp(dist_scores, min=1e-6).sqrt() # 避免 sqrt(0)
                
            # D.3. 计算 total_effect
            # gamma: [H] -> 广播到 [1, H, 1, 1]
            gamma_param = -1. * self.softplus(self.gamma).view(1, H, 1, 1) 
            
            # total_effect = exp(gamma * dist_scores)
            total_effect = torch.clamp((dist_scores * gamma_param).exp(), min=1e-5, max=1e5)
            
            # D.4. 应用衰减
            attn_scores = attn_scores * total_effect # [B, H, T, S]

        # c. 应用掩码 (如果提供了)
        if mask is not None:
            # 广播 mask 到所有批次和所有头
            # mask: [B, T, S] 或 [T, S] -> 扩展到 [B, 1, T, S]
            if mask.ndim == 2:
                 mask = mask.unsqueeze(0).unsqueeze(1) # [1, 1, T, S]
            elif mask.ndim == 3:
                 mask = mask.unsqueeze(1) # [B, 1, T, S]
                 
            # 掩码中需要屏蔽的位置通常用 True 表示，我们将其设置为一个极小值
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            # 注意：PyTorch 的官方实现通常期望 mask 中 True/False 代表是否保留，
            # 这里为了简化，假设输入 mask 中 True/False 与 F.masked_fill 的期望相反
            # 如果是标准的因果掩码，应该用 float('-inf') 来屏蔽未来信息。
            
        # d. Softmax 获得注意力权重
        # 沿 Source 维度 (S) 进行 softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        # attn_weights: [B, H, T, S]
        
        # e. 计算加权和
        # 矩阵乘法: [B, H, T, S] @ [B, H, S, d_k] -> [B, H, T, d_k]
        attn_output = torch.matmul(attn_weights, v_heads)
        # attn_output: [B, H, T, d_k]

        # 3. 拼接和最终线性投影
        
        # a. 拼接多头: [B, H, T, d_k] -> [B, T, H, d_k]
        # (恢复 SeqLen 和 H 的顺序)
        attn_output = attn_output.transpose(1, 2).contiguous() 
        # attn_output: [B, T, H, d_k]

        # b. 压平: [B, T, H, d_k] -> [B, T, E] (因为 H * d_k = E)
        attn_output = attn_output.view(B, T, E)
        
        # c. 最终投影 W_O
        output = self.out_proj(attn_output)
        # output: [B, T, E]
        
        return output

class HawkesKT(nn.Module):
    # def __init__(self, args, corpus):
    def __init__(self, n_skills, n_problems, emb_size, time_log, emb_type="qid",num_heads=1):
        super().__init__()
        self.model_name = "hawkes"
        self.emb_type = emb_type
        self.problem_num = n_problems
        self.skill_num = n_skills
        self.emb_size = emb_size
        self.time_log = time_log
        self.gpu = device
        if 'attn_alpha' in self.emb_type:
            self.alpha_q_proj = nn.Linear(self.emb_size, self.emb_size)
            self.alpha_k_proj = nn.Linear(self.emb_size, self.emb_size)
        if 'attn_beta' in self.emb_type:
            self.beta_q_proj = nn.Linear(self.emb_size, self.emb_size)
            self.beta_k_proj = nn.Linear(self.emb_size, self.emb_size)
        self.problem_base = torch.nn.Embedding(self.problem_num, 1)
        self.skill_base = torch.nn.Embedding(self.skill_num, 1)
        log_decay = 1 if "log_decay" in self.emb_type else 0
        self.mha = MultiHeadAttention(self.emb_size, num_heads,log_decay)

        self.mha_a = MultiHeadAttention(self.emb_size, num_heads,log_decay)
        self.mha_b = MultiHeadAttention(self.emb_size, num_heads,log_decay)

        self.linear_mha = nn.Linear(self.emb_size, 1)
        self.alpha_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
        # print(f"weight: {self.alpha_inter_embeddings.weight}")
        # np.save('alpha_inter_embeddings.npz', self.alpha_inter_embeddings.weight.detach().numpy())
        self.alpha_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)
        self.beta_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
        self.beta_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)
        if 'attn_alpha' in self.emb_type and 'only_linear' in self.emb_type:
            self.alpha_src_bn = nn.BatchNorm1d(self.emb_size)
            self.alpha_target_bn = nn.BatchNorm1d(self.emb_size)
        # self.loss_function = torch.nn.BCELoss()
        # self.init_weights()
        # print(self)
        # self.count = 0
        # self.printparams()
        cos_weights, sin_weights = precompute_freqs_cis(
            dim=self.emb_size, 
            end=200 # 或者您预期的最大长度
        )
        self.register_buffer("cos_weights", cos_weights.unsqueeze(0).unsqueeze(0)) 
        self.register_buffer("sin_weights", sin_weights.unsqueeze(0).unsqueeze(0))
        self.gammas = nn.Parameter(torch.zeros(1, 1, 1))
    
    def apply_alibi(self,seq_len, device):
        """
        Apply ALiBi (Attention with Linear Biases) to the query and key matrices.
        Args:
            q: Query matrix of shape [batch_size, seq_len, emb_size]
            k: Key matrix of shape [batch_size, seq_len, emb_size]
            seq_len: Sequence length
            device: Device ('cuda' or 'cpu')

        Returns:
            Updated attention logits after applying ALiBi.
        """
        # Create a linear bias matrix for ALiBi
        bias = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(0) - torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        bias = torch.clamp(bias, min=0)  # Ensure non-negative bias

        # Apply the bias to the attention logits
        return bias.unsqueeze(0)
    def absolute_position_encoding(self,seq_len, emb_size, device):
        """
        Absolute position encoding using sin/cos functions.
        """
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)  # [seq_len, 1]
        div_term = torch.exp(torch.arange(0, emb_size, 2, device=device).float() * -(math.log(10000.0) / emb_size))  # [emb_size/2]
        pos_enc = torch.zeros(seq_len, emb_size, device=device)
        pos_enc[:, 0::2] = torch.sin(position * div_term)  # [seq_len, emb_size/2]
        pos_enc[:, 1::2] = torch.cos(position * div_term)  # [seq_len, emb_size/2]

        return pos_enc.unsqueeze(0)  # Add batch dimension [1, seq_len, emb_size]
    @staticmethod
    def init_weights(m):
        if type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def printparams(self):
        print("="*20)
        for m in list(self.named_parameters()):
            print(m[0], m[1])
        self.count += 1
        print(f"count: {self.count}")

    def forward(self, skills, problems, times, labels, qtest=False):
        if "qid" in self.emb_type:
            seq_len = skills.shape[1]
            current_cos = self.cos_weights[..., :seq_len, :] 
            current_sin = self.sin_weights[..., :seq_len, :]
            # self.printparams()
            # assert False
            
            # skills = torch.tensor([[1246, 1257, 1251, 1255, 1254]]).long().to(device)
            # problems = torch.tensor([[2493, 2514, 2502, 2510, 2508]]).long().to(device)
            # times = torch.tensor([[1415887648, 1415887655, 1415887663, 1415887667, 1415887671]]).long().to(device)
            # labels = torch.tensor([[1., 0., 1., 1., 0.]]).long().to(device)
            # sm = torch.tensor([[1,1,1,1,1]]).lo ng().to(device)

            # print("skills: ", skills)
            # print("problems: ", problems)
            # print("times: ", times)
            # assert False
            mask_labels = labels# * (sm == 1).long()#labels * (labels > -1).long()
            # print(f"labels: {labels}")
            # print(f"mask_labels: {mask_labels}")
            # print(f"sm: {sm==1}")
            # # assert labels == mask_labels
            inters = skills + mask_labels * self.skill_num
            # print(f"inters: {inters}")

            alpha_src_emb = self.alpha_inter_embeddings(inters)  # [bs, seq_len, emb]
            # print(f"alpha_src_emb:{alpha_src_emb}")
            alpha_target_emb = self.alpha_skill_embeddings(skills)
            # print(f"alpha_target_emb:{alpha_target_emb}")


            if 'attn_alpha' in self.emb_type and 'only_linear' in self.emb_type:
                # 原始线性投影逻辑
                Q = self.alpha_q_proj(alpha_target_emb)
                K = self.alpha_k_proj(alpha_src_emb)
                
                # 注意：原始代码中的BN应用似乎有问题，它将 Q 和 K 都赋给了 alpha_target_emb
                # 并且使用 target_bn 处理 K，src_bn 处理 Q。这里我保留了原始逻辑，但请检查是否正确。
                alpha_target_emb_processed = self.alpha_src_bn(Q.view(-1, self.emb_size)).view(skills.size(0), skills.size(1), self.emb_size)
                alpha_src_emb_processed = self.alpha_target_bn(K.view(-1, self.emb_size)).view(skills.size(0), skills.size(1), self.emb_size)
                
                alphas = torch.matmul(alpha_target_emb_processed, alpha_src_emb_processed.transpose(-2, -1)) # QK^T
                
            elif 'attn_alpha' in self.emb_type and 'use_rope' in self.emb_type:
                # 使用旋转位置编码 (RoPE)

                Q = alpha_target_emb  # Q for target_emb
                K = alpha_src_emb     # K for src_emb
                # print(f"Q: {Q.shape}, K shape: {K.shape}")

                # 应用 RoPE
                Q_rope, K_rope = apply_rope(Q, current_cos, current_sin), apply_rope(K, current_cos, current_sin)

                # 🚀 变化在此：打印 RoPE 后的 Q 和 K 的形状
                # print(f"Q_rope shape: {Q_rope.shape}, K_rope shape: {K_rope.shape}")

                alphas = torch.matmul(Q_rope, K_rope.transpose(-2, -1))
                # print(f"alphas shape: {alphas.shape}")
                alphas = alphas.squeeze(0)

            elif 'attn_alpha' in self.emb_type and 'use_abs_pos' in self.emb_type:
                # 使用绝对位置编码
                # 1. 生成位置编码
                pos_emb = self.absolute_position_encoding(seq_len, self.emb_size, device) 
                
                # 2. 将位置编码加到 Q 和 K 的输入上 (假设是加法集成)
                Q_in = alpha_target_emb + pos_emb
                K_in = alpha_src_emb + pos_emb
                
                # 3. 线性投影
                Q = Q_in
                K = K_in
                
                alphas = torch.matmul(Q, K.transpose(-2, -1))
            elif 'attn_alpha' in self.emb_type and 'use_alibi' in self.emb_type:
                # 使用绝对位置编码
                # 1. 生成位置编码
                pos_emb = self.absolute_position_encoding(seq_len, self.emb_size, device) 
                
                # 2. 将位置编码加到 Q 和 K 的输入上 (假设是加法集成)
                Q_in = alpha_target_emb + pos_emb
                K_in = alpha_src_emb + pos_emb
                
                # 3. 线性投影
                Q = Q_in
                K = K_in
                
                alphas = torch.matmul(Q, K.transpose(-2, -1)) 
                alphas = self.apply_alibi(seq_len,device)+alphas   
            elif 'attn_alpha' in self.emb_type and 'use_decay' in self.emb_type:
                # 使用绝对位置编码
                # 1. 生成位置编码
                # 2. 将位置编码加到 Q 和 K 的输入上 (假设是加法集成)
                Q_in = alpha_target_emb
                K_in = alpha_src_emb
                
                # 3. 线性投影
                Q = Q_in
                K = K_in
                
                alphas = torch.matmul(Q, K.transpose(-2, -1)) 
                total_effect = get_attention_decay_matrix(alphas,gamma=self.gammas.to(device),mask=create_causal_mask(seq_len, device=device))
                alphas = alphas.to(device) * total_effect.to(device)
            else:
                Q = alpha_target_emb  # Q for target_emb
                K = alpha_src_emb     # K for src_emb
                print(f"Q: {Q.shape}, K shape: {K.shape}")
                # 原始点积
                alphas = torch.matmul(Q, K.transpose(-2, -1)) # K^T Q
                print(f"alphas shape: {alphas.shape}")



            # print(f"alphas:{alphas}")
            beta_src_emb = self.beta_inter_embeddings(inters)  # [bs, seq_len, emb]
            # print(f"beta_src_emb:{beta_src_emb}")
            beta_target_emb = self.beta_skill_embeddings(skills)
            # print(f"beta_target_emb:{beta_target_emb}")


            if 'attn_beta' in self.emb_type and 'only_linear' in self.emb_type:
                # 原始线性投影逻辑
                Q = self.beta_q_proj(beta_src_emb)
                K = self.beta_k_proj(beta_target_emb)
                betas = torch.matmul(Q, K.transpose(-2, -1))
                
            elif 'attn_beta' in self.emb_type and 'use_rope' in self.emb_type:
                # 使用旋转位置编码 (RoPE)
                Q = beta_src_emb  # Q for src_emb
                K = beta_target_emb # K for target_emb
                
                # 应用 RoPE
                # 应用 RoPE
                Q_rope, K_rope = apply_rope(Q, current_cos, current_sin),apply_rope(K, current_cos, current_sin)
                
                alphas = torch.matmul(Q_rope, K_rope.transpose(-2, -1))
            elif 'attn_beta' in self.emb_type and 'use_abs_pos' in self.emb_type:
                # 使用绝对位置编码
                # 1. 生成位置编码
                pos_emb = self.absolute_position_encoding(seq_len, self.emb_size, device)
                
                # 2. 将位置编码加到 Q 和 K 的输入上
                Q_in = beta_src_emb + pos_emb
                K_in = beta_target_emb + pos_emb
                
                # 3. 线性投影
                Q = Q_in
                K = K_in
                
                betas = torch.matmul(Q, K.transpose(-2, -1))
            elif 'attn_beta' in self.emb_type and 'use_decay' in self.emb_type:
                # 使用绝对位置编码
                # 1. 生成位置编码
                # 2. 将位置编码加到 Q 和 K 的输入上 (假设是加法集成)
                Q_in = beta_target_emb
                K_in = beta_src_emb
                
                # 3. 线性投影
                Q = Q_in
                K = K_in
                
                beta = torch.matmul(Q, K.transpose(-2, -1)) 
                total_effect = get_attention_decay_matrix(alphas,gamma=self.gammas.to(device),mask=create_causal_mask(seq_len, device=device))
                betas = beta.to(device) * total_effect.to(device) 
            else:
                # 原始点积
                betas = torch.matmul(beta_src_emb, beta_target_emb.transpose(-2, -1)) # [bs, seq_len, seq_len]


            # print(f"betas:{betas}")
            betas = torch.clamp(betas + 1, min=0, max=10)
            # source_idx = inters.unsqueeze(-1).repeat(1, 1, labels.shape[1]).long()
            # target_idx = skills.unsqueeze(1).repeat(1, labels.shape[1], 1).long()
            # alphas = self.alpha[source_idx, target_idx]
            # betas = self.beta[source_idx, target_idx]
            if times.shape[1] > 0:
                times = times.double() / 1000
                delta_t = (times[:, :, None] - times[:, None, :]).abs().double()
                # print(times.shape, delta_t)
                # assert False
            elif "sin_pos" in self.emb_type:
                # 改造点：没有时间数据时，使用基于正弦/余弦的序列位置编码
                seq_len = skills.shape[1]
                batch_size = skills.shape[0]
                
                # 1. 生成序列位置差 (j - i)
                positions = torch.arange(1, seq_len + 1, dtype=torch.double).to(device)
                fake_times = positions.unsqueeze(0).repeat(batch_size, 1)
                delta_p = (fake_times[:, :, None] - fake_times[:, None, :]).abs().double()
                
                # 2. 应用正弦变换作为“时间”的替代品
                # 注意：这里的目标是生成一个标量 delta_t，
                # 它应该随着 delta_p 增大而（大致）单调增大。
                
                # 方案 A: 仅使用 sin/cos 函数，这会引入周期性，可能导致距离远的事件影响反而变大。
                # delta_t = torch.sin(delta_p * (math.pi / 10.0)) + delta_p

                # 方案 B: 在线性/对数距离上添加正弦扰动，引入周期性但保持单调性。
                # 这里的 1e-10 是为了防止 log(0)，并作为基础距离。
                # delta_t = torch.log(delta_p + 1e-10) + torch.sin(delta_p * math.pi / 10.0) * 0.5
                
                # 方案 C: 线性距离 + 正弦扰动 (最简单的引入正弦的方案)
                # 我们必须确保 delta_t > 0 (除了 i=j)。当 delta_p=0 时，delta_t 应当接近 0。
                # 使用一个平滑的、随距离增加而振荡的函数。
                
                # 基础距离 + 振荡项
                base_distance = delta_p
                # 振荡项：随距离增大，振幅可以保持不变或减小
                oscillation = torch.sin(base_distance * math.pi / 5.0) * 0.1 # 周期为10个事件
                
                # 最终 delta_t：确保 i=j 时为 0，且整体保持单调递增
                delta_t = base_distance + oscillation
                
                # 强制处理 i=j 的情况：
                # 由于 delta_p[i, i] = 0，delta_t[i, i] 也会是 0。
                # 如果 delta_t < 1e-10，则替换为 1e-10（防止 log(0)）。
                delta_t = torch.clamp(delta_t, min=1e-10)

                # print("--- Using Sinusoidal-adjusted Sequence Position as Time ---")
            else:
                # 改造点：没有时间数据时，使用序列位置编码
                seq_len = skills.shape[1]
                batch_size = skills.shape[0]
                
                # 生成位置序列：0, 1, 2, ..., seq_len-1
                # 注意：从0开始或从1开始取决于你对间隔的定义，这里我们使用1, 2, ...
                # 为了简化计算 (j-i)，我们可以先生成 1, 2, ..., L
                positions = torch.arange(1, seq_len + 1, dtype=torch.double).to(device)
                # 扩展到批次维度 [bs, seq_len]
                fake_times = positions.unsqueeze(0).repeat(batch_size, 1)
                
                # 计算位置差（序列距离）
                # delta_t[i, j] = j - i (如果 positions 从 1 开始)
                # [bs, seq_len, 1] - [bs, 1, seq_len] => [bs, seq_len, seq_len]
                delta_t = (fake_times[:, :, None] - fake_times[:, None, :]).abs().double()
                
                # 由于 delta_t 中可能包含 0，需要处理。
                # 原始代码中 delta_t = 1 对应的是 log(1) = 0 的效果。
                # 在这里，当 i=j 时 delta_t=0，表示自激，这也是可以的。
                # 我们继续使用这个位置差 delta_t
                
                # print("--- Using Sequence Position as Time ---") # 调试信息
            delta_t = torch.log(delta_t + 1e-10) / np.log(self.time_log)

            # print(f"alphas: {alphas.shape}, betas: {betas.shape}, delta_t: {delta_t.shape}")
            if 'exp_decay' in self.emb_type:
                # 使用指数衰减
                cross_effects = alphas * torch.exp(-betas * delta_t)
            elif 'linear_decay' in self.emb_type:
                # 使用线性衰减
                cross_effects = alphas * (1 / (1 + betas * delta_t))
            elif 'quadratic_decay' in self.emb_type:
                # 使用平方衰减
                cross_effects = alphas * (1 / (1 + (betas * delta_t) ** 2))
            else:
                # 默认使用指数衰减
                cross_effects = alphas * torch.exp(-betas * delta_t)
            # cross_effects = alphas * torch.exp(-self.beta * delta_t)
            # cross_effects = alphas

            seq_len = skills.shape[1]
            valid_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1)
            mask = (torch.from_numpy(valid_mask) == 0)
            mask = mask.cuda() if self.gpu != '' else mask
            sum_t = cross_effects.masked_fill(mask, 0).sum(-2)

            problem_bias = self.problem_base(problems).squeeze(dim=-1)
            skill_bias = self.skill_base(skills).squeeze(dim=-1)
            # print(f"problem_bias: {problem_bias}, skill_bias: {skill_bias}, sum_t: {sum_t}")
            prediction = (problem_bias + skill_bias + sum_t).sigmoid()
            # print(f"prediction:{prediction}")

            # Return predictions and labels from the second position in the sequence
            # out_dict = {'prediction': prediction[:, 1:], 'label': labels[:, 1:].double()}
            # loss = self.loss_function(out_dict["prediction"], out_dict["label"])
            # print(f"out_dict: {out_dict}")
            # print(f"loss: {loss}")
            # assert False
            h = problem_bias + skill_bias + sum_t
            if not qtest:
                return prediction
            else:
                return prediction, h
        elif "mha" in self.emb_type:
            mask_labels = labels
            seq_len = skills.shape[1]
            inters = skills + mask_labels * self.skill_num
            # print(f"inters: {inters}")

            alpha_src_emb = self.alpha_inter_embeddings(inters)  # [bs, seq_len, emb]
            # print(f"alpha_src_emb:{alpha_src_emb}")
            alpha_target_emb = self.alpha_skill_embeddings(skills)
            # print(f"alpha_target_emb:{alpha_target_emb}")
            alphas = torch.matmul(alpha_src_emb, alpha_target_emb.transpose(-2, -1))  # [bs, seq_len, seq_len]
            # print(f"alphas:{alphas}")
            beta_src_emb = self.beta_inter_embeddings(inters)  # [bs, seq_len, emb]
            # print(f"beta_src_emb:{beta_src_emb}")
            beta_target_emb = self.beta_skill_embeddings(skills)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            # causal_mask 中 True 的位置应该被屏蔽 (设置为 -inf)
            # 注意：在我的简化实现中，`masked_fill(mask == 0, float('-inf'))` 的逻辑与此相反
            # 实际应用中，mask应该表示要屏蔽的位置，这里我们调整一下 mask 的定义
            causal_mask = ~causal_mask # 让上三角 (未来信息) 为 True (需要屏蔽)


            causal_mask_for_fill = causal_mask.to(device)
            



            if "abv1" in self.emb_type:
                attn_out = self.mha( q=alpha_src_emb,k=alpha_target_emb,v=beta_target_emb,mask=causal_mask_for_fill)    
            elif "abv2" in self.emb_type:
                attn_out = self.mha( q=alpha_src_emb,k=alpha_target_emb,v=beta_src_emb,mask=causal_mask_for_fill)    
            elif "abv3" in self.emb_type:
                attn_out = self.mha( q=alpha_src_emb,k=beta_target_emb,v=beta_src_emb,mask=causal_mask_for_fill)    
            elif "abv4" in self.emb_type:
                attn_out = self.mha( q=alpha_src_emb,k=beta_target_emb,v=beta_src_emb,mask=causal_mask_for_fill)    
            elif "abv5" in self.emb_type:
                attn_out = self.mha( q=alpha_src_emb,k=alpha_target_emb,v=alpha_target_emb,mask=causal_mask_for_fill)    
            elif "abv6" in self.emb_type:
                attn_out = self.mha( q=alpha_src_emb,k=beta_target_emb,v=beta_target_emb,mask=causal_mask_for_fill)    
            elif "abv7" in self.emb_type:
                attn_out = self.mha( q=alpha_target_emb,k=beta_target_emb,v=beta_src_emb,mask=causal_mask_for_fill)    
            
            
            elif "axb" in self.emb_type:
                attn_out_a = self.mha( q=alpha_src_emb,k=alpha_target_emb,v=alpha_target_emb,mask=causal_mask_for_fill)    
                attn_out_b = self.mha( q=beta_src_emb,k=beta_target_emb,v=beta_target_emb,mask=causal_mask_for_fill)    
                attn_out = torch.matmul(attn_out_a, attn_out_b.transpose(-2, -1))

            elif "only_a" in self.emb_type:
                pass
            elif "only_b" in self.emb_type:
                pass
            
            if "sum_t" in self.emb_type:
                sum_t = attn_out.sum(dim=-1)
                problem_bias = self.problem_base(problems).squeeze(dim=-1)
                skill_bias = self.skill_base(skills).squeeze(dim=-1)
                # print(f"problem_bias: {problem_bias}, skill_bias: {skill_bias}, sum_t: {sum_t}")
                prediction = (problem_bias + skill_bias + sum_t).sigmoid()
                # print(f"prediction:{prediction}")

                # Return predictions and labels from the second position in the sequence
                # out_dict = {'prediction': prediction[:, 1:], 'label': labels[:, 1:].double()}
                # loss = self.loss_function(out_dict["prediction"], out_dict["label"])
                # print(f"out_dict: {out_dict}")
                # print(f"loss: {loss}")
                # assert False
                h = problem_bias + skill_bias + sum_t
            elif "sum_linear" in self.emb_type:
                problem_bias = self.problem_base(problems).squeeze(dim=-1)
                skill_bias = self.skill_base(skills).squeeze(dim=-1)
                sum_t = self.linear_mha(attn_out).squeeze(-1)  #BLD -> BL

                m = nn.Sigmoid()
                skill_bias = self.skill_base(skills).squeeze(dim=-1)
                # print(f"problem_bias: {problem_bias}, skill_bias: {skill_bias}, sum_t: {sum_t}")
                prediction = (problem_bias + skill_bias + sum_t).sigmoid()
                # preds = m(problem_bias + skill_bias + sum_t)
                h = problem_bias + skill_bias + sum_t

            
            if not qtest:
                return prediction
            else:
                return prediction, h