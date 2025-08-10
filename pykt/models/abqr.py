import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import os
import json
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
import copy
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, MultiheadAttention
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy

# This is a mock global variable module to represent the original ABQR's usage.
# In a real implementation, you would need to define and populate 'glo'.
class MockGlo:
    def __init__(self):
        self.data = {}
    def set_value(self, key, value):
        self.data[key] = value
    def get_value(self, key):
        return self.data.get(key)
glo = MockGlo()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

# === 从原始 ABQR 模型中引入的核心模块和函数 ===
class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim, p):
        super(GCNConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w = nn.Parameter(torch.rand((in_dim, out_dim)))
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.rand((out_dim)))
        nn.init.zeros_(self.b)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x, adj):
        x = self.dropout(x)
        x = torch.matmul(x, self.w)
        # Assuming adj is a sparse self.matrix, which is handled in the original ABQR code.
        x = torch.sparse.mm(adj.float(), x)
        x = x + self.b
        return x

class MLP_Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, x):
        return self.net(x)

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

class BGRL(nn.Module):
    def __init__(self, d, p, drop_feat1, drop_feat2, drop_edge1, drop_edge2):
        super(BGRL, self).__init__()
        self.drop_feat1, self.drop_feat2, self.drop_edge1, self.drop_edge2 = drop_feat1, drop_feat2, drop_edge1, drop_edge2
        self.online_encoder = GCNConv(d, d, p)
        self.predictor = MLP_Predictor(d, d, d)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def update_target_network(self, mm):
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, x, adj, perb=None):
        if perb is None:
            # Original ABQR code has a branch for perb=None, but the logic
            # is simplified to just return the GNN output and 0 loss.
            # We will assume a training mode with perb for the contrastive loss part.
            return self.online_encoder(x, adj), 0
        print(f"[DEBUG] x.shape: {x.shape} (type: {type(x.shape)})")
        print(f"[DEBUG] perb.shape: {perb.shape} (type: {type(perb.shape)})")
        # Create two views: original (x1) and perturbed (x2)
        x1, adj1 = x, copy.deepcopy(adj)

        x2, adj2 = x + perb, copy.deepcopy(adj)
        
        online_x = self.online_encoder(x1, adj1)
        online_y = self.online_encoder(x2, adj2)

        with torch.no_grad():
            target_y = self.target_encoder(x1, adj1).detach()
            target_x = self.target_encoder(x2, adj2).detach()

        online_x = self.predictor(online_x)
        online_y = self.predictor(online_y)

        loss = (loss_fn(online_x, target_x) + loss_fn(online_y, target_y)).mean()

        return self.online_encoder(x, adj), loss

# === 修改后的 ABQRv2 类 ===
class ABQR(nn.Module):
    def __init__(self, n_question, n_pid,
             d_model, dropout,  emb_type="qid", 
             step_size=None, step_m=None, grad_clip=None, mm=None,emb_path=""):
        super().__init__()
        self.model_name = "abqr"
        self.n_question = n_question
        self.num_c = n_question

        self.dropout = dropout


        self.n_pid = n_pid

        self.model_type = self.model_name
        self.emb_size = d_model
        self.emb_type = emb_type
        embed_l = d_model
        self.step_size = step_size
        self.step_m = step_m
        self.grad_clip = grad_clip


        # 加载图邻接矩阵
        self.matrix = None
        dataset_name = None
        if self.emb_type.find('as09') != -1:
            pre_load_gcn = "../data/assist2009/ques_skill_gcn_adj.pt"
            self.matrix = torch.load(pre_load_gcn).to(device)
            if not self.matrix.is_sparse:
                self.matrix = self.matrix.to_sparse()
            dataset_name = 'assist2009'
        # ... (此处省略其他数据集的加载逻辑，与原始代码相同)
        elif self.emb_type.find('ni34')!=-1:
            dataset_name = 'nips_task34'
            with open("../configs/data_config.json") as fin:
                data_config = json.load(fin)
            pre_load_gcn = os.path.join(data_config[dataset_name]["dpath"],"ques_skill_gcn_adj.pt")
            
            
            
            self.matrix = torch.load(pre_load_gcn).to(device)
            if not self.matrix.is_sparse:
                self.matrix = self.matrix.to_sparse()
        elif self.emb_type.find('al05')!=-1:
            pre_load_gcn = "../data/algebra2005/ques_skill_gcn_adj.pt"
            self.matrix = torch.load(pre_load_gcn).to(device)
            if not self.matrix.is_sparse:
                self.matrix = self.matrix.to_sparse()
            dataset_name = 'algebra2005'
        elif self.emb_type.find('bd06')!=-1:
            pre_load_gcn = "../data/bridge2algebra2006/ques_skill_gcn_adj.pt"
            self.matrix = torch.load(pre_load_gcn).to(device)
            if not self.matrix.is_sparse:
                self.matrix = self.matrix.to_sparse()
            dataset_name = 'bridge2algebra2006'
        elif self.emb_type.find('py')!=-1:
            pre_load_gcn = "../data/peiyou/ques_skill_gcn_adj.pt"
            self.matrix = torch.load(pre_load_gcn).to(device)
            if not self.matrix.is_sparse:
                self.matrix = self.matrix.to_sparse()
            dataset_name = 'peiyou'
            
        self.matrix = self.matrix.to(device)
        glo.set_value('matrix', self.matrix)
        self.matrix_shape = self.matrix.shape[0]
        # === 核心修改: 替换SFM_CL，并构建原始ABQR的模块 ===
        pro_max = self.n_pid if self.n_pid > 0 else self.n_question
        d = d_model
        p = self.dropout
        
        self.gcl = BGRL(d=d, p=p, drop_feat1=0.2, drop_feat2=0.2, drop_edge1=0.2, drop_edge2=0.2)
        
        self.pro_embed = nn.Parameter(torch.ones((pro_max, d)))
        nn.init.xavier_uniform_(self.pro_embed)
        self.ans_embed = nn.Embedding(2, d)
        
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.dropout_layer = nn.Dropout(p=p)
        
        self.origin_out = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Dropout(p=p),
            # 将输出维度从 1 改为 n_question
            nn.Linear(d, self.n_question) 
        )
                


        
        self.reset()
        self.mm = mm
        if self.mm:
            self.gcl.update_target_network(self.mm)

    def reset(self):
        for p in self.parameters():
            if p.ndimension() > 0 and p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def update_target_network(self):
        if self.mm:
            self.gcl.update_target_network(self.mm)
            
    def forward(self, dcur, qtest=False, train=False, perb=None):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        
        # 使用原始 ABQR 的变量命名
        last_pro = q
        last_ans = r
        next_pro = qshft
        
        # === 核心修改: 按照原始 ABQR 的逻辑进行前向传播 ===
        # 1. 对问题嵌入进行对比学习增强
        pro_embed_with_gcn, contrast_loss = self.gcl(self.pro_embed, self.matrix, perb)
        
        # 2. 将问题嵌入和答案嵌入转化为序列
        last_pro_embed = F.embedding(last_pro, pro_embed_with_gcn)
        next_pro_embed = F.embedding(next_pro, pro_embed_with_gcn)
        ans_embed = self.ans_embed(last_ans)
        
        # 3. 组合问题和答案嵌入作为LSTM输入
        X = last_pro_embed + ans_embed
        X = self.dropout_layer(X)
        
        # 4. 通过LSTM进行时序建模
        X, _ = self.lstm(X)
        
        # 5. 将LSTM输出和下一个问题嵌入拼接，进行最终预测
        # 注意: 这里的拼接和预测逻辑与您提供的原始ABQR代码片段中的forward函数保持一致
        P = torch.sigmoid(self.origin_out(torch.cat([X, next_pro_embed], dim=-1)))
        preds = P
        
        # 6. 返回值与您现有代码的接口保持一致
        if train:
            # ABQRv2的训练返回值为 preds, y2, y3, contrast_loss
            # 原始ABQR没有y2,y3，因此我们返回0
            return preds, 0, 0, contrast_loss
        else:
            # ABQRv2的评测返回值为 preds
            return preds