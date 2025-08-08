# coding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from .utils import ut_mask
from torch.nn.functional import one_hot
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
class AT_DKT(Module):
    def __init__(self, num_c, skill_dim, answer_dim, hidden_dim, attention_dim=80, epsilon=10, beta=0.2, dropout=0.1, emb_type='qid', emb_path="", fix=True, pretrain_dim=768):
        super().__init__()
        self.model_name = "at_dkt"
        self.num_c = num_c
        self.emb_size = hidden_dim
        self.hidden_size = hidden_dim
        self.emb_type = emb_type
        self.skill_dim = skill_dim
        self.answer_dim = answer_dim
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.beta = beta
        
        if "qid_seprate" in emb_type:
            self.skill_emb = nn.Embedding(self.num_c+1, self.skill_dim)
            self.skill_emb.weight.data[-1] = 0
            self.answer_emb = nn.Embedding(2+1, self.answer_dim)
            self.answer_emb.weight.data[-1] = 0
            self.lstm_layer = LSTM(self.skill_dim+self.answer_dim, self.hidden_dim, batch_first=True)
        elif "qid" in emb_type:
            self.interaction_emb = Embedding(self.num_c * 2, self.hidden_dim)
            self.lstm_layer = LSTM(self.hidden_dim, self.hidden_size, batch_first=True)

        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)

    def forward(self, q, r, perturbation=None, pgd_attack=False, loss_fn=None,cshft=None):
        # 基础嵌入计算
        if "qid_seprate" in self.emb_type:
            skill_emb = self.skill_emb(q.long())
            answer_emb = self.answer_emb(r.long())
            xemb = torch.cat((skill_emb, answer_emb), 2)
        elif "qid" in self.emb_type:
            x = (q.long() + self.num_c * r.long())
            xemb = self.interaction_emb(x)
        
        # PGD对抗训练模式
        if pgd_attack and loss_fn is not None:
            return self._pgd_attack(q, r, xemb, loss_fn,cshft)
        
        # 常规前向传播
        if perturbation is not None:
            xemb = xemb + perturbation
            
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        
        return y, xemb

    def _pgd_attack(self, q, r, xemb, loss_fn,cshft):
        """内部PGD攻击实现"""
        perturbation = torch.zeros_like(xemb).uniform_(-self.epsilon, self.epsilon).to(device)
        perturbation.requires_grad_()
        
        for _ in range(7):  # PGD迭代次数
            # 计算对抗损失
            adv_output, _ = self(q, r, perturbation)
            adv_output = (adv_output * one_hot(cshft.long(), self.num_c)).sum(-1)
            adv_loss = loss_fn(adv_output)
            # 梯度上升
            grad = torch.autograd.grad(adv_loss, perturbation, only_inputs=True)[0]
            perturbation.data = perturbation.data + (self.epsilon/4) * grad.sign()
            
            # 投影到扰动范围内
            perturbation.data = torch.clamp(perturbation.data, -self.epsilon, self.epsilon)
        
        return perturbation.detach()