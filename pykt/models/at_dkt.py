# coding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from .utils import ut_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout

class AT_DKT(Module):
    def __init__(self, num_c, skill_dim,answer_dim, hidden_dim,attention_dim=80, epsilon=10, beta=0.2,  dropout=0.1, emb_type='qid', emb_path="",fix=True, pretrain_dim=768):
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

        if emb_type.endswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.hidden_dim)
            self.lstm_layer = LSTM(self.hidden_dim, self.hidden_size, batch_first=True)

        elif emb_type.endswith("qid_seprate"):
            self.skill_emb = nn.Embedding(self.num_c+1, self.skill_dim)
            self.skill_emb.weight.data[-1]= 0
            
            self.answer_emb = nn.Embedding(2+1, self.answer_dim)
            self.answer_emb.weight.data[-1]= 0
            self.lstm_layer = LSTM(self.skill_dim+self.answer_dim, self.hidden_dim, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)
        self.custom_lstm_layer = None
    
    def forward(self, q, r, perturbation=None):
        # print(f"q.shape is {q.shape}")
        emb_type = self.emb_type
        if emb_type.endswith("qid"):
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)
        elif emb_type.endswith("qid_seprate"):
            skill_embedding=self.skill_emb(q)
            answer_embedding=self.answer_emb(r)
            xemb=torch.cat((skill_embedding,answer_embedding), 2)
        # print(f"xemb.shape is {xemb.shape}")
        xemb1 = xemb
        if emb_type.startswith("at"):
            if  perturbation is not None:
                xemb += perturbation


        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        
        return y,xemb1
