# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HawkesLSTM(nn.Module):
    def __init__(self, n_skills, n_problems, emb_size, time_log, emb_type="qid"):
        super().__init__()
        self.model_name = "hawkes_lstm"
        self.emb_type = emb_type
        self.problem_num = n_problems
        self.skill_num = n_skills
        self.emb_size = emb_size
        self.time_log = time_log
        self.gpu = device

        # Parse emb_type
        if "_" in self.emb_type:
            parts = self.emb_type.split("_")
            if len(parts) != 2:
                raise ValueError("emb_type should be in format 'feature_rnn' or a single feature type.")
            self.feature_type = parts[0]
            self.rnn_type = parts[1].lower()
        else:
            self.feature_type = self.emb_type
            self.rnn_type = "lstm"  # Default for backward compatibility

        if self.rnn_type not in ["lstm", "gru"]:
            raise ValueError("Unsupported rnn_type in emb_type; must be 'lstm' or 'gru'.")

        # HawkesKT parameters
        self.alpha_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
        self.alpha_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)
        self.beta_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
        self.beta_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)

        if self.feature_type == "qid":
            self.problem_base = torch.nn.Embedding(self.problem_num, 1)
            self.skill_base = torch.nn.Embedding(self.skill_num, 1)
            rnn_input_size = 1
        elif self.feature_type == "vec":
            self.problem_base = torch.nn.Embedding(self.problem_num, self.emb_size)
            self.skill_base = torch.nn.Embedding(self.skill_num, self.emb_size)
            self.sum_t_proj = nn.Linear(1, self.emb_size)
            rnn_input_size = self.emb_size
            
            # 将RNN特征映射到HawkesKT参数空间 (updated comment for RNN)
            self.alpha_proj = nn.Linear(self.emb_size, self.emb_size)
            self.beta_proj = nn.Linear(self.emb_size, self.emb_size)
        else:
            raise ValueError("Unsupported feature_type in emb_type")

        # RNN backbone
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=self.emb_size, num_layers=1, batch_first=True)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=self.emb_size, num_layers=1, batch_first=True)

        # 输出层
        self.output_linear = nn.Linear(self.emb_size, 1)

    @staticmethod
    def init_weights(m):
        if type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        elif type(m) in (nn.LSTM, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def printparams(self):
        print("="*20)
        for m in list(self.named_parameters()):
            print(m[0], m[1])
        self.count += 1
        print(f"count: {self.count}")

    def forward(self, skills, problems, times, labels, qtest=False):
        mask_labels = labels
        inters = skills + mask_labels * self.skill_num

        if True:
            # 原始HawkesKT实现
            alpha_src_emb = self.alpha_inter_embeddings(inters)
            alpha_target_emb = self.alpha_skill_embeddings(skills)
            alphas = torch.matmul(alpha_src_emb, alpha_target_emb.transpose(-2, -1))
            
            beta_src_emb = self.beta_inter_embeddings(inters)
            beta_target_emb = self.beta_skill_embeddings(skills)
            betas = torch.matmul(beta_src_emb, beta_target_emb.transpose(-2, -1))
            betas = torch.clamp(betas + 1, min=0, max=10)
            
            if times.shape[1] > 0:
                times = times.double() / 1000
                delta_t = (times[:, :, None] - times[:, None, :]).abs().double()
            else:
                delta_t = torch.ones(skills.shape[0], skills.shape[1], skills.shape[1]).double().to(device)
            
            delta_t = torch.log(delta_t + 1e-10) / np.log(self.time_log)
            
            cross_effects = alphas * torch.exp(-betas * delta_t)
            
            seq_len = skills.shape[1]
            valid_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
            mask = (torch.from_numpy(valid_mask) == 0)
            mask = mask.cuda() if self.gpu != '' else mask
            sum_t = cross_effects.masked_fill(mask, 0).sum(-2)
            
            if self.feature_type == "qid":
                problem_bias = self.problem_base(problems).squeeze(dim=-1)
                skill_bias = self.skill_base(skills).squeeze(dim=-1)
                h = problem_bias + skill_bias + sum_t
                h_input = h.unsqueeze(-1)
            elif self.feature_type == "vec":
                problem_emb = self.problem_base(problems)
                skill_emb = self.skill_base(skills)
                sum_t_exp = sum_t.unsqueeze(-1)
                sum_t_proj = self.sum_t_proj(sum_t_exp)
                h = problem_emb + skill_emb + sum_t_proj
                h_input = h
            
            rnn_out, _ = self.rnn(h_input)
            logits = self.output_linear(rnn_out).squeeze(-1)
            prediction = logits.sigmoid()

        if not qtest:
            return prediction
        else:
            return prediction, h