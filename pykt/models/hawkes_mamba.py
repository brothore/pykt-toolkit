import numpy as np
import torch
import torch.nn as nn
# Assuming Mamba is available from a library like mamba_ssm
from mamba_ssm import Mamba

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HawkesMamba(nn.Module):
    def __init__(self, n_skills, n_problems, emb_size, time_log, emb_type="qid"):
        super().__init__()
        self.model_name = "hawkes_mamba"
        self.emb_type = emb_type
        self.problem_num = n_problems
        self.skill_num = n_skills
        self.emb_size = emb_size
        self.time_log = time_log
        self.gpu = device

        self.alpha_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
        self.alpha_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)
        self.beta_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
        self.beta_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)

        if self.emb_type == "qid":
            self.problem_base = torch.nn.Embedding(self.problem_num, 1)
            self.skill_base = torch.nn.Embedding(self.skill_num, 1)
            self.mamba = Mamba(
                d_model=self.emb_size,  # Hidden size, matching LSTM
                d_state=16,             # State dimension (default from Mamba)
                d_conv=4,               # Convolution kernel size
                expand=2,               # Expansion factor
                dt_rank="auto",         # Rank for delta projection
                bias=False              # No bias for simplicity
            )
            self.output_linear = nn.Linear(self.emb_size, 1)
        elif self.emb_type == "vec":
            self.problem_base = torch.nn.Embedding(self.problem_num, self.emb_size)
            self.skill_base = torch.nn.Embedding(self.skill_num, self.emb_size)
            self.sum_t_proj = nn.Linear(1, self.emb_size)
            self.mamba = Mamba(
                d_model=self.emb_size,
                d_state=16,
                d_conv=4,
                expand=2,
                dt_rank="auto",
                bias=False
            )
            self.output_linear = nn.Linear(self.emb_size, 1)
        else:
            raise ValueError("Unsupported emb_type")

    @staticmethod
    def init_weights(m):
        if type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        elif type(m) == Mamba:
            # Mamba's weights are typically initialized internally, but we can customize if needed
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

        alpha_src_emb = self.alpha_inter_embeddings(inters)  # [bs, seq_len, emb]
        alpha_target_emb = self.alpha_skill_embeddings(skills)
        alphas = torch.matmul(alpha_src_emb, alpha_target_emb.transpose(-2, -1))  # [bs, seq_len, seq_len]
        beta_src_emb = self.beta_inter_embeddings(inters)  # [bs, seq_len, emb]
        beta_target_emb = self.beta_skill_embeddings(skills)
        betas = torch.matmul(beta_src_emb, beta_target_emb.transpose(-2, -1))  # [bs, seq_len, seq_len]
        betas = torch.clamp(betas + 1, min=0, max=10)

        if times.shape[1] > 0:
            times = times.double() / 1000
            delta_t = (times[:, :, None] - times[:, None, :]).abs().double()
        else:
            delta_t = torch.ones(skills.shape[0], skills.shape[1], skills.shape[1]).double().to(device)
        delta_t = torch.log(delta_t + 1e-10) / np.log(self.time_log)

        cross_effects = alphas * torch.exp(-betas * delta_t)
        seq_len = skills.shape[1]
        valid_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1)
        mask = (torch.from_numpy(valid_mask) == 0)
        mask = mask.cuda() if self.gpu != '' else mask
        sum_t = cross_effects.masked_fill(mask, 0).sum(-2)

        if self.emb_type == "qid":
            problem_bias = self.problem_base(problems).squeeze(dim=-1)
            skill_bias = self.skill_base(skills).squeeze(dim=-1)
            h = problem_bias + skill_bias + sum_t
            h_input = h.unsqueeze(-1)  # [bs, seq_len, 1]
        elif self.emb_type == "vec":
            problem_emb = self.problem_base(problems)  # [bs, seq_len, emb_size]
            skill_emb = self.skill_base(skills)  # [bs, seq_len, emb_size]
            sum_t_exp = sum_t.unsqueeze(-1)  # [bs, seq_len, 1]
            sum_t_proj = self.sum_t_proj(sum_t_exp)  # [bs, seq_len, emb_size]
            h_input = problem_emb + skill_emb + sum_t_proj  # [bs, seq_len, emb_size]

        # Pass h_input through Mamba
        mamba_out = self.mamba(h_input)  # [bs, seq_len, emb_size]
        logits = self.output_linear(mamba_out).squeeze(-1)  # [bs, seq_len]
        prediction = logits.sigmoid()

        if not qtest:
            return prediction
        else:
            return prediction, h