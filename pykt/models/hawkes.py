import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HawkesKT(nn.Module):
    def __init__(self, n_skills, n_problems, emb_size, time_log, emb_type="qid"):
        super().__init__()
        self.model_name = "hawkes"
        self.emb_type = emb_type
        self.problem_num = n_problems
        self.skill_num = n_skills
        self.emb_size = emb_size
        self.time_log = time_log
        self.gpu = device

        # Base intensity embeddings
        if self.emb_type not in ["no_base", "no_problem_base"]:
            self.problem_base = torch.nn.Embedding(self.problem_num, 1)
        if self.emb_type not in ["no_base", "no_skill_base"]:
            self.skill_base = torch.nn.Embedding(self.skill_num, 1)

        # Cross-effects embeddings
        if self.emb_type == "no_cf":
            self.alpha = torch.nn.Parameter(torch.randn(2 * self.skill_num, self.skill_num) * 0.01)
            self.beta = torch.nn.Parameter(torch.randn(2 * self.skill_num, self.skill_num) * 0.01)
        else:
            self.alpha_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
            self.alpha_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)
            self.beta_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
            self.beta_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)

        # Global beta for \Cross variant
        if self.emb_type == "global_beta":
            self.global_beta = torch.nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def init_weights(m):
        if type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, skills, problems, times, labels, qtest=False):
        mask_labels = labels
        inters = skills + mask_labels * self.skill_num
        # print(f"[DEBUG] self.emb_type: {self.emb_type} (type: {type(self.emb_type)})")
        # Compute alphas
        if self.emb_type == "no_cf":
            source_idx = inters.unsqueeze(-1).repeat(1, 1, labels.shape[1]).long()
            target_idx = skills.unsqueeze(1).repeat(1, labels.shape[1], 1).long()
            alphas = self.alpha[source_idx, target_idx]
        elif self.emb_type == "no_alphas":
            seq_len = skills.shape[1]
            alphas = torch.ones(skills.shape[0], seq_len, seq_len).to(device)
        else:
            alpha_src_emb = self.alpha_inter_embeddings(inters)
            alpha_target_emb = self.alpha_skill_embeddings(skills)
            alphas = torch.matmul(alpha_src_emb, alpha_target_emb.transpose(-2, -1))

        # Compute betas and delta_t
        if self.emb_type == "no_cf":
            betas = torch.clamp(self.beta[source_idx, target_idx] + 1, min=0, max=10)
        elif self.emb_type == "global_beta":
            betas = self.global_beta
        else:
            beta_src_emb = self.beta_inter_embeddings(inters)
            beta_target_emb = self.beta_skill_embeddings(skills)
            betas = torch.matmul(beta_src_emb, beta_target_emb.transpose(-2, -1))
            betas = torch.clamp(betas + 1, min=0, max=10)

        # Compute delta_t
        if self.emb_type == "constant_delta_t":
            seq_len = skills.shape[1]
            delta_t = torch.ones(skills.shape[0], seq_len, seq_len).double().to(device)
            delta_t = torch.log(delta_t + 1e-10) / np.log(self.time_log)
        else:
            if times.shape[1] > 0:
                times = times.double() / 1000
                delta_t = (times[:, :, None] - times[:, None, :]).abs().double()
            else:
                delta_t = torch.ones(skills.shape[0], skills.shape[1], skills.shape[1]).double().to(device)
            delta_t = torch.log(delta_t + 1e-10) / np.log(self.time_log)

        # Compute cross-effects
        if self.emb_type == "no_temporal":
            cross_effects = alphas  # 移除时间衰减
        else:
            cross_effects = alphas * torch.exp(-betas * delta_t)

        seq_len = skills.shape[1]
        valid_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1)
        mask = (torch.from_numpy(valid_mask) == 0)
        mask = mask.cuda() if self.gpu != '' else mask
        sum_t = cross_effects.masked_fill(mask, 0).sum(-2)

        # Compute base intensity
        if self.emb_type == "no_base":
            prediction = sum_t.sigmoid()
            h = sum_t
        elif self.emb_type == "no_skill_base":
            problem_bias = self.problem_base(problems).squeeze(dim=-1)
            prediction = (problem_bias + sum_t).sigmoid()
            h = problem_bias + sum_t
        elif self.emb_type == "no_problem_base":
            skill_bias = self.skill_base(skills).squeeze(dim=-1)
            prediction = (skill_bias + sum_t).sigmoid()
            h = skill_bias + sum_t
        else:
            problem_bias = self.problem_base(problems).squeeze(dim=-1)
            skill_bias = self.skill_base(skills).squeeze(dim=-1)
            prediction = (problem_bias + skill_bias + sum_t).sigmoid()
            h = problem_bias + skill_bias + sum_t

        if not qtest:
            return prediction
        else:
            return prediction, h