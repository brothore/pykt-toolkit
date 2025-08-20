# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn as nn
from mamba_ssm import Mamba

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HawkesMamba(nn.Module):
    # def __init__(self, args, corpus):
    def __init__(self, n_skills, n_problems, emb_size, time_log, emb_type="qid"):
        super().__init__()
        self.model_name = "hawkes_mamba"
        self.emb_type = emb_type
        self.problem_num = n_problems
        self.skill_num = n_skills
        self.emb_size = emb_size
        self.time_log = time_log
        self.gpu = device

        # Create embeddings - will convert to float32 later
        self.alpha_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
        self.alpha_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)
        self.beta_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
        self.beta_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)

        if self.emb_type == "qid":
            self.problem_base = torch.nn.Embedding(self.problem_num, 1)
            self.skill_base = torch.nn.Embedding(self.skill_num, 1)
            # Replace LSTM with Mamba
            self.mamba = Mamba(
                d_model=1,  # input dimension
                d_state=16,  # state dimension
                d_conv=4,    # convolution kernel size
                expand=2,    # expansion factor
            ).to(torch.float32)
            self.output_linear = nn.Linear(1, 1)
        elif self.emb_type == "vec":
            self.problem_base = torch.nn.Embedding(self.problem_num, self.emb_size)
            self.skill_base = torch.nn.Embedding(self.skill_num, self.emb_size)
            self.sum_t_proj = nn.Linear(1, self.emb_size)
            # Replace LSTM with Mamba
            self.mamba = Mamba(
                d_model=self.emb_size,  # input dimension
                d_state=16,             # state dimension  
                d_conv=4,               # convolution kernel size
                expand=2,               # expansion factor
            ).to(torch.float32)
            self.output_linear = nn.Linear(self.emb_size, 1)
        else:
            raise ValueError("Unsupported emb_type")

        # Initialize weights
        self.apply(self.init_weights)
        
        # Ensure all parameters are float32 - more aggressive approach
        self.to(torch.float32)
        
        # Double check: explicitly convert all linear and embedding layers
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data = module.weight.data.float()
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data = module.bias.data.float()
        
        # Also convert Mamba layers
        self.mamba.to(torch.float32)

    @staticmethod
    def init_weights(m):
        if type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        elif type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            if (m.bias is not None):
                nn.init.zeros_(m.bias)

    def printparams(self):
        print("="*20)
        for m in list(self.named_parameters()):
            print(m[0], m[1])
        self.count += 1
        print(f"count: {self.count}")

    def forward(self, skills, problems, times, labels, qtest=False):
        # Debug: Print input types
        # print(f"Input types - skills: {skills.dtype}, problems: {problems.dtype}, times: {times.dtype}, labels: {labels.dtype}")
        
        # Ensure inputs are on the correct device and dtype
        skills = skills.to(device, dtype=torch.long)
        problems = problems.to(device, dtype=torch.long)
        times = times.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        
        mask_labels = labels
        inters = skills + mask_labels * self.skill_num

        alpha_src_emb = self.alpha_inter_embeddings(inters).float()  # [bs, seq_len, emb]
        # print(f"alpha_src_emb dtype: {alpha_src_emb.dtype}")
        
        alpha_target_emb = self.alpha_skill_embeddings(skills).float()
        # print(f"alpha_target_emb dtype: {alpha_target_emb.dtype}")
        
        alphas = torch.matmul(alpha_src_emb, alpha_target_emb.transpose(-2, -1))  # [bs, seq_len, seq_len]
        # print(f"alphas dtype: {alphas.dtype}")
        
        beta_src_emb = self.beta_inter_embeddings(inters).float()  # [bs, seq_len, emb]
        beta_target_emb = self.beta_skill_embeddings(skills).float()
        betas = torch.matmul(beta_src_emb, beta_target_emb.transpose(-2, -1))  # [bs, seq_len, seq_len]
        betas = torch.clamp(betas + 1, min=0, max=10)
        # print(f"betas dtype: {betas.dtype}")
        
        if times.shape[1] > 0:
            times = times.float()  # Ensure float32
            delta_t = (times[:, :, None] - times[:, None, :]).abs().float()
        else:
            delta_t = torch.ones(skills.shape[0], skills.shape[1], skills.shape[1], dtype=torch.float32, device=device)
        
        # print(f"delta_t dtype before log: {delta_t.dtype}")
        delta_t = torch.log(delta_t + 1e-10) / np.log(self.time_log)
        # print(f"delta_t dtype after log: {delta_t.dtype}")

        cross_effects = (alphas * torch.exp(-betas * delta_t)).float()
        # print(f"cross_effects dtype: {cross_effects.dtype}")

        seq_len = skills.shape[1]
        valid_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1)
        mask = (torch.from_numpy(valid_mask) == 0)
        mask = mask.to(device)
        sum_t = cross_effects.masked_fill(mask, 0).sum(-2).float()
        # print(f"sum_t dtype: {sum_t.dtype}")

        if self.emb_type == "qid":
            problem_bias = self.problem_base(problems).squeeze(dim=-1).float()
            skill_bias = self.skill_base(skills).squeeze(dim=-1).float()
            # print(f"problem_bias dtype: {problem_bias.dtype}, skill_bias dtype: {skill_bias.dtype}")
            
            h = problem_bias + skill_bias + sum_t
            h_input = h.unsqueeze(-1)  # [bs, seq_len, 1]
        elif self.emb_type == "vec":
            problem_emb = self.problem_base(problems).float()  # [bs, seq_len, emb_size]
            skill_emb = self.skill_base(skills).float()  # [bs, seq_len, emb_size]
            # print(f"problem_emb dtype: {problem_emb.dtype}, skill_emb dtype: {skill_emb.dtype}")
            
            sum_t_exp = sum_t.unsqueeze(-1)  # [bs, seq_len, 1]
            # print(f"sum_t_exp dtype: {sum_t_exp.dtype}")
            # print(f"sum_t_proj weight dtype: {self.sum_t_proj.weight.dtype}")
            # print(f"sum_t_proj bias dtype: {self.sum_t_proj.bias.dtype if self.sum_t_proj.bias is not None else 'None'}")
            
            # Force convert linear layer to float32 if needed
            if self.sum_t_proj.weight.dtype != torch.float32:
                # print("Converting sum_t_proj to float32...")
                self.sum_t_proj.weight.data = self.sum_t_proj.weight.data.float()
                if self.sum_t_proj.bias is not None:
                    self.sum_t_proj.bias.data = self.sum_t_proj.bias.data.float()
                # print(f"After conversion - weight dtype: {self.sum_t_proj.weight.dtype}")
            
            sum_t_proj = self.sum_t_proj(sum_t_exp)  # Now both should be float32
            # print(f"sum_t_proj dtype: {sum_t_proj.dtype}")
            
            h = problem_emb + skill_emb + sum_t_proj
            h_input = h  # [bs, seq_len, emb_size]
        
        # Ensure h_input is explicitly float32 for Mamba compatibility
        h_input = h_input.float()
        # print(f"h_input dtype before Mamba: {h_input.dtype}")
        # print(f"h_input shape: {h_input.shape}")
        
        # Force convert Mamba to float32 if needed
        mamba_has_double = False
        for name, param in self.mamba.named_parameters():
            if param.dtype == torch.float64:
                # print(f"Found double dtype in Mamba: {name} - {param.dtype}")
                param.data = param.data.float()
                mamba_has_double = True
        
        # if mamba_has_double:
            # print("Converted Mamba parameters to float32")
        
        # Pass h_input through Mamba instead of LSTM
        mamba_out = self.mamba(h_input)  # [bs, seq_len, d_model]
        
        # Force convert output_linear to float32 if needed
        if self.output_linear.weight.dtype != torch.float32:
            # print("Converting output_linear to float32...")
            self.output_linear.weight.data = self.output_linear.weight.data.float()
            if self.output_linear.bias is not None:
                self.output_linear.bias.data = self.output_linear.bias.data.float()
        
        logits = self.output_linear(mamba_out).squeeze(-1)  # [bs, seq_len]
        prediction = logits.sigmoid()

        if not qtest:
            return prediction
        else:
            return prediction, h