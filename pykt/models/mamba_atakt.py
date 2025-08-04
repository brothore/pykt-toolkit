# # coding: utf-8
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import numpy as np
# from .utils import ut_mask
# from mamba_ssm import Mamba
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# from mamba_ssm.modules.mamba_simple import Mamba as OriginalMamba

# class CustomMamba(nn.Module):
#     def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
#         super().__init__()
#         self.mamba = OriginalMamba(
#             d_model=d_model,
#             d_state=d_state,
#             d_conv=d_conv,
#             expand=expand
#         )
#         self.activation = TeLU()
    
#     def forward(self, x):
#         x = self.mamba(x)
#         return self.activation(x)

# class TeLU(nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, x):
#         return x * torch.tanh(torch.exp(x))        
# class MAMBA_ATAKT(nn.Module):
#     def __init__(self, num_c, skill_dim, answer_dim, hidden_dim, attention_dim=80, epsilon=10, beta=0.2, dropout=0.2, emb_type="qid", emb_path="", fix=True):
#         super(MAMBA_ATAKT, self).__init__()
#         self.model_name = "mamba_atakt"
#         self.fix = fix
#         print(f"fix: {fix}")
#         if self.fix == True:
#             self.model_name = "mamba_atakt_fix"
#         self.emb_type = emb_type
#         self.skill_dim=skill_dim
#         self.answer_dim=answer_dim
#         self.hidden_dim = hidden_dim
#         self.num_c = num_c
#         self.epsilon = epsilon
#         self.beta = beta
        
#         # 根据emb_type选择不同的Mamba配置方案
#         if self.emb_type == "qid_after":
#             # 方案1: Mamba输出需要投影到hidden_dim
#             self.rnn = Mamba(
#                 d_model=self.skill_dim + self.answer_dim,  # 输入维度
#                 d_state=16,  
#                 d_conv=4,    
#                 expand=2,    
#             )
#             # 添加线性层将Mamba输出映射到hidden_dim
#             self.mamba_proj = nn.Linear(self.skill_dim + self.answer_dim, self.hidden_dim)
#             self.input_proj = None
            
#         elif self.emb_type == "qid_before":
#             # 方案2: 直接使用hidden_dim作为d_model
#             self.rnn = Mamba(
#                 d_model=self.hidden_dim,  # 直接使用hidden_dim
#                 d_state=16,  
#                 d_conv=4,    
#                 expand=2,    
#             )
#             # 添加输入投影层
#             self.input_proj = nn.Linear(self.skill_dim + self.answer_dim, self.hidden_dim)
#             self.mamba_proj = None
            
#         else:
#             # 默认使用方案1
#             self.rnn = Mamba(
#                 d_model=self.skill_dim + self.answer_dim,
#                 d_state=16,  
#                 d_conv=4,    
#                 expand=2,    
#             )
#             self.mamba_proj = nn.Linear(self.skill_dim + self.answer_dim, self.hidden_dim)
#             self.input_proj = None
        
#         self.dropout_layer = nn.Dropout(dropout)
#         self.fc = nn.Linear(self.hidden_dim*2, self.num_c)
#         self.sig = nn.Sigmoid()
        
#         self.skill_emb = nn.Embedding(self.num_c+1, self.skill_dim)
#         self.skill_emb.weight.data[-1]= 0
        
#         self.answer_emb = nn.Embedding(2+1, self.answer_dim)
#         self.answer_emb.weight.data[-1]= 0
        
#         self.attention_dim = attention_dim
#         self.mlp = nn.Linear(self.hidden_dim, self.attention_dim)
#         self.similarity = nn.Linear(self.attention_dim, 1, bias=False)

    
#     def attention_module(self, lstm_output):
#         # lstm_output = lstm_output[0:1, :, :]
#         # print(f"lstm_output: {lstm_output.shape}")
#         att_w = self.mlp(lstm_output)
#         # print(f"att_w: {att_w.shape}")
#         att_w = torch.tanh(att_w)
#         att_w = self.similarity(att_w)
#         # print(f"att_w: {att_w.shape}")

#         if self.fix == True:
#             attn_mask = ut_mask(lstm_output.shape[1])
#             att_w = att_w.transpose(1,2).expand(lstm_output.shape[0], lstm_output.shape[1], lstm_output.shape[1]).clone()
#             att_w = att_w.masked_fill_(attn_mask, float("-inf"))
#             alphas = torch.nn.functional.softmax(att_w, dim=-1)
#             attn_ouput = torch.bmm(alphas, lstm_output)
#         else: # 原来的官方实现
#             alphas=nn.Softmax(dim=1)(att_w)
#             # print(f"alphas: {alphas.shape}")    
#             attn_ouput = alphas*lstm_output # 整个seq的attn之和为1，计算前面的的时候，所有的attn都<<1，不会有问题？做的少的时候，历史作用小，做得多的时候，历史作用变大？
#             # print(f"attn_ouput: {attn_ouput.shape}")
#         attn_output_cum=torch.cumsum(attn_ouput, dim=1)
#         # print(f"attn_ouput: {attn_ouput}")
#         # print(f"attn_output_cum: {attn_output_cum}")
#         attn_output_cum_1=attn_output_cum-attn_ouput
#         # print(f"attn_output_cum_1: {attn_output_cum_1}")
#         # print(f"lstm_output: {lstm_output}")

#         final_output=torch.cat((attn_output_cum_1, lstm_output),2)
#         # import sys
#         # sys.exit()

#         return final_output


#     def forward(self, skill, answer, perturbation=None):
#         emb_type = self.emb_type
#         r = answer
        
#         skill_embedding=self.skill_emb(skill)
#         answer_embedding=self.answer_emb(answer)
        
#         skill_answer=torch.cat((skill_embedding,answer_embedding), 2)
#         answer_skill=torch.cat((answer_embedding,skill_embedding), 2)
        
#         answer=answer.unsqueeze(2).expand_as(skill_answer)
        
#         skill_answer_embedding=torch.where(answer==1, skill_answer, answer_skill)
        
#         # print(skill_answer_embedding)
        
#         skill_answer_embedding1=skill_answer_embedding
#         if perturbation is not None:
#             skill_answer_embedding += perturbation
        
#         # 根据emb_type选择不同的处理方式
#         if self.emb_type == "qid_after":
#             # 方案1: 先通过Mamba，再投影到hidden_dim
#             out = self.rnn(skill_answer_embedding)
#             out = self.mamba_proj(out)
            
#         elif self.emb_type == "qid_before":
#             # 方案2: 先投影到hidden_dim，再通过Mamba
#             out = self.input_proj(skill_answer_embedding)
#             out = self.rnn(out)
            
#         else:
#             # 默认使用方案1
#             out = self.rnn(skill_answer_embedding)
#             out = self.mamba_proj(out)
        
#         # print(f"out: {out.shape}")
#         out=self.attention_module(out)
#         # print(f"after attn out: {out.shape}")
#         res = self.sig(self.fc(self.dropout_layer(out)))

#         # res = res[:, :-1, :]
#         # pred_res = self._get_next_pred(res, skill)
        
#         return res, skill_answer_embedding1

# from torch.autograd import Variable

# def _l2_normalize_adv(d):
#     if isinstance(d, Variable):
#         d = d.data.cpu().numpy()
#     elif isinstance(d, torch.FloatTensor) or isinstance(d, torch.cuda.FloatTensor):
#         d = d.cpu().numpy()
#     d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
#     return torch.from_numpy(d)


