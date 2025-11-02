import os
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .que_base_model import QueBaseModel,QueEmb
from pykt.utils import debug_print
from sklearn import metrics
from torch.utils.data import DataLoader
from .loss import Loss
from scipy.special import softmax
# from mamba_ssm import Mamba
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    '''
    classifier decoder implemented with mlp
    '''
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()

        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p = dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))



def get_outputs(self, emb_qc_shift, h, data, add_name="", model_type='question'):
    outputs = {}
  
    if model_type == 'question':
        h_next = torch.cat([emb_qc_shift, h], axis=-1)
        y_question_next = torch.sigmoid(self.out_question_next(h_next))
        y_question_all = torch.sigmoid(self.out_question_all(h))
        # 确保 data['qshft'] 在 self.device 上并转换为 long 类型
        qshft = data['qshft'].to(self.device).long()
        outputs["y_question_next" + add_name] = y_question_next.squeeze(-1)
        outputs["y_question_all" + add_name] = (y_question_all * F.one_hot(qshft, self.num_q)).sum(-1)
    else: 
        h_next = torch.cat([emb_qc_shift, h], axis=-1)
        y_concept_next = torch.sigmoid(self.out_concept_next(h_next))
        y_concept_all = torch.sigmoid(self.out_concept_all(h))
        # 确保 data['cshft'] 在 self.device 上
        cshft = data['cshft'].to(self.device).long()
        outputs["y_concept_next" + add_name] = self.get_avg_fusion_concepts(y_concept_next, cshft)
        outputs["y_concept_all" + add_name] = self.get_avg_fusion_concepts(y_concept_all, cshft)

    return outputs

class QIKTNet(nn.Module):
    def __init__(self, num_q,num_c,emb_size, dropout=0.1, emb_type='qaid', emb_path="", pretrain_dim=768,device='cpu',mlp_layer_num=1,other_config={},num_attn_head=2,version="v0"):
        super().__init__()
        self.model_name = "qikt_mamba"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.mlp_layer_num = mlp_layer_num
        self.device = device
        self.other_config = other_config
        self.output_mode = self.other_config.get('output_mode','an')

        self.version = version
        self.emb_type = emb_type
      

        self.que_emb = QueEmb(num_q=num_q,num_c=num_c,emb_size=emb_size,emb_type=self.emb_type,model_name=self.model_name,device=device,
                             emb_path=emb_path,pretrain_dim=pretrain_dim,num_attn_head=num_attn_head)
       
        if self.version == "lstm":
            self.que_lstm_layer = nn.GRU(self.emb_size*4, self.hidden_size, batch_first=True)
            self.concept_lstm_layer = nn.GRU(self.emb_size*2, self.hidden_size, batch_first=True)
        elif self.version in ["encoder",'decoder','full']:
            #transformer
            self.que_lstm_layer = TransformerBranch(emb_size_in=self.emb_size*4, emb_size_out=self.hidden_size,mode=self.version, causal=True)
            self.concept_lstm_layer = TransformerBranch(emb_size_in=self.emb_size*2, emb_size_out=self.hidden_size,mode=self.version, causal=True)
        elif self.version == "unzip_lstm":
            self.four2two = nn.Linear(self.emb_size*4, self.emb_size*2)
            self.que_lstm_layer = nn.LSTM(self.emb_size*2, self.hidden_size, batch_first=True)
            self.concept_lstm_layer = nn.LSTM(self.emb_size*2, self.hidden_size, batch_first=True)
        elif self.version == "zip_lstm":
            # 压缩lstm
            self.four2two = nn.Linear(self.emb_size*2, self.emb_size*4)
            self.que_lstm_layer = nn.LSTM(self.emb_size*4, self.hidden_size, batch_first=True)
            self.concept_lstm_layer = nn.LSTM(self.emb_size*4, self.hidden_size, batch_first=True)
        elif self.version == "public_lstm":
            #共用lstm
            self.four2two = nn.Linear(self.emb_size*2, self.emb_size*4)
            self.que_lstm_layer = nn.LSTM(self.emb_size*4, self.hidden_size, num_layers=2, batch_first=True)
            self.concept_lstm_layer = self.que_lstm_layer
        elif self.version == "public_mamba":
            #共用lstm
            self.four2two = nn.Linear(self.emb_size*2, self.emb_size*4)
            self.que_lstm_layer = Mamba(d_model=self.emb_size*4, d_state=self.hidden_size) 
            self.que_proj = nn.Linear(self.emb_size*4, self.hidden_size)  # 1024 -> 256
            self.concept_proj = nn.Linear(self.emb_size*4, self.hidden_size)  # 512 -> 256
            self.concept_lstm_layer = self.que_lstm_layer
        elif self.version == "public_lstm_large":
            #共用lstm
            self.four2two = nn.Linear(self.emb_size*2, self.emb_size*4)
            self.que_lstm_layer = nn.LSTM(self.emb_size*4, self.hidden_size, num_layers=2, batch_first=True)
            self.concept_lstm_layer = self.que_lstm_layer
        else:
            self.que_lstm_layer = nn.GRU(self.emb_size*4, self.hidden_size, batch_first=True)
            self.concept_lstm_layer = nn.GRU(self.emb_size*2, self.hidden_size, batch_first=True)
            # self.que_lstm_layer = Mamba(d_model=self.emb_size*4, d_state=self.hidden_size) 
            # self.concept_lstm_layer = Mamba(d_model=self.emb_size*2, d_state=self.hidden_size)

            # self.que_proj = nn.Linear(self.emb_size*4, self.hidden_size)  # 1024 -> 256
            # self.concept_proj = nn.Linear(self.emb_size*2, self.hidden_size)  # 512 -> 256
        self.dropout_layer = nn.Dropout(dropout)
        

        

        self.out_question_next = MLP(self.mlp_layer_num,self.hidden_size*3,1,dropout)
        self.out_question_all = MLP(self.mlp_layer_num,self.hidden_size,num_q,dropout)

        self.out_concept_next = MLP(self.mlp_layer_num,self.hidden_size*3,num_c,dropout)
        self.out_concept_all = MLP(self.mlp_layer_num,self.hidden_size,num_c,dropout)

        self.que_disc = MLP(self.mlp_layer_num,self.hidden_size*2,1,dropout)
        
        

    def get_avg_fusion_concepts(self,y_concept,cshft):
        """获取知识点 fusion 的预测结果
        """
        max_num_concept = cshft.shape[-1]
        concept_mask = torch.where(cshft.long()==-1,False,True)
        concept_index = F.one_hot(torch.where(cshft!=-1,cshft,0),self.num_c)
        concept_sum = (y_concept.unsqueeze(2).repeat(1,1,max_num_concept,1)*concept_index).sum(-1)
        concept_sum = concept_sum*concept_mask#remove mask
        y_concept = concept_sum.sum(-1)/torch.where(concept_mask.sum(-1)!=0,concept_mask.sum(-1),1)
        return y_concept

    def forward(self, q, c, r, data=None):
        # 确保输入张量在 self.device 上
        q = q.to(self.device).long()
        c = c.to(self.device).long()
        r = r.to(self.device).long()
        if data is not None:
            data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

        _, emb_qca, emb_qc, _, emb_c = self.que_emb(q, c, r)  # [batch_size,emb_size*4],[batch_size,emb_size*2],...
        
        emb_qc_shift = emb_qc[:, 1:, :]
        emb_qca_current = emb_qca[:, :-1, :]
        # question model
        if self.version == "lstm":

            que_h = self.dropout_layer(self.que_lstm_layer(emb_qca_current)[0])
        elif self.version in ["encoder",'decoder','full']:
            #transformer
            que_h = self.dropout_layer(self.que_lstm_layer(emb_qca_current))
        elif self.version == "zip_lstm":
            # 压缩lstm
            que_h = self.dropout_layer(self.que_lstm_layer((emb_qca_current))[0])
        elif self.version == "unzip_lstm":
            # 压缩lstm
            que_h = self.dropout_layer(self.que_lstm_layer(self.four2two(emb_qca_current))[0])
        elif self.version == "public_lstm":

            #共用lstm
            que_h = self.dropout_layer(self.que_lstm_layer((emb_qca_current))[0])
        elif self.version == "public_mamba":

            que_h = self.dropout_layer(self.que_lstm_layer((emb_qca_current)))
            que_h = self.que_proj(que_h)
        else:
            #原版
            
            que_h = self.dropout_layer(self.que_lstm_layer(emb_qca_current)[0])
        # print(f"[DEBUG] que_h.shape: {que_h.shape} (type: {type(que_h.shape)})")
        que_outputs = get_outputs(self, emb_qc_shift, que_h, data, add_name="", model_type="question")
        outputs = que_outputs

        # concept model
        emb_ca = torch.cat([
            emb_c.mul((1 - r).unsqueeze(-1).repeat(1, 1, self.emb_size)),
            emb_c.mul(r.unsqueeze(-1).repeat(1, 1, self.emb_size))
        ], dim=-1)
        
        emb_ca_current = emb_ca[:, :-1, :]
        if self.version == "lstm":
            #mamba
            concept_h = self.dropout_layer(self.concept_lstm_layer(emb_ca_current)[0])
        elif self.version in ["encoder",'decoder','full']:
            #transformer
            concept_h = self.dropout_layer(self.concept_lstm_layer(emb_ca_current))
        elif self.version == "zip_lstm":
            # 压缩lstm
            concept_h = self.dropout_layer(self.concept_lstm_layer(self.four2two(emb_ca_current))[0])
        elif self.version == "unzip_lstm":
            # 压缩lstm
            concept_h = self.dropout_layer(self.concept_lstm_layer((emb_ca_current))[0])
        elif self.version == "public_lstm":
            #共用lstm
            concept_h = self.dropout_layer(self.concept_lstm_layer(self.four2two(emb_ca_current))[0])

        elif self.version == "public_mamba":
            #共用mamba
            concept_h = self.concept_lstm_layer(self.four2two(emb_ca_current))  # [32, 199, 512]
            concept_h = self.concept_proj(concept_h)  # [32, 199, 256]
            concept_h = self.dropout_layer(concept_h)
        else:
            #原版
            
            concept_h = self.dropout_layer(self.concept_lstm_layer(emb_ca_current)[0])
        concept_outputs = get_outputs(self, emb_qc_shift, concept_h, data, add_name="", model_type="concept")
        outputs['y_concept_all'] = concept_outputs['y_concept_all']
        outputs['y_concept_next'] = concept_outputs['y_concept_next']
        
        return outputs

class QIKT_MAMBA(QueBaseModel):
    def __init__(self, num_q,num_c, emb_size, dropout=0.1, emb_type='qaid', emb_path="", pretrain_dim=768,device='cpu',seed=0,mlp_layer_num=1,other_config={},version="v0",num_attn_head=2,**kwargs):
        model_name = "qikt_mamba"
       
        debug_print(f"emb_type is {emb_type}",fuc_name="QIKT")

        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)
        self.model = QIKTNet(num_q=num_q,num_c=num_c,emb_size=emb_size,dropout=dropout,emb_type=emb_type,
                               emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,mlp_layer_num=mlp_layer_num,other_config=other_config,num_attn_head=num_attn_head,version=version)
        # 新增：支持动态权重的模块
        self.version = version
        if "auto_uncertainty" in self.version:
            self.output_uncertainty = UncertaintyWeightedLoss(num_tasks=3)  # 3个输出：q_all, c_all, c_next
        if "dynamic_attention" in self.version:
            self.attention_layer = nn.MultiheadAttention(embed_dim=1, num_heads=2)  # 假设输出是1D，调整embed_dim如果需要
        if "capsule_routing" in self.version:
            self.routing_iters = 3  # 胶囊路由迭代次数
            self.capsule_dim = 1  # 简化维度
        self.num_attn_head = num_attn_head
        self.model = self.model.to(device)
        self.emb_type = self.model.emb_type
        self.loss_func = self._get_loss_func("binary_crossentropy")
        # self.loss_func = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        self.eval_result = {}
    


    def train_one_step(self,data,process=True,return_all=False):
        outputs,data_new = self.predict_one_step(data,return_details=True,process=process)
        # all -
        loss_q_all = self.get_loss(outputs['y_question_all'],data_new['rshft'],data_new['sm'])
        loss_c_all = self.get_loss(outputs['y_concept_all'],data_new['rshft'],data_new['sm'])
        # next
        loss_q_next = self.get_loss(outputs['y_question_next'],data_new['rshft'],data_new['sm'])#question level loss
        loss_c_next = self.get_loss(outputs['y_concept_next'],data_new['rshft'],data_new['sm'])#kc level loss
        # over all
        loss_kt = self.get_loss(outputs['y'],data_new['rshft'],data_new['sm'])

        def get_loss_lambda(x):
            return self.model.other_config.get(f'loss_{x}',0)*self.model.other_config.get(f'output_{x}',0)
            
        # loss weight
        loss_c_all_lambda = get_loss_lambda("c_all_lambda")
        loss_c_next_lambda = get_loss_lambda("c_next_lambda")
        loss_q_all_lambda = get_loss_lambda("q_all_lambda")
        loss_q_next_lambda = get_loss_lambda("q_next_lambda")

        
        if self.model.output_mode=="an_irt":
            

            if self.version == "all":
                # 完整版本 - 所有损失项都保留
                loss = loss_kt + loss_q_all_lambda * loss_q_all + loss_c_all_lambda * loss_c_all + loss_c_next_lambda * loss_c_next
            elif self.version == "no_kt":
                # 消融知识迁移损失
                loss = loss_q_all_lambda * loss_q_all + loss_c_all_lambda * loss_c_all + loss_c_next_lambda * loss_c_next
            elif self.version == "no_q_all":
                # 消融所有问题损失
                loss = loss_kt + loss_c_all_lambda * loss_c_all + loss_c_next_lambda * loss_c_next
            elif self.version == "no_c_all":
                # 消融所有上下文损失
                loss = loss_kt + loss_q_all_lambda * loss_q_all + loss_c_next_lambda * loss_c_next
            elif self.version == "no_c_next":
                # 消融下一上下文损失
                loss = loss_kt + loss_q_all_lambda * loss_q_all + loss_c_all_lambda * loss_c_all
            elif self.version == "no_c":
                loss = loss_kt + loss_q_all_lambda * loss_q_all
            elif self.version == "kt_only":
                # 仅保留知识迁移损失
                loss = loss_kt
            elif self.version == "q_only":
                # 仅保留问题相关损失
                loss = loss_q_all_lambda * loss_q_all
            elif self.version == "c_only":
                # 仅保留上下文相关损失
                loss = loss_c_all_lambda * loss_c_all + loss_c_next_lambda * loss_c_next
            elif self.version == "no_next_context":
                # 消融下一上下文但保留当前上下文
                loss = loss_kt + loss_q_all_lambda * loss_q_all + loss_c_all_lambda * loss_c_all
            elif self.version == "minimal":
                # 最小组合 - 只保留知识迁移和问题损失
                loss = loss_kt + loss_q_all_lambda * loss_q_all
            elif self.version == "auto_uncertainty":
                loss_func = UncertaintyWeightedLoss(num_tasks=4)
                loss = loss_func([loss_kt, loss_q_all, loss_c_all, loss_c_next])
            else:
                # 默认完整版本
                loss = loss_kt + loss_q_all_lambda * loss_q_all + loss_c_all_lambda * loss_c_all + loss_c_next_lambda * loss_c_next


        else:
            loss = loss_kt  + loss_q_all_lambda * loss_q_all + loss_c_all_lambda * loss_c_all + loss_c_next_lambda* loss_c_next + loss_q_next_lambda*loss_q_next
        # print(f"loss={loss:.3f},loss_kt={loss_kt:.3f},loss_q_all={loss_q_all:.3f},loss_c_all={loss_c_all:.3f},loss_q_next={loss_q_next:.3f},loss_c_next={loss_c_next:.3f}")
        return outputs['y'],loss#y_question没用


    def predict(self,dataset,batch_size,return_ts=False,process=True):
        test_loader = DataLoader(dataset, batch_size=batch_size,shuffle=False)
        self.model.eval()
        with torch.no_grad():
            y_trues = []
            y_pred_dict = {}
            for data in test_loader:
                new_data = self.batch_to_device(data,process=process)
                outputs,data_new = self.predict_one_step(data,return_details=True)
               
                for key in outputs:
                    if not key.startswith("y") or key in ['y_qc_predict']:
                        continue
                    elif key not in y_pred_dict:
                       y_pred_dict[key] = []
                    y = torch.masked_select(outputs[key], new_data['sm']).detach().cpu()#get label
                    y_pred_dict[key].append(y.numpy())
                
                t = torch.masked_select(new_data['rshft'], new_data['sm']).detach().cpu()
                y_trues.append(t.numpy())


        results = y_pred_dict
        for key in results:
            results[key] = np.concatenate(results[key], axis=0)
        ts = np.concatenate(y_trues, axis=0)
        results['ts'] = ts
        return results

    def evaluate(self,dataset,batch_size,acc_threshold=0.5):
        results = self.predict(dataset,batch_size=batch_size)
        eval_result = {}
        ts = results["ts"]
        for key in results:
            if not key.startswith("y") or key in ['y_qc_predict']:
                pass
            else:
                ps = results[key]
                kt_auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
                prelabels = [1 if p >= acc_threshold else 0 for p in ps]
                kt_acc = metrics.accuracy_score(ts, prelabels)
                if key!="y":
                    eval_result["{}_kt_auc".format(key)] = kt_auc
                    eval_result["{}_kt_acc".format(key)] = kt_acc
                else:
                    eval_result["auc"] = kt_auc
                    eval_result["acc"] = kt_acc
        
        self.eval_result = eval_result
        return eval_result

    def predict_one_step(self, data, return_details=False, process=True, return_raw=False):
        data_new = self.batch_to_device(data, process=process)
        outputs = self.model(data_new['cq'].long(), data_new['cc'], data_new['cr'].long(), data=data_new)
        
        # 原有固定权重
        output_c_all_lambda = self.model.other_config.get('output_c_all_lambda', 1)
        output_c_next_lambda = self.model.other_config.get('output_c_next_lambda', 1)
        output_q_all_lambda = self.model.other_config.get('output_q_all_lambda', 1)
        output_q_next_lambda = self.model.other_config.get('output_q_next_lambda', 0)  # not use this
        
        # 提取三个输出，便于动态加权
        y_q_all = outputs['y_question_all']
        y_c_all = outputs['y_concept_all']
        y_c_next = outputs['y_concept_next']
        
        # 根据 self.version 使用不同动态权重方法
        if self.version == "auto_uncertainty":
            # 方法1: 不确定性加权 - 计算动态权重（使用 exp(-log_var) 作为权重比例）
            log_vars = self.output_uncertainty.log_vars  # 可学习参数
            weights = torch.exp(-log_vars)  # [3] for q_all, c_all, c_next
            weights = weights / weights.sum()  # 归一化
            output_q_all_lambda, output_c_all_lambda, output_c_next_lambda = weights[0], weights[1], weights[2]
        
        elif self.version == "dynamic_attention":
            # 方法2: 注意力机制 - 将三个输出堆叠为序列，计算注意力权重
            ys = torch.stack([y_q_all.unsqueeze(0), y_c_all.unsqueeze(0), y_c_next.unsqueeze(0)], dim=0)  # [3, batch, seq]
            attn_output, attn_weights = self.attention_layer(ys, ys, ys)  # 注意: 假设embed_dim=1，需调整如果输出shape不同
            weights = attn_weights.mean(dim=1)[0]  # 平均注意力分数作为权重 [3]
            output_q_all_lambda, output_c_all_lambda, output_c_next_lambda = weights[0], weights[1], weights[2]
        
        elif self.version == "capsule_routing":
            # 方法3: 胶囊网络动态路由 - 简化实现
            # 假设低级胶囊 u = [y_q_all, y_c_all, y_c_next]，路由到1个高级胶囊
            u = torch.stack([y_q_all, y_c_all, y_c_next], dim=-1)  # [batch, seq, 3]
            b = torch.zeros_like(u)  # 初始 logits [batch, seq, 3]
            for _ in range(self.routing_iters):
                c = F.softmax(b, dim=-1)  # 耦合系数 [batch, seq, 3]
                s = (c * u).sum(dim=-1, keepdim=True)  # 加权和 [batch, seq, 1]
                v = self.squash(s)  # squash激活 [batch, seq, 1]
                a = (u * v).sum(dim=-2, keepdim=True)  # 协议更新 [batch, 1, 3]
                b = b + a
            weights = F.softmax(b.mean(dim=1), dim=-1).squeeze(1)  # 最终权重 [batch, 3]，但这里简化取mean
            output_q_all_lambda = weights[..., 0].mean()
            output_c_all_lambda = weights[..., 1].mean()
            output_c_next_lambda = weights[..., 2].mean()
        
        # else: 使用原有固定权重（不动态）
        
        # 应用权重到输出（兼容 an_irt 模式）
        if self.model.output_mode == "an_irt":
            def sigmoid_inverse(x, epsilon=1e-8):
                return torch.log(x / (1 - x + epsilon) + epsilon) if "no_sigmoid_inverse" not in self.version else x
            y = (sigmoid_inverse(y_q_all) * output_q_all_lambda +
                 sigmoid_inverse(y_c_all) * output_c_all_lambda +
                 sigmoid_inverse(y_c_next) * output_c_next_lambda)
            y = torch.sigmoid(y)
        else:
            y = (y_q_all * output_q_all_lambda +
                 y_c_all * output_c_all_lambda +
                 y_c_next * output_c_next_lambda)
            y = y / (output_q_all_lambda + output_c_all_lambda + output_c_next_lambda + 1e-8)  # 避免除零
        
        outputs['y'] = y

        if return_details:
            return outputs, data_new
        else:
            return y

    # 新增辅助函数（用于胶囊路由）
    def squash(self, x, dim=-1):
        squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / torch.sqrt(squared_norm + 1e-8)

# UncertaintyWeightedLoss 类保持原样...
class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))  # 可学习参数

    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + 0.5 * self.log_vars[i]
        return total_loss