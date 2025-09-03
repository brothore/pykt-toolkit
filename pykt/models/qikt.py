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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from mamba_ssm import Mamba

class TransformerBranch(nn.Module):
    """
    A flexible Transformer class that can operate in three modes:
    - 'encoder': Only the encoder branch, now modified to support causality (causal masking) to imitate LSTM-like behavior.
    - 'decoder': Only the decoder branch (supports optional memory for cross-attention; if no memory, acts as a pure autoregressive decoder).
    - 'full': Complete encoder-decoder Transformer.

    Inputs and outputs are expected in shape: (batch, seq_len, emb_size).
    Uses PyTorch's built-in Transformer modules for simplicity.

    Parameters:
    - mode: str, one of 'encoder', 'decoder', 'full'.
    - emb_size_in: int, input embedding dimension.
    - emb_size_out: int, output embedding dimension (if different from emb_size_in, a projection layer is added).
    - n_layers: int, number of layers (default: 6).
    - n_heads: int, number of attention heads (default: 8).
    - ff_dim: int, feed-forward hidden dimension (default: 2048).
    - dropout: float, dropout rate (default: 0.1).
    - causal: bool, whether to apply causal masking in encoder mode (default: False for standard bidirectional; set to True for LSTM-like causality).
    """
    def __init__(self, mode: str, emb_size_in: int, emb_size_out: int, n_layers: int = 6, n_heads: int = 8, ff_dim: int = 2048, dropout: float = 0.1, causal: bool = False):
        super(TransformerBranch, self).__init__()
        self.mode = mode
        self.emb_size_in = emb_size_in
        self.emb_size_out = emb_size_out
        self.causal = causal  # New parameter to toggle causality in encoder

        # Validate mode
        if mode not in ['encoder', 'decoder', 'full']:
            raise ValueError("Mode must be one of 'encoder', 'decoder', or 'full'.")

        # Encoder (used in 'encoder' and 'full' modes)
        if mode in ['encoder', 'full']:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=emb_size_in,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='relu'  # Default in PyTorch
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Decoder (used in 'decoder' and 'full' modes)
        if mode in ['decoder', 'full']:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=emb_size_in,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='relu'
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Optional projection layer if emb_size_out differs
        self.proj = nn.Linear(emb_size_in, emb_size_out) if emb_size_in != emb_size_out else None

    def _generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        Generate a causal mask for the sequence: upper triangle is -inf, lower is 0.
        Shape: (seq_len, seq_len)
        """
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src: torch.Tensor = None, tgt: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass based on the mode.

        - For 'encoder': Requires src (batch, seq_len, emb_size_in). Returns (batch, seq_len, emb_size_out).
          If self.causal is True, applies causal masking to imitate LSTM-like sequential processing.
        - For 'decoder': Requires tgt (batch, seq_len, emb_size_in). Optional src as memory for cross-attention.
          If src is None, acts as pure decoder (self-attention only, with causal mask).
          Returns (batch, seq_len, emb_size_out).
        - For 'full': Requires src and tgt. Encodes src, then decodes tgt with encoder output as memory.
          Returns (batch, seq_len, emb_size_out).

        Note: PyTorch Transformer modules expect input in (seq_len, batch, emb_size), so we transpose internally.
        Causal masking is automatically handled in the decoder; for encoder, it's optional via self.causal.
        """
        if self.mode == 'encoder':
            if src is None:
                raise ValueError("src is required for 'encoder' mode.")
            # Transpose to (seq_len, batch, emb)
            src = src.transpose(0, 1)
            seq_len = src.size(0)
            mask = self._generate_causal_mask(seq_len) if self.causal else None
            out = self.encoder(src, mask=mask)
            # Transpose back to (batch, seq_len, emb)
            out = out.transpose(0, 1)
        elif self.mode == 'decoder':
            if tgt is None:
                raise ValueError("tgt is required for 'decoder' mode.")
            tgt = tgt.transpose(0, 1)
            memory = src.transpose(0, 1) if src is not None else None
            # Decoder handles causal mask internally (tgt_mask=None implies causal)
            out = self.decoder(tgt, memory)
            out = out.transpose(0, 1)
        elif self.mode == 'full':
            if src is None or tgt is None:
                raise ValueError("Both src and tgt are required for 'full' mode.")
            src = src.transpose(0, 1)
            tgt = tgt.transpose(0, 1)
            # For full mode, encoder remains bidirectional unless causal is set (but typically not for full Transformer)
            seq_len = src.size(0)
            enc_mask = self._generate_causal_mask(seq_len) if self.causal else None
            memory = self.encoder(src, mask=enc_mask)
            out = self.decoder(tgt, memory)
            out = out.transpose(0, 1)
        else:
            raise ValueError("Invalid mode.")

        # Apply projection if needed
        if self.proj is not None:
            out = self.proj(out)

        return out

class DebugLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1,debug_print=1):
        super(DebugLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self. debug_print = debug_print
        self.valid_indices = 0
        
    def forward(self, x):
        """
        简化版前向传播，不需要lengths参数
        x: 输入张量，形状为(batch_size, seq_len, input_size)
        """
        print(f"[DEBUG] x[0]: {x[0]} (type: {type(x[0])})")
        self.valid_indices = [i for i in range(x[0].size(0)) if not torch.all(x[0][i] == -1)]
        # 标准LSTM前向传播
        output, (hidden, cell) = self.lstm(x)
        
        # 计算隐藏状态距离
        if self.debug_print:
            self._analyze_hidden_states(output)
        
        return output, (hidden, cell)
    
    def _analyze_hidden_states(self, hidden_states, padding_value=-1):
        """
        分析隐藏状态的变化，跳过填充值部分
        hidden_states: (batch_size, seq_len, hidden_size)
        padding_value: 用于标识填充位置的值
        """
        if hidden_states.dim() == 3:
            h = hidden_states[0]  # 取第一个样本 (seq_len, hidden_size)
        else:
            h = hidden_states
        
        # 找出非填充位置的有效索引
        valid_indices = self.valid_indices
        valid_seq = h[valid_indices]
        valid_len = len(valid_indices)
        
        print(f"\n===== LSTM隐藏状态分析 (有效长度: {valid_len}/{h.size(0)}) =====")
        
        if valid_len == 0:
            print("警告: 所有时间步都是填充值!")
            return
        
        # 预计算所有需要的距离
        distance_cache = {}
        for i in range(valid_len):
            for j in range(i, valid_len):  # 只计算上三角部分
                distance_cache[(i,j)] = self._compute_distance(valid_seq[i], valid_seq[j])
        
        # 1. 相邻时间步距离
        if valid_len > 1:
            print("\n相邻时间步距离 (有效步):")
            total_euclidean = 0
            total_cosine = 0
            for t in range(1, valid_len):
                e, c = distance_cache[(t-1, t)]
                orig_idx = f"{valid_indices[t-1]}->{valid_indices[t]}"
                print(f"t={orig_idx}: 欧氏={e:.4f}, 余弦距离={c:.4f}")
                total_euclidean += e
                total_cosine += c
        
        # # 2. 距离矩阵（仅显示有效部分）
        # if valid_len > 1:
        #     print("\n欧氏距离矩阵 (有效步):")
        #     euclidean_matrix = torch.zeros((valid_len, valid_len))
        #     cosine_matrix = torch.zeros((valid_len, valid_len))
        #     for i in range(valid_len):
        #         for j in range(valid_len):
        #             e, c = distance_cache.get((i,j), distance_cache.get((j,i)))
        #             euclidean_matrix[i,j] = e
        #             cosine_matrix[i,j] = c
            
            # # 打印时带上原始索引
            # header = " " * 6 + " ".join([f"{valid_indices[j]:>5}" for j in range(valid_len)])
            # print(header)
            # for i in range(valid_len):
            #     row = [f"{euclidean_matrix[i,j]:.2f}" for j in range(valid_len)]
            #     print(f"{valid_indices[i]:>4} [" + " ".join(f"{x:>5}" for x in row) + "]")
            
            # print("\n余弦距离矩阵 (有效步):")
            # print(header)
            # for i in range(valid_len):
            #     row = [f"{cosine_matrix[i,j]:.2f}" for j in range(valid_len)]
            #     print(f"{valid_indices[i]:>4} [" + " ".join(f"{x:>5}" for x in row) + "]")
        
        # 3. 首尾距离（从缓存获取）
        if valid_len > 1:
            e, c = distance_cache[(0, valid_len-1)]
            orig_idx = f"{valid_indices[0]}->{valid_indices[-1]}"
            print(f"\n首尾时间步距离 (t={orig_idx}): 欧氏={e:.4f}, 余弦距离={c:.4f}")
            
            # 4. 平均变化
            avg_euclidean = total_euclidean / (valid_len - 1)
            avg_cosine = total_cosine / (valid_len - 1)
            print(f"\n平均相邻时间步变化: 欧氏={avg_euclidean:.4f}, 余弦距离={avg_cosine:.4f}")
    def _compute_distance(self, h1, h2):
        """返回元组(欧氏距离, 余弦距离)而不是字符串"""
        euclidean = torch.norm(h1 - h2, p=2)
        cosine = 1 - torch.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0))
        return euclidean.item(), cosine.item()  # 返回两个数值

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
    def __init__(self, num_q,num_c,emb_size, dropout=0.1, emb_type='qaid', emb_path="", pretrain_dim=768,device='cpu',mlp_layer_num=1,other_config={},qikt_type=None):
        super().__init__()
        self.qikt_type = qikt_type
        self.model_name = "qikt"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.mlp_layer_num = mlp_layer_num
        self.device = device
        self.other_config = other_config
        self.output_mode = self.other_config.get('output_mode','an')


        self.emb_type = emb_type
      

        self.que_emb = QueEmb(num_q=num_q,num_c=num_c,emb_size=emb_size,emb_type=self.emb_type,model_name=self.model_name,device=device,
                             emb_path=emb_path,pretrain_dim=pretrain_dim)
        if self.qikt_type == "mamba":
            pass
            self.que_lstm_layer = Mamba(d_model=self.emb_size*4, d_state=self.hidden_size) 
            self.concept_lstm_layer = Mamba(d_model=self.emb_size*2, d_state=self.hidden_size)
            self.que_proj = nn.Linear(self.emb_size*4, self.hidden_size)  # 1024 -> 256
            self.concept_proj = nn.Linear(self.emb_size*2, self.hidden_size)  # 512 -> 256
        elif self.qikt_type in ["encoder",'decoder','full']:
            #transformer
            self.que_lstm_layer = TransformerBranch(emb_size_in=self.emb_size*4, emb_size_out=self.hidden_size,mode=self.qikt_type, causal=True)
            self.concept_lstm_layer = TransformerBranch(emb_size_in=self.emb_size*2, emb_size_out=self.hidden_size,mode=self.qikt_type, causal=True)
        elif self.qikt_type == "unzip_lstm":
            self.four2two = nn.Linear(self.emb_size*4, self.emb_size*2)
            self.que_lstm_layer = nn.LSTM(self.emb_size*2, self.hidden_size, batch_first=True)
            self.concept_lstm_layer = nn.LSTM(self.emb_size*2, self.hidden_size, batch_first=True)
        elif self.qikt_type == "zip_lstm":
            # 压缩lstm
            self.four2two = nn.Linear(self.emb_size*2, self.emb_size*4)
            self.que_lstm_layer = nn.LSTM(self.emb_size*4, self.hidden_size, batch_first=True)
            self.concept_lstm_layer = nn.LSTM(self.emb_size*4, self.hidden_size, batch_first=True)
        elif self.qikt_type == "public_lstm":
            #共用lstm
            self.four2two = nn.Linear(self.emb_size*2, self.emb_size*4)
            self.que_lstm_layer = nn.LSTM(self.emb_size*4, self.hidden_size, num_layers=2, batch_first=True)
            self.concept_lstm_layer = self.que_lstm_layer
        elif self.qikt_type == "public_mamba":
            #共用lstm
            self.four2two = nn.Linear(self.emb_size*2, self.emb_size*4)
            self.que_lstm_layer = Mamba(d_model=self.emb_size*4, d_state=self.hidden_size) 
            self.que_proj = nn.Linear(self.emb_size*4, self.hidden_size)  # 1024 -> 256
            self.concept_proj = nn.Linear(self.emb_size*4, self.hidden_size)  # 512 -> 256
            self.concept_lstm_layer = self.que_lstm_layer
        elif self.qikt_type == "public_lstm_large":
            #共用lstm
            self.four2two = nn.Linear(self.emb_size*2, self.emb_size*4)
            self.que_lstm_layer = nn.LSTM(self.emb_size*4, self.hidden_size, num_layers=2, batch_first=True)
            self.concept_lstm_layer = self.que_lstm_layer
        else:
            #原版
            self.que_lstm_layer = nn.LSTM(self.emb_size*4, self.hidden_size, batch_first=True)
            self.concept_lstm_layer = nn.LSTM(self.emb_size*2, self.hidden_size, batch_first=True)

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

        _, emb_qca, emb_qc, emb_q, emb_c = self.que_emb(q, c, r)  # [batch_size,emb_size*4],[batch_size,emb_size*2],...
        
        emb_qc_shift = emb_qc[:, 1:, :]
        emb_qca_current = emb_qca[:, :-1, :]
        # question model
        if self.qikt_type == "mamba":
            #mamba
            que_h = self.dropout_layer(self.que_lstm_layer(emb_qca_current))
            que_h = self.que_proj(que_h)
        elif self.qikt_type in ["encoder",'decoder','full']:
            #transformer
            que_h = self.dropout_layer(self.que_lstm_layer(emb_qca_current))
        elif self.qikt_type == "zip_lstm":
            # 压缩lstm
            que_h = self.dropout_layer(self.que_lstm_layer((emb_qca_current))[0])
        elif self.qikt_type == "unzip_lstm":
            # 压缩lstm
            que_h = self.dropout_layer(self.que_lstm_layer(self.four2two(emb_qca_current))[0])
        elif self.qikt_type == "public_lstm":

            #共用lstm
            que_h = self.dropout_layer(self.que_lstm_layer((emb_qca_current))[0])
        elif self.qikt_type == "public_mamba":

            que_h = self.dropout_layer(self.que_lstm_layer((emb_qca_current)))
            que_h = self.que_proj(que_h)
        else:
            #原版
            que_h = self.dropout_layer(self.que_lstm_layer(emb_qca_current)[0])
        print(f"[DEBUG] que_h.shape: {que_h.shape} (type: {type(que_h.shape)})")
        que_outputs = get_outputs(self, emb_qc_shift, que_h, data, add_name="", model_type="question")
        outputs = que_outputs

        # concept model
        emb_ca = torch.cat([
            emb_c.mul((1 - r).unsqueeze(-1).repeat(1, 1, self.emb_size)),
            emb_c.mul(r.unsqueeze(-1).repeat(1, 1, self.emb_size))
        ], dim=-1)
        
        emb_ca_current = emb_ca[:, :-1, :]
        if self.qikt_type == "mamba":
            #mamba
            concept_h = self.concept_lstm_layer(emb_ca_current)  # [32, 199, 512]
            concept_h = self.concept_proj(concept_h)  # [32, 199, 256]
            concept_h = self.dropout_layer(concept_h)
        elif self.qikt_type in ["encoder",'decoder','full']:
            #transformer
            concept_h = self.dropout_layer(self.concept_lstm_layer(emb_ca_current))
        elif self.qikt_type == "zip_lstm":
            # 压缩lstm
            concept_h = self.dropout_layer(self.concept_lstm_layer(self.four2two(emb_ca_current))[0])
        elif self.qikt_type == "unzip_lstm":
            # 压缩lstm
            concept_h = self.dropout_layer(self.concept_lstm_layer((emb_ca_current))[0])
        elif self.qikt_type == "public_lstm":
            #共用lstm
            concept_h = self.dropout_layer(self.concept_lstm_layer(self.four2two(emb_ca_current))[0])

        elif self.qikt_type == "public_mamba":
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

class QIKT(QueBaseModel):
    def __init__(self, num_q,num_c, emb_size, dropout=0.1, emb_type='qaid', emb_path="", pretrain_dim=768,device='cpu',seed=0,mlp_layer_num=1,other_config={},qikt_type=None,**kwargs):
        model_name = "qikt"
       
        debug_print(f"emb_type is {emb_type}",fuc_name="QIKT")

        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)
        self.model = QIKTNet(num_q=num_q,num_c=num_c,emb_size=emb_size,dropout=dropout,emb_type=emb_type,
                               emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,mlp_layer_num=mlp_layer_num,other_config=other_config,qikt_type=qikt_type)
       
        self.model = self.model.to(device)
        self.emb_type = self.model.emb_type
        self.loss_func = self._get_loss_func("binary_crossentropy")
        self.eval_result = {}
        


    def train_one_step(self,data,process=True,return_all=False):
        outputs,data_new = self.predict_one_step(data,return_details=True,process=process)
        # all 
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
            loss = loss_kt  + loss_q_all_lambda * loss_q_all + loss_c_all_lambda * loss_c_all+ loss_c_next_lambda* loss_c_next
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

    def predict_one_step(self,data,return_details=False,process=True,return_raw=False):
        data_new = self.batch_to_device(data,process=process)
        outputs = self.model(data_new['cq'].long(),data_new['cc'],data_new['cr'].long(),data=data_new)
        output_c_all_lambda = self.model.other_config.get('output_c_all_lambda',1)
        output_c_next_lambda = self.model.other_config.get('output_c_next_lambda',1)
        output_q_all_lambda = self.model.other_config.get('output_q_all_lambda',1)
        output_q_next_lambda = self.model.other_config.get('output_q_next_lambda',0)#not use this
       
        if self.model.output_mode=="an_irt":
            def sigmoid_inverse(x,epsilon=1e-8):
                return torch.log(x/(1-x+epsilon)+epsilon)
            y = sigmoid_inverse(outputs['y_question_all'])*output_q_all_lambda + sigmoid_inverse(outputs['y_concept_all'])*output_c_all_lambda + sigmoid_inverse(outputs['y_concept_next'])*output_c_next_lambda
            y = torch.sigmoid(y)
        else:
            # output weight
            y = outputs['y_question_all'] * output_q_all_lambda + outputs['y_concept_all'] * output_c_all_lambda + outputs['y_concept_next'] * output_c_next_lambda
            y = y/(output_q_all_lambda + output_c_all_lambda + output_c_next_lambda)
        outputs['y'] = y

        if return_details:
            return outputs,data_new
        else:
            return y