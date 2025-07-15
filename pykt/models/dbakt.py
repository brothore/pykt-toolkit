import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from torch.nn import LayerNorm
from torch.nn.parameter import Parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.variance_epsilon)
        return self.weight * x + self.bias

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)

    def forward(self, x):
        return self.conv(x)[:, :, :-(self.conv.padding[0])]  # 因果卷积，切掉右侧填充部分


class FrequencyLayer(nn.Module):
    def __init__(self, dropout, hidden_size, kernel_sizes):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(dropout)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        kernel_sizess = [kernel_sizes, kernel_sizes+2]
        # 创建多个因果卷积层
        self.causal_convs = nn.ModuleList([
            CausalConv1d(hidden_size, hidden_size, ks) for ks in kernel_sizess
        ])
        
        
        # 可学习的权重参数用于合并
        self.combine_weights = nn.Parameter(torch.randn(len(kernel_sizess)))
        self.sigmoid_input=nn.Linear(hidden_size+hidden_size,hidden_size)
        self.tanh_iuput=nn.Linear(hidden_size,hidden_size)
    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        
        # 准备输入以适应卷积操作
        input_tensor_conv = input_tensor.permute(0, 2, 1)  # [batch, hidden, seq_len]
        
        # 存储每个卷积层的输出
        sequence_emb_fft_list = []
        
        for i, causal_conv in enumerate(self.causal_convs):
            # 低通滤波
            low_pass = causal_conv(input_tensor_conv)  # [batch, hidden, seq_len]
            low_pass = low_pass.permute(0, 2, 1)  # [batch, seq_len, hidden]
            
            # # 高通滤波
            high_pass = input_tensor - low_pass  # [batch, seq_len, hidden]
            
            #########
            X=torch.concat([input_tensor,high_pass],dim=-1)
            low_forget_gate = torch.sigmoid(self.sigmoid_input(X))
            high_candidate = torch.tanh(self.tanh_iuput(high_pass))
            low_output = low_pass*low_forget_gate#决定在长期信息里保留多少信息
            high_output= (torch.ones(low_output.shape).to(device)-low_forget_gate) * high_candidate
            sequence_emb_fft=low_output+high_output
            #############
            
            sequence_emb_fft_list.append(sequence_emb_fft)
        
        # 将多个卷积层的输出堆叠 [num_kernels, batch, seq_len, hidden]
        sequence_emb_fft_stack = torch.stack(sequence_emb_fft_list, dim=0)
        
        # 对合并权重进行softmax归一化
        combine_weights = torch.softmax(self.combine_weights, dim=0)  # [num_kernels]
        combine_weights = combine_weights.view(-1, 1, 1, 1)  # [num_kernels, 1, 1, 1]
        
        # 加权求和合并
        combined_sequence_emb_fft = (combine_weights * sequence_emb_fft_stack).sum(dim=0)  # [batch, seq_len, hidden]
        
        # Dropout和层归一化
        hidden_states = self.out_dropout(combined_sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        
        return hidden_states

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class timeGap2(nn.Module):
    def __init__(self, num_rgap, num_sgap, num_pcount, emb_size) -> None:
        super().__init__()
        self.num_rgap, self.num_sgap, self.num_pcount = num_rgap, num_sgap, num_pcount
        if num_rgap != 0:
            self.rgap_eye = torch.eye(num_rgap)
        if num_sgap != 0:
            self.sgap_eye = torch.eye(num_sgap)
        if num_pcount != 0:
            self.pcount_eye = torch.eye(num_pcount)

        input_size = num_rgap + num_sgap + num_pcount
        
        print(f"self.num_rgap: {self.num_rgap}, self.num_sgap: {self.num_sgap}, self.num_pcount: {self.num_pcount}, input_size: {input_size}")

        self.time_emb = nn.Linear(input_size, emb_size, bias=False)

    def forward(self, rgap, sgap, pcount):
        infs = []
        if self.num_rgap != 0:
            rgap = self.rgap_eye[rgap].to(device)
            infs.append(rgap)
        if self.num_sgap != 0:
            sgap = self.sgap_eye[sgap].to(device)
            infs.append(sgap)
        if self.num_pcount != 0:
            pcount = self.pcount_eye[pcount].to(device)
            infs.append(pcount)

        tg = torch.cat(infs, -1)
        tg_emb = self.time_emb(tg)

        return tg_emb

class DBAKT(nn.Module):
    def __init__(self, n_question, n_pid, num_rgap, num_sgap, num_pcount, 
            d_model, n_blocks, dropout, d_ff=256, 
            loss1=0.5, loss2=0.5, loss3=0.5, start=50, num_layers=2, nheads=4, seq_len=200, kernel_size=3, freq=True,
            kq_same=1, final_fc_dim=512, final_fc_dim2=256, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "dbakt"
        print(f"model_name: {self.model_name}, emb_type: {emb_type}")
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_l = d_model
        self.n_blocks=n_blocks
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, embed_l) # 题目难度
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) # question emb, 总结了包含当前question（concept）的problems（questions）的变化
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l) # interaction emb, 同上

        if emb_type.startswith("qid"):
            # n_question+1 ,d_model
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa: 
                    self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
            else: # false default
                self.qa_embed = nn.Embedding(2, embed_l)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len, emb_type=self.emb_type, kernel_size=kernel_size, freq=freq,time=False).to(device)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                    final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )

        if  "single" not in emb_type:
            self.c_weight = nn.Linear(d_model, d_model)
            self.t_weight = nn.Linear(d_model, d_model)
            self.time_emb = timeGap(num_rgap, num_sgap, num_pcount, d_model)
            self.model2 = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads,
                                    dropout=dropout, d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,
                                    kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len,emb_type=emb_type, kernel_size=kernel_size, freq=freq,time=True)
        
        #如果是单分支且不使用时间信息，则不需要添加时间序列的信息
        elif "single"  in emb_type:
            self.time_emb = timeGap(num_rgap, num_sgap, num_pcount, d_model)
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  d_model# c_ct
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(target)+q_embed_data
        return q_embed_data, qa_embed_data

    def get_attn_pad_mask(self, sm):
        batch_size, l = sm.size()
        pad_attn_mask = sm.data.eq(0).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, l, l)
        return pad_attn_mask.repeat(self.nhead, 1, 1)

    def forward(self, dcur, dgaps, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((r[:,0:1], rshft), dim=1)
        emb_type = self.emb_type
        q_data = q_data.to(device)
        target = target.to(device)
        
        rg, sg, p = dgaps["rgaps"].long(), dgaps["sgaps"].long(), dgaps["pcounts"].long()
        rgshft, sgshft, pshft = dgaps["shft_rgaps"].long(), dgaps["shft_sgaps"].long(), dgaps["shft_pcounts"].long()
        r_gaps = torch.cat((rg[:, 0:1], rgshft), dim=1)
        s_gaps = torch.cat((sg[:, 0:1], sgshft), dim=1)
        pcounts = torch.cat((p[:, 0:1], pshft), dim=1)
        # Batch Firs
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)#KC层面的嵌入
        if self.n_pid > 0: # have problem id
            q_embed_diff_data = self.q_embed_diff(q_data)  # KC的难度
            pid_embed_data = self.difficult_param(pid_data.to(device))  # uq 当前question的难度
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # uq *d_ct + c_ct # question encoder

        if "single" not in emb_type:
            #时间信息分支的注意力网络
            temb = self.time_emb(r_gaps, s_gaps, pcounts)#输出三个时间特征的混合表示
            #是否选择将时间信息代替kernel_bias的相对位置信息
            if "time" in self.emb_type:
                timestamps = torch.cumsum(s_gaps, dim=1).long().to(device)
                time_diff = torch.abs(timestamps[:, :, None] - timestamps[:, None, :])  # 
            else:
                time_diff = None
                
            t_out = self.model2(temb, qa_embed_data,time_diff=time_diff)
        #如果是单分支，则直接将混合时间表示加到答题信息表示上
        elif "single" in emb_type :
            #是否选择将时间信息代替kernel_bias的相对位置信息
            if "time" in self.emb_type:
                timestamps = torch.cumsum(s_gaps, dim=1).long().to(device)
                time_diff = torch.abs(timestamps[:, :, None] - timestamps[:, None, :])  # 
            else:
                time_diff = None
            temb = self.time_emb(r_gaps, s_gaps, pcounts)#输出三个时间特征的混合表示
        y2, y3 = 0, 0
        
        if "single" not in emb_type:
            d_output = self.model(q_embed_data, qa_embed_data,time_diff=time_diff)

            w = torch.sigmoid(self.c_weight(d_output) + self.t_weight(t_out)) # w = sigmoid(基本信息编码 + 时间信息编码)，每一维设置为0-1之间的数值
            d_output = w * d_output + (1 - w) * t_out # 每一维加权平均后的综合信息
            q_embed_data = q_embed_data + temb # 原始的题目信息和时间信息

            concat_q = torch.cat([d_output, q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)
        elif "single" in emb_type :
            q_embed_data = q_embed_data + temb
            d_output = self.model(q_embed_data, qa_embed_data,time_diff=time_diff)
            # 原始的题目信息和时间信息
            concat_q = torch.cat([d_output, q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)
            
        if train:
            return preds, y2, y3
        else:
            if qtest:
                return preds, concat_q
            else:
                return preds

class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                d_ff, n_heads, dropout, kq_same, model_type, seq_len, emb_type, kernel_size, freq,time):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type
        self.emb_type =emb_type
        self.freq = freq
        self.time=time

        self.blocks_2 = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                            d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same,emb_type=emb_type,time=time)
            for _ in range(n_blocks)])
        
        if "true" in self.emb_type:
            self.filter_layer = FrequencyLayer(dropout,d_model, kernel_size)

    def forward(self, q_embed_data, qa_embed_data,time_diff=None):
        
        y = qa_embed_data
        x = q_embed_data

        if "true" in self.emb_type:
            x = self.filter_layer(x)
            y = self.filter_layer(y)
        
        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True,time_diff=time_diff) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
        return x

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                d_ff, n_heads, dropout,  kq_same,emb_type,time=True):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        self.time = time
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same,emb_type=emb_type,time=self.time)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, mask, query, key, values, apply_pos=True,time_diff=None):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True,time_diff=time_diff) # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False,time_diff=time_diff)
        query = query + self.dropout1((query2)) # 残差1
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2)) # 残差
            query = self.layer_norm2(query) # lay norm
        return query 


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, emb_type="qidtrue",bias=True,time=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.emb_type=emb_type
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        # 只在答题序列上添加kernelbias
        if emb_type.find("ker") != -1 and "time" in self.emb_type:
            self.kernel_bias_timediff = ParallelKerpleLogTimediff(n_heads)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad,time_diff=None):

        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)#BS*seq_len*d_model -->BS*seq_len*num_heads*d_model/num_heads
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        if self.emb_type.find("ker") != -1 and "time" in self.emb_type :                      
            scores = attention(q, k, v, self.d_k,mask, self.dropout, zero_pad,kernel_bias_timediff=self.kernel_bias_timediff,
                            time_diff=time_diff)
        else:
            #输出注意力分数矩阵
            scores = attention(q, k, v, self.d_k,mask, self.dropout, zero_pad)
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output
        


def attention(q, k, v, d_k, mask, dropout, zero_pad,kernel_bias_timediff=None
            ,time_diff=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    if time_diff != None:
        scores = kernel_bias_timediff(scores,time_diff=time_diff)
    
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output
    


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)

class timeGap(nn.Module):
    def __init__(self, num_rgap, num_sgap, num_pcount, emb_size) -> None:
        super().__init__()
        self.rgap_eye = torch.eye(num_rgap)#生成一个主对角线为1的单位矩阵，维度为num_rgap
        self.sgap_eye = torch.eye(num_sgap)
        self.pcount_eye = torch.eye(num_pcount)

        input_size = num_rgap + num_sgap + num_pcount

        self.time_emb = nn.Linear(input_size, emb_size, bias=False)

    def forward(self, rgap, sgap, pcount):
  
        rgap = self.rgap_eye[rgap].to(device)#其实就相当于one-hot表示
        sgap = self.sgap_eye[sgap].to(device)
        pcount = self.pcount_eye[pcount].to(device)

        tg = torch.cat((rgap, sgap, pcount), -1)
        tg_emb = self.time_emb(tg)

        return tg_emb

  
class ParallelKerpleLogTimediff(nn.Module):
    """Kernel Bias"""
    def __init__(self, num_attention_heads):
        super().__init__()
        self.heads = num_attention_heads  # int
        self.num_heads_per_partition = self.heads  # int
        # self.pos_emb = pos_emb  # str
        self.eps = 1e-2
        
        # Allocate weights and initialize.
        # The kernel has the form -p*log(1+a*|m-n|)
        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return Parameter(torch.ones(
                    self.num_heads_per_partition,
                    dtype=torch.float32,
                )[:, None, None] * scale)
            elif init_method == 'uniform':
                return Parameter(torch.rand(
                    self.num_heads_per_partition,
                    dtype=torch.float32,
                )[:, None, None] * scale)
        
        self.bias_p = get_parameter(2, 'uniform')
        self.bias_a = get_parameter(1, 'uniform')
    
    def stats(self):
        def get_stats(name, obj):
            return {
                name + '_mean': obj.mean().detach().cpu(),
                name + '_std': obj.std().detach().cpu(),
                name + '_max': obj.max().detach().cpu(),
                name + '_min': obj.min().detach().cpu()
            }
        dd = {}
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        dd.update(get_stats('bias_a', self.bias_a))
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        dd.update(get_stats('bias_p', self.bias_p))
        return dd
    
    def forward(self, x, time_diff=None):
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        #如果时间信息不为空，即使用时间信息代替相对位置信息
        
        time_diff_expanded = time_diff.unsqueeze(1)
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        bias_a_expanded = self.bias_a.view(1, self.num_heads_per_partition, 1, 1)
        bias_p_expanded = self.bias_p.view(1, self.num_heads_per_partition, 1, 1)
        bias = -bias_p_expanded * torch.log(1 + bias_a_expanded * time_diff_expanded)  # log kernel
        
        if seq_len_q != seq_len_k:  
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            
            if not isinstance(bias, float):
                bias = bias[:, seq_len_k - 1, :].view(bias.shape[0], 1, bias.shape[2])
        return x + bias

