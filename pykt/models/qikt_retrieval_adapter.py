"""Frozen QIKT plus strictly causal current-question history retrieval."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .qikt_mamba import QIKT_MAMBA


class HistoryRetrieval(nn.Module):
    def __init__(self,embedding_dim,mode,dropout=.1):
        super().__init__();self.mode=mode;self.dropout=dropout
        self.query=nn.Linear(embedding_dim,64,bias=False)
        self.key=nn.Linear(embedding_dim,64,bias=False)
        self.response=nn.Embedding(2,16)
        self.value=nn.Linear(80,64)
        self.hidden=nn.Linear(192,64);self.output=nn.Linear(64,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,question,responses,concepts):
        length=question.shape[1];query=self.query(question[:,1:]);key=self.key(question)
        value=self.value(torch.cat([key,self.response(responses.long().clamp(0,1))],-1))
        time=torch.arange(length,device=question.device)
        allowed=(time[None,:]<time[1:,None]).unsqueeze(0)
        known=((responses==0)|(responses==1))&(concepts>=0).any(-1)
        allowed=allowed&known.unsqueeze(1)
        scores=query@key.transpose(1,2)/8
        if self.mode!='content':
            a=concepts[:,1:,None,:,None];b=concepts[:,None,:,None,:]
            shared=((a==b)&(a>=0)&(b>=0)).any(-1).any(-1)
            scores=scores+2*shared.float()
        def retrieve(mask):
            weights=torch.softmax(scores.masked_fill(~mask,-1e9),-1)
            weights=torch.where(mask.any(-1,keepdim=True),weights,torch.zeros_like(weights))
            return F.dropout(weights,p=self.dropout,training=self.training)@value
        long=retrieve(allowed)
        short=(retrieve(allowed&(time[None,:]>=time[1:,None]-32).unsqueeze(0))
               if self.mode=='multi_scale' else torch.zeros_like(long))
        return self.output(F.gelu(self.hidden(torch.cat([query,long,short],-1)))).squeeze(-1)


class QIKTRetrievalAdapter(QIKT_MAMBA):
    def __init__(self,*args,other_config=None,**kwargs):
        config=dict(other_config or {});mode=config.get('retrieval_adapter_mode','off')
        if mode not in {'content','kc_bias','multi_scale'}:raise ValueError('Invalid retrieval mode')
        if config.get('output_mode')!='an_irt' or config.get('causal_adapter_mode','off')!='off' or any(config.get(k,0) for k in ['item_residual','concept_memory_residual','head_fusion_residual','history_response_dropout','pairwise_loss_weight','anchor_kl_weight']):
            raise ValueError('Retrieval isolates an unchanged an_irt backbone')
        config['freeze_backbone']=False
        super().__init__(*args,other_config=config,**kwargs)
        self.retrieval_scale=float(config.get('retrieval_adapter_scale',.25))
        self.retrieval_kl=float(config.get('retrieval_adapter_kl',0.))
        if not math.isfinite(self.retrieval_scale) or not 0<self.retrieval_scale<=1 or not math.isfinite(self.retrieval_kl) or self.retrieval_kl<0:
            raise ValueError('Invalid retrieval bounds')
        for parameter in self.parameters():parameter.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            self.model.retrieval_adapter=HistoryRetrieval(self.model.emb_size,mode).to(self.device)
        self.model.freeze_backbone=True;self.model.train(self.model.training);self._collect=False;self._anchor=None

    def predict_one_step(self,data,return_details=False,process=True,return_raw=False):
        outputs,current=super().predict_one_step(data,return_details=True,process=process)
        inverse=lambda value:torch.log(value/(1-value+1e-8)+1e-8)
        config=self.model.other_config
        baseline=(inverse(outputs['y_question_all'])*config.get('output_q_all_lambda',1)
                  +inverse(outputs['y_concept_all'])*config.get('output_c_all_lambda',1)
                  +inverse(outputs['y_concept_next'])*config.get('output_c_next_lambda',1)).detach()
        question=self.model.que_emb.que_emb(current['cq'].long()).detach()
        delta=self.retrieval_scale*torch.tanh(self.model.retrieval_adapter(question,current['cr'],current['cc']))
        student=baseline+delta;outputs['y']=torch.sigmoid(student)
        outputs['retrieval_delta']=delta
        if self._collect:self._anchor=(baseline,student,current['sm'].bool())
        return (outputs,current) if return_details else outputs['y']

    def train_one_step(self,data,process=True,return_all=False):
        self._collect=True
        try:
            prediction,loss=super().train_one_step(data,process=process,return_all=return_all)
            if self.retrieval_kl:
                baseline,student,mask=self._anchor;b=baseline[mask];s=student[mask];p=torch.sigmoid(b)
                loss=loss+self.retrieval_kl*((F.softplus(s)-p*s)-(F.softplus(b)-p*b)).mean()
            return prediction,loss
        finally:self._collect=False;self._anchor=None
