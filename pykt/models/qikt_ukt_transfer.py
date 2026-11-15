"""UKT-inspired causal stochastic retrieval beside two unchanged GRU branches."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .qikt_mamba import QIKT_MAMBA


def diagonal_w2(mean, variance):
    """Squared 2-Wasserstein distance between all diagonal Gaussian pairs."""
    if mean.shape != variance.shape or bool((variance <= 0).any()):
        raise ValueError('Invalid diagonal Gaussian parameters')
    def squared_distance(x):
        norm=x.square().sum(-1,keepdim=True)
        return (norm+norm.transpose(-1,-2)-2*x@x.transpose(-1,-2)).clamp_min(0)
    return squared_distance(mean)+squared_distance(variance.sqrt())


class UKTReadout(nn.Module):
    def __init__(self, input_dim, mode, width=64):
        super().__init__()
        if mode not in {'mean','w2','w2_legacy'}:raise ValueError('Unknown UKT readout mode')
        self.mode=mode;self.width=width
        self.mean=nn.Linear(input_dim,width)
        self.cov=nn.Linear(input_dim,width)
        self.response_mean=nn.Embedding(2,width)
        self.response_cov=nn.Embedding(2,width)
        self.gamma=nn.Parameter(torch.zeros(()))
        self.out=nn.Sequential(nn.Linear(4*width,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.out[-1].weight);nn.init.zeros_(self.out[-1].bias)

    def forward(self, embedding, responses, valid):
        mu=self.mean(embedding)
        variance=F.elu(self.cov(embedding))+1+1e-6 if self.mode!='mean' else torch.ones_like(mu)
        safe=responses.clamp(0,1).long()
        value_mean=mu+self.response_mean(safe)
        value_variance=F.elu(self.cov(embedding)+self.response_cov(safe))+1+1e-6 if self.mode!='mean' else torch.ones_like(mu)
        length=embedding.shape[1]
        allowed=torch.ones(length,length,dtype=torch.bool,device=embedding.device).tril(-1)
        allowed=allowed.unsqueeze(0)&valid[:,None,:]&valid[:,:,None]
        score=-diagonal_w2(mu,variance)/math.sqrt(self.width)
        if self.mode=='w2_legacy':
            with torch.no_grad():
                initial=score.masked_fill(~allowed,-1e9).softmax(-1)*allowed
                gap=(torch.arange(length,device=embedding.device)[:,None]-torch.arange(length,device=embedding.device)[None,:]).abs()
                distance=((initial.sum(-1,keepdim=True)-initial.cumsum(-1))*gap).clamp_min(0).sqrt()
            score=score*torch.exp(-F.softplus(self.gamma)*distance).clamp(1e-5,1e5)
        attention=score.masked_fill(~allowed,-1e9).softmax(-1)*allowed
        attention=attention/attention.sum(-1,keepdim=True).clamp_min(1e-8)
        context_mean=attention@value_mean
        context_variance=attention.square()@value_variance
        features=torch.cat([mu,variance.sqrt(),context_mean,context_variance.clamp_min(1e-8).sqrt()],-1)
        delta=self.out(features).squeeze(-1)*allowed.any(-1)
        return delta,context_variance,attention


class QIKTUKTTransfer(QIKT_MAMBA):
    def __init__(self,*args,other_config=None,**kwargs):
        cfg=dict(other_config or {})
        mode=cfg.get('ukt_transfer_mode','off')
        if kwargs.get('version') not in {'gru','lstm'} or cfg.get('output_mode')!='an_irt':
            raise ValueError('UKT transfer requires the independent dual-GRU an_irt baseline')
        if any(cfg.get(k,0) for k in ['freeze_backbone','item_residual','concept_memory_residual','head_fusion_residual','pairwise_loss_weight','history_response_dropout']):
            raise ValueError('UKT V1 isolates the transfer mechanism on trainable GRUs')
        super().__init__(*args,other_config=cfg,**kwargs)
        self.ukt_scale=float(cfg.get('ukt_transfer_scale',.1))
        self.ukt_fusion=cfg.get('ukt_transfer_fusion','additive')
        if not math.isfinite(self.ukt_scale) or not 0<self.ukt_scale<=1 or self.ukt_fusion not in {'additive','uncertainty'}:
            raise ValueError('Invalid UKT fusion settings')
        if not isinstance(self.model.que_lstm_layer,nn.GRU) or self.model.que_lstm_layer is self.model.concept_lstm_layer:
            raise ValueError('Two independent GRUs are required')
        freeze=bool(cfg.get('ukt_transfer_freeze',False))
        if freeze:
            for p in self.parameters():p.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            self.model.ukt_transfer=UKTReadout(2*self.model.emb_size,mode).to(self.device)
        self.model.freeze_backbone=freeze
        self.model.train(self.model.training)

    def predict_one_step(self,data,return_details=False,process=True,return_raw=False):
        outputs,current=super().predict_one_step(data,return_details=True,process=process)
        qe=self.model.que_emb
        question=qe.que_emb(current['cq'].long())
        concepts=current['cc'].long();mask=(concepts>=0)&(concepts<self.model.num_c)
        values=qe.concept_emb[concepts.clamp(0,self.model.num_c-1)]
        kc=(values*mask.unsqueeze(-1)).sum(-2)/mask.sum(-1,keepdim=True).clamp_min(1)
        raw,cov,attention=self.model.ukt_transfer(torch.cat([question,kc],-1),current['cr'],mask.any(-1))
        delta=self.ukt_scale*torch.tanh(raw[:,1:])
        if self.ukt_fusion=='uncertainty':delta=delta/(1+cov[:,1:].mean(-1))
        inverse=lambda x:torch.log(x/(1-x+1e-8)+1e-8)
        cfg=self.model.other_config
        baseline=sum(inverse(outputs[k])*cfg.get(w,1.) for k,w in [('y_question_all','output_q_all_lambda'),('y_concept_all','output_c_all_lambda'),('y_concept_next','output_c_next_lambda')])
        outputs['y']=torch.sigmoid(baseline+delta)
        outputs['ukt_delta']=delta;outputs['ukt_context_covariance']=cov[:,1:]
        return (outputs,current) if return_details else outputs['y']
