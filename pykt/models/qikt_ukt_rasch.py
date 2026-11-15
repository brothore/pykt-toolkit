"""UKT-style problem modulation placed at GRU input or bounded readout."""
import math
import torch
from torch import nn
from .qikt_mamba import QIKT_MAMBA


class RaschFactors(nn.Module):
    def __init__(self,num_q,num_c,width,mode,scale):
        super().__init__()
        if mode not in {'input_scalar','input_scalar_additive','input_vector','input_additive','query_vector','output_vector'} or not math.isfinite(scale) or scale<=0:
            raise ValueError('Invalid Rasch factors')
        self.mode=mode;self.scale=scale;self.num_c=num_c;self.width=width
        self.difficulty=nn.Embedding(num_q,1 if mode in {'input_scalar','input_scalar_additive'} else width)
        self.concept_delta=nn.Embedding(num_c,width)
        nn.init.zeros_(self.difficulty.weight)
        if mode in {'input_additive','input_scalar_additive'}:self.register_buffer('concept_origin',self.concept_delta.weight.detach().clone())

    def modulation(self,questions,concepts):
        valid=(concepts>=0)&(concepts<self.num_c)
        direction=(self.concept_delta(concepts.clamp(0,self.num_c-1))*valid.unsqueeze(-1)).sum(-2)
        direction=direction/valid.sum(-1,keepdim=True).clamp_min(1)
        difficulty=self.difficulty(questions.long())
        if self.mode in {'input_additive','input_scalar_additive'}:
            origin=(self.concept_origin[concepts.clamp(0,self.num_c-1)]*valid.unsqueeze(-1)).sum(-2)/valid.sum(-1,keepdim=True).clamp_min(1)
            return (difficulty+(direction-origin))*valid.any(-1,keepdim=True)
        return difficulty*direction

    def input_delta(self,questions,concepts):
        return self.scale*self.modulation(questions,concepts)

    def output_delta(self,questions,concepts):
        return self.scale*torch.tanh(self.modulation(questions,concepts).sum(-1)/math.sqrt(self.width))


class QIKTRaschTransfer(QIKT_MAMBA):
    def __init__(self,*args,other_config=None,**kwargs):
        cfg=dict(other_config or {})
        mode=cfg.get('ukt_rasch_mode','off');freeze=bool(cfg.get('ukt_rasch_freeze',False))
        if kwargs.get('version') not in {'gru','lstm'} or cfg.get('output_mode')!='an_irt':
            raise ValueError('Rasch transfer requires independent dual-GRU an_irt')
        if any(cfg.get(k,0) for k in ['freeze_backbone','item_residual','concept_memory_residual','head_fusion_residual','history_response_dropout','pairwise_loss_weight']) or cfg.get('ukt_transfer_mode','off')!='off':
            raise ValueError('Rasch V2 isolates the new factors')
        if freeze and mode!='output_vector':raise ValueError('V2 freezes only output Rasch factors')
        self.baseline_warmstart_allowed=bool(cfg.get('ukt_rasch_warmstart',False))
        if self.baseline_warmstart_allowed and mode!='query_vector':raise ValueError('V3 trainable warm-start is scoped to query-only Rasch')
        super().__init__(*args,other_config=cfg,**kwargs)
        if self.emb_type not in {'attn','iekt'}:raise ValueError('Rasch V2 requires explicit question/KC embeddings')
        if freeze:
            for p in self.parameters():p.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            self.model.rasch_transfer=RaschFactors(self.model.num_q,self.model.num_c,self.model.emb_size,mode,float(cfg.get('ukt_rasch_scale',1.))).to(self.device)
        self.model.freeze_backbone=freeze;self.model.train(self.model.training)

    def predict_one_step(self,data,return_details=False,process=True,return_raw=False):
        outputs,current=super().predict_one_step(data,return_details=True,process=process)
        factors=self.model.rasch_transfer
        if factors.mode=='output_vector':
            inverse=lambda x:torch.log(x/(1-x+1e-8)+1e-8)
            cfg=self.model.other_config
            baseline=sum(inverse(outputs[k])*cfg.get(w,1.) for k,w in [('y_question_all','output_q_all_lambda'),('y_concept_all','output_c_all_lambda'),('y_concept_next','output_c_next_lambda')])
            delta=factors.output_delta(current['qshft'].long(),current['cshft'].long())
            outputs['y']=torch.sigmoid(baseline+delta);outputs['rasch_delta']=delta
        return (outputs,current) if return_details else outputs['y']
