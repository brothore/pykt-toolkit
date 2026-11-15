"""Frozen QIKT plus a bounded current-question / causal-prefix adapter.

Loaded only by the explicitly configured isolated B2 runtime. No UID or true
sequence AUC enters predictions. Existing backbone checkpoint keys are retained.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .qikt_mamba import QIKT_MAMBA


def prefix_features(concepts,responses,num_concepts):
    query_valid=(concepts>=0)&(concepts<num_concepts);safe=concepts.clamp(0,num_concepts-1).long()
    query=responses.new_zeros((*responses.shape,num_concepts),dtype=torch.float32)
    query.scatter_add_(-1,safe,query_valid.float());query=query.clamp_max(1)
    known=(responses==0)|(responses==1);incidence=query*known.unsqueeze(-1)
    successes=incidence*torch.where(known,responses,0).float().unsqueeze(-1)
    counts=incidence.cumsum(1)-incidence;correct=successes.cumsum(1)-successes
    divisor=query.sum(-1).clamp_min(1)
    rate=(((correct+5)/(counts+10))*query).sum(-1)/divisor
    support=(counts*query).sum(-1)/divisor
    positions=torch.arange(responses.shape[1],device=responses.device).view(1,-1,1)
    touched=torch.where(incidence.bool(),positions,torch.full_like(incidence,-1))
    latest=touched.cummax(1).values
    previous=torch.cat([torch.full_like(latest[:,:1],-1),latest[:,:-1]],1)
    age=((positions-previous)*query).sum(-1)/divisor
    coverage=((counts>0)*query).sum(-1)/divisor
    values=torch.stack([torch.logit(rate.clamp(1e-6,1-1e-6)),torch.log1p(support),torch.log1p(age),coverage],-1)
    return torch.where((query.sum(-1)>0).unsqueeze(-1),values,torch.zeros_like(values))


class PrefixAdapter(nn.Module):
    def __init__(self,embedding_dim,train_projection=False):
        super().__init__();self.project=nn.Linear(embedding_dim,16,bias=False)
        self.hidden=nn.Linear(20,16);self.output=nn.Linear(16,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.project.requires_grad_(train_projection)

    def forward(self,question,features):
        return self.output(F.gelu(self.hidden(torch.cat([self.project(question),features],-1)))).squeeze(-1)


class QIKTPrefixAdapter(QIKT_MAMBA):
    def __init__(self,*args,other_config=None,**kwargs):
        config=dict(other_config or {})
        if config.get('output_mode')!='an_irt' or any(config.get(k,0) for k in ['item_residual','concept_memory_residual','head_fusion_residual','history_response_dropout','pairwise_loss_weight']):
            raise ValueError('B2 isolates the prefix adapter on the unchanged an_irt baseline')
        config['freeze_backbone']=False;config['anchor_kl_weight']=0.
        super().__init__(*args,other_config=config,**kwargs)
        if self.emb_type not in ['attn','iekt']:raise ValueError('B2 supports attn/iekt question embeddings')
        self.adapter_scale=float(config.get('v2_adapter_scale',.05));self.adapter_kl=float(config.get('v2_adapter_kl',0.))
        if not 0<self.adapter_scale<=.1 or self.adapter_kl<0:raise ValueError('Invalid B2 bounds')
        for p in self.parameters():p.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            self.model.prefix_adapter=PrefixAdapter(self.model.emb_size,bool(config.get('v2_projection_trainable',False))).to(self.device)
        self.model.freeze_backbone=True;self.model.train(self.model.training);self._collect_anchor=False;self._anchor=None

    def predict_one_step(self,data,return_details=False,process=True):
        outputs,current=super().predict_one_step(data,return_details=True,process=process)
        def inverse(value):return torch.log(value/(1-value+1e-8)+1e-8)
        config=self.model.other_config
        baseline=(inverse(outputs['y_question_all'])*config.get('output_q_all_lambda',1)+
                  inverse(outputs['y_concept_all'])*config.get('output_c_all_lambda',1)+
                  inverse(outputs['y_concept_next'])*config.get('output_c_next_lambda',1)).detach()
        question=self.model.que_emb.que_emb(current['qshft'].long()).detach()
        features=prefix_features(current['cc'],current['cr'],self.model.num_c)[:,1:]
        delta=self.adapter_scale*torch.tanh(self.model.prefix_adapter(question,features))
        student=baseline+delta;outputs['y']=torch.sigmoid(student)
        outputs['v2_baseline_logit']=baseline;outputs['v2_student_logit']=student;outputs['v2_delta']=delta
        if self._collect_anchor:self._anchor=(baseline,student)
        return (outputs,current) if return_details else outputs['y']

    def train_one_step(self,data,process=True,return_all=False):
        self._collect_anchor=True
        try:
            prediction,loss=super().train_one_step(data,process=process,return_all=return_all)
            if self.adapter_kl:
                baseline,student=self._anchor;mask=self.batch_to_device(data,process=process)['sm'].bool()
                teacher=torch.sigmoid(baseline[mask]);s=student[mask];b=baseline[mask]
                loss=loss+self.adapter_kl*((F.softplus(s)-teacher*s)-(F.softplus(b)-teacher*b)).mean()
            return prediction,loss
        finally:self._collect_anchor=False;self._anchor=None
