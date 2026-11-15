"""Isolated MTKT/LEFOKT-inspired additions to the independent dual GRUs.

Modules expose context/input/hidden interfaces; defaults leave the core intact.
Added projections start at zero. Explicit history abstraction is a structural
ablation and intentionally changes baseline predictions before training.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .qikt_mamba import QIKT_MAMBA


def alibi_slopes(heads):
    if heads<1:raise ValueError('Positive head count required')
    def power(n):
        start=2**(-2**-(math.log2(n)-3));return [start*start**i for i in range(n)]
    if math.log2(heads).is_integer():return power(heads)
    nearest=2**math.floor(math.log2(heads));return power(nearest)+power(2*nearest)[::2][:heads-nearest]


def zero_output(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:nn.init.zeros_(layer.bias)
    return layer


class CausalInteractionConv(nn.Module):
    def __init__(self,width,hidden=64,kernel=3,dropout=.3,context='past'):
        super().__init__()
        if kernel<1:raise ValueError('Positive causal kernel required')
        self.context_mode=context
        if context not in ['past','current']:raise ValueError('Invalid CIC context')
        self.kernel=kernel;self.long=nn.Conv1d(width,hidden,kernel);self.short=nn.Conv1d(width,hidden,1);self.out=zero_output(nn.Conv1d(hidden,width,1));self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        z=x.transpose(1,2)
        a=self.long(F.pad(z,(self.kernel-1,0))) if self.context_mode=='past' else F.conv1d(z,self.long.weight.sum(-1,keepdim=True),self.long.bias)
        b=self.short(z)
        return self.out(a*self.dropout(F.gelu(b))+b*self.dropout(F.gelu(a))).transpose(1,2)


class CausalTimeEncoding(nn.Module):
    def __init__(self,width=64,gap_bins=32,count_bins=15):
        super().__init__();self.gap_bins=gap_bins;self.count_bins=count_bins;self.projection=nn.Linear(2*gap_bins+count_bins,width,bias=False)
    def features(self,concepts,timestamps,valid):
        if concepts.ndim!=3 or timestamps.shape!=concepts.shape[:2] or valid.shape!=timestamps.shape:raise ValueError('Time feature shapes differ')
        batch,length,kcs=concepts.shape;device=concepts.device;cv=(concepts>=0)&valid[...,None]
        match=(concepts[:,:,None,:,None]==concepts[:,None,:,None,:])&cv[:,:,None,:,None]&cv[:,None,:,None,:]
        match=match.any(-1)&torch.ones(length,length,device=device,dtype=torch.bool).tril(-1)[None,:,:,None]
        index=torch.arange(length,device=device)[None,None,:,None];last=torch.where(match,index,-1).amax(2)
        last_t=timestamps.gather(1,last.clamp_min(0).reshape(batch,-1)).reshape(batch,length,kcs)
        difference=timestamps[...,None]-last_t;support=last>=0
        if bool((difference[support]<0).any()):raise ValueError('Nonmonotone repeated-KC timestamps')
        rgap=torch.where(support,torch.round(torch.log2(difference.clamp_min(0).double()/60000+1)).long()+1,0)
        counts=torch.round(torch.log2(match.sum(2).double()+1)).long()
        previous=torch.cat([timestamps[:,:1],timestamps[:,:-1]],1);previous_valid=torch.cat([torch.zeros_like(valid[:,:1]),valid[:,:-1]],1)&valid
        delta=timestamps-previous
        if bool((delta[previous_valid]<0).any()):raise ValueError('Nonmonotone sequence timestamps')
        sgap=torch.where(previous_valid,torch.round(torch.log2(delta.clamp_min(0).double()/60000+1)).long()+1,0)
        if bool((rgap[cv]>=self.gap_bins).any()) or bool((sgap[valid]>=self.gap_bins).any()) or bool((counts[cv]>=self.count_bins).any()):raise ValueError('Time category exceeds registered bins')
        average=lambda z:(z*cv[...,None]).sum(-2)/cv.sum(-1,keepdim=True).clamp_min(1)
        features=torch.cat([average(F.one_hot(rgap,self.gap_bins)),F.one_hot(sgap,self.gap_bins),average(F.one_hot(counts,self.count_bins))],-1)
        return features*valid[...,None]
    def forward(self,concepts,timestamps,valid):return self.projection(self.features(concepts,timestamps,valid).to(self.projection.weight.dtype))


class CausalAlibiReadout(nn.Module):
    def __init__(self,input_dim,width=64,heads=8,query_dim=None,use_bias=True):
        super().__init__()
        if width%heads:raise ValueError('Attention width must divide head count')
        self.width=width;self.heads=heads;self.use_bias=use_bias;self.query=nn.Linear(query_dim or input_dim,width);self.key=nn.Linear(query_dim or input_dim,width);self.value=nn.Linear(input_dim,width);self.response=nn.Embedding(2,width)
        self.register_buffer('slopes',torch.tensor(alibi_slopes(heads)))
    def forward(self,embedding,responses,valid,query_features=None):
        features=embedding if query_features is None else query_features;batch,length,_=embedding.shape;dim=self.width//self.heads
        reshape=lambda z:z.view(batch,length,self.heads,dim).transpose(1,2)
        query=reshape(self.query(features));key=reshape(self.key(features));value=reshape(self.value(embedding)+self.response(responses.clamp(0,1).long()))
        score=query@key.transpose(-1,-2)/math.sqrt(dim)
        positions=torch.arange(length,device=embedding.device);lag=(positions[:,None]-positions[None,:]).abs()
        if self.use_bias:score=score-self.slopes[None,:,None,None]*lag[None,None]
        allowed=torch.ones(length,length,device=embedding.device,dtype=torch.bool).tril(-1)[None,None]&valid[:,None,None,:]&valid[:,None,:,None]
        weights=score.masked_fill(~allowed,-1e9).softmax(-1)*allowed;weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        return (weights@value).transpose(1,2).reshape(batch,length,self.width),weights


class ReferenceFusion(nn.Module):
    def __init__(self,width,mode,scale,bias='alibi',kernel=3,cic_context='past',num_q=None,num_c=None,history_weight=0.):
        super().__init__();self.mode=mode;self.scale=scale
        self.share_history=mode in ['shared_history','shared_scalar'];self.history_weight=history_weight
        if mode in ['scalar_query','shared_history','shared_scalar','scalar_additive_query','scalar_additive_input']:
            if mode!='shared_history':
                from .qikt_ukt_rasch import RaschFactors
                factor_mode='input_scalar_additive' if mode.startswith('scalar_additive_') else 'input_scalar'
                self.scalar=RaschFactors(num_q,num_c,width,factor_mode,scale)
            return
        if mode.startswith('cic_'):
            branches=['question','concept'] if mode=='cic_both' else [mode.split('_',1)[1]]
            self.adapters=nn.ModuleDict({b:CausalInteractionConv(width,kernel=kernel,context=cic_context) for b in branches})
        else:
            if mode.startswith('time_'):self.time=CausalTimeEncoding()
            if mode in ['time_gate','alibi_logit','alibi_question','alibi_concept']:
                self.readout=CausalAlibiReadout(2*width,query_dim=64 if mode=='time_gate' else None,use_bias=bias=='alibi')
            if mode in ['time_input','time_input_decode','time_sparse_input']:
                self.input=zero_output(nn.Linear(64,width))
            if mode in ['time_input','time_sparse_input']:pass
            elif mode=='alibi_logit':self.logit=zero_output(nn.Linear(64,1))
            else:
                context_width=128 if mode=='time_gate' else 64
                branch='concept' if mode=='alibi_concept' else 'question'
                self.adapters=nn.ModuleDict({branch:zero_output(nn.Linear(width+context_width,width))})
                if mode=='time_gate':self.gate=nn.Linear(width+context_width,width)
    def context(self,embedding,concepts,responses,timestamps,valid,questions=None):
        if self.mode=='shared_history':return None
        if self.mode in ['scalar_query','shared_scalar','scalar_additive_query','scalar_additive_input']:
            if questions is None:raise ValueError('Current-question factors require question IDs')
            return self.scalar.input_delta(questions,concepts)
        if self.mode.startswith('cic_'):return None
        if self.mode=='time_sparse_input':
            features=self.time.features(concepts,timestamps,valid).to(self.time.projection.weight.dtype)
            time=self.time.projection(features)
            bins=torch.arange(self.time.count_bins,device=features.device,dtype=features.dtype)
            support=(features[...,2*self.time.gap_bins:]*bins).sum(-1,keepdim=True)
            return torch.cat([time,1/(1+support)],-1)
        time=self.time(concepts,timestamps,valid) if hasattr(self,'time') else None
        if hasattr(self,'readout'):
            value,weights=self.readout(embedding,responses,valid,time)
            return torch.cat([time,value],-1) if self.mode=='time_gate' else value
        return time
    def decoder_delta(self,context):return context if self.mode in ['scalar_query','shared_scalar','scalar_additive_query'] else None
    def input_delta(self,context):
        if self.mode=='scalar_additive_input':return context
        if self.mode=='time_sparse_input':
            return self.scale*context[...,64:]*torch.tanh(self.input(context[...,:64]))
        return self.scale*torch.tanh(self.input(context)) if self.mode in ['time_input','time_input_decode'] else None
    def hidden(self,hidden,context,branch):
        if not hasattr(self,'adapters') or branch not in self.adapters:return hidden
        if self.mode.startswith('cic_'):delta=self.adapters[branch](hidden)
        else:
            features=torch.cat([hidden,context[:,1:]],-1);delta=self.adapters[branch](features)
            if self.mode=='time_gate':delta=delta*torch.sigmoid(self.gate(features))
        decoder_ratio=.2 if self.mode=='time_input_decode' else 1.
        return hidden+self.scale*decoder_ratio*torch.tanh(delta)


class QIKTReferenceFusion(QIKT_MAMBA):
    def __init__(self,*args,other_config=None,**kwargs):
        cfg=dict(other_config or {});mode=cfg.get('reference_fusion_mode','off');scale=float(cfg.get('reference_fusion_scale',.1));bias=cfg.get('reference_fusion_bias','alibi');kernel=int(cfg.get('reference_cic_kernel',3))
        modes={'cic_question','cic_concept','cic_both','time_input','time_decode','time_gate','time_input_decode','time_sparse_input','alibi_logit','alibi_question','alibi_concept','scalar_query','shared_history','shared_scalar','scalar_additive_query','scalar_additive_input'}
        if mode not in modes or not math.isfinite(scale) or not 0<scale<=1 or bias not in ['alibi','none']:raise ValueError('Invalid reference fusion configuration')
        if kwargs.get('version') not in ['gru','lstm'] or cfg.get('output_mode')!='an_irt':raise ValueError('Reference fusion requires independent dual-GRU an_irt')
        if any(cfg.get(k,0) for k in ['freeze_backbone','item_residual','concept_memory_residual','head_fusion_residual','history_response_dropout','pairwise_loss_weight']) or any(cfg.get(k,'off')!='off' for k in ['ukt_transfer_mode','ukt_rasch_mode','causal_adapter_mode','retrieval_adapter_mode']):raise ValueError('Isolate the reference module in the first campaign')
        history_weight=float(cfg.get('reference_history_weight',0.))
        if not math.isfinite(history_weight) or not 0<=history_weight<=1:raise ValueError('History weight must be in[0,1]')
        super().__init__(*args,other_config=cfg,**kwargs)
        if mode in ['scalar_query','shared_scalar','scalar_additive_query','scalar_additive_input']:
            from pathlib import Path
            self.resume_source_dependencies=[str(Path(__file__).with_name('qikt_ukt_rasch.py'))]
        with torch.random.fork_rng(devices=[]):self.model.reference_fusion=ReferenceFusion(self.model.emb_size,mode,scale,bias,kernel,cfg.get('reference_cic_context','past'),num_q=self.model.num_q,num_c=self.model.num_c,history_weight=float(cfg.get('reference_history_weight',0.))).to(self.device)
    def predict_one_step(self,data,return_details=False,process=True,return_raw=False):
        outputs,current=super().predict_one_step(data,return_details=True,process=process)
        if self.model.reference_fusion.mode=='alibi_logit':
            # The context is returned explicitly; no graph cache survives the forward.
            delta=self.model.reference_fusion.scale*torch.tanh(self.model.reference_fusion.logit(outputs.pop('reference_context')[:,1:]).squeeze(-1))
            inverse=lambda x:torch.log(x/(1-x+1e-8)+1e-8);cfg=self.model.other_config
            baseline=sum(inverse(outputs[k])*cfg.get(w,1.) for k,w in [('y_question_all','output_q_all_lambda'),('y_concept_all','output_c_all_lambda'),('y_concept_next','output_c_next_lambda')]);outputs['y']=torch.sigmoid(baseline+delta)
        return (outputs,current) if return_details else outputs['y']
