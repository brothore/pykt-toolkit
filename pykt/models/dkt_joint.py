"""Dataset-specific joint objectives with one DKT state and one main BCE.

The original validated loss implementations are reused through immutable
per-call views. Views expose cached state/head outputs, not extra backbones.
Training heads can all be removed using export_dkt_state_dict().
"""
import copy
import inspect
import math
import types
import torch
from torch import nn
from torch.nn import functional as F
from .dkt import DKT
from .dkt_first_action import DKTFirstAction
from .dkt_process import DKTProcess
from .dkt_distill import DKTdistill
from .dkt_transfer import DKTTransfer
from .dkt_confidence import DKTConfidence
from .dkt_option import DKTOption
from .dkt_profile import DKTProfile

COMPONENTS={'action':DKTFirstAction,'process':DKTProcess,'distillation':DKTdistill,
            'transfer':DKTTransfer,'confidence':DKTConfidence,'option':DKTOption,'profile':DKTProfile}
TASKS={'assist2009':('action','process','distillation','transfer'),
       'nips_task34':('confidence','option','profile')}
TERMS={'action':('action',),'process':('time','attempt','hint'),'distillation':('distillation',),
       'transfer':('transfer',),'confidence':('confidence',),'option':('option',),'profile':('profile',)}
BACKBONE=('interaction_emb','lstm_layer','dropout_layer','out_layer')
WEIGHT_ATTRS={'action':('action_weight',),'process':('time_weight','attempt_weight','hint_weight'),
              'distillation':('kd_weight',),'transfer':('transfer_weight',),
              'confidence':('confidence_weight',),'option':('option_weight',),'profile':('profile_weight',)}


class _LossView:
    """Read-only adapter binding an original loss to cached shared predictions."""
    def __init__(self, model, task, output, hidden, concept):
        self.model,self.task,self.output,self.hidden,self.concept=model,task,output,hidden,concept
    def parameters(self):return self.model.parameters()
    def __call__(self,*args,**kwargs):return self.output
    def __getattr__(self,name):
        model=self.model;task=self.task
        if name in model.component_attributes[task]:return model.component_attributes[task][name]
        if name in model.aux_heads[task]:return model.aux_heads[task][name]
        if name in model.component_buffers[task]:return getattr(model,model.component_buffers[task][name])
        cls=COMPONENTS[task]
        attr=inspect.getattr_static(cls,name,None)
        if isinstance(attr,staticmethod):return attr.__func__
        if callable(attr):return types.MethodType(attr,self)
        return getattr(model,name)
    def forward_details(self,c,r,target_c,*extra):
        heads=self.model.aux_heads[self.task];features=torch.cat((self.hidden,self.concept),dim=-1)
        if self.task=='action':
            logits=heads['action_out'](heads['action_act'](heads['action_proj'](features)))
            return {'predictions':self.output,'action_logits':logits}
        if self.task=='process':
            z=heads['process_act'](heads['process_proj'](features))
            return {'predictions':self.output,'time_prediction':heads['time_out'](z).squeeze(-1),
                    'attempt_logits':heads['attempt_out'](z),'hint_logits':heads['hint_out'](z).squeeze(-1)}
        if self.task=='confidence':
            logits=heads['confidence_out'](heads['confidence_act'](heads['confidence_proj'](features)))
            if self.confidence_mode=='conditional':logits=logits.view(*logits.shape[:-1],2,5)
            return {'predictions':self.output,'confidence_logits':logits}
        if self.task=='option':
            tq=extra[0].long();valid=(tq>=0)&(tq<self.num_q);safe=tq.clamp(0,self.num_q-1)
            z=torch.cat((features,heads['option_question_emb'](safe)),dim=-1)
            logits=heads['option_out'](heads['option_act'](heads['option_proj'](z)))
            if self.option_mode=='prior':logits=logits+torch.where(valid.unsqueeze(-1),self.option_log_prior[safe],0)
            return self.output,logits
        raise RuntimeError(f'{self.task} has no auxiliary head')


class DKTJoint(DKT):
    def __init__(self,num_c,emb_size,dropout=.1,emb_type='qid',emb_path='',pretrain_dim=768,
                 dataset_name='assist2009',task_configs=None,loss_weights=None):
        if dataset_name not in TASKS:raise ValueError('Unsupported joint dataset')
        if emb_type!='qid':raise ValueError('Joint DKT requires qid')
        initial_rng=torch.random.get_rng_state()
        super().__init__(num_c,emb_size,dropout,emb_type,emb_path,pretrain_dim)
        self.dataset_name=dataset_name
        self.model_name='dkt_joint_assist' if dataset_name=='assist2009' else 'dkt_joint_nips'
        self.task_configs=copy.deepcopy(task_configs or {})
        if set(self.task_configs)!=set(TASKS[dataset_name]):raise ValueError('Provide configs for every dataset-specific component')
        terms=[term for task in TASKS[dataset_name] for term in TERMS[task]]
        if set(loss_weights or {})-set(terms):raise ValueError('Unknown joint loss term')
        self.loss_weights={key:float((loss_weights or {}).get(key,0)) for key in terms}
        if any(not math.isfinite(v) or v<0 for v in self.loss_weights.values()):raise ValueError('Weights must be finite nonnegative')
        self.aux_heads=nn.ModuleDict();self.component_attributes={};self.component_buffers={}
        for task in TASKS[dataset_name]:
            cfg=copy.deepcopy(self.task_configs[task])
            for attr in WEIGHT_ATTRS[task]:cfg[attr]=1.
            # Match each original single-task head's initialization exactly,
            # without advancing the response backbone's global RNG stream.
            with torch.random.fork_rng(devices=[]):
                torch.random.set_rng_state(initial_rng)
                template=COMPONENTS[task](num_c,emb_size,dropout=dropout,emb_type=emb_type,
                                          emb_path=emb_path,pretrain_dim=pretrain_dim,**cfg)
            self.component_attributes[task]={k:v for k,v in template.__dict__.items() if not k.startswith('_')}
            self.aux_heads[task]=nn.ModuleDict({name:module for name,module in template._modules.items() if name not in BACKBONE})
            self.component_buffers[task]={}
            for name,buffer in template._buffers.items():
                registered=f'aux_{task}__{name}';self.register_buffer(registered,buffer)
                self.component_buffers[task][name]=registered

    def loss_components(self,data,return_per_sample=False):
        device=next(self.parameters()).device
        c,r=data['cseqs'].to(device).long(),data['rseqs'].to(device).long()
        if any(self.loss_weights.values()):
            h,_=self.lstm_layer(self.interaction_emb(c+self.num_c*r));h=self.dropout_layer(h)
            y=torch.sigmoid(self.out_layer(h))
        else:
            y=self(c,r);h=None
        tc=data['shft_cseqs'].to(device).long();target=data['shft_rseqs'].to(device)
        mask=data['masks'].to(device).bool()&data['smasks'].to(device).bool()
        p=y.gather(-1,tc.unsqueeze(-1)).squeeze(-1)
        if return_per_sample:
            main=(F.binary_cross_entropy(p.double(),target.double(),reduction='none')*mask).sum(1)/mask.sum(1).clamp_min(1)
        else:main=F.binary_cross_entropy(p[mask].double(),target[mask].double()) if mask.any() else p.sum().double()*0
        parts={'response':main}
        concept=None
        if h is not None:
            safe=tc.clamp(0,self.num_c-1);concept=(self.interaction_emb(safe)+self.interaction_emb(safe+self.num_c))*.5
        for task in TASKS[self.dataset_name]:
            terms=TERMS[task]
            if not any(self.loss_weights[t] for t in terms):
                parts.update({t:torch.zeros_like(main) for t in terms});continue
            view=_LossView(self,task,y,h,concept)
            _,original_parts=COMPONENTS[task].compute_loss(view,data,return_per_sample=return_per_sample,return_components=True)
            # Component response losses are deliberately discarded. There is
            # exactly one original BCE in the final objective.
            parts.update({t:original_parts[t] for t in terms})
        return parts

    def compute_loss(self,data,return_per_sample=False,return_components=False):
        parts=self.loss_components(data,return_per_sample)
        total=parts['response']+sum(self.loss_weights[t]*parts[t] for t in self.loss_weights)
        return (total,parts) if return_components else total

    def export_dkt_state_dict(self):
        return {name:value for name,value in self.state_dict().items() if name.split('.')[0] in BACKBONE}
