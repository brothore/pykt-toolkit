"""Training-only GRU reference constraints; the teacher is absent at inference."""
import copy
import math
import types
import torch
from torch.nn import functional as F
from .qikt_mamba import QIKT_MAMBA


def confident_pair_penalty(teacher,student,labels,mask,margin=.1):
    """Mean per-fragment penalty, matching confident teacher-correct pairs."""
    if not math.isfinite(margin) or not 0<=margin<=1:raise ValueError('Invalid pair probability margin')
    if teacher.shape!=student.shape or labels.shape!=student.shape or mask.shape!=student.shape:raise ValueError('Pair shapes differ')
    valid=mask.bool();t=torch.where(valid,teacher.detach(),torch.zeros_like(teacher));s=torch.where(valid,student,torch.zeros_like(student))
    original=t.unsqueeze(-1)-t.unsqueeze(-2);changed=s.unsqueeze(-1)-s.unsqueeze(-2)
    eligible=valid.unsqueeze(-1)&valid.unsqueeze(-2)&(labels==1).unsqueeze(-1)&(labels==0).unsqueeze(-2)&(original>margin)
    counts=eligible.sum((-1,-2));loss=(F.relu(original-changed)*eligible).sum((-1,-2))/counts.clamp_min(1)
    return (loss*(counts>0)).sum()/(counts>0).sum().clamp_min(1)


def install_baseline_anchor(model,protocol):
    kl=float(protocol.get('baseline_anchor_kl',0.));pair=float(protocol.get('baseline_pair_weight',0.));margin=float(protocol.get('baseline_pair_margin',.1))
    if not all(math.isfinite(v) and v>=0 for v in [kl,pair]) or not math.isfinite(margin) or not 0<=margin<=1:raise ValueError('Invalid baseline constraints')
    if not (kl or pair):return None
    if hasattr(model,'_baseline_anchor_metadata'):raise ValueError('Baseline constraints already installed')
    if not getattr(model,'baseline_warmstart_allowed',False) or not protocol.get('initial_checkpoint'):raise ValueError('Reference constraints require explicit trainable GRU warm-start')
    original=torch.load(protocol['initial_checkpoint'],map_location='cpu',weights_only=True)
    if any(k not in model.state_dict() or not torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in original.items()):raise ValueError('Student must still equal B0 when creating teacher')
    teacher=copy.deepcopy(model)
    teacher.model.rasch_transfer=None
    teacher.predict_one_step=types.MethodType(QIKT_MAMBA.predict_one_step,teacher)
    teacher.requires_grad_(False);teacher.model.eval()
    original_step=model.train_one_step
    def step(self,data,process=True,return_all=False):
        predictions,loss=original_step(data,process=process,return_all=return_all)
        current=teacher.batch_to_device(data,process=process)
        with torch.no_grad():
            teacher.model.eval();target=teacher.predict_one_step(current,process=False)
        mask=current['sm'].bool();z=torch.logit(predictions[mask].double().clamp(1e-6,1-1e-6));t=target[mask].double().clamp(1e-6,1-1e-6);b=torch.logit(t)
        divergence=((F.softplus(z)-t*z)-(F.softplus(b)-t*b)).mean()
        preservation=confident_pair_penalty(target.double(),predictions.double(),current['rshft'],current['sm'],margin) if pair else predictions.sum()*0.
        return predictions,loss+kl*divergence+pair*preservation
    model.train_one_step=types.MethodType(step,model)
    model._baseline_anchor_metadata={'kl_weight':kl,'pair_weight':pair,'pair_margin':margin,'teacher':'Frozen original B0, training only; no added Rasch; recreated before loading resumed student','teacher_in_inference_state':False,'ranking_scope':'Teacher-correct positive/negative pairs within original training fragments; not full UID or OOF'}
    return model._baseline_anchor_metadata
