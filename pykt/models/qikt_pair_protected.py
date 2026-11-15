"""Frozen-teacher adapter with training-label-defined confident-pair protection."""
import math
import torch
from torch.nn import functional as F
from .qikt_prefix_adapter import QIKTPrefixAdapter


def pair_preservation_loss(teacher_logits,student_logits,labels,mask,margin=.1):
    if not math.isfinite(margin) or margin<0:raise ValueError('Invalid teacher probability margin')
    losses=[]
    for teacher,student,y,valid in zip(teacher_logits.detach(),student_logits,labels,mask.bool()):
        teacher=torch.sigmoid(teacher[valid]);student=torch.sigmoid(student[valid]);y=y[valid]
        positive=y==1;negative=y==0
        if not positive.any() or not negative.any():continue
        original=teacher[positive,None]-teacher[None,negative]
        changed=student[positive,None]-student[None,negative]
        eligible=original>margin
        if eligible.any():losses.append(F.relu(original[eligible]-changed[eligible]).mean())
    return torch.stack(losses).mean() if losses else student_logits[mask.bool()].sum()*0.


class QIKTProtectedAdapter(QIKTPrefixAdapter):
    def __init__(self,*args,other_config=None,**kwargs):
        config=dict(other_config or {});self.protection_weight=float(config.get('v2_pair_weight',1.))
        if not math.isfinite(self.protection_weight) or self.protection_weight<0:raise ValueError('Invalid protection weight')
        super().__init__(*args,other_config=config,**kwargs)
        self._collect_protection=False;self._protection_cache=None

    def predict_one_step(self,data,return_details=False,process=True):
        outputs,current=super().predict_one_step(data,return_details=True,process=process)
        if self._collect_protection:self._protection_cache=(outputs['v2_baseline_logit'],outputs['v2_student_logit'],current)
        return (outputs,current) if return_details else outputs['y']

    def train_one_step(self,data,process=True,return_all=False):
        self._collect_protection=True
        try:
            prediction,loss=super().train_one_step(data,process=process,return_all=return_all)
            teacher,student,current=self._protection_cache
            penalty=pair_preservation_loss(teacher,student,current['rshft'],current['sm'],margin=.1)
            self.last_pair_preservation_loss=float(penalty.detach().cpu())
            return prediction,loss+self.protection_weight*penalty
        finally:self._collect_protection=False;self._protection_cache=None
