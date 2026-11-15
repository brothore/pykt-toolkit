"""Label-value-independent, bounded KC-frequency weighting for training only."""
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F


class FrequencyObjective:
    def __init__(self,dataset,alpha=.25,cap=2.,power=.25):
        if not 0<=alpha<=1 or cap<1 or power<0:raise ValueError('Invalid frequency objective settings')
        d=dataset.dori
        if 'cseqs' not in d:raise ValueError('Training KC identities required')
        r=d['rseqs'];self.mask=(r[:,:-1]!=-1)&(r[:,1:]!=-1)&(d['smasks'][:,1:]!=-1)
        c=d['cseqs'][:,1:,:].long();unique=self.unique_mask(c)
        active=unique&self.mask.unsqueeze(-1)
        if not active.any():raise ValueError('No scored training KCs')
        if (self.mask&~unique.any(-1)).any():raise ValueError('Scored target has no KC')
        counts=torch.bincount(c[active],minlength=int(c[active].max())+1).double()
        reference=counts[counts>0].median()
        self.weights=(reference/counts.clamp_min(1)).pow(power).clamp(1,cap)
        self.counts=counts;self.alpha=alpha;self.cap=cap;self.power=power
        values=self.event_weights(c)
        self.normalizer=float(values[self.mask].mean())
        self.context=None

    @staticmethod
    def unique_mask(concepts):
        result=concepts>=0
        for index in range(1,concepts.shape[-1]):
            result[...,index]&=~(concepts[...,:index]==concepts[...,index,None]).any(-1)
        return result

    def event_weights(self,concepts):
        unique=self.unique_mask(concepts);table=self.weights.to(concepts.device)
        indices=concepts.clamp(0,len(table)-1)
        return (table[indices]*unique).sum(-1)/unique.sum(-1).clamp_min(1)

    def weighted_bce(self,predictions,labels,mask):
        if self.context is None:raise RuntimeError('KC frequency context absent')
        mask=mask.bool();weights=self.context.to(predictions.device)[mask]
        if self.cap==1:
            return F.binary_cross_entropy(predictions[mask].double(),labels[mask].double())
        losses=F.binary_cross_entropy(predictions[mask].double(),labels[mask].double(),reduction='none')
        return (weights*losses).mean()/self.normalizer

    def metadata(self):
        return {'alpha':self.alpha,'cap':self.cap,'power':self.power,'counts':self.counts.tolist(),
                'global_target_mean_weight':self.normalizer,'raw_weight_min':float(self.weights.min()),'raw_weight_max':float(self.weights.max()),
                'scope':'Scored training targets only; duplicate KCs count once per event; no response values, validation or test frequencies',
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
