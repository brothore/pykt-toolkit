"""Complete-supplied-UID ranking; no validation membership or teacher.

All scored fragments must occur together. 'Complete' means the original
training coverage, not dropped preprocessing tails or unlimited model context.
"""
import hashlib
import math
from pathlib import Path
import torch
import torch.nn.functional as F


class CompleteUIDObjective:
    def __init__(self,dataset,sampler,protocol,counts):
        self.counts=counts;self.batches=len(sampler);self.mode=protocol.get('complete_rank_mode','logistic')
        self.temperature=float(protocol.get('complete_rank_temperature',.5))
        self.reliability=float(protocol.get('complete_rank_reliability',0.))
        self.focus=float(protocol.get('complete_rank_focus',0.))
        if (self.mode not in ('logistic','smooth_auc') or self.temperature<=0 or self.reliability<0
                or not 0<=self.focus<=1 or not all(math.isfinite(v) for v in (self.temperature,self.reliability,self.focus))):
            raise ValueError('Invalid complete ranking protocol')
        labels=dataset.dori['rseqs'][:,1:]
        mask=(dataset.dori['rseqs'][:,:-1]!=-1)&(labels!=-1)&(dataset.dori['smasks'][:,1:]!=-1)
        pos={u:0 for u in counts}
        for uid,y,m in zip(dataset.dori['uid'],labels,mask):pos[int(uid)]+=int((y[m]==1).sum())
        self.minority={u:min(pos[u],counts[u]-pos[u]) for u in counts}
        self.eligible={u:n for u,n in self.minority.items() if n>0}
        raw={u:n/(n+self.reliability) for u,n in self.eligible.items()}
        mean=sum(raw.values())/len(raw) if raw else 1.
        self.weights={u:w/mean for u,w in raw.items()}

    def rank_loss(self,predictions,labels,mask,uids):
        valid=mask.bool();groups={}
        for index,uid in enumerate(uids):groups.setdefault(int(uid),[]).append(index)
        ranks=[];weights=[]
        for uid,indices in groups.items():
            if uid not in self.counts:raise ValueError('UID has no counted training targets')
            active=valid[indices];p=predictions[indices][active].double();y=labels[indices][active].double()
            if len(p)!=self.counts[uid]: raise ValueError('Incomplete UID in training batch')
            if not torch.isfinite(p).all() or not ((y==0)|(y==1)).all():raise ValueError('Invalid active predictions/labels')
            if uid not in self.eligible:continue
            scores=torch.logit(p.clamp(1e-6,1-1e-6));positive=scores[y==1];negative=scores[y==0]
            difference=negative[None,:]-positive[:,None]
            risk=(F.softplus(difference).mean() if self.mode=='logistic' else torch.sigmoid(difference/self.temperature).mean())
            ranks.append(risk)
            # Online detached fragment-free rank difficulty, not OOF reliability.
            bounded_risk=torch.sigmoid(difference/self.temperature).mean().detach()
            focus=torch.exp(self.focus*(bounded_risk-.5)).clamp(.75,1.25)
            weights.append((focus*self.weights[uid]).clamp(.25,1.5))
        zero=predictions[valid].sum()*0.
        rank=(torch.stack(ranks)*torch.stack(weights)).sum()*(self.batches/len(self.eligible)) if ranks else zero
        return rank

    def metadata(self):
        sampler=Path(__file__).resolve().parents[1]/'datasets/complete_uid_sampler.py'
        return {'scope':'All scored training fragments of a UID together; cross-fragment pairs included',
                'rank_mode':self.mode,'temperature':self.temperature,'reliability_strength':self.reliability,
                'focus_strength':self.focus,'UID_batches':self.batches,'rank_eligible_UIDs':len(self.eligible),
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'sampler_sha256':hashlib.sha256(sampler.read_bytes()).hexdigest()}
