"""Capped, epoch-lagged UID-risk tilting; a TERM-inspired approximation.

Complete UID risk is measured on training data in eval mode at epoch start.
Detached weights are then fixed for that epoch, not exact online TERM gradients.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F


def bounded_tilt_weights(losses,temperature=.5,cap=2.,floor=.5):
    values=np.asarray(losses,dtype=float)
    if values.size==0 or not np.isfinite(values).all() or temperature<0 or not 0<floor<=1<=cap:
        raise ValueError('Invalid bounded tilt inputs')
    logits=temperature*(values-values.mean());lo=float(logits.min()-10);hi=float(logits.max()+10)
    for _ in range(100):
        center=(lo+hi)/2;weights=np.clip(np.exp(np.clip(logits-center,-700,700)),floor,cap)
        if weights.mean()>1:lo=center
        else:hi=center
    return np.clip(np.exp(np.clip(logits-(lo+hi)/2,-700,700)),floor,cap)


class TiltObjective:
    def __init__(self,owner,mix=.25,temperature=.5,cap=2.):
        if not 0<=mix<=1:raise ValueError('Invalid tilt mixture')
        self.owner=owner;self.mix=mix;self.temperature=temperature;self.cap=cap
        self.weights={u:1. for u in owner.counts}

    def metadata(self):
        return {'mix':self.mix,'temperature':self.temperature,'cap':self.cap,'floor':.5,
                'estimator':'Complete training-UID risk, eval-mode epoch-start measurement; detached bounded weights fixed for next epoch; not exact TERM',
                'normalization':'UID weights have mean1 with floor/cap enforced by scalar bisection',
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def prepare_epoch(self,model,loader,epoch,folder):
        started=time.perf_counter();totals={u:0. for u in self.owner.counts};counts={u:0 for u in self.owner.counts}
        was_training=model.model.training;model.model.eval();config=model.model.other_config
        heads=[('y_question_all','q_all'),('y_concept_all','c_all'),('y_concept_next','c_next')]
        if model.model.output_mode!='an_irt':heads.append(('y_question_next','q_next'))
        try:
            with torch.no_grad():
                for batch in loader:
                    outputs,current=model.predict_one_step(batch,return_details=True)
                    mask=current['sm'].bool();labels=current['rshft'];losses=torch.zeros_like(labels,dtype=torch.double)
                    losses[mask]=F.binary_cross_entropy(outputs['y'][mask].double(),labels[mask].double(),reduction='none')
                    for output,name in heads:
                        coefficient=config.get(f'loss_{name}_lambda',0)*config.get(f'output_{name}_lambda',0)
                        if coefficient:losses[mask]+=coefficient*F.binary_cross_entropy(outputs[output][mask].double(),labels[mask].double(),reduction='none')
                    for uid,total,n in zip(batch['uid'].tolist(),losses.sum(1).cpu().tolist(),mask.sum(1).cpu().tolist()):
                        totals[int(uid)]+=total;counts[int(uid)]+=n
        finally:model.model.train(was_training)
        if counts!=self.owner.counts:raise ValueError('Measured UID targets differ from frozen training target counts')
        uids=sorted(totals);risks=[totals[u]/counts[u] for u in uids]
        values=bounded_tilt_weights(risks,self.temperature,self.cap);self.weights=dict(zip(uids,map(float,values)))
        record={'epoch':epoch,'metadata':self.metadata(),'uid_risks':dict(zip(uids,risks)),'uid_weights':self.weights,
                'measure_seconds':time.perf_counter()-started,'model_mode':'eval','data':'training only'}
        path=Path(folder)/f'tilt_weights_epoch{epoch:03d}.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record));os.replace(temp,path)

    def weighted_bce(self,predictions,labels,mask):
        mask=mask.bool();losses=torch.zeros_like(predictions,dtype=torch.double)
        losses[mask]=F.binary_cross_entropy(predictions[mask].double(),labels[mask].double(),reduction='none')
        denominators=predictions.new_tensor([self.owner.counts[int(u)] for u in self.owner.uids],dtype=torch.double)
        weights=predictions.new_tensor([self.weights[int(u)] for u in self.owner.uids],dtype=torch.double)
        return (weights*losses.sum(1)/denominators).mean()*(self.owner.windows/len(self.owner.counts))
