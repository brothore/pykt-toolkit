"""Training-only complete-UID weighting and within-fragment ranking objectives.

No inference routing, learned parameters or test/validation membership is used.
"""
from collections import defaultdict
import types
import torch
import torch.nn.functional as F


class SequenceObjective:
    def __init__(self, dataset, alpha=0., rank_weight=0.):
        if not 0 <= alpha <= 1 or rank_weight < 0: raise ValueError('Invalid objective coefficients')
        if any(float(v) != 0 for v in getattr(dataset, 'aug_probs', {}).values()):
            raise ValueError('Sequence target counts require training augmentation disabled')
        d=dataset.dori
        if not all(k in d for k in ['uid','rseqs','smasks']): raise ValueError('Complete UID metadata required')
        self.counts=defaultdict(int);self.pairs=defaultdict(int);self.windows=len(dataset)
        labels=d['rseqs'];mask=(labels[:,:-1]!=-1)&(labels[:,1:]!=-1)&(d['smasks'][:,1:]!=-1)
        for uid,r,valid in zip(d['uid'],labels[:,1:],mask):
            uid=int(uid)
            if uid<0: raise ValueError('UIDs cannot be missing')
            n=int(valid.sum());pos=int((r[valid]==1).sum());self.counts[uid]+=n;self.pairs[uid]+=pos*(n-pos)
        self.counts={u:n for u,n in self.counts.items() if n}
        self.pairs={u:n for u,n in self.pairs.items() if n}
        if not self.counts:raise ValueError('No scored training targets')
        self.alpha=alpha;self.rank_weight=rank_weight;self.uids=None

    def equal_sequence_bce(self, predictions,labels,mask):
        mask=mask.bool()
        # Keep native double BCE numerics, but exclude padding before reduction.
        losses=torch.zeros_like(predictions,dtype=torch.double)
        losses[mask]=F.binary_cross_entropy(predictions[mask].double(),labels[mask].double(),reduction='none')
        denominators=predictions.new_tensor([self.counts[int(u)] for u in self.uids],dtype=torch.double)
        return (losses.sum(1)/denominators).mean()*(self.windows/len(self.counts))

    def sequence_weighted_fragment_rank(self,predictions,labels,mask):
        total=predictions.sum()*0.
        if not self.pairs:return total
        scores=torch.logit(predictions.double().clamp(1e-6,1-1e-6))
        for uid,score,y,valid in zip(self.uids,scores,labels,mask.bool()):
            positive=score[valid&(y==1)];negative=score[valid&(y==0)]
            if positive.numel() and negative.numel():
                total=total+F.softplus(negative[None,:]-positive[:,None]).sum()/self.pairs[int(uid)]
        return total*(self.windows/(len(self.uids)*len(self.pairs)))

    def metadata(self):
        value={'alpha':self.alpha,'rank_weight':self.rank_weight,'training_windows':self.windows,
                'training_orirows':len(self.counts),'scored_targets':sum(self.counts.values()),
                'rank_eligible_orirows':len(self.pairs),'target_counts':self.counts,
                'within_fragment_pair_counts':self.pairs,'ranking_scope':'Complete UID weighting of within-fragment pairs; cross-fragment pairs not included'}
        if getattr(self,'frequency',None) is not None:value['frequency']=self.frequency.metadata()
        if getattr(self,'tilt',None) is not None:value['tilt']=self.tilt.metadata()
        if getattr(self,'complete',None) is not None:
            value['complete_uid']=self.complete.metadata()
            value['ranking_scope']='Complete scored UID pairs including cross-fragment pairs'
            value['rank_eligible_orirows']=len(self.complete.eligible)
        if getattr(self,'gradient_cache',None) is not None:
            value['gradient_cache']=self.gradient_cache.metadata()
            value['ranking_scope']='Epoch-lag complete-UID gradient anchors with original fragment sampling'
            value['rank_eligible_orirows']=len(self.gradient_cache.eligible)
        return value


def install_training_objective(model,dataset,protocol):
    if hasattr(model,'_sequence_objective'):raise ValueError('Objective already installed')
    objective=SequenceObjective(dataset,float(protocol.get('alpha',0)),float(protocol.get('rank_weight',0)))
    objective.frequency=None
    objective.tilt=None
    objective.complete=None
    objective.gradient_cache=None
    if protocol.get('frequency_alpha',0):
        from .qikt_frequency_objective import FrequencyObjective
        if objective.alpha or (objective.rank_weight and not protocol.get('allow_frequency_rank_combination',False)):raise ValueError('Frequency/rank combination requires explicit discovery opt-in')
        objective.frequency=FrequencyObjective(dataset,float(protocol['frequency_alpha']),float(protocol.get('frequency_cap',2)),float(protocol.get('frequency_power',.25)))
    if protocol.get('tilt_mix',0):
        from .qikt_tilt_objective import TiltObjective
        if objective.alpha or objective.rank_weight or objective.frequency is not None:raise ValueError('A3 isolates bounded tilt')
        objective.tilt=TiltObjective(objective,float(protocol['tilt_mix']),float(protocol.get('tilt_temperature',.5)),float(protocol.get('tilt_cap',2)))
        objective.prepare_epoch=objective.tilt.prepare_epoch
    if protocol.get('full_uid_batch_size'):
        from .qikt_complete_uid_objective import CompleteUIDObjective
        sampler=getattr(dataset,'_qikt_complete_uid_sampler',None)
        if sampler is None or sampler.uids_per_batch!=protocol['full_uid_batch_size']:
            raise ValueError('Complete-UID objective requires its matching batch sampler')
        if objective.alpha or objective.frequency is not None or objective.tilt is not None:
            raise ValueError('Complete-UID discovery keeps native BCE; no mixed coverage or tilt')
        objective.complete=CompleteUIDObjective(dataset,sampler,protocol,objective.counts)
    if protocol.get('epoch_rank_cache'):
        from .qikt_rank_gradient_cache import RankGradientCache
        if objective.alpha or objective.frequency is not None or objective.tilt is not None or objective.complete is not None:
            raise ValueError('Cached rank discovery keeps native BCE and original fragment sampling')
        objective.gradient_cache=RankGradientCache(dataset,protocol,objective.counts)
        if objective.rank_weight:objective.prepare_epoch=objective.gradient_cache.prepare_epoch
    original_loss=model.get_loss;original_step=model.train_one_step
    def loss(self,ys,rshft,sm):
        native=original_loss(ys,rshft,sm)
        if objective.tilt is not None:return (1-objective.tilt.mix)*native+objective.tilt.mix*objective.tilt.weighted_bce(ys,rshft,sm)
        if objective.frequency is not None:
            alpha=objective.frequency.alpha
            return (1-alpha)*native+alpha*objective.frequency.weighted_bce(ys,rshft,sm)
        if not objective.alpha:return native
        return (1-objective.alpha)*native+objective.alpha*objective.equal_sequence_bce(ys,rshft,sm)
    def train_step(self,data,process=True,return_all=False):
        objective.uids=data['uid'].detach().cpu().tolist()
        clean={k:v for k,v in data.items() if k!='_fragment_index'} if '_fragment_index' in data else data
        if objective.frequency is not None:
            current=self.batch_to_device(data,process=process)
            objective.frequency.context=objective.frequency.event_weights(current['cshft'])
        try:
            prediction,value=original_step(clean,process=process,return_all=return_all)
            if objective.rank_weight:
                current=self.batch_to_device(clean,process=process)
                if objective.gradient_cache is not None:
                    ranking=objective.gradient_cache.batch_surrogate(prediction,current['sm'],data['_fragment_index'],objective.uids)
                else:
                    ranking=(objective.complete.rank_loss(prediction,current['rshft'],current['sm'],objective.uids)
                             if objective.complete is not None else objective.sequence_weighted_fragment_rank(prediction,current['rshft'],current['sm']))
                value=value+objective.rank_weight*ranking
            return prediction,value
        finally:
            objective.uids=None
            if objective.frequency is not None:objective.frequency.context=None
    model.get_loss=types.MethodType(loss,model);model.train_one_step=types.MethodType(train_step,model)
    model._sequence_objective=objective
    return objective
