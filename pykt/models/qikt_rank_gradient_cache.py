"""Epoch-lag full-UID rank gradients with unchanged native fragment batches.

The surrogate is a bounded first-order approximation, not the current exact
ranking risk. Anchors use training-only eval predictions; no teacher or OOF.
"""
import hashlib
import json
import math
import time
import datetime
from pathlib import Path
import shutil
import numpy as np
from scipy.special import expit
import torch


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value));temporary.replace(path)


def rank_risk_gradient(logits,labels,mode='smooth_auc',temperature=.5):
    """Exact derivative wrt supplied logits, with bounded temporary matrices."""
    z=np.asarray(logits,dtype=np.float64);y=np.asarray(labels)
    if mode not in ('logistic','smooth_auc') or not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Invalid rank surrogate')
    if not np.isfinite(z).all() or not np.isin(y,[0,1]).all():raise ValueError('Invalid scored values')
    positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0);gradient=np.zeros_like(z)
    if not len(positive) or not len(negative):return 0.,gradient
    denom=len(positive)*len(negative);risk=0.
    for start in range(0,len(positive),256):
        indices=positive[start:start+256];difference=z[negative][None,:]-z[indices][:,None]
        if mode=='logistic':
            risk+=np.logaddexp(0,difference).sum();derivative=expit(difference)
        else:
            value=expit(difference/temperature);risk+=value.sum();derivative=value*(1-value)/temperature
        gradient[indices]=-derivative.sum(1)/denom
        gradient[negative]+=derivative.sum(0)/denom
    return float(risk/denom),gradient


def tensor_identity(values):
    h=hashlib.sha256()
    for key,value in sorted(values.items()):
        if isinstance(value,(list,tuple)):
            try:value=torch.as_tensor(value)
            except (ValueError,TypeError):continue
        if not isinstance(value,torch.Tensor):continue
        h.update(str((key,str(value.dtype),tuple(value.shape))).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def risk_focus_weights(risks, mode='off', strength=0.):
    """Bounded detached weights at an epoch anchor, not an OOF teacher."""
    if mode not in {'off', 'exponential', 'variance'} or not math.isfinite(strength) or strength < 0:
        raise ValueError('Invalid risk focus settings')
    if not risks:
        return {}
    keys = list(risks)
    values = np.asarray([risks[k] for k in keys], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite rank risks')
    if mode == 'off' or strength == 0:
        return {k: 1. for k in keys}
    centered = values - values.mean()
    raw = np.exp(np.clip(strength * centered, -50, 50)) if mode == 'exponential' else 1 + strength * centered
    clipped = np.clip(raw, .5, 1.5)
    normalized = np.clip(clipped / clipped.mean(), .5, 1.5)
    return dict(zip(keys, map(float, normalized)))


class RankGradientCache:
    def __init__(self,dataset,protocol,counts):
        if not getattr(dataset,'_qikt_fragment_indices',False):raise ValueError('Stable fragment indices required')
        self.counts=counts;self.windows=len(dataset);self.mode=protocol.get('cached_rank_mode','smooth_auc')
        self.temperature=float(protocol.get('cached_rank_temperature',.5));self.reliability=float(protocol.get('cached_rank_reliability',0.))
        self.radius=float(protocol.get('cached_rank_trust_radius',1.))
        self.focus_mode=protocol.get('cached_rank_focus_mode','off')
        self.focus_strength=float(protocol.get('cached_rank_focus_strength',0.))
        risk_focus_weights({},self.focus_mode,self.focus_strength)
        if self.mode not in ('logistic','smooth_auc') or not all(math.isfinite(v) for v in [self.temperature,self.reliability,self.radius]) or self.temperature<=0 or self.reliability<0 or self.radius<=0:
            raise ValueError('Invalid cached rank protocol')
        self.archive=Path(protocol['gradient_cache_directory'])
        data=dataset.dori;self.uids=torch.as_tensor(data['uid']).cpu().reshape(-1).numpy()
        labels=data['rseqs'][:,1:];mask=(data['rseqs'][:,:-1]!=-1)&(labels!=-1)&(data['smasks'][:,1:]!=-1)
        self.valid=mask.cpu();self.labels=labels.cpu().numpy();self.shape=tuple(labels.shape)
        groups={}
        for i,uid in enumerate(self.uids):groups.setdefault(int(uid),[]).extend((i*self.shape[1]+np.flatnonzero(self.valid[i].numpy())).tolist())
        self.groups={u:np.asarray(ix,dtype=np.int64) for u,ix in groups.items()}
        flat=self.labels.reshape(-1)
        if set(self.groups)!=set(counts) or any(len(ix)!=counts[u] for u,ix in self.groups.items()):
            raise ValueError('Fragment coverage differs from UID target counts')
        if not np.isin(flat[self.valid.numpy().reshape(-1)],[0,1]).all():raise ValueError('Nonbinary active labels')
        self.minority={u:min(int(flat[ix].sum()),len(ix)-int(flat[ix].sum())) for u,ix in self.groups.items()}
        self.eligible={u:n for u,n in self.minority.items() if n>0}
        raw={u:n/(n+self.reliability) for u,n in self.eligible.items()};mean=np.mean(list(raw.values())) if raw else 1.
        self.weights={u:float(np.clip(w/mean,.25,1.5)) for u,w in raw.items()}
        self.data_identity=tensor_identity(data);self.cache=None

    def metadata(self):
        wrapper=Path(__file__).resolve().parents[1]/'datasets/indexed_que_dataset.py'
        return {'scope':'Full scored training UID gradients; original native fragment sampler retained',
                'mode':self.mode,'temperature':self.temperature,'reliability_strength':self.reliability,
                'risk_focus_mode':self.focus_mode,'risk_focus_strength':self.focus_strength,
                'trust_radius_logit':self.radius,'eligible_UIDs':len(self.eligible),'dataset_tensors_sha256':self.data_identity,
                'cache_directory':str(self.archive),'source_sha256':digest(Path(__file__)),'wrapper_sha256':digest(wrapper),
                'precision':'FP32 anchor probabilities/gradient storage; FP64 analytic gradient and batch reduction',
                'limitation':'Epoch-lag eval anchors and clipped linear surrogate; not exact current train-mode AUC risk'}

    def prepare_epoch(self,model,loader,epoch,folder):
        started=time.perf_counter()
        self.archive.mkdir(parents=True,exist_ok=True);local=Path(folder)/'rank_gradient_anchors'
        if not local.exists() and not local.is_symlink():local.symlink_to(self.archive,target_is_directory=True)
        if local.resolve()!=self.archive.resolve():raise ValueError('Gradient archive identity differs')
        path=self.archive/f'epoch_{epoch:03d}.pt';identity={'epoch':epoch,'model_sha256':tensor_identity(model.state_dict()),'metadata':self.metadata(),
            'sampler_type':type(loader.sampler).__name__,'batch_count':len(loader),'batch_size':loader.batch_size}
        if path.exists():
            stored=torch.load(path,map_location='cpu',weights_only=True)
            if stored['identity']!=identity:raise ValueError('Existing gradient anchor identity differs')
            metadata=path.with_suffix('.json')
            if metadata.exists() and json.loads(metadata.read_text())['cache_sha256']!=digest(path):
                raise ValueError('Gradient anchor checksum differs')
            self.cache=stored
            if not metadata.exists():self.write_metadata(path,identity,recovered=True)
            self.write_timing(epoch,path,started,reused=True)
            return
        if shutil.disk_usage(self.archive).free<5*1024**3:raise OSError('Gradient archive has less than 5GiB free')
        anchors=torch.full(self.shape,.5,dtype=torch.float32);seen=torch.zeros(self.windows,dtype=torch.bool)
        model.model.eval()
        with torch.no_grad():
            for batch in loader:
                rows=batch['_fragment_index'].long().cpu()
                if len(torch.unique(rows))!=len(rows) or int(rows.min())<0 or int(rows.max())>=self.windows:
                    raise ValueError('Invalid fragment indices')
                if seen[rows].any():raise ValueError('Duplicate fragment in anchor pass')
                clean={k:v for k,v in batch.items() if k!='_fragment_index'}
                predictions=model.predict_one_step(clean).detach().cpu()
                if tuple(predictions.shape)!=tuple(anchors[rows].shape):raise ValueError('Anchor shape differs')
                anchors[rows]=predictions;seen[rows]=True
        if not seen.all():raise ValueError('Anchor pass missed fragments')
        if not torch.isfinite(anchors[self.valid]).all():raise ValueError('Nonfinite active anchor probabilities')
        anchors=torch.where(self.valid,anchors,torch.full_like(anchors,.5))
        probabilities=anchors.double().clamp(1e-6,1-1e-6);z=torch.logit(probabilities).numpy().reshape(-1)
        coefficients=np.zeros(self.windows*self.shape[1],dtype=np.float32);risks={}
        labels=self.labels.reshape(-1)
        for uid,indices in self.groups.items():
            if uid not in self.eligible:continue
            risk,gradient=rank_risk_gradient(z[indices],labels[indices],self.mode,self.temperature)
            coefficients[indices]=gradient*self.weights[uid];risks[uid]=risk*self.weights[uid]
        focus=risk_focus_weights({u:r/self.weights[u] for u,r in risks.items()},self.focus_mode,self.focus_strength)
        if self.focus_mode != 'off' and self.focus_strength:
            for uid,indices in self.groups.items():
                if uid not in focus:continue
                coefficients[indices]*=focus[uid];risks[uid]*=focus[uid]
        self.cache={'identity':identity,'batch_count':len(loader),'anchor_probabilities':anchors,'gradient':torch.from_numpy(coefficients.reshape(self.shape)),
                    'weighted_risk':risks,'risk_focus_weights':focus}
        temporary=path.with_suffix('.tmp');torch.save(self.cache,temporary);temporary.replace(path)
        self.write_metadata(path,identity)
        self.write_timing(epoch,path,started,reused=False)

    def write_metadata(self,path,identity,recovered=False):
        risks=self.cache['weighted_risk']
        atomic_json(path.with_suffix('.json'),{'identity':identity,'cache_sha256':digest(path),'metadata_recovered_from_cache':recovered,
                    'macro_weighted_rank_risk':float(sum(risks.values())/len(risks)) if risks else 0.,'weighted_UID_risks':risks,
                    'risk_focus_weights':self.cache.get('risk_focus_weights',{}),
                    'model_weights_retained':'Only native current resume and selected checkpoints; intermediate anchor model weights are not all retained.'})

    def write_timing(self,epoch,path,started,reused):
        with (self.archive/f'epoch_{epoch:03d}_timing.jsonl').open('a') as handle:
            handle.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cache_reused':reused,
                'prepare_seconds':time.perf_counter()-started,'cache_sha256':digest(path)})+'\n')

    def batch_surrogate(self,predictions,mask,rows,uids):
        if self.cache is None:raise ValueError('Ranking anchors not prepared')
        rows=torch.as_tensor(rows,dtype=torch.long).cpu();valid=mask.bool()
        if not torch.equal(valid.detach().cpu(),self.valid[rows]):raise ValueError('Batch mask differs from anchor coverage')
        if list(map(int,uids))!=list(map(int,self.uids[rows.numpy()])):raise ValueError('Fragment UID mapping differs')
        safe=torch.where(valid,predictions,torch.full_like(predictions,.5)).double()
        current=torch.logit(safe.clamp(1e-6,1-1e-6))
        anchor=self.cache['anchor_probabilities'][rows].to(predictions.device,dtype=torch.double).clamp(1e-6,1-1e-6)
        anchor=torch.where(valid,anchor,torch.full_like(anchor,.5))
        coefficients=self.cache['gradient'][rows].to(predictions.device,dtype=torch.double)
        change=(current-torch.logit(anchor)).clamp(-self.radius,self.radius)
        constant=sum(self.cache['weighted_risk'].get(int(u),0.)*int(self.valid[i].sum())/self.counts[int(u)] for i,u in zip(rows,uids))
        value=(coefficients*change).sum()+constant
        return value*(self.cache['batch_count']/len(self.eligible)) if self.eligible else predictions[valid].sum()*0.
