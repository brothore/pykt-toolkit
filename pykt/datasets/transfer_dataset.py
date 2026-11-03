"""Future independent-performance supervision; no new model inputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .first_action_dataset import FirstActionDataset, build_first_action_sidecar


MODES = ('same_unassisted', 'same_all', 'cross_all', 'next_event')


def build_transfer_sidecar(directory, pair_mode='same_unassisted', horizon=10):
    import fcntl
    if pair_mode not in MODES or not isinstance(horizon, int) or not 1 <= horizon <= 50:
        raise ValueError('Invalid transfer pairing configuration')
    directory = Path(directory).resolve()
    event_path = build_first_action_sidecar(directory)
    sources = json.loads(event_path.with_suffix('.json').read_text())['sources']
    output = directory/'dkt_transfer'; output.mkdir(exist_ok=True)
    archive = output/f'{pair_mode}_h{horizon}_v1.npz'; manifest=archive.with_suffix('.json')
    with (output/'.build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if archive.exists() and manifest.exists() and json.loads(manifest.read_text()).get('sources') == sources:
            return archive
        with np.load(event_path, allow_pickle=False) as data:
            events,uids,folds=data['events'].copy(),data['uids'].copy(),data['folds'].copy()
        mapping=json.loads((directory/'keyid2idx.json').read_text())
        allowed=set(uids.tolist());inverse={str(raw):int(mapped) for raw,mapped in mapping['uid'].items() if int(mapped) in allowed}
        raw=pd.read_csv(directory/'skill_builder_data_corrected_collapsed.csv',dtype=str).dropna(
            subset=['user_id','problem_id','skill_id','correct','order_id'])
        raw=raw[raw.user_id.isin(inverse)].copy()
        raw['_uid']=raw.user_id.map(inverse);raw['_order']=raw.order_id.astype(np.int64)
        if raw['_order'].duplicated().any(): raise ValueError('Ambiguous transfer event identity')
        first=pd.to_numeric(raw.first_action,errors='coerce');hints=pd.to_numeric(raw.hint_count,errors='coerce')
        attempts=pd.to_numeric(raw.attempt_count,errors='coerce')
        raw['_independent']=(first==0)&(hints==0)&(attempts>=1)&(attempts==np.floor(attempts))
        raw['_correct']=raw.correct.astype(int)
        raw['_success']=(raw['_correct']==1)&raw['_independent']
        raw['_template']=pd.to_numeric(raw.template_id,errors='coerce')
        # Max concept count comes from actual mapping, not target labels.
        max_concepts=int(mapping.get('max_concepts',4))
        pairs={};event_metadata={}
        for uid,group in raw.groupby('_uid',sort=False):
            group=group.sort_values('_order',kind='stable');records=group.to_dict('records')
            q=[int(mapping['questions'][row['problem_id']]) for row in records]
            concepts=[tuple(sorted(set(int(mapping['concepts'][c]) for c in row['skill_id'].split('_')))) for row in records]
            if any(len(c)>max_concepts for c in concepts): raise ValueError('Future concept count exceeds mapping')
            for i,row in enumerate(records):
                event=int(row['_order']);event_metadata[event]=(int(uid),q[i])
                for j in range(i+1,min(len(records),i+horizon+1)):
                    future=records[j]
                    if pair_mode=='next_event':eligible=j==i+1
                    elif not(np.isfinite(row['_template']) and np.isfinite(future['_template'])):eligible=False
                    elif pair_mode=='cross_all':
                        eligible=q[j]!=q[i] and row['_template']!=future['_template'] and concepts[j]==concepts[i]
                    else:
                        eligible=q[j]!=q[i] and row['_template']==future['_template']
                        if pair_mode=='same_unassisted':eligible=eligible and future['_independent']
                    if eligible:
                        label=float(future['_correct'] if pair_mode=='same_unassisted' else future['_success'])
                        pairs[event]=(int(future['_order']),concepts[j],label,j-i,q[j]);break
        shape=events[:,:-1].shape
        future_concepts=np.full((*shape,max_concepts),-1,dtype=np.int64)
        labels=np.zeros(shape,dtype=np.float32);mask=np.zeros(shape,dtype=bool)
        target_events=np.full(shape,-1,dtype=np.int64);distances=np.zeros(shape,dtype=np.int64)
        for i,row in enumerate(events):
            for j,event in enumerate(row[:-1]):
                if event<0 or row[j+1]<0 or event==row[j+1] or int(event) not in pairs:continue
                target,cs,label,distance,target_q=pairs[int(event)]
                # History is anchored after the complete current event; its
                # future label never becomes an input at this position.
                assert event_metadata[int(event)][0]==int(uids[i])
                future_concepts[i,j,:len(cs)]=cs;labels[i,j]=label;mask[i,j]=True
                target_events[i,j]=target;distances[i,j]=distance
        with archive.with_suffix('.npz.tmp').open('wb') as stream:
            np.savez_compressed(stream,concepts=future_concepts,labels=labels,mask=mask,
                anchors=events[:,:-1],targets=target_events,distances=distances,uids=uids,folds=folds)
        archive.with_suffix('.npz.tmp').replace(archive)
        stored={'version':1,'sources':sources,'pair_mode':pair_mode,'horizon_original_events':horizon,
                'max_concepts':max_concepts,'raw_events':len(raw),'raw_observed_unassisted':int(raw['_independent'].sum()),
                'independent_success_differs_from_correct':int((raw['_success'].astype(int)!=raw['_correct']).sum()),
                'eligible_anchors':int(mask.sum()),'positive_rate':float(labels[mask].mean()) if mask.any() else None,
                'mean_future_distance':float(distances[mask].mean()) if mask.any() else None,
                'note':'Same student only; input anchor is LAST concept of completed event; future fields used only as loss labels/query indices.'}
        temporary=manifest.with_suffix('.json.tmp');temporary.write_text(json.dumps(stored,indent=2)+'\n');temporary.replace(manifest)
    return archive


class TransferDataset(Dataset):
    def __init__(self,file_path,input_type,folds,pair_mode='same_unassisted',horizon=10,mode='train'):
        source=Path(file_path).resolve()
        aligned=FirstActionDataset(source,input_type,folds,mode=mode,aug_probs={});self.base=aligned.base
        archive=build_transfer_sidecar(source.parent,pair_mode,horizon);self.pair_mode=pair_mode
        with np.load(archive,allow_pickle=False) as data:
            selected=np.isin(data['folds'],list(folds))
            if not np.array_equal(data['uids'][selected],np.asarray(self.base.dori['uid'])):
                raise ValueError('Transfer target student order mismatch')
            self.concepts=torch.from_numpy(data['concepts'][selected].copy())
            self.labels=torch.from_numpy(data['labels'][selected].copy())
            self.mask=torch.from_numpy(data['mask'][selected].copy())
            self.anchor_events=torch.from_numpy(data['anchors'][selected].copy())
            self.target_events=torch.from_numpy(data['targets'][selected].copy())
            self.distances=torch.from_numpy(data['distances'][selected].copy())
        self.mask &= self.base.dori['masks'] & self.base.dori['smasks']
        self.stats={'anchors':int(self.mask.sum()),'positive_rate':float(self.labels[self.mask].mean()) if self.mask.any() else None,
                    'mean_original_event_distance':float(self.distances[self.mask].float().mean()) if self.mask.any() else None,
                    'uids':len(set(self.base.dori['uid'])),'mode':pair_mode,'horizon':horizon}

    @property
    def dori(self):return self.base.dori

    def __len__(self):return len(self.base)

    def __getitem__(self,index):
        data=self.base[index]
        data.update(future_concepts=self.concepts[index],future_labels=self.labels[index],future_mask=self.mask[index])
        return data
