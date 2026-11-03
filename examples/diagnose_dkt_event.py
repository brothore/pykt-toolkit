"""Isolated diagnostic loss: one BCE per original question at its PRE-event state."""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from pykt.models.dkt_joint import DKTJoint
from pykt.datasets.joint_dataset import joint_collate


def event_fields(source,folds,max_concepts):
    source=__import__('pathlib').Path(source);full=pd.read_csv(source.parent/'train_valid.csv');repeat_full={int(row.uid):[int(x) for x in row.is_repeat.split(',')] for row in full.itertuples(index=False)}
    chunks=pd.read_csv(source);cursors={};records=[];repeat_rows=[];stats={'complete_target_questions':0,'excluded_partial_target_tails':0}
    for row in chunks.itertuples(index=False):
        uid=int(row.uid);c=np.asarray([int(x) for x in row.concepts.split(',')]);r=np.asarray([int(x) for x in row.responses.split(',')]);rep=np.asarray([int(x) for x in row.is_repeat.split(',')]);n=int((r!=-1).sum());offset=cursors.get(uid,0);cursors[uid]=offset+n
        assert rep[:n].tolist()==repeat_full[uid][offset:offset+n]
        if int(row.fold) not in folds:continue
        anchors=[];concepts=[];labels=[];starts=np.flatnonzero(rep[:n]==0)
        for j,a in enumerate(starts):
            b=int(starts[j+1]) if j+1<len(starts) else n
            if not a:continue
            if b==n and offset+n<len(repeat_full[uid]) and repeat_full[uid][offset+n]==1:
                stats['excluded_partial_target_tails']+=1;continue
            cs=sorted(set(c[a:b].tolist()));assert len(cs)<=max_concepts and (r[a:b]==r[a]).all()
            anchors.append(int(a-1));concepts.append(cs+[-1]*(max_concepts-len(cs)));labels.append(int(r[a]))
        records.append({'safe_anchors':torch.tensor(anchors,dtype=torch.long),'safe_concepts':torch.tensor(concepts,dtype=torch.long).reshape(-1,max_concepts),
                        'safe_labels':torch.tensor(labels,dtype=torch.float32),'safe_mask':torch.ones(len(anchors),dtype=torch.bool)})
        repeat_rows.append(torch.tensor(rep[1:],dtype=torch.long));stats['complete_target_questions']+=len(anchors)
    return records,repeat_rows,stats


class EventDataset(Dataset):
    def __init__(self,base,source,folds,max_concepts):
        self.base=base;self.records,self.repeat_rows,self.stats=event_fields(source,folds,max_concepts)
        assert len(base)==len(self.records)
    def __len__(self):return len(self.base)
    def __getitem__(self,index):
        data=self.base[index];data.update(self.records[index]);data['target_repeat']=self.repeat_rows[index];return data


def event_collate(batch):
    keys=[k for k in batch[0] if k.startswith('safe_')]
    result=joint_collate([{k:v for k,v in row.items() if k not in keys} for row in batch])
    for key in keys:result[key]=pad_sequence([row[key] for row in batch],batch_first=True,padding_value=-1 if key=='safe_concepts' else 0)
    return result


def event_probabilities(output,data):
    device=output.device;anchors=data['safe_anchors'].to(device);cs=data['safe_concepts'].to(device);mask=data['safe_mask'].to(device)
    if (mask&((anchors<0)|(anchors>=output.shape[1]))).any():raise ValueError('Invalid event anchor')
    if (mask.unsqueeze(-1)&((cs< -1)|(cs>=output.shape[-1]))).any():raise ValueError('Invalid event concept')
    valid=cs>=0
    if (mask&~valid.any(-1)).any():raise ValueError('Empty active event concept set')
    queried=output.gather(1,anchors.unsqueeze(-1).expand(-1,-1,output.shape[-1])).gather(-1,cs.clamp_min(0))
    p=(queried.double()*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    return torch.where(mask,p,torch.full_like(p,.5)),mask


class DiagnosticJoint(DKTJoint):
    """No new modules, params or inference features. Old models left unchanged."""
    def __init__(self,*args,primary_mode='original',**kwargs):
        super().__init__(*args,**kwargs)
        if primary_mode not in ('original','event'):raise ValueError('Unknown primary mode')
        self.primary_mode=primary_mode
    def loss_components(self,data,return_per_sample=False):
        if self.primary_mode=='original':return super().loss_components(data,return_per_sample)
        captured=[];hook=self.out_layer.register_forward_hook(lambda module,args,output:captured.append(output))
        try:parts=super().loss_components(data,return_per_sample)
        finally:hook.remove()
        assert len(captured)==1
        p,mask=event_probabilities(torch.sigmoid(captured[0]),data);y=data['safe_labels'].to(p.device).double()
        loss=F.binary_cross_entropy(p,y,reduction='none')
        parts['response']=(loss*mask).sum(-1)/mask.sum(-1).clamp_min(1) if return_per_sample else (loss*mask).sum()/mask.sum().clamp_min(1)
        return parts
