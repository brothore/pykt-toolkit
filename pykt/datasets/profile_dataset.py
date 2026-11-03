"""Observed NIPS assessment sessions and exposure-weighted parent outcomes.

Only complete sessions within the SAME original chunk receive supervision.
The original state immediately BEFORE the session predicts every session target.
"""
import ast
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from .confidence_dataset import ConfidenceDataset,build_confidence_sidecar
from .first_action_dataset import _signature

MODES=('score','hierarchy','parent','pseudo')


def aggregate_targets(concepts,responses,parent_map,num_c):
    """Each event has unit exposure, shared across its unique leaf concepts."""
    exposure=np.zeros(num_c,dtype=np.float64);correct=np.zeros(num_c,dtype=np.float64)
    for cs,r in zip(concepts,responses):
        cs=np.asarray(sorted(set(cs)),dtype=int)
        exposure[cs]+=1/len(cs);correct[cs]+=float(r)/len(cs)
    n=len(responses)
    result=[(0,exposure/n,float(np.sum(responses)/n),1.)]
    for parent in sorted(set(parent_map)):
        chosen=np.asarray(parent_map)==parent
        mass=exposure[chosen].sum()
        if mass:
            weights=np.where(chosen,exposure,0)/mass
            result.append((1,weights,float(correct[chosen].sum()/mass),float(mass/n)))
    return result


def candidate_sessions(events):
    """A candidate is a contiguous QuizId/day run; ambiguous runs are removed."""
    runs=[];start=0
    for i in range(1,len(events)+1):
        if i==len(events) or events[i]['key']!=events[start]['key']:
            runs.append((start,i));start=i
    multiplicity=Counter(events[a]['key'] for a,b in runs)
    eligible=[];excluded=Counter()
    for a,b in runs:
        block=events[a:b];reason=None
        if not block[0]['valid_quiz']:reason='missing_quiz'
        elif multiplicity[block[0]['key']]!=1:reason='noncontiguous_revisit'
        elif a==0:reason='no_prior_history'
        elif b==len(events):reason='right_censored_user_tail'
        elif len(block)<2:reason='singleton'
        elif len({e['q'] for e in block})!=len(block):reason='duplicate_question'
        elif events[a-1]['ts']>=block[0]['ts'] or block[-1]['ts']>=events[b]['ts']:reason='tied_session_boundary'
        elif any(block[j]['ts']-block[j-1]['ts']>1800000 for j in range(1,len(block))):reason='internal_gap_over_30min'
        if reason:excluded[reason]+=1
        else:eligible.append((a,b))
    return eligible,dict(excluded)


def build_profile_sidecar(directory):
    import fcntl
    directory=Path(directory).resolve();out=directory/'dkt_profile';out.mkdir(exist_ok=True)
    archive,manifest=out/'profiles_v1.npz',out/'profiles_v1.json'
    aligned=build_confidence_sidecar(directory)
    sources=[directory/'train_task_3_4.csv',directory/'metadata/answer_metadata_task_3_4.csv',
             directory/'metadata/question_metadata_task_3_4.csv',directory/'metadata/subject_metadata.csv',
             directory/'keyid2idx.json',directory/'train_valid.csv',directory/'train_valid_sequences.csv',aligned]
    signatures={str(p.relative_to(directory)):_signature(p) for p in sources}
    with (out/'.build.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if archive.exists() and manifest.exists() and json.loads(manifest.read_text()).get('sources')==signatures:
            return archive
        mapping=json.loads((directory/'keyid2idx.json').read_text());num_c=len(mapping['concepts'])
        subjects=pd.read_csv(directory/'metadata/subject_metadata.csv');parents=dict(zip(subjects.SubjectId.astype(int),subjects.ParentId))
        true_parent=np.full(num_c,-1,dtype=np.int64)
        for raw,c in mapping['concepts'].items():
            parent=parents[int(raw)]
            if pd.isna(parent):raise ValueError('Missing leaf parent')
            true_parent[int(c)]=int(parent)
        if (true_parent<0).any():raise ValueError('Incomplete parent map')
        # Fixed partition control: permute concept assignments, preserving parent group sizes.
        pseudo_parent=true_parent[np.random.default_rng(20261027).permutation(num_c)]
        with np.load(aligned,allow_pickle=False) as f:stored={k:f[k].copy() for k in ('events','uids','folds')}
        original=pd.read_csv(directory/'train_valid.csv');allowed=set(original.uid.astype(int))
        uids={int(raw):int(mapped) for raw,mapped in mapping['uid'].items() if int(mapped) in allowed}
        raw=pd.read_csv(directory/'train_task_3_4.csv',usecols=['UserId','QuestionId','AnswerId','IsCorrect'])
        raw['_raw_index']=np.arange(len(raw));raw=raw[raw.UserId.isin(uids)].copy()
        answers=pd.read_csv(directory/'metadata/answer_metadata_task_3_4.csv',usecols=['AnswerId','DateAnswered','QuizId'])
        if answers.AnswerId.duplicated().any() or raw.AnswerId.duplicated().any():raise ValueError('Ambiguous AnswerId')
        raw=raw.merge(answers,on='AnswerId',how='left',validate='one_to_one',indicator=True,sort=False)
        if not raw['_merge'].eq('both').all():raise ValueError('Missing metadata')
        raw['_uid']=raw.UserId.map(uids);raw['_ts']=pd.to_datetime(raw.DateAnswered,errors='raise').dt.as_unit('ms').astype('int64')
        questions=pd.read_csv(directory/'metadata/question_metadata_task_3_4.csv')
        qconcepts={int(row.QuestionId):[int(mapping['concepts'][str(int(c))]) for c in ast.literal_eval(row.SubjectId) if str(int(c)) in mapping['concepts']] for row in questions.itertuples(index=False)}
        blocks={};excluded=Counter();event_to_block={};block_id=0
        for uid,group in raw.groupby('_uid',sort=False):
            group=group.sort_values(['_ts','_raw_index'],kind='stable');events=[]
            for row in group.to_dict('records'):
                quiz=row['QuizId'];valid=pd.notna(quiz) and float(quiz).is_integer()
                ts=int(row['_ts'])
                events.append({'event':int(row['AnswerId']),'q':int(row['QuestionId']),'cs':qconcepts[int(row['QuestionId'])],
                               'r':int(row['IsCorrect']),'ts':ts,'key':(int(quiz) if valid else -1,ts//86400000),'valid_quiz':valid})
            eligible,reasons=candidate_sessions(events);excluded.update(reasons)
            for a,b in eligible:
                block=events[a:b];ids=[e['event'] for e in block]
                blocks[block_id]={'uid':int(uid),'ids':ids,'sizes':{e['event']:len(set(e['cs'])) for e in block},'previous':events[a-1]['event'],
                    'score_parent':aggregate_targets([e['cs'] for e in block],[e['r'] for e in block],true_parent,num_c),
                    'pseudo':aggregate_targets([e['cs'] for e in block],[e['r'] for e in block],pseudo_parent,num_c)}
                for event in ids:event_to_block[event]=block_id
                block_id+=1
        rows=[];anchors=[];kinds=[];weights=[];labels=[];masses=[];block_ids=[];lengths=[];seen=set();missing=Counter()
        for row_index,event_row in enumerate(stored['events']):
            valid_events=event_row[event_row>=0];positions={int(e):np.flatnonzero(valid_events==e) for e in dict.fromkeys(valid_events)}
            candidates={event_to_block[int(e)] for e in positions if int(e) in event_to_block}
            for bid in sorted(candidates):
                block=blocks[bid];ids=block['ids'];first=positions.get(ids[0]);last=positions.get(ids[-1])
                if first is None or last is None or any(e not in positions or len(positions[e])!=block['sizes'][e] for e in ids):missing['cross_chunk_or_partial']+=1;continue
                a=int(first[0]);b=int(last[-1])
                if a==0 or int(valid_events[a-1])!=block['previous']:missing['no_complete_prefix_in_chunk']+=1;continue
                if b+1<len(valid_events) and valid_events[b+1]==ids[-1]:raise ValueError('Unclosed event')
                # Raw block cardinality must equal the chunk's unique target events.
                if list(dict.fromkeys(valid_events[a:b+1]))!=ids:raise ValueError('Noncontiguous block alignment')
                if int(stored['uids'][row_index])!=block['uid']:raise ValueError('UID mismatch')
                anchor=a-1
                if anchor>=len(event_row)-1:raise ValueError('Anchor outside model outputs')
                for namespace,key in [(0,'score_parent'),(2,'pseudo')]:
                    for kind,exposure,label,mass in block[key]:
                        if namespace==2 and kind==0:continue
                        rows.append(row_index);anchors.append(anchor);kinds.append(kind if namespace==0 else 2)
                        weights.append(exposure);labels.append(label);masses.append(mass);block_ids.append(bid);lengths.append(len(ids))
                seen.add(bid)
        np.savez_compressed(archive,rows=np.asarray(rows,dtype=np.int64),anchors=np.asarray(anchors,dtype=np.int64),
            kinds=np.asarray(kinds,dtype=np.int64),concepts=np.asarray(weights,dtype=np.float32),labels=np.asarray(labels,dtype=np.float32),
            masses=np.asarray(masses,dtype=np.float32),blocks=np.asarray(block_ids,dtype=np.int64),lengths=np.asarray(lengths,dtype=np.int64),
            folds=stored['folds'],uids=stored['uids'])
        metadata={'version':1,'sources':signatures,'raw_train_valid_events':len(raw),'candidate_sessions':len(blocks),
                  'eligible_complete_chunk_sessions':len(seen),'excluded_raw_runs':dict(excluded),'excluded_chunk_occurrences':dict(missing),
                  'true_parent_map':true_parent.tolist(),'pseudo_parent_map':pseudo_parent.tolist(),'pseudo_seed':20261027,
                  'num_parents':len(set(true_parent)),'records':len(rows),'min_events':2,'max_internal_gap_minutes':30,
                  'definition':'Contiguous student QuizId/day run, no revisits/duplicates, strictly earlier prefix, complete within chunk; last user run excluded'}
        manifest.write_text(json.dumps(metadata,indent=2)+'\n')
    return archive


class ProfileDataset(Dataset):
    def __init__(self,file_path,input_type,folds,profile_mode='score',mode='train'):
        if profile_mode not in MODES:raise ValueError('Unknown profile mode')
        source=Path(file_path).resolve();aligned=ConfidenceDataset(source,input_type,folds,mode=mode)
        self.base=aligned.base;self.profile_mode=profile_mode
        archive=build_profile_sidecar(source.parent)
        with np.load(archive,allow_pickle=False) as f:values={k:f[k].copy() for k in f.files}
        chosen_rows=np.flatnonzero(np.isin(values['folds'],list(folds)))
        if not np.array_equal(values['uids'][chosen_rows],np.asarray(self.base.dori['uid'])):raise ValueError('UID order mismatch')
        kind=values['kinds'];keep={'score':kind==0,'hierarchy':kind<2,'parent':kind==1,'pseudo':(kind==0)|(kind==2)}[profile_mode]
        indices=np.flatnonzero(keep);byrow={}
        for i in indices:byrow.setdefault(int(values['rows'][i]),[]).append(i)
        self.records=[];all_indices=[]
        for local,row in enumerate(chosen_rows):
            ix=np.asarray(byrow.get(int(row),[]),dtype=np.int64);all_indices.extend(ix.tolist())
            masses=values['masses'][ix].copy()
            if profile_mode in ('hierarchy','pseudo'):masses*=.5
            self.records.append({'profile_anchors':torch.from_numpy(values['anchors'][ix]),
                'profile_concepts':torch.from_numpy(values['concepts'][ix]),'profile_labels':torch.from_numpy(values['labels'][ix]),
                'profile_mass':torch.from_numpy(masses),'profile_mask':torch.ones(len(ix),dtype=torch.bool),
                'profile_kind':torch.from_numpy(values['kinds'][ix]),'profile_blocks':torch.from_numpy(values['blocks'][ix])})
            anchors=values['anchors'][ix]
            if len(ix) and not (self.base.dori['masks'][local][anchors]&self.base.dori['smasks'][local][anchors]).all():
                raise ValueError('Inactive KT anchor')
        ix=np.asarray(all_indices,dtype=np.int64)
        self.stats={'sequences':len(self.base),'groups':len(ix),'sessions':len(set(values['blocks'][ix].tolist())),
                    'mean_session_events':float(values['lengths'][ix][values['kinds'][ix]==0].mean()) if (values['kinds'][ix]==0).any() else None,
                    'mass':float(sum(float(r['profile_mass'].sum()) for r in self.records)),
                    'students':len(set(values['uids'][chosen_rows].tolist()))}

    @property
    def dori(self):return self.base.dori
    def __len__(self):return len(self.base)
    def __getitem__(self,index):
        data=self.base[index];data.update(self.records[index]);return data


def profile_collate(batch):
    profile_keys=[k for k in batch[0] if k.startswith('profile_')]
    result=default_collate([{k:v for k,v in row.items() if k not in profile_keys} for row in batch])
    for key in profile_keys:result[key]=pad_sequence([row[key] for row in batch],batch_first=True,padding_value=0)
    return result
