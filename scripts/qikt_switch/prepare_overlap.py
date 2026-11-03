"""Training-only overlap windows preserving the baseline scored event set.
No validation/test labels or dataset files are rewritten.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from audit_inputs import sha


def build(raw, fragments, valid_fold=0, length=200, overlap=64):
    if not 0 < overlap < length-1:
        raise ValueError('overlap must leave at least two target slots')
    raw=raw[raw.fold!=valid_fold].set_index('uid')
    assert raw.index.is_unique
    rows=[];mapping=[];total_baseline=0;context_events=0
    fields=['questions','concepts','responses']+[c for c in ['timestamps','usetimes'] if c in raw]
    for uid,g in fragments[fragments.fold!=valid_fold].groupby('uid',sort=False):
        source=raw.loc[uid];seq={k:str(source[k]).split(',') for k in fields};n=len(seq['questions'])
        assert all(len(v)==n for v in seq.values())
        selected=set();offset=0
        for _,part in g.iterrows():
            values={k:str(part[k]).split(',') for k in fields};masks=str(part.selectmasks).split(',')
            size=sum(v!='-1' for v in values['responses'])
            for k in fields:assert values[k][:size]==seq[k][offset:offset+size],(uid,k,offset)
            selected.update(offset+j for j in range(1,size) if masks[j]!='-1')
            offset+=size
        # Baseline preprocessing can drop a short final tail; preserve its actual set.
        targets=sorted(selected);total_baseline+=len(targets);cursor=0
        while cursor<len(targets):
            first=targets[cursor];start=max(0,first-overlap)
            end=min(n,start+length)
            active=[]
            while cursor<len(targets) and targets[cursor]<end:
                active.append(targets[cursor]);cursor+=1
            assert active and first>start
            active_set=set(active);width=end-start
            item={'fold':int(source.fold),'uid':int(uid)}
            for k in fields:item[k]=','.join(seq[k][start:end]+['-1']*(length-width))
            item['selectmasks']=','.join(['1' if i in active_set else '-1' for i in range(start,end)]+['-1']*(length-width))
            for qidx in active:mapping.append({'window':len(rows),'uid':int(uid),'qidx':qidx,'local_position':qidx-start})
            context_events+=width-len(active);rows.append(item)
    aligned=pd.DataFrame(mapping)
    assert not aligned.duplicated(['uid','qidx']).any()
    assert len(aligned)==total_baseline
    return pd.DataFrame(rows),aligned,{'baseline_scored_events':total_baseline,'overlap_scored_events':len(aligned),'overlap_windows':len(rows),'context_only_events':context_events,'validation_fold':valid_fold,'max_length':length,'overlap':overlap,'target_sets_equal':True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',default='nips_task34');ap.add_argument('--fold',type=int,default=0);ap.add_argument('--overlap',type=int,default=64);args=ap.parse_args()
    root=Path(__file__).resolve().parents[2];data=root/'data'/args.dataset;out=root/f'logs/qikt_switch_20261103/overlap_{args.dataset}_fold{args.fold}_{args.overlap}';out.mkdir(exist_ok=True)
    raw=data/'train_valid_quelevel.csv';fragments=data/'train_valid_sequences_quelevel.csv'
    windows,mapping,info=build(pd.read_csv(raw),pd.read_csv(fragments),args.fold,overlap=args.overlap)
    target=out/'train_sequences.csv';windows.to_csv(target,index=False);mapping.to_csv(out/'scored_positions.csv',index=False)
    info.update(dataset=args.dataset,raw_sha256=sha(raw),baseline_fragments_sha256=sha(fragments),generated_sha256=sha(target),source_sha256=sha(Path(__file__)),test_loaded=False,validation_rows_written=0)
    (out/'manifest.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)

if __name__=='__main__':main()
